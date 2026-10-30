#!/usr/bin/env python3
"""Acquire pinned default-task assets from the repository's source registry.

Downloads retain their upstream terms. This tool verifies external files and
CPU-only derived assets; use prepare_assets.py afterwards to import wrappers
and check renderer dependencies. It never starts a simulator or GPU session.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BLOCK_BYTES = 1024 * 1024
MAX_CACHED_ARCHIVE_BYTES = 1024 * 1024 * 1024
VARIANT_ASSETS = {"franka_mimic.usd", "nut_m16.usd", "bolt_m16.usd"}
AUTHORED = "forcebench_authored_wrapper_or_primitive"
BUILDERS = {
    "tools/asset_builders/build_conveyor_a04.py",
    "tools/asset_builders/build_small_warehouse_rack.py",
    "tools/asset_builders/build_single_stovetop.py",
    "tools/asset_builders/restore_lightwheel_layers.py",
}


def _safe_path(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
            or not root.joinpath(*relative.parts).resolve().is_relative_to(root.resolve())):
        raise ValueError(f"Non-portable or escaping path: {name}")
    return root.joinpath(*relative.parts)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _pin(value, label):
    if (type(value.get("bytes")) is not int or value["bytes"] < 0
            or not isinstance(value.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
        raise ValueError(f"Missing or invalid size/SHA256 pin: {label}")
    return {"bytes": value["bytes"], "sha256": value["sha256"]}


def _approved(row):
    pins = [_pin(row, row["path"])]
    variants = row.get("verified_equivalent_exports", [])
    if variants and row["path"] not in VARIANT_ASSETS:
        raise ValueError(f"Equivalent exports are not approved for {row['path']}")
    for variant in variants:
        if not variant.get("verification_report"):
            raise ValueError(f"Equivalent export lacks a verification report: {row['path']}")
        pins.append(_pin(variant, row["path"]))
    return pins


def _download_pin(row, recipe):
    pins = _approved(row)
    if "bytes" not in recipe and "sha256" not in recipe:
        return pins[0]
    selected = _pin(recipe, row["path"])
    if selected not in pins:
        raise ValueError(f"Source recipe selects an unapproved export: {row['path']}")
    return selected


def _url(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"Source registry requires an HTTPS URL without credentials: {value}")
    return value


def _existing(destination, pins):
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_file():
        size = destination.stat().st_size
        candidates = [pin for pin in pins if pin["bytes"] == size]
        if candidates:
            digest = _sha256(destination)
            if any(pin["sha256"] == digest for pin in candidates):
                return {"bytes": size, "sha256": digest}
    raise ValueError(f"Existing file differs from the approved exports; preserved: {destination}")


def _publish(source, destination, pin, *, approved=None):
    approved = approved or [pin]
    existing = _existing(destination, approved)
    if existing is not None:
        return {"status": "already_present", **existing}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".forcebench-download-", suffix=".part", dir=destination.parent,
                                         delete=False) as output:
            temporary = Path(output.name)
            digest, size = hashlib.sha256(), 0
            while block := source.read(BLOCK_BYTES):
                size += len(block)
                if size > pin["bytes"]:
                    raise ValueError(f"Download exceeds the pinned size: {destination}")
                digest.update(block)
                output.write(block)
            actual = digest.hexdigest()
            if size != pin["bytes"] or actual != pin["sha256"]:
                raise ValueError(f"Size/SHA256 verification failed: {destination}; bytes={size}, sha256={actual}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _existing(destination, approved)
            if existing is None:
                raise
            return {"status": "already_present", **existing}
        return {"status": "downloaded", "bytes": size, "sha256": actual}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _download_url(url, destination, pin, *, approved=None, timeout=60.0):
    existing = _existing(destination, approved or [pin])
    if existing is not None:
        return {"status": "already_present", **existing}
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"Unexpected HTTP {response.status}: {url}")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != pin["bytes"]:
            raise ValueError(f"Unexpected Content-Length: {url}")
        return _publish(response, destination, pin, approved=approved)


def _range_module():
    spec = importlib.util.spec_from_file_location("all_task_range_reader", ROOT / "tools/download_weight_sort_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive_recipe(identifier, registry):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier):
        raise ValueError(f"Invalid archive identifier: {identifier}")
    archive = registry.get("archives", {}).get(identifier)
    if archive is None:
        raise ValueError(f"Missing archive recipe: {identifier}")
    _url(archive["url"])
    if type(archive.get("bytes")) is not int or archive["bytes"] <= 0:
        raise ValueError(f"Invalid archive length: {identifier}")
    if archive.get("access") == "range":
        if not archive.get("etag") or not isinstance(archive["etag"], str):
            raise ValueError(f"Range archive requires a pinned ETag: {identifier}")
    elif archive.get("access") == "cached":
        _pin(archive, identifier)
        if archive["bytes"] > MAX_CACHED_ARCHIVE_BYTES:
            raise ValueError(f"Archive is too large for full caching; require range access: {identifier}")
    else:
        raise ValueError(f"Unknown archive access mode: {identifier}")
    return archive


def _builder(recipe):
    relative = recipe.get("builder")
    if relative not in BUILDERS:
        raise ValueError(f"Builder is not one of the repository's audited asset builders: {relative}")
    path = _safe_path(ROOT, relative)
    expected = recipe.get("builder_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"Repository builder SHA256 differs from its recipe: {relative}")
    return path


def _expand_args(args, values):
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("Build args must be a list of strings")
    expanded = []
    for argument in args:
        for key, value in values.items():
            argument = argument.replace("{" + key + "}", str(value))
        if "{" in argument or "}" in argument:
            raise ValueError(f"Unknown build argument placeholder: {argument}")
        expanded.append(argument)
    return expanded


def _plan(manifest, registry, tasks, source_dir, cache_dir):
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported asset source registry schema_version")
    rows = {row["path"]: row for row in manifest["files"]}
    if len(rows) != len(manifest["files"]):
        raise ValueError("Duplicate file paths in asset manifest")
    if set(tasks) - set(manifest["tasks"]):
        raise ValueError("Unknown task selection")
    selected = {name for task in tasks for name in manifest["tasks"][task]["files"]}
    nodes, build_order, visiting, inputs = {}, [], set(), {}

    def visit(name):
        row = rows[name]
        if not row.get("present") or row.get("source_group") == AUTHORED:
            return
        if name in visiting:
            raise ValueError(f"Cycle in derived asset recipes: {name}")
        if name in nodes:
            return
        visiting.add(name)
        _approved(row)
        destination = _safe_path(source_dir, name)
        recipe = registry.get("files", {}).get(name)
        if recipe is None:
            raise ValueError(f"Missing source recipe for selected asset: {name}")
        pin = _download_pin(row, recipe)
        node = {"row": row, "recipe": recipe, "pin": pin, "destination": destination}
        kind = recipe.get("kind")
        if kind == "url":
            _url(recipe["url"])
        elif kind == "zip":
            _archive_recipe(recipe["archive"], registry)
            _safe_path(cache_dir, recipe["member"])
        elif kind == "build":
            build = registry.get("builds", {}).get(recipe.get("recipe"))
            if build is None:
                raise ValueError(f"Missing build recipe for {name}")
            node["build"] = build
            _builder(build)
            _expand_args(build.get("args"), dict.fromkeys(("input", "input_dir", "output", "metadata"), "placeholder"))
            _safe_path(cache_dir, build.get("output_name", PurePosixPath(name).name))
            if not build.get("inputs"):
                raise ValueError(f"Build recipe has no pinned inputs: {name}")
            node["inputs"] = []
            input_names = set()
            for item in build["inputs"]:
                _safe_path(cache_dir, item["name"])
                if item["name"] in input_names:
                    raise ValueError(f"Duplicate build input name: {item['name']}")
                input_names.add(item["name"])
                sources = [key for key in ("asset", "url", "archive") if key in item]
                if len(sources) != 1:
                    raise ValueError(f"Build input requires exactly one asset, URL or archive: {item['name']}")
                if sources[0] == "asset":
                    asset = item["asset"]
                    input_row = rows[asset]
                    if not input_row.get("present"):
                        raise ValueError(f"Build input is not a pinned payload: {asset}")
                    _approved(input_row)
                    visit(asset)
                    input_path = (_safe_path(ROOT / "assets", asset) if input_row.get("source_group") == AUTHORED
                                  else _safe_path(source_dir, asset))
                    node["inputs"].append({"name": item["name"], "path": input_path, "pins": _approved(input_row)})
                else:
                    input_pin = _pin(item, item["name"])
                    input_path = _safe_path(cache_dir / "inputs" / input_pin["sha256"], item["name"])
                    if sources[0] == "url":
                        _url(item["url"])
                        source = {"kind": "url", "url": item["url"]}
                    else:
                        _archive_recipe(item["archive"], registry)
                        _safe_path(cache_dir, item["member"])
                        source = {"kind": "zip", "archive": item["archive"], "member": item["member"]}
                    inputs[str(input_path)] = {"path": input_path, "pin": input_pin, **source}
                    node["inputs"].append({"name": item["name"], "path": input_path, "pins": [input_pin]})
            build_order.append(name)
        else:
            raise ValueError(f"Unknown source recipe kind for {name}: {kind}")
        visiting.remove(name)
        nodes[name] = node

    for name in sorted(selected):
        visit(name)
    return nodes, build_order, inputs


def _require_usd_core():
    try:
        version = importlib.metadata.version("usd-core")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != "25.11":
        raise RuntimeError(
            f"Derived asset builders require usd-core==25.11 (found {version or 'not installed'}). "
            "Install it in this Python environment: python -m pip install usd-core==25.11"
        )


def _cached_archive_path(cache_dir, identifier, recipe):
    return cache_dir / "archives" / f"{identifier}-{recipe['sha256']}.zip"


def _manual_archive_inputs(archive_files, registry, needed_archives, cache_dir):
    selected = {}
    for identifier, filename in (archive_files or {}).items():
        recipe = _archive_recipe(identifier, registry)
        if recipe["access"] != "cached":
            raise ValueError(f"Manual archive import is limited to cached archives; range archive refused: {identifier}")
        if identifier not in needed_archives:
            raise ValueError(f"Manual archive is unrelated to the selected task plan: {identifier}")
        pin = _pin(recipe, identifier)
        try:
            source = Path(filename).expanduser().resolve()
            if _existing(source, [pin]) is None:
                raise ValueError("File is missing")
        except (OSError, ValueError):
            # Keep the user's private input path out of acquisition receipts.
            raise ValueError(f"Manual archive is missing, unreadable or has the wrong size/SHA256: {identifier}") from None
        _existing(_cached_archive_path(cache_dir, identifier, recipe), [pin])
        selected[identifier] = source
    return selected


class _Archives:
    def __init__(self, registry, cache_dir, timeout, archive_files=None):
        self.registry, self.cache_dir, self.timeout = registry, cache_dir, timeout
        self.archive_files = archive_files or {}
        self.manual_reports = []
        self.local, self.lock, self.opened = threading.local(), threading.Lock(), []
        self.range_tools = _range_module()
        self.caches = {}

    def cached_path(self, identifier):
        archive = self.registry["archives"][identifier]
        return _cached_archive_path(self.cache_dir, identifier, archive)

    def prepare(self, identifiers):
        for identifier in sorted(identifiers):
            archive = self.registry["archives"][identifier]
            if archive["access"] == "cached":
                if identifier in self.archive_files:
                    with self.archive_files[identifier].open("rb") as stream:
                        identity = _publish(stream, self.cached_path(identifier), _pin(archive, identifier))
                    self.manual_reports.append({"archive_id": identifier, "source_url": archive["url"],
                                                **identity, "status": "imported" if identity["status"] == "downloaded"
                                                else identity["status"]})
                else:
                    _download_url(archive["url"], self.cached_path(identifier), _pin(archive, identifier), timeout=self.timeout)
            else:
                self.caches[identifier] = self.range_tools._DirectoryCache()

    def get(self, identifier):
        if not hasattr(self.local, "archives"):
            self.local.archives = {}
        if identifier not in self.local.archives:
            recipe = self.registry["archives"][identifier]
            reader = None
            if recipe["access"] == "range":
                reader = self.range_tools.HTTPRangeReader(
                    timeout=self.timeout, cache=self.caches[identifier], url=recipe["url"],
                    archive_bytes=recipe["bytes"], etag=recipe["etag"])
                try:
                    archive = zipfile.ZipFile(reader)
                except Exception:
                    reader.close()
                    raise
            else:
                archive = zipfile.ZipFile(self.cached_path(identifier))
            self.local.archives[identifier] = archive
            with self.lock:
                self.opened.append((archive, reader))
        return self.local.archives[identifier]

    def close(self):
        for archive, reader in self.opened:
            archive.close()
            if reader is not None:
                reader.close()


def _run_build(node, cache_dir):
    build, destination, pin = node["build"], node["destination"], node["pin"]
    builder = _builder(build)
    existing = _existing(destination, _approved(node["row"]))
    if existing is not None:
        return {"status": "already_present", **existing}
    work_root = cache_dir / "builds"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="asset-build-", dir=work_root) as temporary:
        work = Path(temporary).resolve()
        for item in node["inputs"]:
            identity = _existing(item["path"], item["pins"])
            if identity is None:
                raise ValueError(f"Build input is missing: {item['path']}")
            staged = _safe_path(work, item["name"])
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["path"], staged)
            _existing(staged, [identity])
        # Explicit-output builders may restore a file with the same basename as
        # their input. Keep that output separate from every staged input. A
        # fixed-output builder using only input_dir still writes in work itself.
        output_root = (Path(tempfile.mkdtemp(prefix="outputs-", dir=work))
                       if any("{output}" in argument for argument in build["args"]) else work)
        output = _safe_path(output_root, build.get("output_name", PurePosixPath(node["row"]["path"]).name))
        output.parent.mkdir(parents=True, exist_ok=True)
        values = {"input": _safe_path(work, node["inputs"][0]["name"]), "input_dir": work,
                  "output": output, "metadata": work / "build_metadata.json"}
        arguments = _expand_args(build["args"], values)
        completed = subprocess.run([sys.executable, str(builder), *arguments], cwd=str(work), shell=False,
                                   capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Asset builder exited {completed.returncode}: {(completed.stderr or completed.stdout)[-4000:]}")
        if not output.is_file():
            raise ValueError(f"Asset builder did not create its expected output: {output.name}")
        with output.open("rb") as stream:
            result = _publish(stream, destination, pin, approved=_approved(node["row"]))
        if result["status"] == "downloaded":
            result["status"] = "built"
        return {**result, "builder": build["builder"], "builder_sha256": build["builder_sha256"]}


def download_assets(source_dir, *, cache_dir=None, tasks=None, workers=4, timeout=60.0, manifest=None, registry=None,
                    archive_files=None):
    source_dir = Path(source_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else source_dir / ".forcebench-cache"
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("--workers must be between 1 and 16")
    if manifest is None:
        manifest = json.loads((ROOT / "assets/asset_manifest.json").read_text(encoding="utf-8"))
    if registry is None:
        registry = json.loads((ROOT / "assets/asset_sources.json").read_text(encoding="utf-8"))
    tasks = sorted(set(tasks or manifest["tasks"]))
    report = {"selected_tasks": tasks, "source_dir": str(source_dir), "cache_dir": str(cache_dir),
              "checks_passed": False, "files": [], "build_inputs": [], "manual_archives": [], "issues": [],
              "scope": "Pinned external payloads and CPU asset builders; import with tools/prepare_assets.py afterwards. "
                       "Original asset terms apply; no simulator or GPU validation is performed."}
    try:
        nodes, build_order, inputs = _plan(manifest, registry, tasks, source_dir, cache_dir)
    except (KeyError, OSError, ValueError) as error:
        report["issues"].append({"kind": "source_recipe_preflight_failed", "detail": str(error)})
        return report
    needed_archives = {node["recipe"]["archive"] for node in nodes.values() if node["recipe"]["kind"] == "zip"}
    needed_archives.update(item["archive"] for item in inputs.values() if item["kind"] == "zip")
    try:
        manual_archives = _manual_archive_inputs(archive_files, registry, needed_archives, cache_dir)
    except (KeyError, OSError, ValueError) as error:
        report["issues"].append({"kind": "manual_archive_preflight_failed", "detail": str(error)})
        return report
    results, pending = {}, []
    for name, node in nodes.items():
        recipe = node["recipe"]
        result = {"file": name, "path": str(node["destination"]), "expected_bytes": node["pin"]["bytes"],
                  "expected_sha256": node["pin"]["sha256"], "source": recipe, "status": "missing"}
        try:
            existing = _existing(node["destination"], _approved(node["row"]))
            if existing is not None:
                result.update(status="already_present", **existing)
            elif recipe["kind"] != "build":
                pending.append(name)
        except (OSError, ValueError) as error:
            result.update(status="failed", error=str(error))
        results[name] = result
    needed_builds = [name for name in build_order if results[name]["status"] == "missing"]
    needed_inputs = {str(item["path"]) for name in needed_builds for item in nodes[name]["inputs"]}
    input_pending = []
    input_results = {}
    for identifier, item in inputs.items():
        if identifier not in needed_inputs:
            continue
        result = {"path": identifier, "status": "missing", **item["pin"],
                  "source": {key: item[key] for key in ("kind", "url", "archive", "member") if key in item}}
        if item["kind"] == "url":
            result["source_url"] = item["url"]
        try:
            existing = _existing(item["path"], [item["pin"]])
            if existing is not None:
                result.update(status="already_present", **existing)
            else:
                input_pending.append(identifier)
        except (OSError, ValueError) as error:
            result.update(status="failed", error=str(error))
        input_results[identifier] = result
    if needed_builds:
        try:
            _require_usd_core()
        except RuntimeError as error:
            report["issues"].append({"kind": "build_runtime_unavailable", "detail": str(error)})
    failed = any(item["status"] == "failed" for item in (*results.values(), *input_results.values()))
    if not failed and not report["issues"]:
        archives = _Archives(registry, cache_dir, timeout, archive_files=manual_archives)
        try:
            archive_ids = {nodes[name]["recipe"]["archive"] for name in pending if nodes[name]["recipe"]["kind"] == "zip"}
            archive_ids.update(inputs[key]["archive"] for key in input_pending if inputs[key]["kind"] == "zip")
            archive_ids.update(manual_archives)
            archives.prepare(archive_ids)

            def acquire(recipe, destination, pin, approved=None):
                if recipe["kind"] == "url":
                    return _download_url(recipe["url"], destination, pin, approved=approved, timeout=timeout)
                archive = archives.get(recipe["archive"])
                info = archive.getinfo(recipe["member"])
                if info.file_size != pin["bytes"]:
                    raise ValueError(f"ZIP entry size differs from its pin: {recipe['member']}")
                with archive.open(info) as stream:
                    return _publish(stream, destination, pin, approved=approved)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(acquire, nodes[name]["recipe"], nodes[name]["destination"], nodes[name]["pin"],
                                           _approved(nodes[name]["row"])): results[name] for name in pending}
                futures.update({executor.submit(acquire, inputs[key], inputs[key]["path"], inputs[key]["pin"]):
                                input_results[key] for key in input_pending})
                for future in as_completed(futures):
                    result = futures[future]
                    try:
                        result.update(future.result())
                    except Exception as error:
                        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        except Exception as error:
            report["issues"].append({"kind": "archive_acquisition_failed", "detail": f"{type(error).__name__}: {error}"})
        finally:
            archives.close()
            report["manual_archives"] = archives.manual_reports
        for name in needed_builds:
            try:
                results[name].update(_run_build(nodes[name], cache_dir))
            except Exception as error:
                results[name].update(status="failed", error=f"{type(error).__name__}: {error}")
    report["files"] = [results[name] for name in sorted(results)]
    report["build_inputs"] = [input_results[name] for name in sorted(input_results)]
    report["required_external_files"] = len(results)
    report["downloaded_files"] = sum(item["status"] == "downloaded" for item in results.values())
    report["built_files"] = sum(item["status"] == "built" for item in results.values())
    report["already_present_files"] = sum(item["status"] == "already_present" for item in results.values())
    report["checks_passed"] = (not report["issues"] and all(item["status"] in ("downloaded", "built", "already_present")
                                                           for item in (*results.values(), *input_results.values())))
    return report


def _parse_archive_files(arguments):
    archives = {}
    for argument in arguments or []:
        parts = argument.split("=", 1)
        if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] in archives:
            raise ValueError("--archive-file requires a unique ARCHIVE_ID=/local/file.zip value")
        identifier, filename = parts
        archives[identifier] = filename
    return archives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--task", action="append", help="Repeat to select tasks; default is all eight")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--archive-file", action="append", metavar="ARCHIVE_ID=PATH",
                        help="Import a verified local ZIP for a selected cached archive; repeat as needed")
    args = parser.parse_args()
    try:
        report = download_assets(args.source_dir, cache_dir=args.cache_dir, tasks=args.task,
                                 workers=args.workers, timeout=args.timeout, archive_files=_parse_archive_files(args.archive_file))
    except (OSError, ValueError, KeyError) as error:
        report = {"checks_passed": False, "error": str(error)}
    print(json.dumps(report, indent=2))
    return 0 if report["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
