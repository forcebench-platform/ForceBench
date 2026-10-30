#!/usr/bin/env python3
"""Fetch only the pinned Weight Sort external assets from NVIDIA sources.

Selected SimReady files are read with HTTP ranges from the historical ZIP;
the 21 GB archive is never downloaded in full. NVIDIA asset terms still apply.
Authored wrappers and runtime materials are handled by prepare_assets.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = "https://d4i3qtqj3r0z5.cloudfront.net/SimReady_Containers_Shipping_01_NVD%4010010.zip"
ARCHIVE_BYTES = 21_907_097_879
ARCHIVE_ETAG = "822c58760f11488f10f4c2f0c2bb6bce-2612"
ARCHIVE_MEMBER_PREFIX = "Assets/simready_content/common_assets/props/"
TABLE_URL_PREFIX = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/ArchVis/Residential/Furniture/DiningSets/EastRural/"
)
ZIP_PREFIXES = ("simready/flatbox_a02/", "simready/container_f27/")
TABLE_PREFIX = "residential/east_rural/"
TABLE_FILES = {"EastRural_Table.usd"} | {
    f"Textures/{material}_{channel}.png"
    for material in ("ER_TableTop", "ER_Table_WhiteWood")
    for channel in ("BaseColor", "Normal", "Roughness")
}
MAX_RANGE_BYTES = 8 * 1024 * 1024
BLOCK_BYTES = 1024 * 1024


class _DirectoryCache:
    """Share immutable archive-tail reads; file payloads are never cached."""

    def __init__(self):
        self.lock = threading.Lock()
        self.blocks = {}
        self.bytes = 0


class HTTPRangeReader(io.RawIOBase):
    """A bounded seekable reader for a pinned historical ZIP."""

    def __init__(self, *, timeout=60.0, cache=None, url=None, archive_bytes=None, etag=None):
        super().__init__()
        self.position = 0
        self.timeout = timeout
        self.cache = cache or _DirectoryCache()
        self.url = ARCHIVE_URL if url is None else url
        self.archive_bytes = ARCHIVE_BYTES if archive_bytes is None else archive_bytes
        self.etag = ARCHIVE_ETAG if etag is None else etag

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.archive_bytes + offset
        else:
            raise ValueError("Invalid seek mode")
        if position < 0:
            raise ValueError("Cannot seek before the archive")
        self.position = position
        return position

    def _fetch(self, start, length):
        end = start + length - 1
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={start}-{end}", "If-Match": f'"{self.etag}"',
            "Accept-Encoding": "identity",
        })
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status != 206:
                raise ValueError(f"Server ignored the bounded Range request (HTTP {response.status})")
            expected_range = f"bytes {start}-{end}/{self.archive_bytes}"
            if response.headers.get("Content-Range") != expected_range:
                raise ValueError(f"Unexpected Content-Range; expected {expected_range}")
            if response.headers.get("ETag", "").strip('"') != self.etag:
                raise ValueError("Historical ZIP ETag changed or was not supplied")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != length:
                raise ValueError("Range Content-Length differs from the requested byte count")
            blocks, received = [], 0
            while received <= length:
                block = response.read(min(BLOCK_BYTES, length + 1 - received))
                if not block:
                    break
                blocks.append(block)
                received += len(block)
            if received != length:
                raise ValueError(f"Range body has {received} bytes; expected {length}")
            return b"".join(blocks)

    def read(self, size=-1):
        self._checkClosed()
        remaining = max(0, self.archive_bytes - self.position)
        length = remaining if size is None or size < 0 else min(size, remaining)
        if length > MAX_RANGE_BYTES:
            raise ValueError(f"Archive read exceeds the {MAX_RANGE_BYTES}-byte Range limit")
        if length == 0:
            return b""
        start = self.position
        key = (self.url, self.archive_bytes, self.etag, start, length)
        if start >= self.archive_bytes - MAX_RANGE_BYTES:
            with self.cache.lock:
                data = self.cache.blocks.get(key)
                if data is None:
                    data = self._fetch(start, length)
                    if self.cache.bytes + length <= MAX_RANGE_BYTES:
                        self.cache.blocks[key] = data
                        self.cache.bytes += length
        else:
            data = self._fetch(start, length)
        self.position += length
        return data


def _robot_tool():
    spec = importlib.util.spec_from_file_location("weight_sort_robot_downloader", ROOT / "tools/download_robot_asset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_destination(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
            or not root.joinpath(*relative.parts).resolve().is_relative_to(root)):
        raise ValueError(f"Non-portable or escaping asset path: {name}")
    return root.joinpath(*relative.parts)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _variants(row):
    variants = [row]
    if row["path"] == "franka_mimic.usd":
        variants += row.get("verified_equivalent_exports", [])
    for variant in variants:
        if (type(variant.get("bytes")) is not int or variant["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", variant.get("sha256", ""))):
            raise ValueError(f"Invalid pinned asset metadata: {row['path']}")
    return variants


def _existing(destination, row):
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_file():
        candidates = [item for item in _variants(row) if item["bytes"] == destination.stat().st_size]
        if candidates:
            digest = _sha256(destination)
            if any(item["sha256"] == digest for item in candidates):
                return {"bytes": destination.stat().st_size, "sha256": digest}
    raise ValueError(f"Existing asset differs from the pinned exports; preserved: {destination}")


def _selected_rows(manifest):
    files = {row["path"]: row for row in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise ValueError("Duplicate file paths in manifest")
    selected = []
    for name in sorted(set(manifest["tasks"]["weight_sort"]["files"])):
        row = files[name]
        if not row.get("present") or row.get("source_group") == "forcebench_authored_wrapper_or_primitive":
            continue
        _variants(row)
        if name == "franka_mimic.usd":
            selected.append((row, "robot", None))
        elif name.startswith(ZIP_PREFIXES):
            selected.append((row, "zip", ARCHIVE_MEMBER_PREFIX + name.removeprefix("simready/")))
        elif name.startswith(TABLE_PREFIX) and name.removeprefix(TABLE_PREFIX) in TABLE_FILES:
            selected.append((row, "table", TABLE_URL_PREFIX + name.removeprefix(TABLE_PREFIX)))
        else:
            raise ValueError(f"No audited Weight Sort source recipe for {name}")
    return selected


def _publish_stream(source, destination, row):
    existing = _existing(destination, row)
    if existing is not None:
        return {"status": "already_present", **existing}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".weight-sort-", suffix=".part", dir=destination.parent,
                                         delete=False) as output:
            temporary = Path(output.name)
            digest, size = hashlib.sha256(), 0
            while block := source.read(BLOCK_BYTES):
                size += len(block)
                if size > row["bytes"]:
                    raise ValueError(f"Download exceeds pinned size: {row['path']}")
                digest.update(block)
                output.write(block)
            actual = digest.hexdigest()
            if size != row["bytes"] or actual != row["sha256"]:
                raise ValueError(f"Size/SHA256 verification failed for {row['path']}: bytes={size}, sha256={actual}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _existing(destination, row)
            if existing is None:
                raise
            return {"status": "already_present", **existing}
        return {"status": "downloaded", "bytes": size, "sha256": actual}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_weight_sort_assets(source_dir, *, manifest=None, workers=4, timeout=60.0):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("--workers must be between 1 and 16")
    source_dir = Path(source_dir).expanduser().resolve()
    if manifest is None:
        manifest = json.loads((ROOT / "assets/asset_manifest.json").read_text(encoding="utf-8"))
    selected = _selected_rows(manifest)
    results, pending = {}, []
    for row, kind, location in selected:
        destination = _safe_destination(source_dir, row["path"])
        entry = {"file": row["path"], "path": str(destination), "expected_bytes": row["bytes"],
                 "expected_sha256": row["sha256"], "source_url": ARCHIVE_URL if kind == "zip" else location}
        if kind == "zip":
            entry["archive_member"] = location
        try:
            existing = _existing(destination, row)
            if existing is None:
                entry["status"] = "missing"
                pending.append((row, kind, location, destination))
            else:
                entry.update(status="already_present", **existing)
        except (OSError, ValueError) as error:
            entry.update(status="failed", error=str(error))
        results[row["path"]] = entry

    report = {
        "task": "weight_sort", "source_dir": str(source_dir), "required_external_files": len(selected),
        "archive_url": ARCHIVE_URL, "archive_bytes": ARCHIVE_BYTES, "archive_etag": ARCHIVE_ETAG,
        "terms": "NVIDIA asset terms apply; these assets are not covered by ForceBench's MIT license.",
        "authored_wrappers": "Supplied by tools/prepare_assets.py; not downloaded here.",
    }
    if not any(item["status"] == "failed" for item in results.values()):
        cache, local = _DirectoryCache(), threading.local()
        opened, opened_lock = [], threading.Lock()

        def download_one(row, kind, location, destination):
            if kind == "robot":
                robot_report = _robot_tool().download_robot(source_dir, timeout=timeout)
                identity = _existing(destination, row)
                if identity is None:
                    raise ValueError("Robot downloader returned without a verified asset")
                return {**robot_report, **identity}
            if kind == "table":
                with urllib.request.urlopen(location, timeout=timeout) as response:
                    if response.status != 200:
                        raise ValueError(f"Unexpected HTTP {response.status} for {row['path']}")
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) != row["bytes"]:
                        raise ValueError(f"Unexpected Content-Length for {row['path']}")
                    return _publish_stream(response, destination, row)
            if not hasattr(local, "archive"):
                reader = HTTPRangeReader(timeout=timeout, cache=cache)
                try:
                    archive = zipfile.ZipFile(reader)
                except Exception:
                    reader.close()
                    raise
                local.archive = archive
                with opened_lock:
                    opened.append((archive, reader))
            info = local.archive.getinfo(location)
            if info.file_size != row["bytes"]:
                raise ValueError(f"ZIP entry size differs from the manifest: {location}")
            with local.archive.open(info) as stream:
                # ZipExtFile checks the entry's CRC when EOF is reached.
                return _publish_stream(stream, destination, row)

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(download_one, *item): item[0]["path"] for item in pending}
                for future in as_completed(futures):
                    entry = results[futures[future]]
                    try:
                        entry.update(future.result())
                    except Exception as error:
                        entry.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            for archive, reader in opened:
                archive.close()
                reader.close()
    report["files"] = [results[name] for name in sorted(results)]
    report["checks_passed"] = all(row["status"] in ("downloaded", "already_present") for row in report["files"])
    report["downloaded_files"] = sum(row["status"] == "downloaded" for row in report["files"])
    report["already_present_files"] = sum(row["status"] == "already_present" for row in report["files"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    try:
        report = download_weight_sort_assets(args.source_dir, workers=args.workers, timeout=args.timeout)
    except (OSError, ValueError, KeyError, http.client.HTTPException) as error:
        report = {"task": "weight_sort", "checks_passed": False, "error": str(error)}
    print(json.dumps(report, indent=2))
    return 0 if report["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
