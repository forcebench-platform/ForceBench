#!/usr/bin/env python3
"""Inventory authored USD, material and texture dependencies without starting Sim.

Requires pxr. It never downloads or changes assets. Missing, non-portable and
runtime module references remain explicit in the output rather than being
silently considered complete.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse


USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
TEXTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tga", ".exr", ".hdr", ".dds", ".bmp", ".tif", ".tiff", ".ktx", ".ktx2"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def usd_references(path: Path) -> list[tuple[str, str]]:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise ValueError("USD layer cannot be opened")
    found = {(str(ref), "composition") for ref in layer.GetCompositionAssetDependencies() if str(ref)}
    found.update((str(ref), "layer_external") for ref in layer.GetExternalAssetDependencies() if str(ref))

    def paths(value):
        if isinstance(value, Sdf.AssetPath):
            if value.path:
                found.add((value.path, "asset_attribute"))
        elif isinstance(value, dict):
            for child in value.values():
                paths(child)
        elif isinstance(value, (list, tuple)) or type(value).__name__ == "AssetPathArray":
            for child in value:
                paths(child)

    def visit(spec_path):
        obj = layer.GetObjectAtPath(spec_path)
        if isinstance(obj, Sdf.AttributeSpec) and str(obj.typeName).startswith("asset"):
            paths(obj.default)
            for time in layer.ListTimeSamplesForPath(spec_path):
                paths(layer.QueryTimeSample(spec_path, time))
        # Assets in clip dictionaries and other metadata are also dependencies.
        if obj is not None:
            for key in ("clips", "assetInfo", "customData"):
                if obj.HasInfo(key):
                    paths(obj.GetInfo(key))

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return sorted(found)


def mdl_references(path: Path) -> list[tuple[str, str]]:
    content = path.read_text(encoding="utf-8", errors="replace")
    found = set()
    for value in re.findall(r'"([^"\n]+)"', content):
        if Path(value).suffix.lower() in TEXTURE_SUFFIXES | {".mdl"}:
            found.add((value, "mdl_resource"))
    for module in re.findall(r"\b(?:import|using)\s+((?:::)?[A-Za-z_][A-Za-z0-9_:]*)", content):
        found.add((module, "mdl_module"))
    return sorted(found)


def collect(asset_dir: Path, roots: dict) -> dict:
    root = asset_dir.expanduser().resolve()
    tasks = roots["tasks"]
    common = roots.get("common", [])
    pending = deque(sorted(set(common + [value for values in tasks.values() for value in values])))
    records = {}
    gaps = []
    edges = []
    while pending:
        relative = pending.popleft()
        if relative in records:
            continue
        alias = roots.get("source_aliases", {}).get(relative)
        path = root / relative
        if not path.is_file() and alias:
            path = root / alias
        if not path.is_file():
            records[relative] = {"path": relative, "present": False, "dependencies": []}
            gaps.append({"file": relative, "kind": "missing_file"})
            continue
        record = {"path": relative, "present": True, "bytes": path.stat().st_size,
                  "sha256": sha256_file(path), "dependencies": []}
        if alias:
            record["source_alias"] = alias
        records[relative] = record
        try:
            if path.suffix.lower() in USD_SUFFIXES:
                references = usd_references(path)
            elif path.suffix.lower() == ".mdl":
                references = mdl_references(path)
            else:
                references = []
        except Exception as error:
            gaps.append({"file": relative, "kind": "parse_error", "detail": str(error)})
            continue
        for authored, kind in references:
            edge = {"from": relative, "authored": authored, "kind": kind}
            parsed = urlparse(authored)
            if kind == "mdl_module":
                edge["status"] = "runtime_module"
                # These must be supplied by the supported renderer's MDL paths.
                gaps.append({"file": relative, "kind": "mdl_runtime_module", "reference": authored})
            elif parsed.scheme and parsed.scheme not in ("file",):
                edge["status"] = "external_uri"
                gaps.append({"file": relative, "kind": "external_uri", "reference": authored})
            elif "<UDIM>" in authored or "<UVTILE" in authored or "[" in authored:
                edge["status"] = "unexpanded_asset_pattern"
                gaps.append({"file": relative, "kind": edge["status"], "reference": authored})
            else:
                authored_path = Path(parsed.path if parsed.scheme == "file" else authored)
                resolved = (path.parent / authored_path).resolve()
                if not resolved.is_relative_to(root):
                    edge["status"] = "outside_asset_root"
                    gaps.append({"file": relative, "kind": edge["status"], "reference": authored,
                                 "exists": resolved.is_file()})
                else:
                    target = resolved.relative_to(root).as_posix()
                    edge["to"] = target
                    edge["status"] = "local" if resolved.is_file() else "missing_file"
                    if authored_path.is_absolute() or parsed.scheme == "file":
                        edge["nonportable_absolute_reference"] = True
                        gaps.append({"file": relative, "kind": "absolute_reference", "reference": authored})
                    record["dependencies"].append(target)
                    pending.append(target)
            edges.append(edge)
        record["dependencies"] = sorted(set(record["dependencies"]))

    def reachable(names):
        seen = set()
        todo = deque(names)
        while todo:
            name = todo.popleft()
            if name not in seen:
                seen.add(name)
                todo.extend(records.get(name, {}).get("dependencies", []))
        return seen

    task_rows = {}
    for task, direct in sorted(tasks.items()):
        selected = reachable(common + direct)
        selected_gaps = [gap for gap in gaps if gap["file"] in selected]
        task_rows[task] = {"roots": sorted(set(common + direct)), "files": sorted(selected),
                           "closure_complete": not selected_gaps, "gaps": selected_gaps}
    return {
        "schema_version": 1,
        "scope": "Eight-task authored USD/layer/asset attributes, MDL file resources and imports, with shared robot and selected scenes. No network fetching or GPU simulation.",
        "roots": roots,
        "files": [records[name] for name in sorted(records)],
        "references": edges,
        "tasks": task_rows,
        "gaps": gaps,
        "summary": {"files_present": sum(bool(row["present"]) for row in records.values()),
                    "files_missing": sum(not row["present"] for row in records.values()),
                    "bytes": sum(row.get("bytes", 0) for row in records.values()),
                    "references": len(edges), "gaps": len(gaps),
                    "closure_complete": not gaps},
        "limitations": ["Authored references include inactive branches and variants; a gap does not alone establish an active runtime failure.",
                        "Renderer built-in MDL modules require the supported Isaac Sim runtime and are listed explicitly.",
                        "No asset ownership, redistribution permission or paper-score reproduction is asserted."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--roots-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.asset_dir, json.loads(args.roots_json.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
