#!/usr/bin/env python3
"""Verify/copy pinned task assets that the user has obtained separately.

No downloading or license grant is performed. All input hashes, documented
dependency reviews and existing destination files are checked before copying.
Only files selected by the manifest can be written under --asset-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import runpy
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Non-portable manifest path: {relative!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path resolves outside requested directory: {relative}")
    return target


def _runtime_material_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("isaacsim")
    except (ImportError, ValueError):
        spec = None
    if spec and spec.submodule_search_locations:
        for parent in spec.submodule_search_locations:
            candidate = Path(parent) / "kit/mdl/core/Base"
            if candidate.is_dir():
                return candidate
    return None


def _gap_key(row: dict) -> tuple[str, str, str]:
    return row["file"], row["kind"], row.get("reference", "")


def _pinned_exports(row: dict) -> list[dict]:
    """Keep historical pins; only the robot has audited equivalent exports."""
    exports = [{"bytes": row.get("bytes"), "sha256": row.get("sha256", ""), "export": "primary"}]
    equivalents = row.get("verified_equivalent_exports", [])
    if equivalents and row["path"] != "franka_mimic.usd":
        raise ValueError(f"Equivalent exports are only supported for the audited robot: {row['path']}")
    for variant in equivalents:
        if not variant.get("verification_report"):
            raise ValueError(f"Equivalent robot export lacks a verification report: {row['path']}")
        exports.append({**variant, "export": "verified_equivalent"})
    for variant in exports:
        if (not isinstance(variant.get("sha256"), str) or len(variant["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in variant["sha256"])
                or type(variant.get("bytes")) is not int or variant["bytes"] < 0):
            raise ValueError(f"Invalid pinned metadata: {row['path']}")
    return exports


def _matching_export(path: Path, exports: list[dict]) -> dict | None:
    if not path.is_file():
        return None
    size = path.stat().st_size
    candidates = [variant for variant in exports if variant["bytes"] == size]
    if candidates:
        digest = sha256_file(path)
        return next((variant for variant in candidates if variant["sha256"] == digest), None)
    return None


def _export_identity(path: Path, variant: dict, *, present: bool) -> dict:
    return {"path": str(path), "present": present, **variant}


def prepare(
    source_dir: Path,
    asset_dir: Path,
    manifest: dict,
    review: dict,
    *,
    tasks: list[str] | None = None,
    runtime_material_dir: Path | None = None,
    check_only: bool = False,
    bundled_assets: Path = ROOT / "assets",
) -> dict:
    source_dir = source_dir.expanduser().resolve()
    asset_dir = asset_dir.expanduser().resolve()
    bundled_assets = bundled_assets.expanduser().resolve()
    tasks = tasks or list(manifest["tasks"])
    if set(tasks) - set(manifest["tasks"]):
        raise ValueError("Unknown task selection")
    selected = {name for task in tasks for name in manifest["tasks"][task]["files"]}
    rows = {row["path"]: row for row in manifest["files"]}
    if len(rows) != len(manifest["files"]):
        raise ValueError("Duplicate file paths in manifest")
    issues = []
    copies = []
    verified = set()
    verified_assets = []
    required_bytes = 0
    for name in sorted(selected):
        row = rows.get(name)
        if row is None:
            issues.append({"file": name, "kind": "missing_manifest_entry"})
            continue
        if not row.get("present"):
            if not any(gap["file"] == name and gap["kind"] == "missing_file" for gap in manifest.get("gaps", [])):
                issues.append({"file": name, "kind": "unexplained_missing_manifest_file"})
            continue
        for dependency in row.get("dependencies", []):
            if dependency not in selected:
                issues.append({"file": name, "kind": "dependency_not_selected", "detail": dependency})
        exports = _pinned_exports(row)
        destination = safe_path(asset_dir, name)
        candidates = [safe_path(source_dir, name)]
        if row.get("source_alias"):
            candidates.append(safe_path(source_dir, row["source_alias"]))
        # Authored wrappers are shipped with the source; third-party binaries
        # never fall back to this checkout unless explicitly marked authored.
        if row.get("source_group") == "forcebench_authored_wrapper_or_primitive":
            candidates.append(safe_path(bundled_assets, name))
        found = None
        source_export = None
        candidate_present = False
        for candidate in candidates:
            if not candidate.is_file():
                continue
            candidate_present = True
            source_export = _matching_export(candidate, exports)
            if source_export is not None:
                found = candidate
                break
        if found is None:
            issues.append({"file": name, "kind": "source_sha256_mismatch" if candidate_present else "missing_source"})
            continue
        verified.add(name)
        required_bytes += source_export["bytes"]
        destination_export = _matching_export(destination, exports)
        if destination.exists() and destination_export is None:
            issues.append({"file": name, "kind": "different_existing_destination"})
        identity = {
            "file": name,
            "source": _export_identity(found, source_export, present=True),
            "destination": _export_identity(destination, destination_export or source_export,
                                            present=destination.exists()),
            "action": "keep" if destination_export else "copy",
        }
        if destination.exists() and destination_export is None:
            identity["destination"] = {"path": str(destination), "present": True, "export": "unrecognized"}
            identity["action"] = "blocked"
        verified_assets.append(identity)
        copies.append((found, destination, source_export, identity))

    review_rows = {_gap_key(row): row for row in review.get("reviews", [])}
    selected_gaps = [row for row in manifest.get("gaps", []) if row["file"] in selected]
    reviewed = []
    for gap in selected_gaps:
        decision = review_rows.get(_gap_key(gap))
        if not decision:
            issues.append({"file": gap["file"], "kind": "unreviewed_dependency_gap", "detail": gap["kind"]})
            continue
        pins = decision.get("owner_sha256", {})
        if not pins or any(rows.get(owner, {}).get("sha256") != expected for owner, expected in pins.items()):
            issues.append({"file": gap["file"], "kind": "stale_dependency_review"})
            continue
        reviewed.append(decision)

    materials = []
    runtime_material_dir = runtime_material_dir or _runtime_material_dir()
    required_materials = review.get("runtime_materials", []) if any(
        row["classification"] in ("runtime_material", "runtime_mdl_module") for row in reviewed
    ) else []
    for material in required_materials:
        source = safe_path(runtime_material_dir.resolve(), material["name"]) if runtime_material_dir else None
        passed = bool(source and source.is_file() and source.stat().st_size == material["bytes"]
                      and sha256_file(source) == material["sha256"])
        materials.append({"name": material["name"], "verified": passed})
        if not passed:
            issues.append({"file": material["name"], "kind": "runtime_material_not_verified",
                           "detail": "Use the supported Isaac Sim runtime or --runtime-material-dir pointing to kit/mdl/core/Base."})

    # The exact Nut source/wrapper values are also enforced by the task runtime.
    if "nut_thread" in tasks:
        contract = runpy.run_path(str(ROOT / "forcebench/nut_assets.py"))
        SOURCE_SHA256, WRAPPER_SHA256 = contract["SOURCE_SHA256"], contract["WRAPPER_SHA256"]
        for name, expected in list(SOURCE_SHA256.items()) + [
            ("normalized/nut_thread/" + name, expected) for name, expected in WRAPPER_SHA256.items()
        ]:
            if rows.get(name, {}).get("sha256") != expected:
                issues.append({"file": name, "kind": "nut_contract_mismatch"})

    report = {
        "selected_tasks": sorted(tasks), "source_dir": str(source_dir), "asset_dir": str(asset_dir),
        "verified_files": len(verified), "required_payload_files": sum(bool(rows.get(name, {}).get("present")) for name in selected),
        "verified_bytes": required_bytes, "reviewed_authored_gaps": len(reviewed),
        "verified_assets": verified_assets,
        "raw_authored_closure_complete": not selected_gaps,
        "runtime_materials": materials, "issues": issues,
        "prepared": False, "copied_files": 0,
        "scope": "Pinned files and reviewed dependencies for the eight default task scenes; no redistribution permission, GPU physics or paper reproduction is asserted.",
    }
    if issues or check_only:
        report["checks_passed"] = not issues
        return report
    for source, destination, source_export, identity in copies:
        if destination.exists() or destination.is_symlink():
            destination_export = _matching_export(destination, _pinned_exports(rows[identity["file"]]))
            if destination_export is None:
                raise OSError(f"Existing destination differs from approved exports; preserved: {destination}")
            identity["destination"] = _export_identity(destination, destination_export, present=True)
            identity["action"] = "keep"
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with source.open("rb") as incoming, destination.open("xb") as output:
                created = True
                shutil.copyfileobj(incoming, output, length=1024 * 1024)
            if _matching_export(destination, [source_export]) is None:
                destination.unlink()
                raise OSError(f"Copied asset differs from selected export: {identity['file']}")
        except Exception:
            # Completed files remain valid; no existing destination is replaced.
            if created and destination.exists():
                destination.unlink()
            raise
        report["copied_files"] += 1
        identity["destination"]["present"] = True
    report["prepared"] = True
    report["checks_passed"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "assets/asset_manifest.json")
    parser.add_argument("--dependency-review", type=Path, default=ROOT / "assets/asset_dependency_review.json")
    parser.add_argument("--runtime-material-dir", type=Path)
    parser.add_argument("--task", action="append", help="Repeat to select tasks; default is all eight")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare(args.source_dir, args.asset_dir,
                         json.loads(args.manifest.read_text(encoding="utf-8")),
                         json.loads(args.dependency_review.read_text(encoding="utf-8")),
                         tasks=args.task, runtime_material_dir=args.runtime_material_dir,
                         check_only=args.check_only)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
