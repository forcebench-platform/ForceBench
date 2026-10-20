#!/usr/bin/env python3
"""Install pinned Nut Thread references from a separately licensed asset directory.

No network access, simulator import, geometry substitution, or source conversion
occurs.  This copies the known USD bytes and the authored reference wrappers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forcebench.nut_assets import SOURCE_SHA256, WRAPPER_SOURCES, WRAPPER_SHA256, inspect_nut_assets, sha256_file


def prepare(source_dir: Path, asset_dir: Path) -> dict:
    source_dir = source_dir.expanduser().resolve()
    asset_dir = asset_dir.expanduser().resolve()
    copies = []
    # Check every input and destination before changing any destination file.
    for name, expected in SOURCE_SHA256.items():
        candidates = [source_dir / name, source_dir / f"factory_{name}"]
        found = [path for path in candidates if path.is_file() and sha256_file(path) == expected]
        if not found:
            raise ValueError(
                f"No exact {name} found in {source_dir}; expected SHA256 {expected}. "
                "Use the pinned self-contained Factory/FORGE export; a similarly named "
                "USD from another release cannot be silently substituted. See docs/nut_assets.md."
            )
        copies.append((found[0], asset_dir / name))
    for name in WRAPPER_SOURCES:
        rel = Path("normalized/nut_thread") / name
        wrapper = ROOT / "assets" / rel
        if not wrapper.is_file():
            raise FileNotFoundError(f"Authored wrapper missing from source checkout: {wrapper}")
        if sha256_file(wrapper) != WRAPPER_SHA256[name]:
            raise ValueError(f"Authored wrapper differs from pinned source: {wrapper}")
        copies.append((wrapper, asset_dir / rel))
    for source, destination in copies:
        if not destination.resolve().is_relative_to(asset_dir):
            raise ValueError(f"Asset destination resolves outside the asset directory: {destination}")
        if destination.exists() and (not destination.is_file() or sha256_file(destination) != sha256_file(source)):
            raise FileExistsError(f"Refusing to replace different existing asset: {destination}")
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination and not destination.exists():
            shutil.copyfile(source, destination)
    report = inspect_nut_assets(asset_dir)
    if not report["ready"]:
        raise ValueError("Installed assets failed validation: " + "; ".join(report["issues"]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Directory containing separately obtained source USDs")
    parser.add_argument("--asset-dir", required=True, type=Path, help="ForceBench asset directory")
    parser.add_argument("--check-only", action="store_true", help="Validate without copying")
    args = parser.parse_args()
    try:
        if args.check_only:
            report = inspect_nut_assets(args.asset_dir)
        else:
            if args.source_dir is None:
                parser.error("--source-dir is required unless --check-only is used")
            report = prepare(args.source_dir, args.asset_dir)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
