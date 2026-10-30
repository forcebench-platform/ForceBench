#!/usr/bin/env python3
"""Install pinned Nut Thread references from a separately licensed asset directory.

No network access, simulator import, geometry substitution, or source conversion
occurs.  This copies the known USD bytes and the authored reference wrappers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forcebench.nut_assets import (
    SOURCE_ALLOWED_EXPORTS, WRAPPER_SOURCES, WRAPPER_SHA256,
    inspect_nut_assets, matching_source_export, sha256_file,
)


def _existing(destination: Path, source_name: str | None, selected: dict, asset_dir: Path) -> dict | None:
    if not destination.resolve().is_relative_to(asset_dir):
        raise ValueError(f"Asset destination resolves outside the asset directory: {destination}")
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_file():
        size, digest = destination.stat().st_size, sha256_file(destination)
        accepted = (matching_source_export(source_name, size, digest) is not None if source_name
                    else size == selected["bytes"] and digest == selected["sha256"])
        if accepted:
            return {"bytes": size, "sha256": digest}
    raise FileExistsError(f"Refusing to replace different existing asset: {destination}")


def _copy_verified(source: Path, destination: Path, source_name: str | None,
                   selected: dict, asset_dir: Path) -> None:
    if _existing(destination, source_name, selected, asset_dir) is not None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".forcebench-nut-", suffix=".part", dir=destination.parent,
                                         delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, output, length=1024 * 1024)
        if temporary.stat().st_size != selected["bytes"] or sha256_file(temporary) != selected["sha256"]:
            raise OSError(f"Copied asset differs from the selected source export: {destination}")
        try:
            # Publish only complete verified bytes; a concurrent file is never replaced.
            os.link(temporary, destination)
        except FileExistsError:
            if _existing(destination, source_name, selected, asset_dir) is None:
                raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(source_dir: Path, asset_dir: Path) -> dict:
    source_dir = source_dir.expanduser().resolve()
    asset_dir = asset_dir.expanduser().resolve()
    copies = []
    # Check every input and destination before changing any destination file.
    for name, exports in SOURCE_ALLOWED_EXPORTS.items():
        candidates = [source_dir / name, source_dir / f"factory_{name}"]
        found = [path for path in candidates if path.is_file()
                 and matching_source_export(name, path.stat().st_size, sha256_file(path)) is not None]
        if not found:
            raise ValueError(
                f"No exact {name} found in {source_dir}; expected one of the audited exports {exports}. "
                "Use the pinned self-contained Factory/FORGE export; a similarly named "
                "USD from another release cannot be silently substituted. See docs/nut_assets.md."
            )
        selected = matching_source_export(name, found[0].stat().st_size, sha256_file(found[0]))
        if selected is None:
            raise ValueError(f"Source changed after export verification: {found[0]}")
        copies.append((found[0], asset_dir / name, name, selected))
    for name in WRAPPER_SOURCES:
        rel = Path("normalized/nut_thread") / name
        wrapper = ROOT / "assets" / rel
        if not wrapper.is_file():
            raise FileNotFoundError(f"Authored wrapper missing from source checkout: {wrapper}")
        if sha256_file(wrapper) != WRAPPER_SHA256[name]:
            raise ValueError(f"Authored wrapper differs from pinned source: {wrapper}")
        copies.append((wrapper, asset_dir / rel, None,
                       {"bytes": wrapper.stat().st_size, "sha256": WRAPPER_SHA256[name]}))
    for source, destination, source_name, selected in copies:
        _existing(destination, source_name, selected, asset_dir)
    for source, destination, source_name, selected in copies:
        _copy_verified(source, destination, source_name, selected, asset_dir)
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
