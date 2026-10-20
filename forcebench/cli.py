"""Inspect task and asset requirements without importing simulator/model runtimes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from importlib.resources import files
from pathlib import Path

from .registry import task_manifests
from .nut_assets import inspect_nut_assets


def inspect_assets(asset_dir: Path, tasks: list[str]) -> dict:
    catalog = json.loads(files("forcebench").joinpath("asset_requirements.json").read_text(encoding="utf-8"))
    root = asset_dir.expanduser().resolve()
    rows = {}
    for task in tasks:
        paths = catalog["tasks"][task]
        rows[task] = {
            "required_direct_files": len(paths),
            "missing": [path for path in paths if not (root / path).is_file()],
        }
        if task == "nut_thread":
            rows[task]["source_check"] = inspect_nut_assets(root)
    return {
        "asset_dir": str(root),
        "tasks": rows,
        "direct_files_present": all(not row["missing"] for row in rows.values()),
        "pinned_sources_valid": all(row.get("source_check", {}).get("ready", True) for row in rows.values()),
        "scope": "Direct file existence only. USD references, textures, licensing, physics and GPU execution are not validated.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list-tasks", help="Show the eight paper task entry points")
    listing.add_argument("--json", action="store_true")
    doctor = sub.add_parser("doctor", help="Report missing direct assets and simulator modules; does not launch them")
    doctor.add_argument("--task", choices=tuple(task_manifests()))
    doctor.add_argument("--asset-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    manifests = task_manifests()
    if args.command == "list-tasks":
        rows = [
            {"task": name, "prompt": item.prompt, "scene_profile": item.scene_profile,
             "max_control_steps": item.evaluation_max_steps}
            for name, item in manifests.items()
        ]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(f"{row['task']}\t{row['max_control_steps']} steps\t{row['scene_profile']}")
        return 0
    asset_dir = args.asset_dir or (Path(os.environ["FORCEBENCH_ASSET_DIR"])
                                  if os.environ.get("FORCEBENCH_ASSET_DIR") else None)
    if asset_dir is None:
        parser.error("doctor requires --asset-dir PATH or FORCEBENCH_ASSET_DIR")
    report = inspect_assets(asset_dir, [args.task] if args.task else list(manifests))
    report["modules"] = {name: importlib.util.find_spec(name) is not None for name in ("isaaclab", "isaacsim")}
    report["gpu_execution_validated"] = False
    report["paper_results_reproduced"] = False
    print(json.dumps(report, indent=2))
    return 0 if report["direct_files_present"] and report["pinned_sources_valid"] and all(report["modules"].values()) else 1
