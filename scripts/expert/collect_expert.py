#!/usr/bin/env python3
"""Collect validated ForceBench expert episodes through one standard interface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from forcebench.expert_collection import (
    ExpertCollectionPlan,
    execute_expert_collection,
    parse_seed_spec,
)
from forcebench.expert_registry import (
    all_expert_backend_manifests,
    default_expert_backend_name,
    expert_backend_manifests,
    expert_backend_names,
)
from forcebench.paths import CODE_ROOT, SCRATCH_ROOT, dataset_dir
from forcebench.registry import get_task_manifest, task_names


PARALLEL_CLONE_TASKS = frozenset(
    (
        "weight_sort",
        "rattle_sort",
        "stove_turn_on",
        "radmanso_drawer_sort_table",
        "peg_insert",
        "nut_thread",
        "gear_mesh",
        "faucet_turn",
        "faucet_force_turn",
        "stove_press_turn",
    )
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--list", action="store_true", help="list task-bound expert plugins")
    result.add_argument("--task", choices=task_names())
    result.add_argument("--backend", help="backend registered for this task; defaults to scripted")
    result.add_argument(
        "--seeds",
        default="0",
        help="seed list/range, for example 0-39, 0,2,5, or 0:40:2",
    )
    result.add_argument("--episodes-per-seed", type=int, default=1)
    result.add_argument("--dataset-dir", type=Path)
    result.add_argument("--run-name", help="safe name below the expert scratch root")
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--devices", default="cuda:0", help="comma-separated devices")
    result.add_argument("--max-attempts", type=int, default=2)
    result.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help=(
            "per-process timeout; defaults to 5400 s for cloned-environment collectors "
            "and 900 s for single-environment collectors"
        ),
    )
    result.add_argument("--launch-stagger-s", type=float, default=20.0)
    result.add_argument("--main-image-size", type=int, default=224)
    result.add_argument("--wrist-image-size", type=int, default=224)
    result.add_argument(
        "--record-videos",
        action="store_true",
        help=(
            "write per-episode diagnostic videos below the scratch run directory; "
            "disabled by default for high-throughput formal collection"
        ),
    )
    result.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse one Isaac application for a worker's seed shard; "
            "use --no-persistent-workers for one process per seed"
        ),
    )
    result.add_argument(
        "--seeds-per-worker-process",
        type=int,
        default=8,
        help="restart each persistent Isaac process after this many seeds",
    )
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--backend-options", default="{}", help="task-plugin JSON options")
    result.add_argument("--env-spacing", type=float, default=6.0)
    result.add_argument("--video-tile-size", type=int, default=192)
    result.add_argument(
        "--trajectory-mode",
        choices=("task_default", "half_split", "all_expert"),
        default="task_default",
        help="Trajectory distribution for nut_thread/gear_mesh clone collectors.",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def list_plugins() -> None:
    manifests = expert_backend_manifests()
    all_manifests = all_expert_backend_manifests()
    for task in task_names():
        if task in PARALLEL_CLONE_TASKS:
            print(f"{task}\tparallel32\tsource available; GPU validation pending")
            continue
        names = expert_backend_names(task)
        if not names:
            disabled = [
                manifest
                for (registered_task, _), manifest in all_manifests.items()
                if registered_task == task and not manifest.enabled
            ]
            if disabled:
                for manifest in disabled:
                    print(
                        f"{task}\t{manifest.name}\tdisabled\t"
                        f"{manifest.disabled_reason}"
                    )
            else:
                print(f"{task}\t-\tno automated expert plugin")
            continue
        for name in names:
            manifest = manifests[(task, name)]
            checkpoint = "checkpoint required" if manifest.requires_checkpoint else "ready"
            print(f"{task}\t{name}\t{checkpoint}\t{manifest.description}")


def safe_run_name(value: str) -> str:
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid run name: {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.list:
        list_plugins()
        return 0
    if args.task is None:
        cli.error("--task is required unless --list is used")
    if args.task in PARALLEL_CLONE_TASKS:
        if args.backend not in (None, "scripted"):
            cli.error("parallel tasks use only the default 32-clone scripted collector")
        if args.episodes_per_seed != 1:
            cli.error("the 32-clone collector writes exactly one episode per seed")
        if args.workers != 1 or not args.persistent_workers:
            cli.error(
                "parallel tasks use one Isaac process with cloned workspaces; "
                "worker/process alternatives have been retired"
            )
        devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
        if len(devices) != 1:
            cli.error("the cloned-environment collector requires exactly one CUDA device")
        from scripts.expert.parallel_sort_collection import collect

        run_name = safe_run_name(
            args.run_name
            or datetime.now(timezone.utc).strftime("parallel32_%Y%m%dT%H%M%SZ")
        )
        summary = collect(
            argparse.Namespace(
                task=args.task,
                seeds=args.seeds,
                dataset_dir=args.dataset_dir,
                run_name=run_name,
                device=devices[0],
                max_attempts=args.max_attempts,
                timeout_s=args.timeout_s if args.timeout_s is not None else 5400.0,
                main_image_size=args.main_image_size,
                wrist_image_size=args.wrist_image_size,
                env_spacing=args.env_spacing,
                video_tile_size=args.video_tile_size,
                trajectory_mode=args.trajectory_mode,
                dry_run=args.dry_run,
            )
        )
        print(
            "FORCEBENCH_EXPERT_COLLECTION="
            + json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
        return 1 if summary.get("failed_seeds") else 0
    if not expert_backend_names(args.task):
        cli.error(
            f"expert collection for {args.task!r} is not packaged in this preview; "
            "task specifications and evaluation source are available, but collection is unsupported"
        )
    try:
        options = json.loads(args.backend_options)
    except json.JSONDecodeError as error:
        cli.error(f"--backend-options must be valid JSON: {error}")
    if not isinstance(options, dict):
        cli.error("--backend-options must decode to an object")
    manifest = get_task_manifest(args.task)
    backend = args.backend or default_expert_backend_name(args.task)
    if backend not in expert_backend_names(args.task):
        cli.error(
            f"backend {backend!r} is unavailable for {args.task!r}; "
            f"available={expert_backend_names(args.task)}"
        )
    run_name = safe_run_name(
        args.run_name
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    plan = ExpertCollectionPlan(
        task=args.task,
        backend=backend,
        seeds=parse_seed_spec(args.seeds),
        dataset_dir=(args.dataset_dir or dataset_dir(manifest.dataset_subdir)).resolve(),
        run_dir=(SCRATCH_ROOT / "expert" / args.task / run_name).resolve(),
        isaaclab_root=(CODE_ROOT / "runtime/IsaacLab").resolve(),
        devices=devices,
        workers=args.workers,
        episodes_per_seed=args.episodes_per_seed,
        max_attempts=args.max_attempts,
        timeout_s=args.timeout_s if args.timeout_s is not None else 900.0,
        launch_stagger_s=args.launch_stagger_s,
        main_image_size=args.main_image_size,
        wrist_image_size=args.wrist_image_size,
        record_videos=args.record_videos,
        persistent_workers=args.persistent_workers,
        seeds_per_worker_process=args.seeds_per_worker_process,
        checkpoint=args.checkpoint.resolve() if args.checkpoint is not None else None,
        backend_options=options,
    )
    try:
        summary = execute_expert_collection(plan, dry_run=args.dry_run)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        cli.error(str(error))
    print(
        "FORCEBENCH_EXPERT_COLLECTION="
        + json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    return 1 if summary.get("failed_seeds") else 0


if __name__ == "__main__":
    raise SystemExit(main())
