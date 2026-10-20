"""Resume-safe supervision for the cloned-environment task collectors."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from forcebench.expert_collection import _run_process, parse_seed_spec, successful_episodes
from forcebench.paths import CODE_ROOT, SCRATCH_ROOT, dataset_dir
from forcebench.registry import get_task_manifest


PARALLEL_TASKS = (
    "weight_sort",
    "rattle_sort",
    "stove_turn_on",
    "radmanso_drawer_sort_table",
    "peg_insert",
    "nut_thread",
    "gear_mesh",
    "faucet_turn",
)
DEFAULT_CLONE_COUNT = 32


def clone_count_for_task(task: str) -> int:
    """All supported automatic experts use the canonical 32-clone topology."""

    if task not in PARALLEL_TASKS:
        raise ValueError(f"unsupported parallel task: {task!r}")
    return DEFAULT_CLONE_COUNT


@dataclass(frozen=True)
class CloneBatch:
    index: int
    seeds: tuple[int, ...]
    write_seeds: tuple[int, ...]


def build_clone_batches(
    requested_seeds: tuple[int, ...],
    successful_seeds: set[int],
    clone_count: int = DEFAULT_CLONE_COUNT,
) -> tuple[CloneBatch, ...]:
    """Group missing seeds into exact clone batches without duplicate writes."""

    missing = [seed for seed in requested_seeds if seed not in successful_seeds]
    if not missing:
        return ()
    fillers = [seed for seed in requested_seeds if seed in successful_seeds]
    batches: list[CloneBatch] = []
    next_external_filler = max(requested_seeds, default=-1) + 1
    for index, offset in enumerate(range(0, len(missing), clone_count), start=1):
        write_seeds = missing[offset : offset + clone_count]
        seeds = list(write_seeds)
        available = [seed for seed in fillers if seed not in seeds]
        seeds.extend(available[: clone_count - len(seeds)])
        while len(seeds) < clone_count:
            while next_external_filler in seeds:
                next_external_filler += 1
            seeds.append(next_external_filler)
            next_external_filler += 1
        if len(seeds) != clone_count or len(set(seeds)) != clone_count:
            raise RuntimeError(f"failed to construct {clone_count} distinct clone seeds")
        batches.append(CloneBatch(index, tuple(seeds), tuple(write_seeds)))
    return tuple(batches)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task", choices=PARALLEL_TASKS, required=True)
    result.add_argument(
        "--trajectory-mode",
        choices=("task_default", "half_split", "all_expert"),
        default="task_default",
        help="Trajectory distribution for nut_thread/gear_mesh collectors.",
    )
    result.add_argument("--seeds", default="0-199")
    result.add_argument("--dataset-dir", type=Path)
    result.add_argument("--run-name")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--max-attempts", type=int, default=2)
    # A full 32-clone 224/224 batch can spend most of its wall time serializing
    # HDF5 images, so keep a conservative per-process timeout.
    result.add_argument("--timeout-s", type=float, default=5400.0)
    result.add_argument("--main-image-size", type=int, default=224)
    result.add_argument("--wrist-image-size", type=int, default=224)
    result.add_argument("--env-spacing", type=float, default=6.0)
    result.add_argument("--video-tile-size", type=int, default=192)
    result.add_argument("--dry-run", action="store_true")
    return result


def safe_run_name(value: str) -> str:
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid run name: {value!r}")
    return value


def collect(args: argparse.Namespace) -> dict[str, object]:
    if args.max_attempts < 1 or args.timeout_s <= 0:
        raise ValueError("max-attempts and timeout must be positive")
    if args.main_image_size < 16 or args.wrist_image_size < 16:
        raise ValueError("image sizes must be at least 16")
    if args.env_spacing < 5.5:
        raise ValueError("env spacing must be at least 5.5 m")
    requested = parse_seed_spec(args.seeds)
    task_manifest = get_task_manifest(args.task)
    if args.task == "gear_mesh":
        trajectory_mode = (
            "half_split" if args.trajectory_mode == "task_default" else args.trajectory_mode
        )
    elif args.task == "nut_thread":
        trajectory_mode = (
            "all_expert" if args.trajectory_mode == "task_default" else args.trajectory_mode
        )
    else:
        if args.trajectory_mode != "task_default":
            raise ValueError("--trajectory-mode is only valid for nut_thread and gear_mesh")
        trajectory_mode = "task_default"
    clone_count = clone_count_for_task(args.task)
    formal_dir = (args.dataset_dir or dataset_dir(task_manifest.dataset_subdir)).resolve()
    run_name = safe_run_name(
        args.run_name
        or datetime.now(timezone.utc).strftime("parallel32_%Y%m%dT%H%M%SZ")
    )
    run_dir = (SCRATCH_ROOT / "expert" / args.task / run_name).resolve()
    formal_root = (formal_dir.parents[1] if formal_dir.parent.name == "datasets" else None)
    if formal_root is None or formal_dir.parent != formal_root / "datasets":
        raise ValueError("parallel datasets must be a direct child of the formal datasets root")
    run_dir.mkdir(parents=True, exist_ok=True)
    formal_dir.mkdir(parents=True, exist_ok=True)

    initial = successful_episodes(formal_dir, args.task)
    initial_success = set(initial)
    batches = build_clone_batches(requested, initial_success, clone_count)
    manifest_path = run_dir / "manifest.jsonl"
    summary_path = run_dir / "summary.json"
    plan_payload = {
        "schema": "forcebench_parallel_sort_collection_32env_v1",
        "task": args.task,
        "trajectory_mode": trajectory_mode,
        "clone_count": clone_count,
        "requested_seeds": list(requested),
        "existing_successful_seeds": sorted(initial_success),
        "missing_seeds": [seed for seed in requested if seed not in initial_success],
        "batches": [
            {
                "index": batch.index,
                "seeds": list(batch.seeds),
                "write_seeds": list(batch.write_seeds),
            }
            for batch in batches
        ],
    }
    (run_dir / "plan.json").write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        return {**plan_payload, "dry_run": True, "run_dir": str(run_dir)}

    isaaclab_root = Path(
        os.environ.get("FORCEBENCH_ISAACLAB_ROOT", str(CODE_ROOT / "runtime/IsaacLab"))
    ).expanduser().resolve()
    launcher = isaaclab_root / "isaaclab.sh"
    generators = {
        "stove_turn_on": "generate_parallel_stove.py",
        "peg_insert": "generate_parallel_peg_insert.py",
        "radmanso_drawer_sort_table": "generate_parallel_radmanso.py",
        "weight_sort": "generate_parallel_sort.py",
        "rattle_sort": "generate_parallel_sort.py",
        "nut_thread": "generate_parallel_privileged_placement.py",
        "gear_mesh": "generate_parallel_privileged_placement.py",
        "faucet_turn": "generate_parallel_faucet.py",
    }
    generator = CODE_ROOT / "scripts/expert" / generators[args.task]
    if not launcher.is_file() or not generator.is_file():
        raise FileNotFoundError("Isaac Lab launcher or 32-env task generator is missing")
    environment = os.environ.copy()
    environment["FORCEBENCH_TASK"] = args.task
    environment["FORCEBENCH_EXPERT_COLLECTION_CONTRACT"] = "forcebench_expert_collection"
    batch_results: list[dict[str, object]] = []
    failed_seeds: list[int] = []

    for batch in batches:
        remaining = list(batch.write_seeds)
        attempts: list[dict[str, object]] = []
        for attempt in range(1, args.max_attempts + 1):
            current = set(successful_episodes(formal_dir, args.task))
            remaining = [seed for seed in remaining if seed not in current]
            if not remaining:
                break
            fillers = [seed for seed in batch.seeds if seed not in remaining]
            launch_seeds = list(remaining)
            launch_seeds.extend(fillers[: clone_count - len(launch_seeds)])
            next_filler = max((*requested, *launch_seeds), default=-1) + 1
            while len(launch_seeds) < clone_count:
                if next_filler not in launch_seeds:
                    launch_seeds.append(next_filler)
                next_filler += 1
            # Peg/Gear contact outcomes can be sensitive to the clone slot even
            # though the seed and task geometry are unchanged.  Keeping a
            # failed seed in env_0 on every retry simply reproduces the same
            # deterministic contact state.  Rotate the complete launch order
            # by a coprime stride so retries sample a different clone slot;
            # write_seeds and per-seed randomization remain unchanged.
            clone_rotation = 0
            if args.task in ("peg_insert", "gear_mesh") and attempt > 1:
                clone_rotation = ((attempt - 1) * 7) % clone_count
                launch_seeds = (
                    launch_seeds[clone_rotation:] + launch_seeds[:clone_rotation]
                )
            artifact_dir = run_dir / f"batch_{batch.index:03d}" / f"attempt_{attempt:02d}"
            log_path = run_dir / "logs" / f"batch_{batch.index:03d}_attempt_{attempt:02d}.log"
            command = [
                str(launcher),
                "-p",
                str(generator),
            ]
            if args.task in ("weight_sort", "rattle_sort", "nut_thread", "gear_mesh"):
                command.extend(("--task", args.task))
            if args.task in ("nut_thread", "gear_mesh"):
                command.extend(("--trajectory_mode", trajectory_mode))
            command.extend([
                "--output_dir",
                str(formal_dir),
                "--artifact_dir",
                str(artifact_dir),
                "--seeds",
                ",".join(str(seed) for seed in launch_seeds),
                "--write_seeds",
                ",".join(str(seed) for seed in remaining),
                "--env_spacing",
                str(args.env_spacing),
                "--main_image_size",
                str(args.main_image_size),
                "--wrist_image_size",
                str(args.wrist_image_size),
                "--video_tile_size",
                str(args.video_tile_size),
                "--headless",
                "--enable_cameras",
                "--device",
                args.device,
                "--kit_args=--/UJITSO/geometry=false --/persistent/UJITSO/geometry=false",
            ])
            return_code, timed_out, elapsed_s = _run_process(
                command,
                cwd=isaaclab_root,
                environment=environment,
                log_path=log_path,
                timeout_s=args.timeout_s,
            )
            current_after = set(successful_episodes(formal_dir, args.task))
            completed = [seed for seed in remaining if seed in current_after]
            attempt_result = {
                "attempt": attempt,
                "return_code": return_code,
                "timed_out": timed_out,
                "elapsed_s": elapsed_s,
                "launch_seeds": launch_seeds,
                "requested_write_seeds": list(remaining),
                "completed_seeds": completed,
                "clone_rotation": clone_rotation,
                "artifact_dir": str(artifact_dir),
                "log": str(log_path),
            }
            attempts.append(attempt_result)
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(attempt_result, ensure_ascii=False) + "\n")
            remaining = [seed for seed in remaining if seed not in current_after]
            if not remaining:
                break
        failed_seeds.extend(remaining)
        batch_results.append(
            {
                "batch": batch.index,
                "write_seeds": list(batch.write_seeds),
                "success": not remaining,
                "failed_seeds": list(remaining),
                "attempts": attempts,
            }
        )
        if remaining:
            break

    final = successful_episodes(formal_dir, args.task)
    successful = [seed for seed in requested if seed in final]
    summary = {
        "schema": "forcebench_parallel_sort_collection_32env_v1",
        "task": args.task,
        "trajectory_mode": trajectory_mode,
        "clone_count": clone_count,
        "dataset_dir": str(formal_dir),
        "run_dir": str(run_dir),
        "requested_seeds": list(requested),
        "skipped_existing_seeds": sorted(initial_success.intersection(requested)),
        "new_successful_seeds": sorted(set(successful) - initial_success),
        "successful_seeds": successful,
        "failed_seeds": sorted(set(failed_seeds)),
        "valid_episode_count": len(final),
        "batches": batch_results,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary = collect(args)
    print(
        "FORCEBENCH_PARALLEL_SORT_COLLECTION="
        + json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    return 1 if summary.get("failed_seeds") else 0


if __name__ == "__main__":
    raise SystemExit(main())
