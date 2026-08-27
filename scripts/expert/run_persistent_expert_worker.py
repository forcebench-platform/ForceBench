#!/usr/bin/env python3
"""Run several expert episodes while keeping one Isaac application alive."""

from __future__ import annotations

import argparse
import gc
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jobs", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import torch

from forcebench.isaac_session import (
    PersistentWorkerRecycleRequired,
    clear_shared_simulation_app,
    close_shared_isaac_resources,
    register_shared_simulation_app,
)


ALLOWED_GENERATORS = {
}
LONG_LIVED_ENV_GENERATORS: set[str] = set()
EXPERT_SCRIPT_ROOT = Path(__file__).resolve().parent
MAX_RSS_GROWTH_MB = float(os.environ.get("FORCEBENCH_PERSISTENT_MAX_RSS_GROWTH_MB", "1024"))
MAX_RSS_MB = float(os.environ.get("FORCEBENCH_PERSISTENT_MAX_RSS_MB", "24576"))


def prepare_episode_stage() -> None:
    """Attach a fresh USD stage after the previous env cleared its context."""

    import omni.usd
    from pxr import UsdUtils

    from isaaclab.sim import close_stage, create_new_stage, update_stage

    if omni.usd.get_context().get_stage() is not None:
        close_stage()
    stage = create_new_stage()
    stage_cache = UsdUtils.StageCache.Get()
    stage_id = stage_cache.GetId(stage)
    if stage_id.ToLongInt() < 0:
        stage_id = stage_cache.Insert(stage)
    context = omni.usd.get_context()
    if context.get_stage() is not stage:
        context.attach_stage_with_callback(stage_id.ToLongInt())
    update_stage()


def resident_memory_mb() -> float | None:
    statm = Path("/proc/self/statm")
    if not statm.is_file():
        return None
    pages = int(statm.read_text(encoding="utf-8").split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def load_jobs(path: Path) -> list[dict[str, object]]:
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"persistent-worker job file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "forcebench_persistent_expert_jobs_v1":
        raise ValueError("invalid persistent-worker job schema")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("persistent worker needs at least one job")
    normalized: list[dict[str, object]] = []
    seen: set[int] = set()
    for raw in jobs:
        if not isinstance(raw, dict):
            raise TypeError("persistent-worker jobs must be objects")
        seed = raw.get("seed")
        script = Path(str(raw.get("script", ""))).resolve()
        argv = raw.get("argv")
        if not isinstance(seed, int) or seed < 0 or seed in seen:
            raise ValueError(f"invalid or duplicate persistent-worker seed: {seed!r}")
        if (
            script.name not in ALLOWED_GENERATORS
            or script.parent != EXPERT_SCRIPT_ROOT
            or not script.is_file()
        ):
            raise ValueError(f"unsupported persistent expert generator: {script}")
        if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
            raise TypeError(f"persistent-worker argv must be a string list for seed {seed}")
        seen.add(seed)
        normalized.append({"seed": seed, "script": script, "argv": argv})
    return normalized


jobs = load_jobs(args.jobs.resolve())
results: list[dict[str, object]] = []
generator_names = {Path(job["script"]).name for job in jobs}
reuse_environment = (
    len(generator_names) == 1
    and next(iter(generator_names)) in LONG_LIVED_ENV_GENERATORS
)
recycle_reason: str | None = None
rss_baseline_mb: float | None = None
register_shared_simulation_app(simulation_app)
try:
    for index, job in enumerate(jobs, start=1):
        seed = int(job["seed"])
        script = Path(job["script"])
        argv = list(job["argv"])
        print(
            "FORCEBENCH_PERSISTENT_EXPERT_EVENT="
            + json.dumps(
                {"event": "episode_started", "index": index, "total": len(jobs), "seed": seed},
                separators=(",", ":"),
            ),
            flush=True,
        )
        previous_argv = sys.argv
        request_recycle = False
        try:
            # Sort tasks use the separate 32-clone collector. Remaining
            # generators retain their isolated stage lifecycle.
            if index == 1 or not reuse_environment:
                prepare_episode_stage()
            sys.argv = [str(script), *argv]
            runpy.run_path(str(script), run_name=f"__forcebench_expert_seed_{seed}__")
        except PersistentWorkerRecycleRequired as error:
            success = False
            detail = f"{type(error).__name__}:{error}"
            recycle_reason = detail
            request_recycle = True
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 1
            success = code == 0
            detail = None if success else f"SystemExit:{error.code}"
            request_recycle = bool(reuse_environment and not success)
            if request_recycle:
                recycle_reason = detail
        except Exception as error:
            success = False
            detail = f"{type(error).__name__}:{error}"
            traceback.print_exc()
            request_recycle = reuse_environment
            if request_recycle:
                recycle_reason = detail
        else:
            success = True
            detail = None
        finally:
            sys.argv = previous_argv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = {
            "seed": seed,
            "success": success,
            "error": detail,
            "resident_memory_mb": resident_memory_mb(),
        }
        results.append(result)
        print(
            "FORCEBENCH_PERSISTENT_EXPERT_EVENT="
            + json.dumps({"event": "episode_finished", **result}, separators=(",", ":")),
            flush=True,
        )
        current_rss_mb = result["resident_memory_mb"]
        if current_rss_mb is not None:
            current_rss_mb = float(current_rss_mb)
            if rss_baseline_mb is None and success:
                rss_baseline_mb = current_rss_mb
            growth_mb = (
                0.0 if rss_baseline_mb is None else current_rss_mb - rss_baseline_mb
            )
            if current_rss_mb > MAX_RSS_MB or growth_mb > MAX_RSS_GROWTH_MB:
                recycle_reason = (
                    f"resident_memory_mb={current_rss_mb:.1f},"
                    f"baseline_mb={rss_baseline_mb},growth_mb={growth_mb:.1f},"
                    f"limits=({MAX_RSS_MB:.1f},{MAX_RSS_GROWTH_MB:.1f})"
                )
                request_recycle = True
        if request_recycle:
            print(
                "FORCEBENCH_PERSISTENT_EXPERT_EVENT="
                + json.dumps(
                    {
                        "event": "worker_recycle_requested",
                        "seed": seed,
                        "reason": recycle_reason,
                        "remaining_jobs": len(jobs) - index,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            break
    # Write the supervisor-visible result while Kit is still alive.  On some
    # Isaac versions simulation_app.close() raises SystemExit, so code placed
    # after cleanup is not guaranteed to run.
    summary = {
        "schema": "forcebench_persistent_expert_worker_v1",
        "jobs": len(jobs),
        "successful_seeds": [item["seed"] for item in results if item["success"]],
        "failed_seeds": [item["seed"] for item in results if not item["success"]],
        "long_lived_environment": reuse_environment,
        "processed_jobs": len(results),
        "recycled": recycle_reason is not None,
        "recycle_reason": recycle_reason,
        "results": results,
    }
    args.jobs.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "FORCEBENCH_PERSISTENT_EXPERT_WORKER="
        + json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
finally:
    try:
        close_shared_isaac_resources()
    finally:
        clear_shared_simulation_app(simulation_app)
        simulation_app.close()
raise SystemExit(75 if summary["recycled"] else (1 if summary["failed_seeds"] else 0))
