#!/usr/bin/env python3
"""Replay a 60 Hz expert dataset in cloned environments without cameras."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("weight_sort", "rattle_sort"), required=True)
parser.add_argument("--dataset_dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument(
    "--seeds",
    type=str,
    default=None,
    help="Optional comma-separated seed subset, for targeted replay diagnostics.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.batch_size <= 32:
    parser.error("--batch_size must be between 1 and 32")
if args.env_spacing < 5.5:
    parser.error("--env_spacing must be at least 5.5 m")
if args.limit is not None and args.limit <= 0:
    parser.error("--limit must be positive")
if args.enable_cameras:
    parser.error("this audit deliberately forbids --enable_cameras")

os.environ["FORCEBENCH_TASK"] = args.task
launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import h5py
import numpy as np
import torch

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import validate_hdf5_episode
from forcebench.registry import load_task

if args.task == "weight_sort":
    from forcebench.tasks.weight_sort.spec import evaluate_final_state
else:
    from forcebench.tasks.rattle_sort.spec import SPEC, evaluate_final_state


PHYSICS_RATE_HZ = 60
SEED_PATTERN = re.compile(r"^seed_(\d+)_episode_(\d+)\.hdf5$")


@dataclass(frozen=True)
class Episode:
    path: Path
    seed: int
    frames: int
    actions: np.ndarray
    phase_ids: np.ndarray | None
    static_inspect_phase_id: int | None
    metadata: dict[str, object]
    validation: dict[str, object]


def episode_sort_key(path: Path) -> tuple[int, int, str]:
    match = SEED_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"non-canonical episode filename: {path.name}")
    return int(match.group(1)), int(match.group(2)), path.name


def load_episode(path: Path) -> Episode:
    validation = validate_hdf5_episode(path)
    with h5py.File(path, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        actions = np.asarray(handle["actions/canonical"], dtype=np.float32)
        phase_ids = (
            np.asarray(handle["phases/id"], dtype=np.int64)
            if "phases/id" in handle
            else None
        )

    task = str(metadata.get("task"))
    if task != args.task:
        raise ValueError(f"{path} belongs to {task!r}, expected {args.task!r}")
    declared_physics_rate = float(metadata.get("physics_rate_hz", -1.0))
    if declared_physics_rate != PHYSICS_RATE_HZ:
        raise ValueError(
            f"{path} declares {declared_physics_rate:g} Hz, expected current "
            f"{PHYSICS_RATE_HZ} Hz"
        )
    if actions.shape != (int(validation["control_frames"]), 7):
        raise ValueError(f"invalid action shape in {path}: {actions.shape}")

    static_phase_id = None
    if args.task == "rattle_sort":
        schema = metadata.get("phase_schema")
        if not isinstance(schema, dict) or "static_inspect" not in schema:
            raise ValueError(f"rattle episode lacks static_inspect phase: {path}")
        if phase_ids is None or phase_ids.shape != (len(actions),):
            raise ValueError(f"rattle episode lacks phase ids: {path}")
        static_phase_id = int(schema["static_inspect"])

    return Episode(
        path=path,
        seed=int(metadata["seed"]),
        frames=len(actions),
        actions=actions,
        phase_ids=phase_ids,
        static_inspect_phase_id=static_phase_id,
        metadata=metadata,
        validation=validation,
    )


def numeric_deltas(
    current: dict[str, object], recorded: dict[str, object]
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in sorted(current.keys() & recorded.keys()):
        current_value = current[key]
        recorded_value = recorded[key]
        if (
            isinstance(current_value, Real)
            and not isinstance(current_value, bool)
            and isinstance(recorded_value, Real)
            and not isinstance(recorded_value, bool)
        ):
            deltas[key] = float(current_value) - float(recorded_value)
    return deltas


def write_payload(results: list[dict[str, object]], *, complete: bool) -> None:
    successes = sum(bool(result.get("success")) for result in results)
    payload = {
        "schema_version": "forcebench_expert_replay_audit_v1",
        "task": args.task,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "physics_rate_hz": PHYSICS_RATE_HZ,
        "control_rate_hz": 15,
        "camera_rendering": False,
        "batch_size": args.batch_size,
        "complete": complete,
        "episodes_checked": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, args.output)


def update_rattle_inspection(
    env: ForceBenchEnv,
    episodes: list[Episode],
    frame: int,
    completed: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> None:
    box_position, _ = env.entity_local_pose("box")
    box = env.entity("box")
    linear_speed = torch.linalg.vector_norm(box.data.root_lin_vel_w.torch, dim=1)
    angular_speed = torch.linalg.vector_norm(box.data.root_ang_vel_w.torch, dim=1)
    distance = torch.linalg.vector_norm(
        box_position - env.fingertip_midpoint_pos, dim=1
    )
    held = (
        (box_position[:, 2] >= SPEC.inspection_min_lift_height_m)
        & (
            env.gripper_open_fraction_tensor
            <= SPEC.inspection_max_gripper_open_fraction
        )
        & (distance <= SPEC.inspection_max_tcp_box_distance_m)
    )
    stationary = (
        (linear_speed <= SPEC.inspection_max_box_linear_speed_m_s)
        & (angular_speed <= SPEC.inspection_max_box_angular_speed_rad_s)
    )
    valid_state = (held & stationary).detach().cpu().numpy()
    held_state = held.detach().cpu().numpy()
    stationary_state = stationary.detach().cpu().numpy()
    height_values = box_position[:, 2].detach().cpu().numpy()
    gripper_values = env.gripper_open_fraction_tensor.detach().cpu().numpy()
    distance_values = distance.detach().cpu().numpy()
    linear_values = linear_speed.detach().cpu().numpy()
    angular_values = angular_speed.detach().cpu().numpy()
    for env_id, episode in enumerate(episodes):
        if frame >= episode.frames:
            continue
        if int(episode.phase_ids[frame]) == episode.static_inspect_phase_id:
            diagnostics["frames"][env_id] += 1
            diagnostics["held_seen"][env_id] |= bool(held_state[env_id])
            diagnostics["stationary_seen"][env_id] |= bool(
                stationary_state[env_id]
            )
            diagnostics["max_height_m"][env_id] = max(
                diagnostics["max_height_m"][env_id], height_values[env_id]
            )
            diagnostics["min_gripper_open_fraction"][env_id] = min(
                diagnostics["min_gripper_open_fraction"][env_id],
                gripper_values[env_id],
            )
            diagnostics["min_tcp_box_distance_m"][env_id] = min(
                diagnostics["min_tcp_box_distance_m"][env_id],
                distance_values[env_id],
            )
            diagnostics["min_linear_speed_m_s"][env_id] = min(
                diagnostics["min_linear_speed_m_s"][env_id], linear_values[env_id]
            )
            diagnostics["min_angular_speed_rad_s"][env_id] = min(
                diagnostics["min_angular_speed_rad_s"][env_id],
                angular_values[env_id],
            )
            completed[env_id] |= bool(valid_state[env_id])


def evaluate_env(
    env: ForceBenchEnv,
    env_id: int,
    episode: Episode,
    inspection_completed: bool,
    inspection_diagnostics: dict[str, object] | None,
    outcome_evaluator,
) -> dict[str, object]:
    box_position, box_quaternion = env.entity_local_pose("box")
    box_velocity = env.entity("box").data.root_lin_vel_w.torch
    box_angular_velocity = env.entity("box").data.root_ang_vel_w.torch
    gripper = env.gripper_open_fraction_tensor
    kwargs = {
        "box_kind": str(env.layouts[env_id]["box_kind"]),
        "box_position_m": box_position[env_id].detach().cpu().numpy(),
        "box_linear_velocity_m_s": box_velocity[env_id].detach().cpu().numpy(),
        "gripper_open_fraction": float(gripper[env_id].item()),
    }
    if args.task == "rattle_sort":
        kwargs["inspection_completed"] = bool(inspection_completed)
    report = outcome_evaluator(evaluate_final_state(**kwargs).to_dict())
    recorded_report = episode.metadata.get("final_success_report", {})
    result = {
        "path": str(episode.path.resolve()),
        "seed": episode.seed,
        "frames": episode.frames,
        "success": bool(report["success"]),
        "failure_reasons": list(report["failure_reasons"]),
        "report": report,
        "final_box_position_m": box_position[env_id].detach().cpu().tolist(),
        "final_box_quaternion_wxyz": (
            box_quaternion[env_id].detach().cpu().tolist()
        ),
        "final_box_linear_velocity_m_s": (
            box_velocity[env_id].detach().cpu().tolist()
        ),
        "final_box_angular_velocity_rad_s": (
            box_angular_velocity[env_id].detach().cpu().tolist()
        ),
        "recorded_report": recorded_report,
        "replay_minus_recorded": numeric_deltas(report, recorded_report),
        "validation": episode.validation,
    }
    if inspection_diagnostics is not None:
        result["replay_inspection_diagnostics"] = inspection_diagnostics
    return result


dataset_paths = sorted(args.dataset_dir.glob("*.hdf5"), key=episode_sort_key)
if not dataset_paths:
    parser.error(f"no HDF5 episodes found in {args.dataset_dir}")
if len({episode_sort_key(path)[0] for path in dataset_paths}) != len(dataset_paths):
    parser.error("dataset contains duplicate seeds")
if args.seeds is not None:
    try:
        requested_seeds = {int(value) for value in args.seeds.split(",")}
    except ValueError as error:
        parser.error(f"invalid --seeds value: {error}")
    if not requested_seeds or min(requested_seeds) < 0:
        parser.error("--seeds must contain non-negative integers")
    by_seed = {episode_sort_key(path)[0]: path for path in dataset_paths}
    missing_seeds = sorted(requested_seeds - by_seed.keys())
    if missing_seeds:
        parser.error(f"requested seeds are absent: {missing_seeds}")
    dataset_paths = [by_seed[seed] for seed in sorted(requested_seeds)]
if args.limit is not None:
    dataset_paths = dataset_paths[: args.limit]

task = load_task(args.task, ASSET_DIR)
outcome_module = importlib.import_module(f"forcebench.tasks.{args.task}.outcome")
outcome_evaluator = outcome_module.evaluate_report
cfg = ForceBenchEnvCfg()
cfg.seed = episode_sort_key(dataset_paths[0])[0]
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(
    cfg,
    device=args.device,
    physics_rate_hz=60,
    solver_iterations=64,
    num_envs=args.batch_size,
    env_spacing=args.env_spacing,
)
cfg.num_rerenders_on_reset = 0

env: ForceBenchEnv | None = None
results: list[dict[str, object]] = []
try:
    env = ForceBenchEnv(
        cfg=cfg,
        task_runtime=task,
        enable_cameras=False,
    )
    if env._enable_cameras or hasattr(env, "_table_camera"):
        raise RuntimeError("camera-free replay unexpectedly created a camera")
    zero_action = torch.zeros(
        (args.batch_size, 7), dtype=torch.float32, device=env.device
    )

    for batch_index, start in enumerate(range(0, len(dataset_paths), args.batch_size)):
        actual_paths = dataset_paths[start : start + args.batch_size]
        episodes = [load_episode(path) for path in actual_paths]
        filler_count = args.batch_size - len(episodes)
        filler_start = 900_000_000 + batch_index * args.batch_size
        seeds = [episode.seed for episode in episodes] + list(
            range(filler_start, filler_start + filler_count)
        )
        if len(seeds) != args.batch_size or len(set(seeds)) != len(seeds):
            raise RuntimeError("failed to construct distinct replay seeds")

        env.set_episode_seeds(seeds)
        env.reset(seed=seeds[0])
        for env_id, episode in enumerate(episodes):
            if env.layouts[env_id] != episode.metadata.get("initial_layout"):
                raise ValueError(
                    f"seed {episode.seed} replay layout differs from recorded layout"
                )

        inspection_completed = np.zeros(args.batch_size, dtype=bool)
        inspection_diagnostics = {
            "frames": np.zeros(args.batch_size, dtype=np.int64),
            "held_seen": np.zeros(args.batch_size, dtype=bool),
            "stationary_seen": np.zeros(args.batch_size, dtype=bool),
            "max_height_m": np.full(args.batch_size, -np.inf, dtype=np.float64),
            "min_gripper_open_fraction": np.full(
                args.batch_size, np.inf, dtype=np.float64
            ),
            "min_tcp_box_distance_m": np.full(
                args.batch_size, np.inf, dtype=np.float64
            ),
            "min_linear_speed_m_s": np.full(
                args.batch_size, np.inf, dtype=np.float64
            ),
            "min_angular_speed_rad_s": np.full(
                args.batch_size, np.inf, dtype=np.float64
            ),
        }
        snapshots: dict[int, dict[str, object]] = {}
        max_frames = max(episode.frames for episode in episodes)
        for frame in range(max_frames):
            if args.task == "rattle_sort":
                update_rattle_inspection(
                    env,
                    episodes,
                    frame,
                    inspection_completed,
                    inspection_diagnostics,
                )

            policy_state = env.policy_state.detach().cpu().numpy()
            targets = policy_state[:, :6].copy()
            grippers = np.clip((policy_state[:, 6] + 1.0) * 0.5, 0.0, 1.0)
            for env_id, episode in enumerate(episodes):
                if frame < episode.frames:
                    command = episode.actions[frame]
                    targets[env_id] = command[:6]
                    grippers[env_id] = np.clip(
                        (float(command[6]) + 1.0) * 0.5, 0.0, 1.0
                    )
            env.set_cartesian_rpy_target(targets, grippers)
            env.step(zero_action)

            for env_id, episode in enumerate(episodes):
                if frame + 1 == episode.frames:
                    replay_inspection = None
                    if args.task == "rattle_sort":
                        replay_inspection = {
                            key: (
                                int(values[env_id])
                                if key == "frames"
                                else bool(values[env_id])
                                if key in ("held_seen", "stationary_seen")
                                else float(values[env_id])
                            )
                            for key, values in inspection_diagnostics.items()
                        }
                    snapshots[env_id] = evaluate_env(
                        env,
                        env_id,
                        episode,
                        bool(inspection_completed[env_id]),
                        replay_inspection,
                        outcome_evaluator,
                    )

        if len(snapshots) != len(episodes):
            raise RuntimeError("not every episode received a final-state snapshot")
        batch_results = [snapshots[index] for index in range(len(episodes))]
        results.extend(batch_results)
        write_payload(results, complete=False)
        print(
            "FORCEBENCH_REPLAY_AUDIT_BATCH="
            + json.dumps(
                {
                    "task": args.task,
                    "batch_index": batch_index,
                    "episodes": len(batch_results),
                    "checked_total": len(results),
                    "successes_total": sum(
                        bool(result["success"]) for result in results
                    ),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    write_payload(results, complete=True)
    failures = [result for result in results if not result["success"]]
    print(
        "FORCEBENCH_REPLAY_AUDIT_RESULT="
        + json.dumps(
            {
                "task": args.task,
                "episodes": len(results),
                "successes": len(results) - len(failures),
                "failures": len(failures),
                "output": str(args.output.resolve()),
                "camera_rendering": False,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(2)
finally:
    if env is not None:
        env.close()
    simulation_app.close()
