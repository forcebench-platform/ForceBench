#!/usr/bin/env python3
"""Collect 32 weight_sort or rattle_sort episodes in cloned workspaces."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("weight_sort", "rattle_sort"), required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--artifact_dir", type=Path, required=True)
parser.add_argument("--seeds", required=True, help="exactly 32 distinct comma-separated seeds")
parser.add_argument(
    "--write_seeds",
    required=True,
    help="subset of --seeds that is missing from the formal dataset and must be written",
)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--video_tile_size", type=int, default=192)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(seed < 0 for seed in values) or len(set(values)) != len(values):
        raise ValueError("seed lists must contain distinct non-negative integers")
    return values


SEEDS = parse_seed_list(args.seeds)
WRITE_SEEDS = frozenset(parse_seed_list(args.write_seeds))
if len(SEEDS) != 32:
    parser.error("the default collector requires exactly 32 cloned workspaces")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env_spacing must be at least 5.5 m")

os.environ.setdefault("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
os.environ["FORCEBENCH_TASK"] = args.task
os.environ.setdefault("FORCEBENCH_EXPERT_BACKEND", "scripted")

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import cv2
import numpy as np
import torch
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import compute_pose_error

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.episode_qc import weight_sort_visual_errors, weight_sort_visual_metrics
from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task
from forcebench.sort_expert_path import (
    bounded_box_feedback_tcp_targets,
    direct_transport_waypoints,
    limit_tcp_command_step,
)

if args.task == "weight_sort":
    from forcebench.tasks.weight_sort.spec import SPEC, evaluate_final_state
else:
    from forcebench.tasks.rattle_sort.spec import SPEC, evaluate_final_state


NUM_ENVS = 32
PHYSICS_RATE_HZ = 60
TRANSPORT_CLEARANCE_Z_M = 0.24
PLACEMENT_POSITION_TOLERANCE_M = 0.008
RELEASE_BOX_GAP_TARGET_M = 0.040
# The smaller rattle parcel can be lowered to 30--50 mm over its destination.
# The weight-sort gripper is wider than the 100 mm container opening and its
# fingers reach the 75 mm rim before the held box can enter that envelope.
# Keep the box just above the rim instead of commanding through the container.
RELEASE_BOX_GAP_RANGE_M = (
    (0.050, 0.065) if args.task == "weight_sort" else (0.030, 0.050)
)
RELEASE_FEEDBACK_GAIN_XYZ = (0.12, 0.12, 0.08)
RELEASE_MAX_COMMAND_STEP_M = 0.003
RELEASE_STABLE_FRAMES = 6
RELEASE_MAX_CONTROL_LINEAR_SPEED_M_S = 0.030
RELEASE_MAX_CONTROL_ANGULAR_SPEED_RAD_S = 0.30
RELEASE_FORCE_GUARD_N = 20.0
RELEASE_COMMAND_Z_SAFETY_MARGIN_M = 0.010
RELEASE_COMMAND_XY_BOUND_M = 0.040
DIAGNOSE_RELEASE_ENVELOPE = os.environ.get(
    "FORCEBENCH_DIAGNOSE_RELEASE_ENVELOPE", "0"
) == "1"
DOWN_RPY = np.asarray([math.pi, 0.0, 0.0], dtype=np.float32)
TASK_PHASES = {
    "weight_sort": (
        "observe_layout",
        "approach_box",
        "grasp_box",
        "lift_and_weigh",
        "transport_to_container",
        "release_box",
        "final_retract_and_settle",
    ),
    "rattle_sort": (
        "observe_layout",
        "approach_box",
        "grasp_box",
        "lift_box",
        "static_inspect",
        "transport_to_destination",
        "release_box",
        "final_retract_and_settle",
    ),
}
PHASE_SCHEMA = {name: index for index, name in enumerate(TASK_PHASES[args.task])}
SPEED_PROFILE = {
    "convergence_stable_frames": 6,
    "observe_frames": 12,
    "grasp_close_frames": 40,
    "inspection_hold_frames": 35,
    "pre_release_hold_frames": 12,
    "release_open_frames": 40,
    "post_release_hold_frames": 24,
    "final_settle_frames": 70,
}


def as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def rgb_batch(value) -> np.ndarray:
    return as_numpy(value)[..., :3].astype(np.uint8)


def make_mosaic(images: np.ndarray, phase: str, env: ForceBenchEnv) -> np.ndarray:
    size = int(args.video_tile_size)
    rows, columns = 4, 8
    canvas = np.zeros((rows * size, columns * size, 3), dtype=np.uint8)
    wrench = as_numpy(env.force_sensor_joint_raw)
    for env_id, image in enumerate(images):
        tile = cv2.resize(image[..., ::-1], (size, size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (size, 39), (18, 18, 18), -1)
        layout = env.layouts[env_id]
        kind = str(layout["box_kind"])
        write_mark = "REC" if SEEDS[env_id] in WRITE_SEEDS else "FILL"
        cv2.putText(
            tile,
            f"e{env_id:02d} s{SEEDS[env_id]:03d} {kind} {write_mark}",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        signal = (
            f"Fz {wrench[env_id, 2]:+.2f}N"
            if args.task == "weight_sort"
            else f"|T| {np.linalg.norm(wrench[env_id, 3:]):.3f}Nm"
        )
        cv2.putText(
            tile,
            f"{phase[:18]} {signal}",
            (4, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.27,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(env_id, columns)
        canvas[row * size : (row + 1) * size, column * size : (column + 1) * size] = tile
    return canvas


def write_overview(path: Path, env: ForceBenchEnv, phase: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), make_mosaic(rgb_batch(env.table_rgb), phase, env))


def reset_visual_qc(env: ForceBenchEnv) -> list[dict[str, object]]:
    main = rgb_batch(env.table_rgb)
    wrist = rgb_batch(env.wrist_rgb)
    reports: list[dict[str, object]] = []
    failures: list[str] = []
    for env_id in range(NUM_ENVS):
        if args.task == "weight_sort":
            metrics = weight_sort_visual_metrics(main[env_id], wrist[env_id])
            errors = weight_sort_visual_errors(metrics)
        else:
            metrics = {
                "main_mean_intensity": float(np.mean(main[env_id])),
                "main_dark_fraction": float(np.mean(np.max(main[env_id], axis=2) < 45)),
                "wrist_mean_intensity": float(np.mean(wrist[env_id])),
            }
            errors = tuple(
                message
                for condition, message in (
                    (
                        metrics["main_mean_intensity"] < 120.0,
                        f"main mean intensity {metrics['main_mean_intensity']:.2f} below 120",
                    ),
                    (
                        metrics["main_dark_fraction"] > 0.20,
                        f"main dark fraction {metrics['main_dark_fraction']:.3f} above 0.20",
                    ),
                    (
                        metrics["wrist_mean_intensity"] < 15.0,
                        f"wrist mean intensity {metrics['wrist_mean_intensity']:.2f} below 15",
                    ),
                )
                if condition
            )
        if errors:
            failures.append(f"env_{env_id}: {'; '.join(errors)}")
        reports.append({"env_id": env_id, "seed": SEEDS[env_id], "metrics": metrics})
    if failures:
        raise RuntimeError("reset visual QC failed: " + " | ".join(failures))
    return reports


def weight_container_extents() -> dict[str, list[float]]:
    if args.task != "weight_sort":
        return {}
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is unavailable")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], True
    )
    expected = np.asarray(SPEC.scaled_container_size_m, dtype=np.float64)
    result: dict[str, list[float]] = {}
    for env_id in range(NUM_ENVS):
        for color in ("black", "blue"):
            path = f"/World/envs/env_{env_id}/Task_{color}_container"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsActive():
                raise RuntimeError(f"missing active container prim: {path}")
            actual = np.asarray(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize())
            if not np.allclose(actual, expected, rtol=0.08, atol=0.01):
                raise RuntimeError(
                    f"container bounds mismatch at {path}: {actual.tolist()} != {expected.tolist()}"
                )
            result[path] = [float(value) for value in actual]
    return result


class ParallelSortExpert:
    def __init__(self, env: ForceBenchEnv, video_path: Path):
        self.env = env
        self.buffers = [EpisodeBuffer() for _ in range(NUM_ENVS)]
        self.zero_action = torch.zeros((NUM_ENVS, 7), dtype=torch.float32, device=env.device)
        self.down = math_utils.quat_from_euler_xyz(
            torch.full((NUM_ENVS,), torch.pi, device=env.device),
            torch.zeros(NUM_ENVS, device=env.device),
            torch.zeros(NUM_ENVS, device=env.device),
        )
        self.phase_ranges: list[dict[str, int | str]] = []
        self.sampled_lift_heights_m = np.asarray(
            [
                np.random.default_rng(np.random.SeedSequence([seed, 0x4C494654])).uniform(
                    0.12, 0.18
                )
                for seed in SEEDS
            ],
            dtype=np.float32,
        )
        self.lift_tcp_targets_z_m = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.lifted_box_heights_m = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.inspection_box_heights_m = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.load_signals_n = np.zeros(NUM_ENVS, dtype=np.float64)
        self.static_force_samples: list[np.ndarray] = []
        self.static_torque_samples: list[np.ndarray] = []
        self.inspection_hold_s = np.zeros(NUM_ENVS, dtype=np.float32)
        self.inspection_max_hold_s = np.zeros(NUM_ENVS, dtype=np.float32)
        self.inspection_completed = np.zeros(NUM_ENVS, dtype=bool)
        self.inspection_previous_box_position = None
        self.inspection_previous_box_quaternion = None
        self.inspection_max_position_step_m = np.zeros(NUM_ENVS, dtype=np.float32)
        self.inspection_max_rotation_step_rad = np.zeros(NUM_ENVS, dtype=np.float32)
        self.release_envelope_diagnostics: list[dict[str, object]] = []
        video_path.parent.mkdir(parents=True, exist_ok=True)
        size = int(args.video_tile_size)
        self.video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (8 * size, 4 * size)
        )
        if not self.video.isOpened():
            raise RuntimeError(f"failed to create video: {video_path}")

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    def _update_rattle_inspection(self, phase: str) -> None:
        if args.task != "rattle_sort" or phase != "static_inspect":
            return
        box_position, box_quaternion = self.env.entity_local_pose("box")
        distance = torch.linalg.vector_norm(box_position - self.env.fingertip_midpoint_pos, dim=1)
        held = (
            (box_position[:, 2] >= SPEC.inspection_min_lift_height_m)
            & (self.env.gripper_open_fraction_tensor <= SPEC.inspection_max_gripper_open_fraction)
            & (distance <= SPEC.inspection_max_tcp_box_distance_m)
        )
        if self.inspection_previous_box_position is None:
            position_step = torch.full(
                (NUM_ENVS,), float("inf"), dtype=torch.float32, device=self.env.device
            )
            rotation_step = position_step.clone()
        else:
            position_step = torch.linalg.vector_norm(
                box_position - self.inspection_previous_box_position, dim=1
            )
            quaternion_dot = torch.abs(
                torch.sum(
                    box_quaternion * self.inspection_previous_box_quaternion, dim=1
                )
            ).clamp(max=1.0)
            rotation_step = 2.0 * torch.acos(quaternion_dot)
        self.inspection_previous_box_position = box_position.clone()
        self.inspection_previous_box_quaternion = box_quaternion.clone()
        finite_position_step = torch.where(
            torch.isfinite(position_step), position_step, torch.zeros_like(position_step)
        )
        finite_rotation_step = torch.where(
            torch.isfinite(rotation_step), rotation_step, torch.zeros_like(rotation_step)
        )
        self.inspection_max_position_step_m = np.maximum(
            self.inspection_max_position_step_m, as_numpy(finite_position_step)
        )
        self.inspection_max_rotation_step_rad = np.maximum(
            self.inspection_max_rotation_step_rad, as_numpy(finite_rotation_step)
        )
        stationary = (
            (position_step <= SPEC.inspection_max_control_position_step_m)
            & (rotation_step <= SPEC.inspection_max_control_rotation_step_rad)
        )
        valid = as_numpy(held & stationary)
        self.inspection_hold_s = np.where(
            valid, self.inspection_hold_s + self.env.step_dt, 0.0
        ).astype(np.float32)
        self.inspection_max_hold_s = np.maximum(
            self.inspection_max_hold_s, self.inspection_hold_s
        )
        self.inspection_completed |= (
            self.inspection_hold_s + np.float32(1.0e-6)
            >= np.float32(SPEC.inspection_required_hold_s)
        )

    def _privileged(self, box_pose: np.ndarray, env_id: int) -> dict[str, np.ndarray]:
        fields = {"objects/box/pose": box_pose[env_id]}
        if args.task == "rattle_sort":
            fields.update(
                {
                    "objects/box/center_of_mass_offset_local": np.asarray(
                        self.env.layouts[env_id]["center_of_mass_offset_local_m"],
                        dtype=np.float32,
                    ),
                    "inspection/static_hold_state": np.asarray(
                        [
                            self.inspection_hold_s[env_id],
                            float(self.inspection_completed[env_id]),
                        ],
                        dtype=np.float32,
                    ),
                }
            )
        return fields

    def _record_and_step(self, target_m: np.ndarray, gripper, phase: str) -> None:
        targets = np.asarray(target_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        grippers = np.asarray(gripper, dtype=np.float32)
        if grippers.ndim == 0:
            grippers = np.full(NUM_ENVS, float(grippers), dtype=np.float32)
        actions = np.concatenate(
            (
                targets,
                np.repeat(DOWN_RPY[None, :], NUM_ENVS, axis=0),
                (2.0 * grippers - 1.0)[:, None],
            ),
            axis=1,
        ).astype(np.float32)
        self._update_rattle_inspection(phase)
        states = as_numpy(self.env.policy_state)
        wrench_raw = as_numpy(self.env.force_sensor_joint_raw)
        wrench_ema = as_numpy(self.env.force_sensor_joint_smooth)
        main_images = rgb_batch(self.env.table_rgb)
        wrist_images = rgb_batch(self.env.wrist_rgb)
        joint_position = as_numpy(self.env.robot.data.joint_pos.torch)
        joint_velocity = as_numpy(self.env.robot.data.joint_vel.torch)
        tcp_pose = as_numpy(
            torch.cat((self.env.fingertip_midpoint_pos, self.env.fingertip_midpoint_quat), dim=-1)
        )
        box_position, box_quaternion = self.env.entity_local_pose("box")
        box_pose = as_numpy(torch.cat((box_position, box_quaternion), dim=-1))
        timestamp = (len(self.buffers[0]) + 1) * self.env.step_dt
        for env_id, buffer in enumerate(self.buffers):
            buffer.append(
                state=states[env_id],
                action=actions[env_id],
                wrench_raw=wrench_raw[env_id],
                wrench_ema=wrench_ema[env_id],
                timestamp=timestamp,
                image=main_images[env_id],
                wrist_image=wrist_images[env_id],
                joint_position=joint_position[env_id],
                joint_velocity=joint_velocity[env_id],
                tcp_pose=tcp_pose[env_id],
                privileged=self._privileged(box_pose, env_id),
                phase_id=PHASE_SCHEMA[phase],
            )
        if phase == "static_inspect":
            self.static_force_samples.append(np.linalg.norm(wrench_raw[:, :3], axis=1))
            self.static_torque_samples.append(np.linalg.norm(wrench_raw[:, 3:], axis=1))
        self.video.write(make_mosaic(main_images, phase, self.env))
        self.env.set_cartesian_rpy_target(actions[:, :6], grippers)
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        for env_id, buffer in enumerate(self.buffers):
            buffer.set_last_transition_outcome(
                reward=float(reward[env_id]),
                success=False,
                terminated=bool(terminated[env_id]),
                truncated=bool(truncated[env_id]),
            )
        self.env.update_cameras()

    def move(
        self,
        target_m,
        gripper,
        phase: str,
        *,
        max_frames: int = 220,
        tolerance_m: float = 0.004,
    ) -> None:
        targets = np.asarray(target_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        grippers = np.asarray(gripper, dtype=np.float32)
        start = len(self.buffers[0])
        stable = torch.zeros(NUM_ENVS, dtype=torch.int64, device=self.env.device)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=self.env.device)
        position_error = torch.full((NUM_ENVS,), float("inf"), device=self.env.device)
        rotation_error = torch.full_like(position_error, float("inf"))
        for _ in range(max_frames):
            self._record_and_step(targets, grippers, phase)
            position_error = torch.linalg.vector_norm(
                self.env.fingertip_midpoint_pos - target_tensor, dim=1
            )
            _, rotation = compute_pose_error(
                self.env.fingertip_midpoint_pos,
                self.env.fingertip_midpoint_quat,
                target_tensor,
                self.down,
                rot_error_type="axis_angle",
            )
            rotation_error = torch.linalg.vector_norm(rotation, dim=1)
            stable = torch.where(
                position_error <= tolerance_m, stable + 1, torch.zeros_like(stable)
            )
            if bool(torch.all(stable >= SPEED_PROFILE["convergence_stable_frames"])):
                break
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})
        if not bool(torch.all(stable >= SPEED_PROFILE["convergence_stable_frames"])):
            bad = as_numpy(stable < SPEED_PROFILE["convergence_stable_frames"])
            actual_tcp = as_numpy(self.env.fingertip_midpoint_pos)
            box_position, _ = self.env.entity_local_pose("box")
            actual_box = as_numpy(box_position)
            details = [
                {
                    "env_id": index,
                    "seed": SEEDS[index],
                    "position_error_m": float(as_numpy(position_error)[index]),
                    "rotation_error_rad": float(as_numpy(rotation_error)[index]),
                    "target_tcp_position_m": targets[index].tolist(),
                    "actual_tcp_position_m": actual_tcp[index].tolist(),
                    "tcp_minus_target_m": (
                        actual_tcp[index] - targets[index]
                    ).tolist(),
                    "box_position_m": actual_box[index].tolist(),
                    "tcp_minus_box_m": (
                        actual_tcp[index] - actual_box[index]
                    ).tolist(),
                }
                for index in map(int, np.flatnonzero(bad))
            ]
            raise RuntimeError(f"phase {phase} did not converge: {json.dumps(details)}")
        print(
            "FORCEBENCH_PARALLEL_SORT_PHASE="
            + json.dumps(
                {
                    "task": args.task,
                    "phase": phase,
                    "frames": len(self.buffers[0]) - start,
                    "max_position_error_m": float(torch.max(position_error).item()),
                    "max_rotation_error_rad": float(torch.max(rotation_error).item()),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    def hold(self, frames: int, gripper, phase: str, *, measure_load: bool = False) -> None:
        target = as_numpy(self.env.fingertip_midpoint_pos)
        start = len(self.buffers[0])
        samples: list[np.ndarray] = []
        for _ in range(frames):
            self._record_and_step(target, gripper, phase)
            if measure_load:
                samples.append(
                    as_numpy(torch.linalg.vector_norm(self.env.force_sensor_joint_raw[:, :3], dim=1))
                )
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})
        if samples:
            values = np.stack(samples)
            self.load_signals_n = np.mean(values[len(values) // 2 :], axis=0)

    def move_box_to_release_envelope(
        self,
        destination_xy_m: np.ndarray,
        surface_z_m: np.ndarray,
        gripper,
        phase: str,
        *,
        max_frames: int = 300,
    ) -> None:
        destinations = np.asarray(destination_xy_m, dtype=np.float32).reshape(NUM_ENVS, 2)
        surfaces = np.asarray(surface_z_m, dtype=np.float32).reshape(NUM_ENVS)
        target_box = np.column_stack(
            (
                destinations,
                surfaces + np.float32(RELEASE_BOX_GAP_TARGET_M),
            )
        ).astype(np.float32)
        stable = np.zeros(NUM_ENVS, dtype=np.int64)
        converged_frame = np.full(NUM_ENVS, -1, dtype=np.int64)
        maximum_command_step = np.zeros(NUM_ENVS, dtype=np.float32)
        maximum_force = np.zeros(NUM_ENVS, dtype=np.float32)
        force_guard_triggered = np.zeros(NUM_ENVS, dtype=bool)
        force_guard_first_frame = np.full(NUM_ENVS, -1, dtype=np.int64)
        minimum_gap = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        previous_box_position = None
        previous_box_quaternion = None
        box_position, _ = self.env.entity_local_pose("box")
        command_tcp = as_numpy(self.env.fingertip_midpoint_pos).copy()
        tcp_box_offset_z = command_tcp[:, 2] - as_numpy(box_position)[:, 2]
        command_z_floor = (
            surfaces
            + np.float32(RELEASE_BOX_GAP_RANGE_M[0])
            + np.maximum(tcp_box_offset_z, np.float32(0.0))
            + np.float32(RELEASE_COMMAND_Z_SAFETY_MARGIN_M)
        )
        start = len(self.buffers[0])
        final_xy_error = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        final_gap = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        final_linear_speed = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        final_angular_speed = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        for frame in range(max_frames):
            box_position, _ = self.env.entity_local_pose("box")
            targets, command_step = bounded_box_feedback_tcp_targets(
                command_tcp,
                as_numpy(box_position),
                target_box,
                gain=RELEASE_FEEDBACK_GAIN_XYZ,
                max_command_step_m=RELEASE_MAX_COMMAND_STEP_M,
            )
            targets[:, :2] = np.clip(
                targets[:, :2],
                destinations - np.float32(RELEASE_COMMAND_XY_BOUND_M),
                destinations + np.float32(RELEASE_COMMAND_XY_BOUND_M),
            )
            force_before = as_numpy(
                torch.linalg.vector_norm(
                    self.env.force_sensor_joint_raw[:, :3], dim=1
                )
            )
            newly_guarded = (~force_guard_triggered) & (
                force_before > RELEASE_FORCE_GUARD_N
            )
            force_guard_first_frame[newly_guarded] = frame + 1
            force_guard_triggered |= force_before > RELEASE_FORCE_GUARD_N
            descending_while_guarded = force_guard_triggered & (
                targets[:, 2] < command_tcp[:, 2]
            )
            current_gap = as_numpy(box_position)[:, 2] - surfaces
            descending_below_gap_floor = (
                current_gap < RELEASE_BOX_GAP_RANGE_M[0]
            ) & (targets[:, 2] < command_tcp[:, 2])
            descending_while_guarded |= descending_below_gap_floor
            targets[descending_while_guarded, 2] = command_tcp[
                descending_while_guarded, 2
            ]
            targets[:, 2] = np.maximum(targets[:, 2], command_z_floor)
            targets, command_step = limit_tcp_command_step(
                command_tcp,
                targets,
                max_command_step_m=RELEASE_MAX_COMMAND_STEP_M,
            )
            command_norm = np.linalg.norm(command_step, axis=1)
            if np.any(command_norm > RELEASE_MAX_COMMAND_STEP_M + 1.0e-5):
                raise RuntimeError("release command limiter violated its 3D norm bound")
            command_tcp = targets.copy()
            maximum_command_step = np.maximum(
                maximum_command_step, np.linalg.norm(command_step, axis=1)
            )
            self._record_and_step(targets, gripper, phase)
            box_position, box_quaternion = self.env.entity_local_pose("box")
            position = as_numpy(box_position)
            quaternion = box_quaternion
            final_xy_error = np.linalg.norm(position[:, :2] - destinations, axis=1)
            final_gap = position[:, 2] - surfaces
            minimum_gap = np.minimum(minimum_gap, final_gap)
            if previous_box_position is None:
                control_linear_speed = np.full(NUM_ENVS, np.inf, dtype=np.float32)
                control_angular_speed = np.full(NUM_ENVS, np.inf, dtype=np.float32)
            else:
                control_linear_speed = as_numpy(
                    torch.linalg.vector_norm(
                        box_position - previous_box_position, dim=1
                    )
                ) / np.float32(self.env.step_dt)
                quaternion_dot = torch.abs(
                    torch.sum(quaternion * previous_box_quaternion, dim=1)
                ).clamp(max=1.0)
                control_angular_speed = as_numpy(
                    2.0 * torch.acos(quaternion_dot)
                ) / np.float32(self.env.step_dt)
            previous_box_position = box_position.clone()
            previous_box_quaternion = quaternion.clone()
            final_linear_speed = control_linear_speed
            final_angular_speed = control_angular_speed
            maximum_force = np.maximum(
                maximum_force,
                as_numpy(
                    torch.linalg.vector_norm(
                        self.env.force_sensor_joint_raw[:, :3], dim=1
                    )
                ),
            )
            within = (
                (final_xy_error <= PLACEMENT_POSITION_TOLERANCE_M)
                & (final_gap >= RELEASE_BOX_GAP_RANGE_M[0])
                & (final_gap <= RELEASE_BOX_GAP_RANGE_M[1])
                & (final_linear_speed <= RELEASE_MAX_CONTROL_LINEAR_SPEED_M_S)
                & (final_angular_speed <= RELEASE_MAX_CONTROL_ANGULAR_SPEED_RAD_S)
            )
            stable = np.where(within, stable + 1, 0)
            newly_converged = (converged_frame < 0) & (stable >= RELEASE_STABLE_FRAMES)
            converged_frame[newly_converged] = frame + 1
            if np.all(stable >= RELEASE_STABLE_FRAMES):
                break
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})
        diagnostics = [
            {
                "env_id": env_id,
                "seed": SEEDS[env_id],
                "converged": bool(converged_frame[env_id] >= 0),
                "converged_control_frames": int(converged_frame[env_id]),
                "max_command_step_m": float(maximum_command_step[env_id]),
                "box_xy_error_m": float(final_xy_error[env_id]),
                "box_bottom_surface_gap_m": float(final_gap[env_id]),
                "control_observed_linear_speed_m_s": float(
                    final_linear_speed[env_id]
                ),
                "control_observed_angular_speed_rad_s": float(
                    final_angular_speed[env_id]
                ),
                "max_wrist_force_norm_n": float(maximum_force[env_id]),
                "minimum_box_bottom_surface_gap_m": float(minimum_gap[env_id]),
                "force_guard_triggered": bool(force_guard_triggered[env_id]),
                "force_guard_first_frame": int(force_guard_first_frame[env_id]),
                "command_z_floor_m": float(command_z_floor[env_id]),
            }
            for env_id in range(NUM_ENVS)
        ]
        self.release_envelope_diagnostics = diagnostics
        print(
            "FORCEBENCH_RELEASE_ENVELOPE="
            + json.dumps(diagnostics, separators=(",", ":")),
            flush=True,
        )
        if not np.all(stable >= RELEASE_STABLE_FRAMES):
            raise RuntimeError(
                "release envelope did not converge: "
                + json.dumps(
                    [item for item in diagnostics if not item["converged"]],
                    separators=(",", ":"),
                )
            )
        if DIAGNOSE_RELEASE_ENVELOPE:
            raise RuntimeError("release envelope diagnostic completed without opening gripper")

    def _pickup(self, lift_phase: str, inspection_phase: str | None) -> None:
        open_gripper = np.ones(NUM_ENVS, dtype=np.float32)
        closed_gripper = np.zeros(NUM_ENVS, dtype=np.float32)
        box_position, _ = self.env.entity_local_pose("box")
        xyz = as_numpy(box_position)
        above = xyz.copy()
        above[:, 2] = 0.12
        self.move(above, open_gripper, "approach_box")
        grasp = xyz.copy()
        grasp[:, 2] = 0.018 if args.task == "weight_sort" else 0.014
        self.move(grasp, open_gripper, "approach_box")
        self.hold(SPEED_PROFILE["grasp_close_frames"], closed_gripper, "grasp_box")
        grasped, _ = self.env.entity_local_pose("box")
        offsets = as_numpy(self.env.fingertip_midpoint_pos[:, 2] - grasped[:, 2])
        self.lift_tcp_targets_z_m = self.sampled_lift_heights_m + offsets
        lift = xyz.copy()
        lift[:, 2] = self.lift_tcp_targets_z_m
        self.move(lift, closed_gripper, lift_phase, tolerance_m=0.025)
        if args.task == "weight_sort":
            self.hold(
                SPEED_PROFILE["inspection_hold_frames"],
                closed_gripper,
                lift_phase,
                measure_load=True,
            )
        else:
            self.hold(
                SPEED_PROFILE["inspection_hold_frames"],
                closed_gripper,
                str(inspection_phase),
            )
            if not np.all(self.inspection_completed):
                details = [
                    {
                        "env_id": int(env_id),
                        "seed": SEEDS[env_id],
                        "max_consecutive_hold_s": float(
                            self.inspection_max_hold_s[env_id]
                        ),
                        "max_position_step_m": float(
                            self.inspection_max_position_step_m[env_id]
                        ),
                        "max_rotation_step_rad": float(
                            self.inspection_max_rotation_step_rad[env_id]
                        ),
                    }
                    for env_id in np.flatnonzero(~self.inspection_completed)
                ]
                raise RuntimeError(
                    "rattle static inspection did not complete: "
                    + json.dumps(details, separators=(",", ":"))
                )
        lifted, _ = self.env.entity_local_pose("box")
        self.lifted_box_heights_m = as_numpy(lifted[:, 2])
        if np.any(self.lifted_box_heights_m < 0.075):
            bad = np.flatnonzero(self.lifted_box_heights_m < 0.075).tolist()
            raise RuntimeError(f"boxes were not lifted in envs {bad}")
        if args.task == "rattle_sort":
            self.inspection_box_heights_m = self.lifted_box_heights_m.copy()

    def _run_weight(self) -> list[dict[str, object]]:
        open_gripper = np.ones(NUM_ENVS, dtype=np.float32)
        closed_gripper = np.zeros(NUM_ENVS, dtype=np.float32)
        self.hold(SPEED_PROFILE["observe_frames"], open_gripper, "observe_layout")
        self._pickup("lift_and_weigh", None)
        is_light = np.asarray([layout["box_kind"] == "light" for layout in self.env.layouts])
        destination = np.asarray(
            [
                [*SPEC.black_container_center_xy_m, 0.20]
                if light
                else [*SPEC.blue_container_center_xy_m, 0.20]
                for light in is_light
            ],
            dtype=np.float32,
        )
        transport_route = direct_transport_waypoints(
            as_numpy(self.env.fingertip_midpoint_pos),
            destination,
            clearance_z_m=TRANSPORT_CLEARANCE_Z_M,
            release_z_m=0.12,
        )
        for index, waypoint in enumerate(transport_route[:2]):
            self.move(
                waypoint,
                closed_gripper,
                "transport_to_container",
                max_frames=220 if index < 2 else 300,
                tolerance_m=0.020,
            )
        self.move_box_to_release_envelope(
            destination[:, :2],
            np.zeros(NUM_ENVS, dtype=np.float32),
            closed_gripper,
            "transport_to_container",
        )
        self.hold(
            SPEED_PROFILE["pre_release_hold_frames"],
            closed_gripper,
            "transport_to_container",
        )
        self.hold(SPEED_PROFILE["release_open_frames"], open_gripper, "release_box")
        retract = destination.copy()
        retract[:, 2] = 0.27
        self.move(retract, open_gripper, "release_box", max_frames=180)
        self.hold(SPEED_PROFILE["post_release_hold_frames"], open_gripper, "release_box")
        self.hold(
            SPEED_PROFILE["final_settle_frames"], open_gripper, "final_retract_and_settle"
        )
        final_position, _ = self.env.entity_local_pose("box")
        final_velocity = self.env.entity("box").data.root_lin_vel_w.torch
        gripper = as_numpy(self.env.gripper_open_fraction_tensor)
        reports = []
        for env_id, layout in enumerate(self.env.layouts):
            report = evaluate_final_state(
                box_kind=str(layout["box_kind"]),
                box_position_m=as_numpy(final_position[env_id]),
                box_linear_velocity_m_s=as_numpy(final_velocity[env_id]),
                gripper_open_fraction=float(gripper[env_id]),
            ).to_dict()
            reports.append(report)
        return reports

    def _run_rattle(self) -> list[dict[str, object]]:
        open_gripper = np.ones(NUM_ENVS, dtype=np.float32)
        closed_gripper = np.zeros(NUM_ENVS, dtype=np.float32)
        self.hold(SPEED_PROFILE["observe_frames"], open_gripper, "observe_layout")
        self._pickup("lift_box", "static_inspect")
        imbalanced = np.asarray(
            [layout["box_kind"] == "imbalanced" for layout in self.env.layouts]
        )
        destination = np.asarray(
            [
                [*SPEC.black_basket_center_xy_m, 0.20]
                if value
                else [*SPEC.conveyor_center_xy_m, 0.20]
                for value in imbalanced
            ],
            dtype=np.float32,
        )
        transport_route = direct_transport_waypoints(
            as_numpy(self.env.fingertip_midpoint_pos),
            destination,
            clearance_z_m=TRANSPORT_CLEARANCE_Z_M,
            release_z_m=0.12,
        )
        for index, waypoint in enumerate(transport_route[:2]):
            self.move(
                waypoint,
                closed_gripper,
                "transport_to_destination",
                max_frames=220 if index < 2 else 300,
                tolerance_m=0.020,
            )
        release_surfaces = np.where(
            imbalanced,
            np.float32(SPEC.black_basket_visual_floor_z_m),
            np.float32(SPEC.conveyor_surface_z_m),
        )
        self.move_box_to_release_envelope(
            destination[:, :2],
            release_surfaces,
            closed_gripper,
            "transport_to_destination",
        )
        self.hold(
            SPEED_PROFILE["pre_release_hold_frames"],
            closed_gripper,
            "transport_to_destination",
        )
        self.hold(SPEED_PROFILE["release_open_frames"], open_gripper, "release_box")
        retract = destination.copy()
        retract[:, 2] = 0.27
        self.move(retract, open_gripper, "release_box", max_frames=180)
        self.hold(SPEED_PROFILE["post_release_hold_frames"], open_gripper, "release_box")
        self.hold(
            SPEED_PROFILE["final_settle_frames"], open_gripper, "final_retract_and_settle"
        )
        final_position, _ = self.env.entity_local_pose("box")
        final_velocity = self.env.entity("box").data.root_lin_vel_w.torch
        gripper = as_numpy(self.env.gripper_open_fraction_tensor)
        reports = []
        for env_id, layout in enumerate(self.env.layouts):
            report = evaluate_final_state(
                box_kind=str(layout["box_kind"]),
                box_position_m=as_numpy(final_position[env_id]),
                box_linear_velocity_m_s=as_numpy(final_velocity[env_id]),
                inspection_completed=bool(self.inspection_completed[env_id]),
                gripper_open_fraction=float(gripper[env_id]),
            ).to_dict()
            reports.append(report)
        return reports

    def run(self) -> list[dict[str, object]]:
        reports = self._run_weight() if args.task == "weight_sort" else self._run_rattle()
        for buffer, report in zip(self.buffers, reports, strict=True):
            if not report["success"]:
                continue
            buffer.set_last_transition_outcome(
                reward=1.0, success=True, terminated=False, truncated=False
            )
        return reports

    def details(self, env_id: int) -> dict[str, object]:
        layout = self.env.layouts[env_id]
        common = {
            "box_kind": layout["box_kind"],
            "box_mass_kg": float(layout["box_mass_kg"]),
            "lift_box_height_range_m": [0.12, 0.18],
            "sampled_lift_box_height_m": float(self.sampled_lift_heights_m[env_id]),
            "lift_tcp_target_z_m": float(self.lift_tcp_targets_z_m[env_id]),
            "lifted_box_height_m": float(self.lifted_box_heights_m[env_id]),
            "destination": layout["destination"],
            "phase_ranges_control_frames": list(self.phase_ranges),
            "release_envelope": (
                self.release_envelope_diagnostics[env_id]
                if self.release_envelope_diagnostics
                else None
            ),
        }
        if args.task == "weight_sort":
            common["load_signals_n"] = {"box": float(self.load_signals_n[env_id])}
        else:
            force = np.stack(self.static_force_samples)[:, env_id]
            torque = np.stack(self.static_torque_samples)[:, env_id]
            common.update(
                {
                    "total_mass_kg": float(layout["total_mass_kg"]),
                    "center_of_mass_offset_local_m": layout[
                        "center_of_mass_offset_local_m"
                    ],
                    "heavy_end": layout["heavy_end"],
                    "inspection_box_height_m": float(
                        self.inspection_box_heights_m[env_id]
                    ),
                    "static_force_mean_n": float(np.mean(force)),
                    "static_torque_mean_nm": float(np.mean(torque)),
                    "inspection_state": {
                        "inspection_hold_s": float(self.inspection_hold_s[env_id]),
                        "inspection_max_hold_s": float(
                            self.inspection_max_hold_s[env_id]
                        ),
                        "inspection_completed": bool(
                            self.inspection_completed[env_id]
                        ),
                        "required_hold_s": float(SPEC.inspection_required_hold_s),
                        "max_position_step_m": float(
                            self.inspection_max_position_step_m[env_id]
                        ),
                        "max_rotation_step_rad": float(
                            self.inspection_max_rotation_step_rad[env_id]
                        ),
                    },
                }
            )
        return common


task = load_task(args.task, ASSET_DIR)
cfg = ForceBenchEnvCfg()
cfg.seed = SEEDS[0]
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(
    cfg,
    device=args.device,
    physics_rate_hz=PHYSICS_RATE_HZ,
    solver_iterations=64,
    num_envs=NUM_ENVS,
    env_spacing=args.env_spacing,
)
cfg.num_rerenders_on_reset = 1
env = ForceBenchEnv(
    cfg=cfg,
    task_runtime=task,
    main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size,
    enable_cameras=True,
)
expert: ParallelSortExpert | None = None
args.artifact_dir.mkdir(parents=True, exist_ok=True)
video_path = args.artifact_dir / f"{args.task}_32env_mosaic.mp4"
overview_path = args.artifact_dir / f"{args.task}_32env_initial.png"
final_overview_path = args.artifact_dir / f"{args.task}_32env_final.png"

try:
    env.set_episode_seeds(SEEDS)
    env.reset(seed=SEEDS[0])
    env.reset_cameras()
    for _ in range(3):
        env.sim.render()
    env.update_cameras()
    extents = weight_container_extents()
    write_overview(overview_path, env, "initial_reset")
    visual_qc = reset_visual_qc(env)

    env.start_wrench_capture()
    expert = ParallelSortExpert(env, video_path)
    reports = expert.run()
    high_rate_batch = env.stop_wrench_capture()
    expert.close()
    write_overview(final_overview_path, env, "final_settled")

    task_metadata = dict(task.episode_metadata())
    asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
    asset_paths.update(env.scene_asset_paths)
    asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
    writer = HDF5EpisodeWriter(args.output_dir)
    physics_wrench = np.asarray(high_rate_batch["eef_wrench_joint_raw"], dtype=np.float32)
    written: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for env_id, seed in enumerate(SEEDS):
        if seed not in WRITE_SEEDS:
            continue
        if not reports[env_id]["success"]:
            failure = {"env_id": env_id, "seed": seed, "report": reports[env_id]}
            failed.append(failure)
            print(
                "FORCEBENCH_PARALLEL_SORT_FINAL_FAILURE="
                + json.dumps(failure, separators=(",", ":")),
                flush=True,
            )
            continue
        existing = sorted(args.output_dir.glob(f"seed_{seed:03d}_episode_*.hdf5"))
        if existing:
            raise RuntimeError(
                f"refusing to duplicate successful seed {seed}: {[str(path) for path in existing]}"
            )
        layout = env.layouts[env_id]
        if args.task == "weight_sort":
            end_reason = "scripted_parallel_weight_sort_final_check"
            control_source = "scripted_32env_single_process_weight_sort_pose_oracle"
            version = "weight_sort_32env_tiled_v8_dynamic_route_box_feedback_release_physics60"
        else:
            end_reason = "scripted_parallel_rattle_sort_final_check"
            control_source = "scripted_32env_single_process_rattle_sort_pose_oracle"
            version = "rattle_sort_32env_tiled_v8_basket_safe_pickup_rectangle_thick_floor_bounded_pi_box_feedback_static2s_physics60"
        metadata = {
            "task": args.task,
            "prompt": task.manifest.prompt,
            "success": True,
            "end_reason": end_reason,
            "seed": seed,
            "initial_layout": layout,
            "final_success_report": reports[env_id],
            "control_rate_hz": 15.0,
            "physics_rate_hz": float(PHYSICS_RATE_HZ),
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "arm_controller_diagnostics": env.arm_controller_diagnostics,
            "scene_profile": env.scene_profile_name,
            "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "control_source": control_source,
            "transport_path_profile": {
                "type": "current_tcp_raise_layout_destination_midpoint_release_retract",
                "clearance_tcp_z_m": TRANSPORT_CLEARANCE_Z_M,
                "release_box_gap_target_m": RELEASE_BOX_GAP_TARGET_M,
                "release_box_gap_range_m": list(RELEASE_BOX_GAP_RANGE_M),
                "placement_position_tolerance_m": PLACEMENT_POSITION_TOLERANCE_M,
                "release_feedback_gain_xyz": list(RELEASE_FEEDBACK_GAIN_XYZ),
                "release_max_command_step_m": RELEASE_MAX_COMMAND_STEP_M,
                "release_stable_frames": RELEASE_STABLE_FRAMES,
                "release_force_guard_n": RELEASE_FORCE_GUARD_N,
                "release_command_z_safety_margin_m": (
                    RELEASE_COMMAND_Z_SAFETY_MARGIN_M
                ),
            },
            "wrench_profile": env.wrench_profile,
            "wrench_storage": "raw_and_ema",
            "primary_wrist_wrench_dataset": "forces/control/eef_wrench_joint_raw",
            "recorded_wrist_wrench_datasets": [
                "forces/control/eef_wrench_joint_raw",
                "forces/control/eef_wrench_joint_ema",
            ],
            "wrench_semantics": (
                "panda_hand_joint incoming constraint reaction wrench, parent-on-child, "
                "joint frame, Fx_Fy_Fz_Tx_Ty_Tz; raw is primary and EMA(alpha=0.25) "
                "is stored as an auxiliary channel"
            ),
            "success_check_timing": "only_after_box_released_and_settled",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [args.main_image_size, args.main_image_size, 3],
                "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
            },
            "phase_schema": PHASE_SCHEMA,
            "scripted_expert_generator_version": version,
            "scripted_expert_speed_profile": SPEED_PROFILE,
            "scripted_expert_details": expert.details(env_id),
            "parallel_environment": {
                "enabled": True,
                "single_isaac_process": True,
                "num_envs": NUM_ENVS,
                "env_id": env_id,
                "env_spacing_m": args.env_spacing,
                "all_seeds": SEEDS,
                "camera_backend": "TiledCamera",
                "collision_filtering": True,
                "replicate_physics": True,
                "reset_visual_qc": visual_qc[env_id],
                "container_world_extents_m": {
                    key: value
                    for key, value in extents.items()
                    if f"env_{env_id}/" in key
                },
            },
            "scene_roles": list(env.entity_roles),
            "asset_usd_paths": asset_paths,
            **task_metadata,
        }
        high_rate = {
            "timestamp": high_rate_batch["timestamp"],
            "eef_wrench_joint_raw": physics_wrench[:, env_id],
        }
        path = writer.write(expert.buffers[env_id], metadata, high_rate=high_rate)[0]
        validation = validate_hdf5_episode(path)
        written.append(
            {"env_id": env_id, "seed": seed, "path": str(path), "validation": validation}
        )

    summary = {
        "schema": "forcebench_parallel_sort_32env_v1",
        "task": args.task,
        "seeds": SEEDS,
        "write_seeds": sorted(WRITE_SEEDS),
        "num_envs": NUM_ENVS,
        "env_spacing_m": args.env_spacing,
        "frames_per_episode": len(expert.buffers[0]),
        "video": str(video_path.resolve()),
        "overview": str(overview_path.resolve()),
        "final_overview": str(final_overview_path.resolve()),
        "episodes": written,
        "failed_episodes": failed,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "FORCEBENCH_PARALLEL_SORT=" + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
except BaseException as error:
    print(
        f"FORCEBENCH_PARALLEL_SORT_ERROR={type(error).__name__}:{error}",
        flush=True,
    )
    raise
finally:
    if expert is not None:
        expert.close()
    env.discard_wrench_capture()
    env.close()
    simulation_app.close()
