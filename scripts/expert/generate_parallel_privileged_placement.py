#!/usr/bin/env python3
"""Collect 32 nut_thread or gear_mesh demonstrations in one Isaac process."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("nut_thread", "gear_mesh"), required=True)
parser.add_argument(
    "--trajectory_mode",
    choices=("task_default", "half_split", "all_expert"),
    default="task_default",
    help="Explicitly select the per-seed trajectory distribution.",
)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--artifact_dir", type=Path, required=True)
parser.add_argument("--seeds", required=True)
parser.add_argument("--write_seeds", required=True)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--video_tile_size", type=int, default=192)
parser.add_argument("--translation_speed_m_s", type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


def parse_seed_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(seed < 0 for seed in values) or len(set(values)) != len(values):
        raise ValueError("seed lists must contain distinct non-negative integers")
    return values


SEEDS = parse_seed_list(args.seeds)
WRITE_SEEDS = frozenset(parse_seed_list(args.write_seeds))
NUM_ENVS = 32
PHYSICS_RATE_HZ = 60
CONTROL_RATE_HZ = 15
if len(SEEDS) != NUM_ENVS:
    parser.error("the parallel placement collector requires exactly 32 seeds")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env spacing must be at least 5.5 m")

os.environ.setdefault("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
os.environ["FORCEBENCH_TASK"] = args.task
os.environ.setdefault("FORCEBENCH_EXPERT_BACKEND", "scripted")

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import cv2
import numpy as np
import torch
from isaaclab.utils import math as math_utils

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task
from forcebench.tasks.gear_mesh.spec import SPEC as GEAR_SPEC
from forcebench.tasks.nut_thread.spec import SPEC as NUT_SPEC


SOURCE_TYPE_SCHEMA = {"expert": 0, "exploration": 1, "expert_recovery": 2}
EXPERT_SOURCE = SOURCE_TYPE_SCHEMA["expert"]
EXPLORATION_SOURCE = SOURCE_TYPE_SCHEMA["exploration"]
EXPERT_RECOVERY_SOURCE = SOURCE_TYPE_SCHEMA["expert_recovery"]
EXPLORATION_SAMPLER_VERSION = "seed_task_attempt_polar_v1"
EXPLORATION_ATTEMPTS = 1
EXPLORATION_TASK_TAG = {
    "nut_thread": 0x4E555458,
    "gear_mesh": 0x47454158,
}
# Nut probes remain inside 36% of its 2.5 mm final center envelope.  Gear
# probes remain inside 44% of its 5 mm center envelope.  They exceed the
# measured 0.5 mm radial bore/shaft free play while remaining a small offset.
EXPLORATION_RADIUS_RANGE_M = {
    "nut_thread": (
        0.08 * float(NUT_SPEC.max_center_error_m),
        0.20 * float(NUT_SPEC.max_center_error_m),
    ),
    "gear_mesh": (
        0.24 * float(GEAR_SPEC.max_center_error_m),
        0.44 * float(GEAR_SPEC.max_center_error_m),
    ),
}
NUT_EXPLORATION_DESCENT_M = 0.002
NUT_EXPLORATION_VERTICAL_DISENGAGE_M = 0.006
# Keep the two gear-mesh training corpora independently selectable without
# editing source between runs.  Task defaults preserve the established
# behavior: nut-thread is pure expert and gear-mesh is the 100/100 split.
TRAJECTORY_MODE = args.trajectory_mode
if TRAJECTORY_MODE == "task_default":
    TRAJECTORY_MODE = "all_expert" if args.task == "nut_thread" else "half_split"
ALL_EXPERT_BATCH = TRAJECTORY_MODE == "all_expert"
EXPLORATION_MASK = (
    np.zeros(NUM_ENVS, dtype=bool)
    if ALL_EXPERT_BATCH
    else np.asarray([seed % 2 == 1 for seed in SEEDS], dtype=bool)
)
if not ALL_EXPERT_BATCH and int(np.count_nonzero(EXPLORATION_MASK)) != NUM_ENVS // 2:
    parser.error("every 32-clone batch must contain 16 even expert and 16 odd exploration seeds")
PHASES = (
    "approach_pickup",
    "descend_to_grasp",
    "close_gripper",
    "lift_object",
    "transport_above_target",
    "privileged_clearance",
    "privileged_exact_place",
    "exploration_probe",
    "expert_recovery_center",
    "thread_engagement",
    "placement_settle",
    "release",
    "retract",
    "final_settle",
    "grasp_gear",
    "lift_gear",
    "transport_above_fixture",
    "direct_alignment",
    "final_seating",
    "release_gear",
    "retract_and_settle",
)
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASES)}


def phase_key(phase: str) -> str:
    if phase.startswith("privileged_clearance_"):
        return "privileged_clearance"
    if phase not in PHASE_SCHEMA:
        raise ValueError(f"undeclared placement phase: {phase!r}")
    return phase


def as_numpy(value) -> np.ndarray:
    # Isaac Sim can expose tensors through a separately loaded torch class, so
    # isinstance(value, torch.Tensor) is not reliable inside Kit.  Use the
    # tensor protocol instead and always stage CUDA data through host memory.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rgb_batch(value) -> np.ndarray:
    return as_numpy(value)[..., :3].astype(np.uint8)


def wrap_pi(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value)).astype(np.float32)


@dataclass(frozen=True)
class BatchPlan:
    object_role: str
    pickup_tcp_z_m: float
    lift_tcp_z_m: float
    grasp_open_fraction: float
    placement_clearances_m: tuple[float, ...]
    translation_speed_m_s: float
    object_position_tolerance_m: float
    object_yaw_tolerance_rad: float
    thread_rotation_rad: float
    thread_yaw_step_rad: float
    thread_downward_bias_m: float
    retract_height_m: float


class ParallelPlacementExpert:
    def __init__(
        self,
        env,
        task,
        plan: BatchPlan,
        desired_yaws: np.ndarray,
        video_path: Path,
    ):
        self.env = env
        self.task = task
        self.plan = plan
        self.desired_yaws = np.asarray(desired_yaws, dtype=np.float32).reshape(NUM_ENVS)
        self.buffers = [EpisodeBuffer() for _ in range(NUM_ENVS)]
        self.zero_action = torch.zeros((NUM_ENVS, 7), dtype=torch.float32, device=env.device)
        self.translation_step_m = float(plan.translation_speed_m_s / CONTROL_RATE_HZ)
        self.command_yaws = np.zeros(NUM_ENVS, dtype=np.float32)
        self.command_rolls = np.full(NUM_ENVS, math.pi, dtype=np.float32)
        self.command_pitches = np.zeros(NUM_ENVS, dtype=np.float32)
        self.failures = [[] for _ in range(NUM_ENVS)]
        self.phase_ranges: list[dict[str, int | str]] = []
        self.max_object_position_error_m = [dict() for _ in range(NUM_ENVS)]
        self.eef_minus_object = np.zeros((NUM_ENVS, 3), dtype=np.float32)
        self.target_objects = np.zeros((NUM_ENVS, 3), dtype=np.float32)
        self.gear_preinsert_center_error_m = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.exploration_mask = EXPLORATION_MASK.copy()
        self.exploration_radius_range_m = EXPLORATION_RADIUS_RANGE_M[args.task]
        self.exploration_probe_samples: list[list[dict[str, float | int]]] = [
            [] for _ in range(NUM_ENVS)
        ]
        if not ALL_EXPERT_BATCH:
            for env_id, seed in enumerate(SEEDS):
                for attempt in range(EXPLORATION_ATTEMPTS):
                    generator = np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, EXPLORATION_TASK_TAG[args.task], attempt]
                        )
                    )
                    radius = float(generator.uniform(*self.exploration_radius_range_m))
                    theta = float(generator.uniform(-math.pi, math.pi))
                    sample: dict[str, float | int] = {
                        "attempt": attempt,
                        "radius_m": radius,
                        "theta_rad": theta,
                        "dx_m": radius * math.cos(theta),
                        "dy_m": radius * math.sin(theta),
                    }
                    self.exploration_probe_samples[env_id].append(sample)
        self.privileged_roles = (
            ("target_bolt", "nut")
            if args.task == "nut_thread"
            else ("target_base", "small_gear", "large_gear", "medium_gear")
        )
        size = int(args.video_tile_size)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            CONTROL_RATE_HZ,
            (8 * size, 4 * size),
        )
        if not self.video.isOpened():
            raise RuntimeError(f"failed to create video: {video_path}")

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    def make_mosaic(self, images: np.ndarray, phase: str) -> np.ndarray:
        size = int(args.video_tile_size)
        canvas = np.zeros((4 * size, 8 * size, 3), dtype=np.uint8)
        force = as_numpy(
            torch.linalg.vector_norm(self.env.force_sensor_joint_raw[:, :3], dim=1)
        )
        for env_id, image in enumerate(images):
            tile = cv2.resize(
                image[..., ::-1], (size, size), interpolation=cv2.INTER_AREA
            )
            cv2.rectangle(tile, (0, 0), (size, 39), (18, 18, 18), -1)
            mark = "REC" if SEEDS[env_id] in WRITE_SEEDS else "FILL"
            trajectory = "EXP" if self.exploration_mask[env_id] else "EXPERT"
            cv2.putText(
                tile,
                f"e{env_id:02d} s{SEEDS[env_id]:03d} {mark} {trajectory}",
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"{phase[:13]} F {force[env_id]:.1f}N",
                (4, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            row, column = divmod(env_id, 8)
            canvas[
                row * size : (row + 1) * size,
                column * size : (column + 1) * size,
            ] = tile
        return canvas

    @property
    def alive(self) -> np.ndarray:
        return np.asarray([not values for values in self.failures], dtype=bool)

    def source_types(self, exploration: bool = False, recovery: bool = False) -> np.ndarray:
        values = np.full(NUM_ENVS, EXPERT_SOURCE, dtype=np.uint8)
        if exploration:
            values[self.exploration_mask] = EXPLORATION_SOURCE
        if recovery:
            values[self.exploration_mask] = EXPERT_RECOVERY_SOURCE
        return values

    def object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = self.env.entity_local_pose(self.plan.object_role)
        _, _, yaw = math_utils.euler_xyz_from_quat(quaternion)
        return as_numpy(position).astype(np.float32), as_numpy(yaw).astype(np.float32)

    def object_tilt_degrees(self) -> np.ndarray:
        _, quaternion = self.env.entity_local_pose(self.plan.object_role)
        local_z = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=self.env.device)
        local_z[:, 2] = 1.0
        axis = as_numpy(math_utils.quat_apply(quaternion, local_z)).astype(np.float32)
        return np.degrees(np.arccos(np.clip(axis[:, 2], -1.0, 1.0))).astype(np.float32)

    def object_rpy_and_tilt(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _, quaternion = self.env.entity_local_pose(self.plan.object_role)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quaternion)
        local_z = torch.zeros((NUM_ENVS, 3), dtype=torch.float32, device=self.env.device)
        local_z[:, 2] = 1.0
        axis = as_numpy(math_utils.quat_apply(quaternion, local_z)).astype(np.float32)
        tilt = np.degrees(np.arccos(np.clip(axis[:, 2], -1.0, 1.0))).astype(np.float32)
        return (
            as_numpy(roll).astype(np.float32),
            as_numpy(pitch).astype(np.float32),
            as_numpy(yaw).astype(np.float32),
            tilt,
        )

    def align_object_axis(
        self,
        enabled: np.ndarray,
        object_rolls: np.ndarray,
        object_pitches: np.ndarray,
    ) -> None:
        """Tilt the wrist opposite the measured object roll/pitch error."""

        enabled = np.asarray(enabled, dtype=bool).reshape(NUM_ENVS)
        roll_step = np.clip(
            -np.float32(0.45) * object_rolls,
            -np.float32(math.radians(2.0)),
            np.float32(math.radians(2.0)),
        )
        pitch_step = np.clip(
            -np.float32(0.45) * object_pitches,
            -np.float32(math.radians(2.0)),
            np.float32(math.radians(2.0)),
        )
        self.command_rolls[enabled] += roll_step[enabled]
        self.command_pitches[enabled] += pitch_step[enabled]
        self.command_rolls = math.pi + wrap_pi(self.command_rolls - math.pi)
        self.command_pitches = np.clip(
            self.command_pitches, -math.radians(20.0), math.radians(20.0)
        ).astype(np.float32)

    def target_positions(self) -> np.ndarray:
        target = self.task.wrench_reference_position(self.env)
        if isinstance(target, tuple):
            target = target[0]
        return as_numpy(target).astype(np.float32).reshape(NUM_ENVS, 3)

    def slew(self, requested: np.ndarray) -> np.ndarray:
        current = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
        delta = np.asarray(requested, dtype=np.float32) - current
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.translation_step_m / np.maximum(distance, 1.0e-9))
        return current + delta * scale.astype(np.float32)

    def record_and_step(
        self, requested, gripper, phase: str, source_type_ids=None
    ) -> None:
        requested = np.asarray(requested, dtype=np.float32).reshape(NUM_ENVS, 3)
        # The validated Gear expert holds the final Cartesian waypoint on every
        # frame.  Slewing that target to a few millimetres ahead of the measured
        # loaded TCP removes the controller's position error and stalls the lift.
        # Nut retains its already validated velocity-limited command stream.
        command = requested if args.task == "gear_mesh" else self.slew(requested)
        grippers = np.asarray(gripper, dtype=np.float32)
        if grippers.ndim == 0:
            grippers = np.full(NUM_ENVS, float(grippers), dtype=np.float32)
        actions = np.concatenate(
            (
                command,
                self.command_rolls[:, None],
                self.command_pitches[:, None],
                self.command_yaws[:, None],
                (2.0 * grippers - 1.0)[:, None],
            ),
            axis=1,
        ).astype(np.float32)
        states = as_numpy(self.env.policy_state)
        wrench_raw = as_numpy(self.env.force_sensor_joint_raw)
        wrench_ema = as_numpy(self.env.force_sensor_joint_smooth)
        main = rgb_batch(self.env.table_rgb)
        wrist = rgb_batch(self.env.wrist_rgb)
        joints = as_numpy(self.env.joint_pos)
        velocities = as_numpy(self.env.robot.data.joint_vel.torch)
        tcp = as_numpy(
            torch.cat((self.env.fingertip_midpoint_pos, self.env.fingertip_midpoint_quat), dim=-1)
        )
        # The legacy task hooks intentionally expose env 0 for single-env
        # teleoperation.  Read scene entities directly so every clone receives
        # its own privileged pose in the parallel collector.
        privileged = {
            f"objects/{role}/pose": as_numpy(self.env.entity_world_pose(role))
            for role in self.privileged_roles
        }
        timestamp = (len(self.buffers[0]) + 1) * self.env.step_dt
        source_types = np.asarray(
            EXPERT_SOURCE if source_type_ids is None else source_type_ids,
            dtype=np.uint8,
        )
        if source_types.ndim == 0:
            source_types = np.full(NUM_ENVS, int(source_types), dtype=np.uint8)
        if source_types.shape != (NUM_ENVS,):
            raise ValueError("source_type_ids must be scalar or one value per environment")
        current_phase_id = PHASE_SCHEMA[phase_key(phase)]
        for env_id, buffer in enumerate(self.buffers):
            per_env_privileged = {}
            for key, value in privileged.items():
                if value.shape[0] != NUM_ENVS:
                    raise RuntimeError(
                        f"privileged field {key!r} has non-batched shape {value.shape}"
                    )
                per_env_privileged[key] = value[env_id]
            buffer.append(
                state=states[env_id],
                action=actions[env_id],
                wrench_raw=wrench_raw[env_id],
                wrench_ema=wrench_ema[env_id],
                timestamp=timestamp,
                image=main[env_id],
                wrist_image=wrist[env_id],
                joint_position=joints[env_id],
                joint_velocity=velocities[env_id],
                tcp_pose=tcp[env_id],
                privileged=per_env_privileged,
                phase_id=current_phase_id,
                control_source_type_id=int(source_types[env_id]),
            )
        self.video.write(self.make_mosaic(main, phase))
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

    def add_phase(self, name: str, start: int) -> None:
        self.phase_ranges.append({"name": name, "start": start, "end": len(self.buffers[0])})

    def move_tcp(
        self,
        targets,
        gripper: float,
        phase: str,
        *,
        max_frames: int = 220,
        tolerance_m: float = 0.0025,
        source_type_ids=None,
    ) -> np.ndarray:
        targets = np.asarray(targets, dtype=np.float32).reshape(NUM_ENVS, 3)
        enabled = self.alive
        converged = ~enabled.copy()
        start = len(self.buffers[0])
        errors = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        for frame in range(max_frames):
            current = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            requested = targets.copy()
            requested[converged] = current[converged]
            self.record_and_step(requested, gripper, phase, source_type_ids)
            errors = np.linalg.norm(
                as_numpy(self.env.fingertip_midpoint_pos) - targets, axis=1
            )
            if frame >= 4:
                converged |= enabled & (errors <= tolerance_m)
            if np.all(converged):
                break
        self.add_phase(phase, start)
        for env_id in np.flatnonzero(enabled & ~converged):
            self.failures[env_id].append(
                f"phase {phase} TCP did not converge: error={errors[env_id]:.4f}m"
            )
        return converged

    def hold(
        self, frames: int, gripper: float, phase: str, source_type_ids=None
    ) -> None:
        targets = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
        start = len(self.buffers[0])
        for _ in range(frames):
            self.record_and_step(targets, gripper, phase, source_type_ids)
        self.add_phase(phase, start)

    def command_for_frames(
        self,
        targets,
        gripper,
        phase: str,
        *,
        frames: int,
        enabled: np.ndarray | None = None,
        source_type_ids=None,
    ) -> None:
        """Issue a fixed-length, slew-limited batched Cartesian command."""

        targets = np.asarray(targets, dtype=np.float32).reshape(NUM_ENVS, 3)
        if enabled is None:
            enabled = self.alive
        else:
            enabled = np.asarray(enabled, dtype=bool).reshape(NUM_ENVS)
        start = len(self.buffers[0])
        for _ in range(frames):
            current = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            requested = targets.copy()
            requested[~enabled] = current[~enabled]
            self.record_and_step(requested, gripper, phase, source_type_ids)
        self.add_phase(phase, start)

    def servo_object(
        self,
        desired,
        phase: str,
        *,
        max_frames: int = 180,
        tolerance_m: float,
        position_gain: float,
        enabled: np.ndarray | None = None,
        fatal_nonconvergence: bool = True,
        source_type_ids=None,
        desired_yaw_targets=None,
    ) -> np.ndarray:
        desired = np.asarray(desired, dtype=np.float32).reshape(NUM_ENVS, 3)
        target_yaws = np.asarray(
            self.desired_yaws if desired_yaw_targets is None else desired_yaw_targets,
            dtype=np.float32,
        ).reshape(NUM_ENVS)
        enabled = self.alive if enabled is None else self.alive & np.asarray(
            enabled, dtype=bool
        ).reshape(NUM_ENVS)
        converged = ~enabled.copy()
        position_errors = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        yaw_errors = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        start = len(self.buffers[0])
        for frame in range(max_frames):
            positions, yaws = self.object_pose()
            delta = desired - positions
            yaw_delta = wrap_pi(target_yaws - yaws)
            moving = enabled & ~converged
            if args.task == "gear_mesh" and phase == "expert_recovery_center":
                object_rolls, object_pitches, _, _ = self.object_rpy_and_tilt()
                self.align_object_axis(moving, object_rolls, object_pitches)
            self.command_yaws[moving] += np.clip(yaw_delta[moving], -0.04, 0.04)
            self.command_yaws = wrap_pi(self.command_yaws)
            eef = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            requested = eef.copy()
            requested[moving] += np.float32(position_gain) * delta[moving]
            self.record_and_step(
                requested,
                self.plan.grasp_open_fraction,
                phase,
                source_type_ids,
            )
            after_positions, after_yaws = self.object_pose()
            position_errors = np.linalg.norm(desired - after_positions, axis=1)
            yaw_errors = np.abs(wrap_pi(target_yaws - after_yaws))
            for env_id in np.flatnonzero(enabled):
                self.max_object_position_error_m[env_id][phase] = max(
                    self.max_object_position_error_m[env_id].get(phase, 0.0),
                    float(position_errors[env_id]),
                )
            if frame >= 5:
                converged |= enabled & (
                    (position_errors <= tolerance_m)
                    & (yaw_errors <= self.plan.object_yaw_tolerance_rad)
                )
            if np.all(converged):
                break
        self.add_phase(phase, start)
        for env_id in np.flatnonzero(enabled & ~converged):
            destination = self.failures if fatal_nonconvergence else None
            if destination is None:
                continue
            destination[env_id].append(
                f"phase {phase} object servo did not converge: "
                f"position={position_errors[env_id]:.4f}m "
                f"yaw={math.degrees(float(yaw_errors[env_id])):.2f}deg"
            )
        return converged

    def servo_object_xy_at_fixed_tcp_height(
        self,
        desired_xy,
        phase: str,
        *,
        frames: int,
        tolerance_m: float,
        position_gain: float,
        enabled: np.ndarray,
        source_type_ids=None,
    ) -> np.ndarray:
        """Recenter the grasped object without giving back vertical clearance."""

        desired_xy = np.asarray(desired_xy, dtype=np.float32).reshape(NUM_ENVS, 2)
        enabled = self.alive & np.asarray(enabled, dtype=bool).reshape(NUM_ENVS)
        converged = ~enabled.copy()
        fixed_tcp_z = as_numpy(self.env.fingertip_midpoint_pos)[:, 2].astype(np.float32)
        xy_errors = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        yaw_errors = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        start = len(self.buffers[0])
        for frame in range(frames):
            positions, yaws = self.object_pose()
            delta_xy = desired_xy - positions[:, :2]
            yaw_delta = wrap_pi(self.desired_yaws - yaws)
            moving = enabled & ~converged
            self.command_yaws[moving] += np.clip(yaw_delta[moving], -0.04, 0.04)
            self.command_yaws = wrap_pi(self.command_yaws)
            eef = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            requested = eef.copy()
            requested[moving, :2] += np.float32(position_gain) * delta_xy[moving]
            requested[enabled, 2] = fixed_tcp_z[enabled]
            self.record_and_step(
                requested,
                self.plan.grasp_open_fraction,
                phase,
                source_type_ids,
            )
            after_positions, after_yaws = self.object_pose()
            xy_errors = np.linalg.norm(desired_xy - after_positions[:, :2], axis=1)
            yaw_errors = np.abs(wrap_pi(self.desired_yaws - after_yaws))
            if frame >= 5:
                converged |= enabled & (
                    (xy_errors <= tolerance_m)
                    & (yaw_errors <= self.plan.object_yaw_tolerance_rad)
                )
            # Deliberately run the full fixed frame count.  Non-exploration
            # clones depend on the same stabilization interval in this batched
            # trajectory, while converged exploration clones simply hold.
        self.add_phase(phase, start)
        return converged

    def thread_object(self, target_positions: np.ndarray, source_type_ids=None) -> None:
        total_rotation = float(self.plan.thread_rotation_rad)
        if total_rotation == 0.0:
            return
        enabled = self.alive
        target = np.asarray(target_positions, dtype=np.float32).copy()
        target[:, 2] -= np.float32(self.plan.thread_downward_bias_m)
        initial_positions, _ = self.object_pose()
        frames = max(1, math.ceil(abs(total_rotation) / self.plan.thread_yaw_step_rad))
        yaw_step = np.float32(total_rotation / frames)
        start = len(self.buffers[0])
        for frame in range(frames):
            fraction = np.float32((frame + 1) / frames)
            desired = target.copy()
            desired[:, 2] = (
                (np.float32(1.0) - fraction) * initial_positions[:, 2]
                + fraction * target[:, 2]
            )
            positions, _ = self.object_pose()
            eef = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            requested = eef.copy()
            requested[enabled] += np.float32(1.5) * (desired - positions)[enabled]
            self.command_yaws[enabled] = wrap_pi(self.command_yaws[enabled] + yaw_step)
            self.record_and_step(
                requested,
                self.plan.grasp_open_fraction,
                "thread_engagement",
                source_type_ids,
            )
        self.add_phase("thread_engagement", start)

    def controlled_placement_exploration(self) -> None:
        """Make one seeded contact, disengage vertically, then recenter."""

        exploration_enabled = self.exploration_mask & self.alive
        if not np.any(exploration_enabled):
            return
        exploration_sources = self.source_types(exploration=True)
        recovery_sources = self.source_types(recovery=True)
        samples = [values[0] for values in self.exploration_probe_samples]
        dx = np.asarray([sample["dx_m"] for sample in samples], dtype=np.float32)
        dy = np.asarray([sample["dy_m"] for sample in samples], dtype=np.float32)

        if args.task == "gear_mesh":
            # Start from the already aligned shaft-top pose.  Make one short
            # polar-offset contact, return to the same pre-contact height to
            # break contact, then undo the lateral offset.  There is no random
            # yaw perturbation, force-triggered branch, or repeated search.
            before_positions, _ = self.object_pose()
            center_tcp = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            offset_tcp = center_tcp.copy()
            offset_tcp[:, 0] += dx
            offset_tcp[:, 1] += dy
            self.command_for_frames(
                offset_tcp,
                self.plan.grasp_open_fraction,
                "exploration_probe",
                frames=4,
                enabled=exploration_enabled,
                source_type_ids=exploration_sources,
            )
            contact_tcp = offset_tcp.copy()
            contact_tcp[:, 2] -= np.float32(0.016)
            self.command_for_frames(
                contact_tcp,
                self.plan.grasp_open_fraction,
                "exploration_probe",
                frames=6,
                enabled=exploration_enabled,
                source_type_ids=exploration_sources,
            )
            # Lift exactly back to the pre-contact height before moving
            # laterally, so the shaft and gear are no longer rubbing.
            self.command_for_frames(
                offset_tcp,
                self.plan.grasp_open_fraction,
                "expert_recovery_center",
                frames=6,
                enabled=exploration_enabled,
                source_type_ids=recovery_sources,
            )
            self.command_for_frames(
                center_tcp,
                self.plan.grasp_open_fraction,
                "expert_recovery_center",
                frames=6,
                enabled=exploration_enabled,
                source_type_ids=recovery_sources,
            )
            recovery = before_positions.copy()
            recovery[:, :2] = self.target_objects[:, :2]
            self.servo_object(
                recovery,
                "expert_recovery_center",
                max_frames=120,
                tolerance_m=0.0010,
                position_gain=0.25,
                enabled=exploration_enabled,
                fatal_nonconvergence=False,
                source_type_ids=recovery_sources,
                desired_yaw_targets=self.desired_yaws,
            )
            return

        # Nut: approach 3 mm above the measured pose, then descend 2 mm below
        # the measured pre-probe height to guarantee one small bolt contact.
        # After contact, move the TCP only vertically by 6 mm before any
        # lateral recentering.  Mixing XYZ correction while the nut is still
        # touching the bolt leaves enough friction to prevent recentering.
        positions_before, _ = self.object_pose()
        current_clearance = positions_before[:, 2] - self.target_objects[:, 2]
        probe_high = self.target_objects.copy()
        probe_high[:, 0] += dx
        probe_high[:, 1] += dy
        probe_high[:, 2] += current_clearance + np.float32(0.003)
        probe_low = probe_high.copy()
        probe_low[:, 2] = (
            self.target_objects[:, 2]
            + current_clearance
            - np.float32(NUT_EXPLORATION_DESCENT_M)
        )
        self.servo_object(
            probe_high,
            "exploration_probe",
            max_frames=60,
            tolerance_m=0.0015,
            position_gain=0.8,
            enabled=exploration_enabled,
            fatal_nonconvergence=False,
            source_type_ids=exploration_sources,
        )
        self.servo_object(
            probe_low,
            "exploration_probe",
            max_frames=10,
            tolerance_m=0.0008,
            position_gain=0.6,
            enabled=exploration_enabled,
            fatal_nonconvergence=False,
            source_type_ids=exploration_sources,
        )
        contact_tcp = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
        vertical_retreat_tcp = contact_tcp.copy()
        vertical_retreat_tcp[exploration_enabled, 2] += np.float32(
            NUT_EXPLORATION_VERTICAL_DISENGAGE_M
        )
        self.command_for_frames(
            vertical_retreat_tcp,
            self.plan.grasp_open_fraction,
            "expert_recovery_center",
            frames=12,
            enabled=exploration_enabled,
            source_type_ids=recovery_sources,
        )
        self.servo_object_xy_at_fixed_tcp_height(
            self.target_objects[:, :2],
            "expert_recovery_center",
            frames=120,
            tolerance_m=0.0008,
            position_gain=0.8,
            enabled=exploration_enabled,
            source_type_ids=recovery_sources,
        )
        # Preserve the original batched stabilization budget without delaying
        # the vertical-disengage measurement, because the nut can slowly slip
        # inside the closed fingers during a long stationary lift hold.
        self.hold(
            48,
            self.plan.grasp_open_fraction,
            "expert_recovery_center",
            source_type_ids=recovery_sources,
        )

    def run_generic_placement(self) -> None:
        initial_objects, _ = self.object_pose()
        pickup_xy = initial_objects[:, :2]
        approach = np.column_stack(
            (pickup_xy, np.full(NUM_ENVS, self.plan.pickup_tcp_z_m + 0.09))
        )
        grasp = np.column_stack(
            (pickup_xy, np.full(NUM_ENVS, self.plan.pickup_tcp_z_m))
        )
        self.move_tcp(approach, 1.0, "approach_pickup")
        self.move_tcp(grasp, 1.0, "descend_to_grasp")
        self.hold(16, self.plan.grasp_open_fraction, "close_gripper")
        lift = np.column_stack(
            (pickup_xy, np.full(NUM_ENVS, self.plan.lift_tcp_z_m))
        )
        self.move_tcp(
            lift,
            self.plan.grasp_open_fraction,
            "lift_object",
            tolerance_m=0.050,
        )
        lifted_objects, _ = self.object_pose()
        for env_id in np.flatnonzero(lifted_objects[:, 2] <= initial_objects[:, 2] + 0.035):
            self.failures[env_id].append(
                f"{self.plan.object_role} was not lifted: z={lifted_objects[env_id, 2]:.4f}"
            )
        self.eef_minus_object = (
            as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32) - lifted_objects
        )
        self.target_objects = self.target_positions()
        transit = self.target_objects + self.eef_minus_object
        transit[:, 2] = self.plan.lift_tcp_z_m
        self.move_tcp(
            transit,
            self.plan.grasp_open_fraction,
            "transport_above_target",
            tolerance_m=0.050,
        )
        recovery_sources = None
        exploration_complete = False
        for clearance in self.plan.placement_clearances_m:
            if not exploration_complete and clearance <= 0.003:
                self.controlled_placement_exploration()
                recovery_sources = self.source_types(recovery=True)
                exploration_complete = True
            desired = self.target_objects.copy()
            desired[:, 2] += np.float32(clearance)
            phase = (
                "privileged_exact_place"
                if clearance == 0.0
                else f"privileged_clearance_{clearance:.3f}m"
            )
            tolerance = (
                self.plan.object_position_tolerance_m
                if clearance == 0.0
                else max(self.plan.object_position_tolerance_m, 0.003)
            )
            self.servo_object(
                desired,
                phase,
                tolerance_m=tolerance,
                position_gain=2.0 if clearance == 0.0 else 1.0,
                fatal_nonconvergence=False,
                source_type_ids=recovery_sources,
            )
        self.thread_object(self.target_objects, recovery_sources)
        self.hold(
            8,
            self.plan.grasp_open_fraction,
            "placement_settle",
            recovery_sources,
        )
        self.hold(14, 1.0, "release", recovery_sources)
        retract = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
        retract[:, 2] += np.float32(self.plan.retract_height_m)
        self.move_tcp(retract, 1.0, "retract", source_type_ids=recovery_sources)
        self.hold(12, 1.0, "final_settle", recovery_sources)

    def run_gear_mesh(self) -> None:
        """Vectorized form of the validated 60 Hz Gear scripted expert.

        Gear cannot use the Nut-style high straight-line transit: that pose is
        outside the useful loaded-IK workspace near the far-side fixture.  The
        validated expert instead uses a low pickup lift, an elevated midpoint,
        and a reachable pre-insert pose, followed by physical gear-pose feedback.
        """

        initial_objects, initial_yaws = self.object_pose()
        pickup_xy = initial_objects[:, :2]

        approach_high = np.column_stack(
            (pickup_xy, np.full(NUM_ENVS, 0.145, dtype=np.float32))
        )
        approach_low = np.column_stack(
            (pickup_xy, np.full(NUM_ENVS, 0.065, dtype=np.float32))
        )
        grasp = np.column_stack(
            (pickup_xy, np.full(NUM_ENVS, self.plan.pickup_tcp_z_m, dtype=np.float32))
        )
        self.move_tcp(approach_high, 1.0, "approach_pickup")
        self.move_tcp(approach_low, 1.0, "approach_pickup")
        self.move_tcp(grasp, 1.0, "approach_pickup", tolerance_m=0.003)
        self.hold(38, self.plan.grasp_open_fraction, "grasp_gear")
        grasped_objects, _ = self.object_pose()
        self.eef_minus_object = (
            as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32) - grasped_objects
        )
        self.hold(10, self.plan.grasp_open_fraction, "grasp_gear")

        lift = np.column_stack(
            (pickup_xy, np.full(NUM_ENVS, 0.190, dtype=np.float32))
        )
        self.move_tcp(
            lift,
            self.plan.grasp_open_fraction,
            "lift_gear",
            tolerance_m=0.006,
        )
        lifted_objects, _ = self.object_pose()
        for env_id in np.flatnonzero(lifted_objects[:, 2] < 0.085):
            self.failures[env_id].append(
                f"medium_gear was not lifted: z={lifted_objects[env_id, 2]:.4f}"
            )

        # Preserve the grasp-relative tooth phase exactly as the validated
        # single-environment expert does.  The seeded pickup yaw jitter is
        # therefore corrected by a small wrist rotation during this waypoint.
        self.command_yaws = wrap_pi(
            self.command_yaws + wrap_pi(self.desired_yaws - initial_yaws)
        )
        self.move_tcp(
            lift,
            self.plan.grasp_open_fraction,
            "transport_above_fixture",
            tolerance_m=0.008,
        )

        self.target_objects = self.target_positions()
        midpoint_xy = np.float32(0.5) * (pickup_xy + self.target_objects[:, :2])
        midpoint = np.column_stack(
            (midpoint_xy, np.full(NUM_ENVS, 0.215, dtype=np.float32))
        )
        self.move_tcp(
            midpoint,
            self.plan.grasp_open_fraction,
            "transport_above_fixture",
            tolerance_m=0.008,
        )
        preinsert = np.column_stack(
            (
                self.target_objects[:, :2],
                np.full(NUM_ENVS, 0.105, dtype=np.float32),
            )
        )
        self.move_tcp(
            preinsert,
            self.plan.grasp_open_fraction,
            "transport_above_fixture",
            tolerance_m=0.012,
        )

        # Cancel loaded-TCP bias from the physical medium-gear center, not from
        # the commanded TCP.  Replicated scenes retain a few millimetres of
        # steady-state error under a purely proportional target.  Integrate no
        # more than 0.12 mm per frame and require eight consecutive centered
        # frames.  This is slow enough to avoid crossing the shaft with the
        # loaded arm's momentum, while still removing the static bias.
        center_stable = np.zeros(NUM_ENVS, dtype=np.int32)
        command_xy = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)[:, :2]
        center_history: list[dict[str, float | int]] = []
        center_start = len(self.buffers[0])
        for frame in range(300):
            positions, object_yaws = self.object_pose()
            error = self.target_objects[:, :2] - positions[:, :2]
            center_error = np.linalg.norm(error, axis=1)
            yaw_error = wrap_pi(self.desired_yaws - object_yaws)
            moving = self.alive & (center_stable < 8)
            if not np.any(moving):
                break
            norms = np.linalg.norm(error, axis=1, keepdims=True)
            correction = np.float32(0.03) * error * np.minimum(
                1.0, 0.00012 / np.maximum(np.float32(0.03) * norms, 1.0e-9)
            ).astype(np.float32)
            command_xy[moving] += correction[moving]
            self.command_yaws[moving] = wrap_pi(
                self.command_yaws[moving]
                + np.clip(
                    np.float32(0.10) * yaw_error[moving],
                    -np.float32(0.005),
                    np.float32(0.005),
                )
            )
            targets = np.column_stack(
                (command_xy, np.full(NUM_ENVS, 0.105, dtype=np.float32))
            )
            current = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            targets[~self.alive] = current[~self.alive]
            self.record_and_step(
                targets, self.plan.grasp_open_fraction, "direct_alignment"
            )
            after_positions, after_yaws = self.object_pose()
            after_error = np.linalg.norm(
                self.target_objects[:, :2] - after_positions[:, :2], axis=1
            )
            after_yaw_error = np.abs(wrap_pi(self.desired_yaws - after_yaws))
            within = (
                self.alive
                & (after_error <= 0.00035)
                & (after_yaw_error <= math.radians(0.5))
            )
            center_stable = np.where(within, center_stable + 1, 0)
            if frame % 15 == 0:
                alive_errors = after_error[self.alive]
                center_history.append(
                    {
                        "frame": frame + 1,
                        "median_error_m": float(np.median(alive_errors))
                        if len(alive_errors)
                        else 0.0,
                        "max_error_m": float(np.max(alive_errors))
                        if len(alive_errors)
                        else 0.0,
                        "max_yaw_error_deg": math.degrees(
                            float(np.max(after_yaw_error[self.alive]))
                        )
                        if np.any(self.alive)
                        else 0.0,
                    }
                )
        self.add_phase("direct_alignment", center_start)
        positions, object_yaws = self.object_pose()
        center_error = np.linalg.norm(
            self.target_objects[:, :2] - positions[:, :2], axis=1
        )
        yaw_error = np.abs(wrap_pi(self.desired_yaws - object_yaws))
        print(
            "FORCEBENCH_PARALLEL_GEAR_CENTER="
            + json.dumps(
                {
                    "errors_m": [float(value) for value in center_error],
                    "yaw_errors_deg": [math.degrees(float(value)) for value in yaw_error],
                    "stable_frames": [int(value) for value in center_stable],
                    "history": center_history,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        self.gear_preinsert_center_error_m = center_error.astype(np.float32)
        for env_id in np.flatnonzero(self.alive & (center_error > 0.00075)):
            self.failures[env_id].append(
                f"gear center oracle correction did not converge: {center_error[env_id]:.6f}m"
            )
        for env_id in np.flatnonzero(
            self.alive & (yaw_error > self.plan.object_yaw_tolerance_rad)
        ):
            self.failures[env_id].append(
                "gear phase alignment did not converge: "
                f"{math.degrees(float(yaw_error[env_id])):.3f}deg"
            )

        # The two Gear corpora share the same pickup/alignment/seating expert.
        # In half-split mode, odd seeds make exactly one bounded contact and
        # then return vertically to the pre-contact height before privileged
        # recentering.  All subsequent insertion frames are expert recovery
        # for those odd seeds.  Pure-expert mode skips this block entirely.
        if ALL_EXPERT_BATCH:
            seating_sources = self.source_types()
            # Start the clean expert insertion from a contact-free pose.  The
            # direct alignment above can finish while the gear is already
            # grazing the shaft/teeth; descending immediately from there makes
            # otherwise correct oracle targets jam.  Lift vertically first,
            # recenter at that fixed safe height, then let the seating loop
            # descend.  This contains no randomized offset or exploration
            # source frames.
            safe_tcp = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            safe_tcp[:, 2] += np.float32(0.016)
            self.command_for_frames(
                safe_tcp,
                self.plan.grasp_open_fraction,
                "privileged_exact_place",
                frames=6,
                enabled=self.alive,
                source_type_ids=seating_sources,
            )
            safe_positions, _ = self.object_pose()
            exact_center = safe_positions.copy()
            exact_center[:, :2] = self.target_objects[:, :2]
            self.servo_object(
                exact_center,
                "privileged_exact_place",
                max_frames=120,
                tolerance_m=0.0010,
                position_gain=0.25,
                enabled=self.alive,
                fatal_nonconvergence=False,
                source_type_ids=seating_sources,
                desired_yaw_targets=self.desired_yaws,
            )
        else:
            self.controlled_placement_exploration()
            seating_sources = self.source_types(recovery=True)

        # Continue from the privileged center alignment, descend through a short
        # clearance list with physical object-pose feedback, then release/retract.
        for clearance in self.plan.placement_clearances_m:
            desired = self.target_objects.copy()
            desired[:, 2] += np.float32(clearance)
            self.servo_object(
                desired,
                "final_seating",
                max_frames=120,
                tolerance_m=0.005 if clearance == 0.0 else 0.0025,
                position_gain=0.25,
                # The gear can become physically seated even when the closed
                # gripper can no longer servo its center to the commanded pose.
                # Do not kill the environment here: release and retract first,
                # then let the task's complete final-state evaluator decide.
                fatal_nonconvergence=False,
                source_type_ids=seating_sources,
                desired_yaw_targets=self.desired_yaws,
            )
        self.hold(
            12,
            self.plan.grasp_open_fraction,
            "final_seating",
            seating_sources,
        )
        self.hold(42, 1.0, "release_gear", seating_sources)
        retract = np.column_stack(
            (
                self.target_objects[:, :2],
                np.full(NUM_ENVS, 0.165, dtype=np.float32),
            )
        )
        self.move_tcp(
            retract,
            1.0,
            "retract_and_settle",
            tolerance_m=0.006,
            source_type_ids=seating_sources,
        )
        self.hold(30, 1.0, "retract_and_settle", seating_sources)

    def run(self) -> list[dict[str, object]]:
        if args.task == "gear_mesh":
            self.run_gear_mesh()
        else:
            self.run_generic_placement()

        reports = []
        final_positions, final_yaws = self.object_pose()
        for env_id in range(NUM_ENVS):
            report = self.task.evaluate_final_state_for(self.env, env_id)
            success = bool(report["success"] and not self.failures[env_id])
            report = dict(report)
            report["success"] = success
            report["trajectory_failures"] = list(self.failures[env_id])
            if success:
                self.buffers[env_id].set_last_transition_outcome(
                    reward=1.0, success=True, terminated=False, truncated=False
                )
            reports.append(
                {
                    "report": report,
                    "phase_ranges_control_frames": list(self.phase_ranges),
                    "eef_minus_object_after_grasp_m": self.eef_minus_object[env_id].tolist(),
                    "target_object_position_m": self.target_objects[env_id].tolist(),
                    "final_object_position_m": final_positions[env_id].tolist(),
                    "desired_object_yaw_rad": float(self.desired_yaws[env_id]),
                    "final_object_yaw_rad": float(final_yaws[env_id]),
                    "oracle_pose_feedback": True,
                    "physics_intervention": False,
                    "object_state_overwrite": False,
                    "thread_rotation_rad": self.plan.thread_rotation_rad,
                    "thread_downward_bias_m": self.plan.thread_downward_bias_m,
                    "max_object_position_error_m": self.max_object_position_error_m[env_id],
                    "gear_preinsert_center_error_m": float(
                        self.gear_preinsert_center_error_m[env_id]
                    ),
                    "trajectory_type": "exploration"
                    if self.exploration_mask[env_id]
                    else "expert",
                    "exploration_sampler_version": None
                    if ALL_EXPERT_BATCH
                    else EXPLORATION_SAMPLER_VERSION,
                    "exploration_probe_samples": list(
                        self.exploration_probe_samples[env_id]
                    )
                    if self.exploration_mask[env_id]
                    else [],
                }
            )
        return reports


def build_plan(env) -> tuple[BatchPlan, np.ndarray]:
    layouts = env.layouts
    if args.task == "nut_thread":
        speed = 0.075 if args.translation_speed_m_s is None else args.translation_speed_m_s
        plan = BatchPlan(
            object_role="nut",
            pickup_tcp_z_m=0.018,
            lift_tcp_z_m=0.170,
            grasp_open_fraction=0.0,
            placement_clearances_m=(0.050, 0.025, 0.012, 0.006, 0.003, 0.0),
            translation_speed_m_s=float(speed),
            object_position_tolerance_m=0.00175,
            object_yaw_tolerance_rad=math.radians(1.0),
            thread_rotation_rad=-math.pi,
            thread_yaw_step_rad=0.025,
            thread_downward_bias_m=0.0015,
            retract_height_m=0.08,
        )
        yaws = np.asarray(
            [float(layout["target_bolt_yaw_rad"]) for layout in layouts], dtype=np.float32
        )
        return plan, yaws

    from forcebench.tasks.gear_mesh.spec import SPEC

    speed = 0.09 if args.translation_speed_m_s is None else args.translation_speed_m_s
    plan = BatchPlan(
        object_role="medium_gear",
        pickup_tcp_z_m=SPEC.pickup_grasp_tcp_z_m,
        lift_tcp_z_m=0.220,
        grasp_open_fraction=SPEC.grasp_gripper_open_fraction,
        placement_clearances_m=(0.012, 0.006, 0.0),
        translation_speed_m_s=float(speed),
        object_position_tolerance_m=0.0015,
        object_yaw_tolerance_rad=math.radians(1.0),
        thread_rotation_rad=0.0,
        thread_yaw_step_rad=0.025,
        thread_downward_bias_m=0.0,
        retract_height_m=0.12,
    )
    phase_offset = math.radians(SPEC.expert_twist_alignment_offset_deg)
    yaws = np.asarray(
        [float(layout["target_base_yaw_rad"]) + phase_offset for layout in layouts],
        dtype=np.float32,
    )
    return plan, yaws


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
expert = None
capture_active = False
args.artifact_dir.mkdir(parents=True, exist_ok=True)
args.output_dir.mkdir(parents=True, exist_ok=True)
video_path = args.artifact_dir / f"{args.task}_32env_mosaic.mp4"

try:
    print("FORCEBENCH_PARALLEL_PHASE=set_episode_seeds", flush=True)
    env.set_episode_seeds(SEEDS)
    print("FORCEBENCH_PARALLEL_PHASE=reset", flush=True)
    env.reset(seed=SEEDS[0])
    print("FORCEBENCH_PARALLEL_PHASE=restore_arm", flush=True)
    env.restore_fixed_arm_pose(env.joint_pos[0, :7].clone())
    print("FORCEBENCH_PARALLEL_PHASE=reset_cameras", flush=True)
    env.reset_cameras()
    env.sim.render()
    env.update_cameras()
    print("FORCEBENCH_PARALLEL_PHASE=build_plan", flush=True)
    plan, desired_yaws = build_plan(env)
    env.start_wrench_capture()
    capture_active = True
    expert = ParallelPlacementExpert(env, task, plan, desired_yaws, video_path)
    print("FORCEBENCH_PARALLEL_PHASE=run_expert", flush=True)
    results = expert.run()
    print("FORCEBENCH_PARALLEL_PHASE=serialize", flush=True)
    high_rate_batch = env.stop_wrench_capture()
    capture_active = False

    physics_wrench = np.asarray(high_rate_batch["eef_wrench_joint_raw"], dtype=np.float32)
    physics_timestamps = np.asarray(high_rate_batch["timestamp"], dtype=np.float64)
    expected_ratio = PHYSICS_RATE_HZ // CONTROL_RATE_HZ
    if physics_wrench.ndim != 3 or physics_wrench.shape[1:] != (NUM_ENVS, 6):
        raise RuntimeError(f"unexpected batched wrench shape {physics_wrench.shape}")
    if abs(len(physics_timestamps) - len(expert.buffers[0]) * expected_ratio) > expected_ratio:
        raise RuntimeError("high-rate wrench/control sample ratio is invalid")

    writer = HDF5EpisodeWriter(args.output_dir)
    task_metadata = dict(task.episode_metadata())
    asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
    asset_paths.update(env.scene_asset_paths)
    asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
    written, failed = [], []
    for env_id, seed in enumerate(SEEDS):
        if seed not in WRITE_SEEDS:
            continue
        report = results[env_id]["report"]
        if not report["success"]:
            failure = {"env_id": env_id, "seed": seed, "report": report}
            failed.append(failure)
            print(
                "FORCEBENCH_PARALLEL_PLACEMENT_FAILURE="
                + json.dumps(failure, separators=(",", ":")),
                flush=True,
            )
            continue
        existing = sorted(args.output_dir.glob(f"seed_{seed:03d}_episode_*.hdf5"))
        if existing:
            raise RuntimeError(f"refusing to duplicate successful seed {seed}: {existing}")
        details = {key: value for key, value in results[env_id].items() if key != "report"}
        trajectory_type = "exploration" if EXPLORATION_MASK[env_id] else "expert"
        metadata = {
            "task": args.task,
            "trajectory_mode": TRAJECTORY_MODE,
            "prompt": task.manifest.prompt,
            "success": True,
            "end_reason": "parallel_privileged_pose_teacher_final_check",
            "seed": seed,
            "initial_layout": env.layouts[env_id],
            "final_success_report": report,
            "control_rate_hz": float(CONTROL_RATE_HZ),
            "physics_rate_hz": float(PHYSICS_RATE_HZ),
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "scene_profile": env.scene_profile_name,
            "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "control_source": (
                "parallel_all_expert_smooth_position"
                if ALL_EXPERT_BATCH
                else "parallel_half_expert_half_controlled_exploration_smooth_position"
            ),
            "trajectory_type": trajectory_type,
            "control_source_type_schema": SOURCE_TYPE_SCHEMA,
            "control_source_type_dataset": "control/source_type",
            "control_source_type_semantics": (
                "all frames use expert=0 scripted teacher"
                if ALL_EXPERT_BATCH
                else (
                    "expert=scripted teacher; exploration=one bounded seeded contact probe; "
                    "expert_recovery=privileged closed-loop recenter/insert/release after probe"
                )
            ),
            "phase_schema": PHASE_SCHEMA,
            "success_check_timing": "after_place_release_and_retract",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [args.main_image_size, args.main_image_size, 3],
                "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
            },
            "privileged_expert_contract": {
                "reads_object_and_fixture_pose": True,
                "object_state_overwrite": False,
                "robot_state_overwrite": False,
                "commands_through_environment_controller": True,
            },
            "scripted_expert_details": details,
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
            },
            "scripted_expert_generator_version": (
                (
                    "parallel_gear_32env_v3_all_expert_safe_recenter_physics60"
                    if ALL_EXPERT_BATCH
                    else "parallel_gear_32env_v16_half_expert_half_single_polar_contact_physics60"
                )
                if args.task == "gear_mesh"
                else "parallel_nut_32env_v8_all_expert_tight_contact_fixed_height_recenter_physics60"
            ),
            "batch_trajectory_split": (
                {
                    "rule": "all_seeds_expert",
                    "expert_count": NUM_ENVS,
                    "exploration_count": 0,
                }
                if ALL_EXPERT_BATCH
                else {
                    "rule": "even_seed_expert_odd_seed_exploration",
                    "expert_count": NUM_ENVS // 2,
                    "exploration_count": NUM_ENVS // 2,
                }
            ),
            "controlled_exploration_profile": (
                {
                    "enabled": False,
                    "probe_attempts": 0,
                    "reason": f"{args.task}_all_expert_collection",
                }
                if ALL_EXPERT_BATCH
                else {
                    "insertion_only": True,
                    "sampler_version": EXPLORATION_SAMPLER_VERSION,
                    "sampling_key": "SeedSequence(seed, task_tag, attempt)",
                    "probe_pattern": (
                        "single_seeded_polar_offset_contact_vertical_disengage_recenter"
                    ),
                    "probe_attempts": EXPLORATION_ATTEMPTS,
                    "probe_radius_m": list(EXPLORATION_RADIUS_RANGE_M[args.task]),
                    "probe_theta_rad": [-math.pi, math.pi],
                    "radius_derivation": (
                        "bounded to 8-20% of the task's 2.5 mm final center envelope"
                    ),
                    "probe_downward_move_m": (
                        0.016 if args.task == "gear_mesh" else NUT_EXPLORATION_DESCENT_M
                    ),
                    "probe_clearance_m": (
                        "pre_contact_aligned_pose"
                        if args.task == "gear_mesh"
                        else "measured_pre_probe_clearance_minus_0.002"
                    ),
                    "vertical_disengage_m": (
                        0.016
                        if args.task == "gear_mesh"
                        else NUT_EXPLORATION_VERTICAL_DISENGAGE_M
                    ),
                    "recovery_clearance_m": (
                        "pre_contact_height"
                        if args.task == "gear_mesh"
                        else "fixed_tcp_height_after_vertical_disengage_during_xy_recenter"
                    ),
                    "wrench_used_for_control": False,
                    "angle_semantics": (
                        "theta_rad is the randomized polar direction relative to target center"
                    ),
                    "dangerous_random_action": False,
                    "final_success_required": True,
                }
            ),
            "all_expert_preinsert_profile": (
                {
                    "enabled": True,
                    "vertical_disengage_m": 0.016,
                    "xy_target": "privileged_exact_fixture_center",
                    "random_offset": False,
                    "contact_probe": False,
                    "source_type": "expert",
                }
                if ALL_EXPERT_BATCH and args.task == "gear_mesh"
                else {"enabled": False}
            ),
            "scene_roles": list(env.entity_roles),
            "asset_usd_paths": asset_paths,
            **task_metadata,
        }
        high_rate = {
            "timestamp": physics_timestamps,
            "eef_wrench_joint_raw": physics_wrench[:, env_id],
        }
        path = writer.write(expert.buffers[env_id], metadata, high_rate=high_rate)[0]
        written.append(
            {
                "env_id": env_id,
                "seed": seed,
                "path": str(path),
                "validation": validate_hdf5_episode(path),
            }
        )
    summary = {
        "schema": (
            "forcebench_parallel_privileged_placement_32env_v3_all_expert"
            if ALL_EXPERT_BATCH
            else "forcebench_parallel_privileged_placement_32env_v2_mixed_source"
        ),
        "task": args.task,
        "trajectory_mode": TRAJECTORY_MODE,
        "seeds": SEEDS,
        "write_seeds": sorted(WRITE_SEEDS),
        "num_envs": NUM_ENVS,
        "frames_per_episode": len(expert.buffers[0]),
        "episodes": written,
        "failed_episodes": failed,
        "trajectory_type_counts": (
            {"expert": NUM_ENVS, "exploration": 0}
            if ALL_EXPERT_BATCH
            else {"expert": NUM_ENVS // 2, "exploration": NUM_ENVS // 2}
        ),
        "control_source_type_schema": SOURCE_TYPE_SCHEMA,
        "video": str(video_path.resolve()),
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "FORCEBENCH_PARALLEL_PRIVILEGED_PLACEMENT="
        + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
except BaseException as error:
    print(
        "FORCEBENCH_PARALLEL_ABORT="
        + json.dumps({"type": type(error).__name__, "message": str(error)}, separators=(",", ":")),
        flush=True,
    )
    print(traceback.format_exc(), flush=True)
    raise
finally:
    if capture_active:
        env.discard_wrench_capture()
    if expert is not None:
        expert.close()
    env.close()
    simulation_app.close()
