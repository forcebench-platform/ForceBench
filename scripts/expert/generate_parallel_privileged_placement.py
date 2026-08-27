#!/usr/bin/env python3
"""Collect 16 nut_thread or gear_mesh demonstrations in one Isaac process."""

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
NUM_ENVS = 16
PHYSICS_RATE_HZ = 60
CONTROL_RATE_HZ = 15
if len(SEEDS) != NUM_ENVS:
    parser.error("the parallel placement collector requires exactly 16 seeds")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env spacing must be at least 5.5 m")

os.environ.setdefault("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
os.environ["FORCEBENCH_TASK"] = args.task
os.environ.setdefault("FORCEBENCH_EXPERT_BACKEND", "scripted")

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import numpy as np
import torch
from isaaclab.utils import math as math_utils

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task


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
    def __init__(self, env, task, plan: BatchPlan, desired_yaws: np.ndarray):
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
        self.gear_released_during_seating = np.zeros(NUM_ENVS, dtype=bool)
        self.gear_preinsert_center_error_m = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.privileged_roles = (
            ("target_bolt", "nut")
            if args.task == "nut_thread"
            else ("target_base", "small_gear", "large_gear", "medium_gear")
        )

    @property
    def alive(self) -> np.ndarray:
        return np.asarray([not values for values in self.failures], dtype=bool)

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

    def record_and_step(self, requested, gripper, phase: str) -> None:
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
            )
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
            self.record_and_step(requested, gripper, phase)
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

    def hold(self, frames: int, gripper: float, phase: str) -> None:
        targets = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
        start = len(self.buffers[0])
        for _ in range(frames):
            self.record_and_step(targets, gripper, phase)
        self.add_phase(phase, start)

    def command_for_frames(
        self,
        targets,
        gripper,
        phase: str,
        *,
        frames: int,
        enabled: np.ndarray | None = None,
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
            self.record_and_step(requested, gripper, phase)
        self.add_phase(phase, start)

    def servo_object(
        self,
        desired,
        phase: str,
        *,
        max_frames: int = 180,
        tolerance_m: float,
        position_gain: float,
    ) -> np.ndarray:
        desired = np.asarray(desired, dtype=np.float32).reshape(NUM_ENVS, 3)
        enabled = self.alive
        converged = ~enabled.copy()
        position_errors = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        yaw_errors = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        start = len(self.buffers[0])
        for frame in range(max_frames):
            positions, yaws = self.object_pose()
            delta = desired - positions
            yaw_delta = wrap_pi(self.desired_yaws - yaws)
            moving = enabled & ~converged
            self.command_yaws[moving] += np.clip(yaw_delta[moving], -0.04, 0.04)
            self.command_yaws = wrap_pi(self.command_yaws)
            eef = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
            requested = eef.copy()
            requested[moving] += np.float32(position_gain) * delta[moving]
            self.record_and_step(requested, self.plan.grasp_open_fraction, phase)
            after_positions, after_yaws = self.object_pose()
            position_errors = np.linalg.norm(desired - after_positions, axis=1)
            yaw_errors = np.abs(wrap_pi(self.desired_yaws - after_yaws))
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
            self.failures[env_id].append(
                f"phase {phase} object servo did not converge: "
                f"position={position_errors[env_id]:.4f}m "
                f"yaw={math.degrees(float(yaw_errors[env_id])):.2f}deg"
            )
        return converged

    def thread_object(self, target_positions: np.ndarray) -> None:
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
            self.record_and_step(requested, self.plan.grasp_open_fraction, "thread_engagement")
        self.add_phase("thread_engagement", start)

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
        for clearance in self.plan.placement_clearances_m:
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
            )
        self.thread_object(self.target_objects)
        self.hold(8, self.plan.grasp_open_fraction, "placement_settle")
        self.hold(14, 1.0, "release")
        retract = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
        retract[:, 2] += np.float32(self.plan.retract_height_m)
        self.move_tcp(retract, 1.0, "retract")
        self.hold(12, 1.0, "final_settle")

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

        # Descend while cancelling lateral deflection from shaft/tooth contact.
        seated_tcp_z = self.target_objects[:, 2] + self.eef_minus_object[:, 2]
        seating_tcp_floor_z = seated_tcp_z - np.float32(0.006)
        command_z = np.full(NUM_ENVS, 0.070, dtype=np.float32)
        base_mesh_yaws = self.command_yaws.copy()
        twist_offsets = np.radians(
            np.asarray([1.5, -1.5, 3.0, -3.0, 4.5, -4.5, 0.0], dtype=np.float32)
        )
        twist_indices = np.zeros(NUM_ENVS, dtype=np.int32)
        seating_done = ~self.alive
        for iteration in range(40):
            positions, _ = self.object_pose()
            object_rolls, object_pitches, _, tilt_deg = self.object_rpy_and_tilt()
            xy_error = self.target_objects[:, :2] - positions[:, :2]
            center_error = np.linalg.norm(xy_error, axis=1)
            height_error = positions[:, 2] - self.target_objects[:, 2]
            seating_done |= self.alive & (center_error <= 0.0035) & (
                np.abs(height_error) <= 0.006
            ) & (tilt_deg <= 2.0)
            working = self.alive & ~seating_done
            if not np.any(working):
                break
            self.align_object_axis(working, object_rolls, object_pitches)

            release_now = (
                working
                & (iteration >= 17)
                & (command_z <= seating_tcp_floor_z + 1.0e-6)
                & (center_error <= 0.00055)
                & (tilt_deg <= 1.0)
                & (height_error > 0.012)
            )
            if np.any(release_now):
                tilt_deg = self.object_tilt_degrees()
                print(
                    "FORCEBENCH_PARALLEL_GEAR_RELEASE_STATE="
                    + json.dumps(
                        [
                            {
                                "env_id": int(env_id),
                                "center_error_m": float(center_error[env_id]),
                                "height_error_m": float(height_error[env_id]),
                                "tilt_deg": float(tilt_deg[env_id]),
                                "tcp_z_m": float(command_z[env_id]),
                            }
                            for env_id in np.flatnonzero(release_now)
                        ],
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                release_targets = np.column_stack((command_xy, command_z))
                grippers = np.full(
                    NUM_ENVS, self.plan.grasp_open_fraction, dtype=np.float32
                )
                grippers[self.gear_released_during_seating | release_now] = 1.0
                self.command_for_frames(
                    release_targets,
                    grippers,
                    "release_gear",
                    frames=36,
                    enabled=self.alive,
                )
                after_release, _ = self.object_pose()
                release_center_error = np.linalg.norm(
                    after_release[:, :2] - self.target_objects[:, :2], axis=1
                )
                release_height_error = (
                    after_release[:, 2] - self.target_objects[:, 2]
                )
                release_success = (
                    release_now
                    & (release_center_error <= 0.005)
                    & (np.abs(release_height_error) <= 0.006)
                )
                self.gear_released_during_seating |= release_success
                seating_done |= release_success
                for env_id in np.flatnonzero(release_now & ~release_success):
                    self.failures[env_id].append(
                        "aligned gravity release did not seat gear: "
                        f"center={release_center_error[env_id]:.6f}m "
                        f"height={release_height_error[env_id]:.6f}m"
                    )
                working = self.alive & ~seating_done
                if not np.any(working):
                    break
                positions, _ = self.object_pose()
                xy_error = self.target_objects[:, :2] - positions[:, :2]
                center_error = np.linalg.norm(xy_error, axis=1)
                height_error = positions[:, 2] - self.target_objects[:, 2]

            norms = np.linalg.norm(xy_error, axis=1, keepdims=True)
            correction = np.float32(0.50) * xy_error * np.minimum(
                1.0, 0.006 / np.maximum(np.float32(0.50) * norms, 1.0e-9)
            ).astype(np.float32)
            command_xy[working] += correction[working]
            descend = working & (center_error <= 0.0045)
            command_z[descend] = np.maximum(
                seating_tcp_floor_z[descend], command_z[descend] - np.float32(0.005)
            )
            twist = (
                working
                & (iteration >= 10)
                & (command_z <= seating_tcp_floor_z + 1.0e-6)
                & (center_error <= 0.0045)
                & (height_error > 0.012)
            )
            for env_id in np.flatnonzero(twist):
                offset = twist_offsets[twist_indices[env_id] % len(twist_offsets)]
                self.command_yaws[env_id] = wrap_pi(
                    np.asarray([base_mesh_yaws[env_id] + offset], dtype=np.float32)
                )[0]
                twist_indices[env_id] += 1
            targets = np.column_stack((command_xy, command_z))
            grippers = np.full(
                NUM_ENVS, self.plan.grasp_open_fraction, dtype=np.float32
            )
            grippers[self.gear_released_during_seating] = 1.0
            self.command_for_frames(
                targets,
                grippers,
                "final_seating",
                frames=10,
                enabled=self.alive,
            )

        positions, _ = self.object_pose()
        center_error = np.linalg.norm(
            positions[:, :2] - self.target_objects[:, :2], axis=1
        )
        height_error = positions[:, 2] - self.target_objects[:, 2]
        for env_id in np.flatnonzero(
            self.alive & ((center_error > 0.006) | (np.abs(height_error) > 0.012))
        ):
            self.failures[env_id].append(
                "feedback seating did not converge: "
                f"center={center_error[env_id]:.6f}m height={height_error[env_id]:.6f}m"
            )

        seat_targets = np.column_stack((command_xy, seated_tcp_z))
        seat_grippers = np.full(
            NUM_ENVS, self.plan.grasp_open_fraction, dtype=np.float32
        )
        seat_grippers[self.gear_released_during_seating] = 1.0
        self.command_for_frames(
            seat_targets,
            seat_grippers,
            "final_seating",
            frames=18,
            enabled=self.alive,
        )
        positions, _ = self.object_pose()
        center_error = np.linalg.norm(
            positions[:, :2] - self.target_objects[:, :2], axis=1
        )
        height_error = positions[:, 2] - self.target_objects[:, 2]
        for env_id in np.flatnonzero(
            self.alive & ((center_error > 0.008) | (np.abs(height_error) > 0.014))
        ):
            self.failures[env_id].append(
                "gear not plausibly seated before release: "
                f"center={center_error[env_id]:.6f}m height={height_error[env_id]:.6f}m"
            )
        self.hold(42, 1.0, "release_gear")
        retract = np.column_stack(
            (
                self.target_objects[:, :2],
                np.full(NUM_ENVS, 0.165, dtype=np.float32),
            )
        )
        self.move_tcp(retract, 1.0, "retract_and_settle", tolerance_m=0.006)
        self.hold(55, 1.0, "retract_and_settle")

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
                    "gear_released_during_seating": bool(
                        self.gear_released_during_seating[env_id]
                    ),
                    "gear_preinsert_center_error_m": float(
                        self.gear_preinsert_center_error_m[env_id]
                    ),
                    "gear_seating_downward_bias_m": 0.006
                    if args.task == "gear_mesh"
                    else 0.0,
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
        placement_clearances_m=(0.100, 0.050, 0.025, 0.012, 0.006, 0.0),
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
    expert = ParallelPlacementExpert(env, task, plan, desired_yaws)
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
        metadata = {
            "task": args.task,
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
            "control_source": "parallel_privileged_object_pose_closed_loop_smooth_position",
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
                "parallel_privileged_gear_reachable_submm_v2_physics60"
                if args.task == "gear_mesh"
                else "parallel_privileged_placement_v1_physics60"
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
        "schema": "forcebench_parallel_privileged_placement_16env_v1",
        "task": args.task,
        "seeds": SEEDS,
        "write_seeds": sorted(WRITE_SEEDS),
        "num_envs": NUM_ENVS,
        "frames_per_episode": len(expert.buffers[0]),
        "episodes": written,
        "failed_episodes": failed,
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
    env.close()
    simulation_app.close()
