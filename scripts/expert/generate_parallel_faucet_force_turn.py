#!/usr/bin/env python3
"""Collect 32 pure-rotation faucet force demonstrations in one Isaac process."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--artifact_dir", type=Path, required=True)
parser.add_argument("--seeds", required=True)
parser.add_argument("--write_seeds", required=True)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--video_tile_size", type=int, default=192)
parser.add_argument("--probe_min_angle_deg", type=float, default=5.0)
parser.add_argument("--probe_max_angle_deg", type=float, default=12.0)
parser.add_argument("--switch_torque_threshold_nm", type=float, default=0.11)
parser.add_argument("--max_turn_frames", type=int, default=360)
parser.add_argument("--max_settle_frames", type=int, default=240)
parser.add_argument("--hold_frames", type=int, default=24)
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
PHASE_SCHEMA = {
    name: index
    for index, name in enumerate(
        (
            "approach",
            "descend",
            "close",
            "probe_high",
            "switch_direction",
            "turn_selected",
            "settle",
            "open",
            "retreat_away_from_pipe",
            "lift_clear",
            "final_balanced_move",
        )
    )
}
if len(SEEDS) != NUM_ENVS:
    parser.error("the parallel faucet collector requires exactly 32 seeds")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env spacing must be at least 5.5 m")
if args.max_turn_frames < 1 or args.max_settle_frames < 1 or args.hold_frames < 1:
    parser.error("frame counts must be positive")
if not 0.0 < args.probe_min_angle_deg < args.probe_max_angle_deg:
    parser.error("probe angle limits are invalid")

os.environ.setdefault("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
os.environ["FORCEBENCH_TASK"] = "faucet_force_turn"
os.environ.setdefault("FORCEBENCH_EXPERT_BACKEND", "scripted")

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import cv2
import numpy as np
import torch
import warp as wp
from isaaclab.utils import math as math_utils

from forcebench.seed_balance import final_move_sign, VERSION as DIRECTION_VERSION
from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import (
    CONTROL_SOURCE_TYPE_SCHEMA,
    EpisodeBuffer,
    HDF5EpisodeWriter,
    validate_hdf5_episode,
)
from forcebench.registry import load_task
from forcebench.tasks.faucet_force_turn.spec import SPEC, spec_from_layout


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rgb_batch(value) -> np.ndarray:
    return as_numpy(value)[..., :3].astype(np.uint8)


def trajectory_mode(seed: int) -> str:
    """Even seeds are direct experts; odd seeds are measured-force exploration."""

    return "direct_expert" if seed % 2 == 0 else "force_explore_switch"


class ParallelFaucetExpert:
    def __init__(self, env, task):
        self.env = env
        self.task = task
        self.buffers = [EpisodeBuffer() for _ in range(NUM_ENVS)]
        self.zero_action = torch.zeros((NUM_ENVS, 7), dtype=torch.float32, device=env.device)
        self.specs = [spec_from_layout(layout) for layout in env.layouts]
        self.modes = tuple(trajectory_mode(seed) for seed in SEEDS)
        self.easy_sign = np.asarray(
            [spec.easy_direction_sign for spec in self.specs], dtype=np.int8
        )
        self.hard_sign = -self.easy_sign
        self.pivots = np.asarray(
            [
                [base + offset for base, offset in zip(layout["valve_position_m"], spec.lever_pivot_offset_m)]
                for layout, spec in zip(env.layouts, self.specs)
            ],
            dtype=np.float32,
        )
        self.radii = np.asarray([spec.expert_grasp_radius_m for spec in self.specs], dtype=np.float32)
        self.centerline_y = np.asarray(
            [spec.lever_rod_centerline_y_offset_m for spec in self.specs], dtype=np.float32
        )
        self.centerline_x = np.asarray(
            [spec.lever_rod_centerline_x_offset_m for spec in self.specs], dtype=np.float32
        )
        self.critical = np.asarray(
            [spec.detent_center_magnitude_deg for spec in self.specs], dtype=np.float32
        )
        self.drop_end = np.asarray(
            [
                spec.detent_center_magnitude_deg + spec.detent_drop_width_deg
                for spec in self.specs
            ],
            dtype=np.float32,
        )
        self.target = np.asarray(
            [spec.target_final_angle_deg for spec in self.specs], dtype=np.float32
        )
        self.final_angle_tolerance = np.asarray(
            [spec.final_angle_tolerance_deg for spec in self.specs], dtype=np.float32
        )
        self.max_final_speed = np.asarray(
            [spec.max_final_valve_speed_rad_s for spec in self.specs], dtype=np.float32
        )
        self.valve_y = np.asarray(
            [layout["valve_position_m"][1] for layout in env.layouts], dtype=np.float32
        )
        self.water_outlets = np.asarray(
            [layout["water_outlet_position_m"] for layout in env.layouts], dtype=np.float32
        )
        self.min_outlet_distance = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        self.phase_ranges: list[dict[str, int | str]] = []
        self.command_angle_deg = np.zeros(NUM_ENVS, dtype=np.float32)
        self.switched = np.asarray(
            [mode == "direct_expert" for mode in self.modes], dtype=bool
        )
        self.force_switch_triggered = np.zeros(NUM_ENVS, dtype=bool)
        self.probe_failed = np.zeros(NUM_ENVS, dtype=bool)
        self.turn_complete = np.zeros(NUM_ENVS, dtype=bool)
        self.detector_windows = [deque(maxlen=9) for _ in range(NUM_ENVS)]
        self.detector_streak = np.zeros(NUM_ENVS, dtype=np.int32)
        self.switch_frame = np.full(NUM_ENVS, -1, dtype=np.int32)
        self.switch_angle_deg = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.switch_torque_nm = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.maximum_easy_progress_deg = np.zeros(NUM_ENVS, dtype=np.float32)
        self.maximum_hard_progress_deg = np.zeros(NUM_ENVS, dtype=np.float32)
        self.angle_traces = [[] for _ in range(NUM_ENVS)]
        self.torque_norm_traces = [[] for _ in range(NUM_ENVS)]
        self.phase_traces = [[] for _ in range(NUM_ENVS)]
        size = int(args.video_tile_size)
        self.video_path = args.artifact_dir / "faucet_force_turn_32env_mosaic.mp4"
        self.video = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            CONTROL_RATE_HZ,
            (8 * size, 4 * size),
        )
        if not self.video.isOpened():
            raise RuntimeError(f"failed to create video: {self.video_path}")

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    def mechanism_state(self) -> tuple[np.ndarray, np.ndarray]:
        entity = self.env.entity("faucet_valve")
        positions = as_numpy(wp.to_torch(entity.root_view.get_dof_positions()))
        velocities = as_numpy(wp.to_torch(entity.root_view.get_dof_velocities()))
        return (
            positions[:, self.task._turn_joint_id].astype(np.float32),
            velocities[:, self.task._turn_joint_id].astype(np.float32),
        )

    def arc_points(self, angles_rad) -> np.ndarray:
        angles = np.asarray(angles_rad, dtype=np.float32).reshape(NUM_ENVS)
        physical = np.float32(SPEC.lever_rotation_sign) * angles
        cosine = np.cos(physical)
        sine = np.sin(physical)
        points = self.pivots.copy()
        points[:, 0] += self.centerline_x * cosine + self.radii * sine
        points[:, 1] += self.centerline_y
        points[:, 2] += -self.centerline_x * sine + self.radii * cosine
        return points

    def rpy_for_angles(self, angles_rad) -> np.ndarray:
        angles = torch.as_tensor(angles_rad, dtype=torch.float32, device=self.env.device).reshape(-1)
        initial = math_utils.quat_from_euler_xyz(
            torch.full_like(angles, math.pi), torch.zeros_like(angles), torch.zeros_like(angles)
        )
        axis = torch.tensor([[0.0, 1.0, 0.0]], device=self.env.device).repeat(NUM_ENVS, 1)
        swing = math_utils.quat_from_angle_axis(SPEC.lever_rotation_sign * angles, axis)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(math_utils.quat_mul(swing, initial))
        return as_numpy(torch.stack((roll, pitch, yaw), dim=1)).astype(np.float32)

    def source_type(self, env_id: int, phase: str) -> int:
        if self.modes[env_id] == "direct_expert":
            return CONTROL_SOURCE_TYPE_SCHEMA["expert"]
        if phase == "probe_high":
            return CONTROL_SOURCE_TYPE_SCHEMA["exploration"]
        if self.force_switch_triggered[env_id]:
            return CONTROL_SOURCE_TYPE_SCHEMA["expert_recovery"]
        return CONTROL_SOURCE_TYPE_SCHEMA["expert"]

    def make_mosaic(
        self,
        images: np.ndarray,
        phases: list[str],
        angles_rad: np.ndarray,
    ) -> np.ndarray:
        size = int(args.video_tile_size)
        canvas = np.zeros((4 * size, 8 * size, 3), dtype=np.uint8)
        wrench = as_numpy(self.env.force_sensor_joint_smooth)
        torque_norm = np.linalg.norm(wrench[:, 3:6], axis=1)
        for env_id, image in enumerate(images):
            tile = cv2.resize(image[..., ::-1], (size, size), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (size, 47), (18, 18, 18), -1)
            group = "D" if self.modes[env_id] == "direct_expert" else "X"
            mark = "REC" if SEEDS[env_id] in WRITE_SEEDS else "FILL"
            cv2.putText(
                tile,
                f"e{env_id:02d} s{SEEDS[env_id]:04d} {group}/{mark}",
                (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.27,
                (245, 245, 245), 1, cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"{phases[env_id][:8]} A{math.degrees(angles_rad[env_id]):+.1f}",
                (3, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                (230, 230, 230), 1, cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"R{1000.0 * self.radii[env_id]:.0f}mm |M|{torque_norm[env_id]:.3f}",
                (3, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.23,
                (210, 235, 245), 1, cv2.LINE_AA,
            )
            row, column = divmod(env_id, 8)
            canvas[
                row * size:(row + 1) * size,
                column * size:(column + 1) * size,
            ] = tile
        return canvas

    def record_and_step(self, positions, orientation_angles, gripper: float, phase) -> None:
        positions = np.asarray(positions, dtype=np.float32).reshape(NUM_ENVS, 3)
        phases = [phase for _ in range(NUM_ENVS)] if isinstance(phase, str) else list(phase)
        if len(phases) != NUM_ENVS:
            raise ValueError("phase batch must match the 32 cloned environments")
        actions = np.concatenate(
            (
                positions,
                self.rpy_for_angles(orientation_angles),
                np.full((NUM_ENVS, 1), 2.0 * gripper - 1.0, dtype=np.float32),
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
        valve_pose = as_numpy(self.env.entity_world_pose("faucet_valve"))
        entity = self.env.entity("faucet_valve")
        valve_joint_position = as_numpy(entity.data.joint_pos.torch)
        valve_joint_velocity = as_numpy(entity.data.joint_vel.torch)
        turn_position, _ = self.mechanism_state()
        fingertip = as_numpy(self.env.fingertip_midpoint_pos)
        self.min_outlet_distance = np.minimum(
            self.min_outlet_distance,
            np.linalg.norm(fingertip - self.water_outlets, axis=1),
        )
        timestamp = (len(self.buffers[0]) + 1) * self.env.step_dt
        for env_id, buffer in enumerate(self.buffers):
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
                privileged={
                    "objects/faucet_valve/pose": valve_pose[env_id],
                    "objects/faucet_valve/joint_position": valve_joint_position[env_id],
                    "objects/faucet_valve/joint_velocity": valve_joint_velocity[env_id],
                    "objects/faucet_valve/turn_position": np.asarray(
                        [turn_position[env_id]], dtype=np.float32
                    ),
                },
                phase_id=PHASE_SCHEMA[phases[env_id]],
                control_source_type_id=self.source_type(env_id, phases[env_id]),
            )
            self.phase_traces[env_id].append(phases[env_id])
        self.video.write(self.make_mosaic(main, phases, turn_position))
        self.env.set_cartesian_rpy_target(actions[:, :6], np.full(NUM_ENVS, gripper))
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        for env_id, buffer in enumerate(self.buffers):
            buffer.set_last_transition_outcome(
                reward=float(reward[env_id]),
                success=False,
                terminated=bool(terminated[env_id]),
                truncated=bool(truncated[env_id]),
            )
        turn_after, _ = self.mechanism_state()
        angle_deg = np.degrees(turn_after)
        ema_after = as_numpy(self.env.force_sensor_joint_smooth)
        torque_norm = np.linalg.norm(ema_after[:, 3:6], axis=1)
        self.maximum_easy_progress_deg = np.maximum(
            self.maximum_easy_progress_deg, self.easy_sign * angle_deg
        )
        self.maximum_hard_progress_deg = np.maximum(
            self.maximum_hard_progress_deg, self.hard_sign * angle_deg
        )
        for env_id in range(NUM_ENVS):
            self.angle_traces[env_id].append(float(angle_deg[env_id]))
            self.torque_norm_traces[env_id].append(float(torque_norm[env_id]))
        self.env.update_cameras()

    def track(self, positions, angles, gripper: float, frames: int, phase: str) -> None:
        start = len(self.buffers[0])
        for _ in range(frames):
            self.record_and_step(positions, angles, gripper, phase)
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})

    def run(self) -> list[dict[str, object]]:
        zero_angles = np.zeros(NUM_ENVS, dtype=np.float32)
        grasp = self.arc_points(zero_angles)
        approach = grasp.copy()
        approach[:, 2] += 0.090
        self.track(approach, zero_angles, 1.0, 60, "approach")
        self.track(grasp, zero_angles, 1.0, 40, "descend")
        self.track(grasp, zero_angles, 0.0, 20, "close")
        closed_fingers = as_numpy(self.env.robot.data.joint_pos.torch)[:, 7:9].copy()
        grasp_offset = as_numpy(self.env.fingertip_midpoint_pos) - grasp

        # Direct episodes turn immediately in the privileged correct direction.
        # Exploration episodes start on the hidden hard side, but the online
        # reversal decision reads only the measured 9-frame median torque norm.
        start = len(self.buffers[0])
        for _ in range(args.max_turn_frames):
            phases: list[str] = []
            for env_id, spec in enumerate(self.specs):
                if self.turn_complete[env_id]:
                    phases.append("settle")
                    continue
                if self.modes[env_id] == "force_explore_switch" and not self.switched[env_id]:
                    self.command_angle_deg[env_id] += 0.24 * self.hard_sign[env_id]
                    phases.append("probe_high")
                else:
                    self.command_angle_deg[env_id] += 0.24 * self.easy_sign[env_id]
                    live_angle = (
                        self.angle_traces[env_id][-1]
                        if self.angle_traces[env_id]
                        else 0.0
                    )
                    phases.append(
                        "turn_selected"
                        if self.easy_sign[env_id] * live_angle >= 0.0
                        else "switch_direction"
                    )

            live_angle = self.mechanism_state()[0]
            positions = self.arc_points(np.deg2rad(self.command_angle_deg)) + grasp_offset
            self.record_and_step(positions, live_angle, 0.0, phases)
            after_angle, _ = self.mechanism_state()
            after_deg = np.degrees(after_angle)
            wrench_ema = as_numpy(self.env.force_sensor_joint_smooth)
            measured_torque = np.linalg.norm(wrench_ema[:, 3:6], axis=1)

            for env_id, spec in enumerate(self.specs):
                if self.modes[env_id] == "force_explore_switch" and not self.switched[env_id]:
                    hard_progress = self.hard_sign[env_id] * after_deg[env_id]
                    self.detector_windows[env_id].append(float(measured_torque[env_id]))
                    filtered = float(np.median(self.detector_windows[env_id]))
                    if (
                        hard_progress >= args.probe_min_angle_deg
                        and filtered >= args.switch_torque_threshold_nm
                    ):
                        self.detector_streak[env_id] += 1
                    else:
                        self.detector_streak[env_id] = 0
                    if self.detector_streak[env_id] >= 4:
                        self.switched[env_id] = True
                        self.force_switch_triggered[env_id] = True
                        self.switch_frame[env_id] = len(self.buffers[env_id]) - 1
                        self.switch_angle_deg[env_id] = float(after_deg[env_id])
                        self.switch_torque_nm[env_id] = filtered
                    elif hard_progress >= args.probe_max_angle_deg:
                        self.probe_failed[env_id] = True
                        self.switched[env_id] = True

                # Use the just-recorded high-water mark.  The articulation
                # view queried again immediately after the step can lag the
                # state captured inside record_and_step by one control frame.
                easy_progress = self.maximum_easy_progress_deg[env_id]
                # Stop inside the post-drop low-load plateau. A large forward
                # overshoot would require reversing across the sharp detent and
                # can pull the shorter 65 mm grasp back to the high-load side.
                # The active turn is complete once the commanded motion has
                # cleared the sharp 0.6-degree unloading interval.  The
                # following settle controller owns the later plateau target.
                if easy_progress >= self.drop_end[env_id] + 0.15:
                    self.turn_complete[env_id] = True
                    # Other clones may need many more frames to reach their
                    # randomized detent.  Do not leave an already-complete
                    # clone pulling beyond its narrow low-load plateau while
                    # it waits for the rest of the 32-env batch.
                    self.command_angle_deg[env_id] = self.target[env_id]
            if np.all(self.turn_complete):
                break
        self.phase_ranges.append(
            {"name": "force_disambiguated_turn", "start": start, "end": len(self.buffers[0])}
        )

        # Settle every valve at its signed sampled target, then hold a short
        # quiet window for final-speed validation.
        windows: list[np.ndarray] = []
        stable_count = np.zeros(NUM_ENVS, dtype=np.int32)
        converged = np.zeros(NUM_ENVS, dtype=bool)
        # Remove any residual command/plant lag accumulated during the shared
        # vectorized turn loop, then close the remaining target error slowly.
        self.command_angle_deg = self.target.copy()
        start = len(self.buffers[0])
        for _ in range(args.max_settle_frames):
            angles, _ = self.mechanism_state()
            errors = self.target - np.degrees(angles)
            self.command_angle_deg += np.clip(0.16 * errors, -0.30, 0.30)
            positions = self.arc_points(np.deg2rad(self.command_angle_deg)) + grasp_offset
            self.record_and_step(positions, angles, 0.0, "settle")
            after_angles = self.mechanism_state()[0]
            windows.append(np.degrees(after_angles))
            windows = windows[-12:]
            if len(windows) == 12:
                window = np.stack(windows)
                within = (
                    np.max(np.abs(window - self.target), axis=0)
                    <= self.final_angle_tolerance
                )
                quiet = np.ptp(window, axis=0) <= 0.45
                stable_count = np.where(within & quiet, stable_count + 1, 0)
                # Do not permanently latch a clone as converged while other
                # clones are still settling.  It must remain stable in the
                # same final window as the rest of the vectorized batch.
                converged = stable_count >= 8
            if np.all(converged):
                break
        for _ in range(args.hold_frames):
            angles, _ = self.mechanism_state()
            errors = self.target - np.degrees(angles)
            self.command_angle_deg += np.clip(0.10 * errors, -0.08, 0.08)
            positions = self.arc_points(np.deg2rad(self.command_angle_deg)) + grasp_offset
            self.record_and_step(positions, angles, 0.0, "settle")
            windows.append(np.degrees(self.mechanism_state()[0]))
            windows = windows[-12:]
        self.phase_ranges.append(
            {"name": "settle", "start": start, "end": len(self.buffers[0])}
        )

        checkpoint_angles, checkpoint_velocities = self.mechanism_state()
        checkpoint_deg = np.degrees(checkpoint_angles)
        target_error = np.abs(checkpoint_deg - self.target)
        window = np.stack(windows) if windows else checkpoint_deg[None, :]
        window_range = np.ptp(window, axis=0)
        stopped = (window_range <= 0.50) & (np.abs(checkpoint_velocities) <= 0.20)
        checkpoint_success = (target_error <= self.final_angle_tolerance) & stopped

        # Open, move away from the lever axis first, and only then lift.  With
        # the shorter 65 mm grasp, lifting in place can let an open finger rub
        # the rod and rotate the valve after a valid checkpoint.
        release_positions = self.arc_points(np.deg2rad(self.command_angle_deg)) + grasp_offset
        self.track(release_positions, checkpoint_angles, 1.0, 24, "open")
        release_fingers = as_numpy(self.env.robot.data.joint_pos.torch)[:, 7:9].copy()
        retreat = release_positions.copy()
        retreat[:, 1] = self.valve_y - 0.085
        self.track(retreat, checkpoint_angles, 1.0, 60, "retreat_away_from_pipe")
        lift = retreat.copy()
        lift[:, 2] = grasp[:, 2] + 0.155
        self.track(lift, checkpoint_angles, 1.0, 60, "lift_clear")
        final_move = lift.copy()
        final_move[:, 1] = self.valve_y + np.asarray([final_move_sign(s) for s in SEEDS]) * 0.085
        self.track(final_move, checkpoint_angles, 1.0, 60, "final_balanced_move")

        final_angles, final_velocities = self.mechanism_state()
        fingertip = as_numpy(self.env.fingertip_midpoint_pos)
        local_root = as_numpy(self.env.entity_local_pose("faucet_valve")[0])
        results = []
        for env_id in range(NUM_ENVS):
            report = dict(self.task.evaluate_final_state_for(self.env, env_id))
            root_error = float(
                np.linalg.norm(local_root[env_id] - np.asarray(self.env.layouts[env_id]["valve_position_m"]))
            )
            custom_success = bool(
                np.max(closed_fingers[env_id]) < 0.0395
                and (
                    self.modes[env_id] == "direct_expert"
                    or (
                        self.force_switch_triggered[env_id]
                        and not self.probe_failed[env_id]
                    )
                )
                and (
                    self.modes[env_id] != "direct_expert"
                    or self.maximum_hard_progress_deg[env_id] <= 2.0
                )
                and np.min(release_fingers[env_id]) >= 0.0395
                and fingertip[env_id, 2] >= grasp[env_id, 2] + 0.09
                and final_move_sign(SEEDS[env_id]) * (fingertip[env_id, 1] - self.valve_y[env_id]) >= 0.055
                and report["success"]
                and abs(float(final_velocities[env_id])) <= self.max_final_speed[env_id]
                and root_error <= 1.0e-5
                and self.min_outlet_distance[env_id] >= 0.10
            )
            report["success"] = custom_success
            report["parallel_expert_checks"] = {
                "gripped": bool(np.max(closed_fingers[env_id]) < 0.0395),
                "turn_complete": bool(self.turn_complete[env_id]),
                "trajectory_mode": self.modes[env_id],
                "expert_grasp_radius_m": float(self.radii[env_id]),
                "force_switch_triggered": bool(self.force_switch_triggered[env_id]),
                "probe_failed": bool(self.probe_failed[env_id]),
                "switch_frame": int(self.switch_frame[env_id]),
                "switch_angle_deg": (
                    None if np.isnan(self.switch_angle_deg[env_id])
                    else float(self.switch_angle_deg[env_id])
                ),
                "switch_torque_norm_nm": (
                    None if np.isnan(self.switch_torque_nm[env_id])
                    else float(self.switch_torque_nm[env_id])
                ),
                "maximum_hard_progress_deg": float(
                    self.maximum_hard_progress_deg[env_id]
                ),
                "maximum_easy_progress_deg": float(
                    self.maximum_easy_progress_deg[env_id]
                ),
                "target_checkpoint_success": bool(checkpoint_success[env_id]),
                "target_error_deg": float(target_error[env_id]),
                "target_window_range_deg": float(window_range[env_id]),
                "final_stopped": bool(
                    abs(float(final_velocities[env_id]))
                    <= self.max_final_speed[env_id]
                ),
                "released": bool(np.min(release_fingers[env_id]) >= 0.0395),
                "cleared_lever_axis": bool(
                    final_move_sign(SEEDS[env_id]) * (fingertip[env_id, 1] - self.valve_y[env_id]) >= 0.055
                ),
                "lifted_clear": bool(
                    fingertip[env_id, 2] >= grasp[env_id, 2] + 0.09
                ),
                "valve_root_position_error_m": root_error,
                "minimum_fingertip_outlet_distance_m": float(self.min_outlet_distance[env_id]),
            }
            if custom_success:
                self.buffers[env_id].set_last_transition_outcome(
                    reward=1.0, success=True, terminated=False, truncated=False
                )
            results.append(
                {
                    "report": report,
                    "phase_ranges_control_frames": list(self.phase_ranges),
                    "critical_peak_angle_deg": float(self.critical[env_id]),
                    "drop_end_angle_deg": float(self.drop_end[env_id]),
                    "target_final_angle_deg": float(self.target[env_id]),
                    "final_valve_angle_deg": float(np.rad2deg(final_angles[env_id])),
                    "final_valve_velocity_rad_s": float(final_velocities[env_id]),
                }
            )
        return results


task = load_task("faucet_force_turn", ASSET_DIR)
cfg = ForceBenchEnvCfg()
cfg.seed = SEEDS[0]
cfg.scene_profile = task.manifest.scene_profile
cfg.robot.actuators["arm_forearm"].effort_limit_sim = 30.0
cfg.robot.spawn.physics_material.static_friction = 2.0
cfg.robot.spawn.physics_material.dynamic_friction = 2.0
cfg.robot.spawn.physics_material.friction_combine_mode = "max"
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
args.output_dir.mkdir(parents=True, exist_ok=True)
args.artifact_dir.mkdir(parents=True, exist_ok=True)

try:
    env.set_episode_seeds(SEEDS)
    env.reset(seed=SEEDS[0])
    env.restore_fixed_arm_pose(env.joint_pos[0, :7].clone())
    env.reset_cameras()
    env.sim.render()
    env.update_cameras()
    env.start_wrench_capture()
    capture_active = True
    expert = ParallelFaucetExpert(env, task)
    results = expert.run()
    high_rate_batch = env.stop_wrench_capture()
    capture_active = False
    expert.close()

    physics_wrench = np.asarray(high_rate_batch["eef_wrench_joint_raw"], dtype=np.float32)
    physics_timestamps = np.asarray(high_rate_batch["timestamp"], dtype=np.float64)
    expected_ratio = PHYSICS_RATE_HZ // CONTROL_RATE_HZ
    if physics_wrench.ndim != 3 or physics_wrench.shape[1:] != (NUM_ENVS, 6):
        raise RuntimeError(f"unexpected batched wrench shape {physics_wrench.shape}")
    if abs(len(physics_timestamps) - len(expert.buffers[0]) * expected_ratio) > expected_ratio:
        raise RuntimeError("high-rate wrench/control sample ratio is invalid")

    writer = HDF5EpisodeWriter(args.output_dir)
    base_task_metadata = dict(task.episode_metadata())
    asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
    asset_paths.update(env.scene_asset_paths)
    asset_paths.update(base_task_metadata.pop("asset_usd_paths", {}))
    written, failed = [], []
    for env_id, seed in enumerate(SEEDS):
        if seed not in WRITE_SEEDS:
            continue
        report = results[env_id]["report"]
        if not report["success"]:
            failure = {"env_id": env_id, "seed": seed,
            "direction_assignment_version": DIRECTION_VERSION,
            "final_move_sign_y": final_move_sign(seed), "report": report}
            failed.append(failure)
            print(
                "FORCEBENCH_PARALLEL_FAUCET_FAILURE="
                + json.dumps(failure, separators=(",", ":")),
                flush=True,
            )
            continue
        existing = sorted(args.output_dir.glob(f"seed_{seed:03d}_episode_*.hdf5"))
        if existing:
            raise RuntimeError(f"refusing to duplicate successful seed {seed}: {existing}")
        task_metadata = copy.deepcopy(base_task_metadata)
        episode_spec = expert.specs[env_id]
        if "load_model" in task_metadata:
            task_metadata["load_model"].update(
                {
                    "resistance_peak_magnitude_deg": (
                        episode_spec.detent_center_magnitude_deg
                    ),
                    "easy_direction": episode_spec.easy_direction_name,
                    "easy_direction_sign": episode_spec.easy_direction_sign,
                    "hard_direction": episode_spec.hard_direction_name,
                    "hard_direction_sign": episode_spec.hard_direction_sign,
                }
            )
        details = {key: value for key, value in results[env_id].items() if key != "report"}
        metadata = {
            "task": "faucet_force_turn",
            "prompt": task.manifest.prompt,
            "success": True,
            "end_reason": "force_disambiguated_turn_release_retract",
            "seed": seed,
            "direction_assignment_version": DIRECTION_VERSION,
            "final_move_sign_y": final_move_sign(seed),
            "initial_layout": env.layouts[env_id],
            "final_success_report": report,
            "control_rate_hz": float(CONTROL_RATE_HZ),
            "physics_rate_hz": float(PHYSICS_RATE_HZ),
            "physics_steps_per_control": PHYSICS_RATE_HZ // CONTROL_RATE_HZ,
            "solver_position_iterations": 64,
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "scene_profile": env.scene_profile_name,
            "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "control_source": "scripted_32env_faucet_force_teacher",
            "trajectory_type": (
                "expert" if trajectory_mode(seed) == "direct_expert" else "exploration"
            ),
            "dataset_group": trajectory_mode(seed),
            "control_source_type_schema": CONTROL_SOURCE_TYPE_SCHEMA,
            "wrench_storage": "raw_and_ema_control_plus_raw_physics",
            "success_check_timing": "after_turn_settle_release_and_retract",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [args.main_image_size, args.main_image_size, 3],
                "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
            },
            "phase_schema": PHASE_SCHEMA,
            "mechanism_dofs": [SPEC.turn_joint_name],
            "expert_grasp_radius_m": float(episode_spec.expert_grasp_radius_m),
            "scripted_expert_details": details,
            "expert_transition_signal": (
                "median_9_control_frames_norm(eef_wrench_joint_ema[torque_xyz])"
            ),
            "expert_uses_analytic_resistance_for_transition": False,
            "force_switch_logic": {
                "initial_probe_side_allocated_from_hidden_hard_direction": (
                    trajectory_mode(seed) == "force_explore_switch"
                ),
                "hidden_direction_used_for_online_switch": False,
                "measured_signal": "norm(forces/control/eef_wrench_joint_ema[Mx:Mz])",
                "median_window_control_frames": 9,
                "probe_min_angle_deg": args.probe_min_angle_deg,
                "probe_max_angle_deg": args.probe_max_angle_deg,
                "threshold_nm": args.switch_torque_threshold_nm,
                "required_consecutive_frames": 4,
            },
            "randomization_audit": {
                "easy_direction": episode_spec.easy_direction_name,
                "easy_direction_sign": episode_spec.easy_direction_sign,
                "detent_center_magnitude_deg": (
                    episode_spec.detent_center_magnitude_deg
                ),
                "detent_angle_bin_1deg": min(
                    9,
                    int(episode_spec.detent_center_magnitude_deg - 20.0),
                ),
            },
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
                "faucet_force_turn_32env_v2_physics60_control15"
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
            "direction_assignment_version": DIRECTION_VERSION,
            "final_move_sign_y": final_move_sign(seed),
                "path": str(path),
                "validation": validate_hdf5_episode(path),
            }
        )
    summary = {
        "schema": "forcebench_parallel_faucet_force_turn_32env_v2",
        "task": "faucet_force_turn",
        "seeds": SEEDS,
        "write_seeds": sorted(WRITE_SEEDS),
        "num_envs": NUM_ENVS,
        "frames_per_episode": len(expert.buffers[0]),
        "video": str(expert.video_path.resolve()),
        "episodes": written,
        "failed_episodes": failed,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "FORCEBENCH_PARALLEL_FAUCET_PRESS_TURN="
        + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
    if failed:
        raise RuntimeError(f"{len(failed)} requested faucet_force_turn episodes failed")
finally:
    if expert is not None:
        expert.close()
    if capture_active:
        env.discard_wrench_capture()
    env.close()
    simulation_app.close()
