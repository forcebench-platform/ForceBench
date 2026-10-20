#!/usr/bin/env python3
"""Collect 32 stove_press_turn episodes in one cloned Isaac process."""

from __future__ import annotations

import argparse
from independent_release_state import CompletionSnapshot, ReleaseState, trim_episode_buffer
import json
import math
import os
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--artifact_dir", type=Path, required=True)
parser.add_argument("--seeds", required=True, help="exactly 32 distinct comma-separated seeds")
parser.add_argument(
    "--write_seeds",
    required=True,
    help="subset of --seeds eligible to be written to the output directory",
)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--video_tile_size", type=int, default=160)
parser.add_argument("--grasp_height_offset_m", type=float, default=-0.001)
parser.add_argument("--probe_min_angle_deg", type=float, default=8.0)
parser.add_argument("--probe_max_angle_deg", type=float, default=20.0)
parser.add_argument("--switch_torque_threshold_nm", type=float, default=0.034)
parser.add_argument(
    "--trajectory_mode",
    choices=("seed_parity", "direct_expert", "force_explore_switch"),
    default="seed_parity",
    help="override the parity-based trajectory split when collecting one trajectory class",
)
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
PRESS_COMMAND_DEPTH_M = 0.0142
if len(SEEDS) != NUM_ENVS:
    parser.error("the stove_press_turn collector requires exactly 32 cloned workspaces")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env_spacing must be at least 5.5 m")
if not 0.0 < args.probe_min_angle_deg < args.probe_max_angle_deg:
    parser.error("probe angle limits are invalid")

os.environ.setdefault("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
os.environ["FORCEBENCH_TASK"] = "stove_press_turn"
os.environ.setdefault("FORCEBENCH_EXPERT_BACKEND", "scripted")

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import cv2
import numpy as np
import torch
import warp as wp

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
from forcebench.tasks.stove_press_turn.spec import (
    directional_resistance_nm,
    spec_from_layout,
)


PHASES = (
    "approach",
    "descend",
    "close",
    "press",
    "probe_high",
    "switch_direction",
    "turn_selected",
    "settle",
    "release_press",
    "open",
    "lift",
    "retreat",
)
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASES)}
SPEED_PROFILE = {
    "approach_frames": 75,
    "descend_frames": 55,
    "close_frames": 35,
    "press_ramp_frames": 50,
    "press_hold_frames": 15,
    "maximum_turn_frames": 650,
    "maximum_settle_frames": 220,
    "settle_tolerance_deg": 0.50,
    "settle_stable_frames": 5,
    "release_stability_window_frames": 12,
    "release_stability_max_frames": 180,
    "release_stability_range_deg": 0.50,
    "release_stability_mean_step_deg": 0.10,
    "release_press_frames": 45,
    "open_ramp_frames": 60,
    "open_hold_frames": 10,
    "lift_frames": 75,
    "retreat_frames": 45,
    "command_step_deg": 0.32,
    "target_approach_margin_deg": 2.0,
    "settle_command_step_deg": 0.08,
}
QUALITY_HISTORY_FIELDS = (
    "minimum_press_m", "maximum_turn_press_m", "maximum_easy_progress_deg",
    "maximum_hard_progress_deg", "maximum_tracking_error_m",
)


def as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def rgb_batch(value) -> np.ndarray:
    return as_numpy(value)[..., :3].astype(np.uint8)


def trajectory_mode(seed: int) -> str:
    if args.trajectory_mode != "seed_parity":
        return args.trajectory_mode
    return "direct_expert" if seed % 2 == 0 else "force_explore_switch"


class ParallelStovePressTurnExpert:
    def __init__(self, env: ForceBenchEnv, task, specs, video_path: Path):
        self.env = env
        self.task = task
        self.specs = tuple(specs)
        self.modes = tuple(trajectory_mode(seed) for seed in SEEDS)
        self.buffers = [EpisodeBuffer() for _ in range(NUM_ENVS)]
        self.zero_action = torch.zeros((NUM_ENVS, 7), dtype=torch.float32, device=env.device)
        self.easy_sign = np.asarray([spec.easy_direction_sign for spec in specs], dtype=np.int8)
        self.hard_sign = -self.easy_sign
        self.command_yaw = np.zeros(NUM_ENVS, dtype=np.float64)
        self.switched = np.asarray([mode == "direct_expert" for mode in self.modes], dtype=bool)
        self.force_switch_triggered = np.zeros(NUM_ENVS, dtype=bool)
        self.probe_failed = np.zeros(NUM_ENVS, dtype=bool)
        self.turn_complete = np.zeros(NUM_ENVS, dtype=bool)
        self.settle_complete = np.zeros(NUM_ENVS, dtype=bool)
        self.detector_windows = [deque(maxlen=11) for _ in range(NUM_ENVS)]
        self.detector_streak = np.zeros(NUM_ENVS, dtype=np.int64)
        self.switch_frame = np.full(NUM_ENVS, -1, dtype=np.int64)
        self.switch_angle_deg = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.switch_torque_nm = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.minimum_press_m = np.zeros(NUM_ENVS, dtype=np.float64)
        self.maximum_turn_press_m = np.full(NUM_ENVS, -np.inf, dtype=np.float64)
        self.maximum_easy_progress_deg = np.zeros(NUM_ENVS, dtype=np.float64)
        self.maximum_hard_progress_deg = np.zeros(NUM_ENVS, dtype=np.float64)
        self.maximum_tracking_error_m = np.zeros(NUM_ENVS, dtype=np.float64)
        self.closed_finger_positions_m = np.zeros((NUM_ENVS, 2), dtype=np.float64)
        self.open_finger_positions_m = np.zeros((NUM_ENVS, 2), dtype=np.float64)
        self.release_stability_frames = np.zeros(NUM_ENVS, dtype=np.int64)
        self.release_stability_range_deg = np.full(NUM_ENVS, np.inf, dtype=np.float64)
        self.angle_before_open_deg = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.angle_after_open_deg = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.angle_traces = [[] for _ in range(NUM_ENVS)]
        self.press_traces = [[] for _ in range(NUM_ENVS)]
        self.ema_torque_traces = [[] for _ in range(NUM_ENVS)]
        self.phase_traces = [[] for _ in range(NUM_ENVS)]
        self.source_type_traces = [[] for _ in range(NUM_ENVS)]
        self.independent_stage = ["initial" for _ in range(NUM_ENVS)]
        self.stage_events = []
        self.completed_frames = np.zeros(NUM_ENVS, dtype=np.int64)
        self.total_control_frames = 0
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

    def mechanism_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        entity = self.env.entity("stove_knob")
        positions = wp.to_torch(entity.root_view.get_dof_positions())
        velocities = wp.to_torch(entity.root_view.get_dof_velocities())
        return (
            as_numpy(positions[:, self.task._press_joint_id]),
            as_numpy(positions[:, self.task._turn_joint_id]),
            as_numpy(velocities[:, self.task._press_joint_id]),
            as_numpy(velocities[:, self.task._turn_joint_id]),
        )

    def privileged_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position, quaternion = self.env.entity_local_pose("stove_knob")
        entity = self.env.entity("stove_knob")
        pose = as_numpy(torch.cat((position, quaternion), dim=-1))
        return pose, as_numpy(entity.data.joint_pos.torch), as_numpy(entity.data.joint_vel.torch)

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
        press_m: np.ndarray,
        angles_rad: np.ndarray,
    ) -> np.ndarray:
        size = int(args.video_tile_size)
        canvas = np.zeros((4 * size, 8 * size, 3), dtype=np.uint8)
        wrench = as_numpy(self.env.force_sensor_joint_smooth)
        for env_id, image in enumerate(images):
            tile = cv2.resize(image[..., ::-1], (size, size), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (size, 47), (18, 18, 18), -1)
            group = "D" if self.modes[env_id] == "direct_expert" else "X"
            mark = "REC" if SEEDS[env_id] in WRITE_SEEDS else "FILL"
            cv2.putText(
                tile,
                f"e{env_id:02d} s{SEEDS[env_id]:04d} {group}/{mark}",
                (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (245, 245, 245), 1, cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"{(phases[env_id] if self.independent_stage[env_id] == 'initial' else self.independent_stage[env_id])[:8]} A{math.degrees(angles_rad[env_id]):+.1f}",
                (3, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (230, 230, 230), 1, cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"P{-1000.0 * press_m[env_id]:.1f}mm |Tz|{abs(wrench[env_id, 5]):.3f}",
                (3, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (210, 235, 245), 1, cv2.LINE_AA,
            )
            row, column = divmod(env_id, 8)
            canvas[row * size:(row + 1) * size, column * size:(column + 1) * size] = tile
        return canvas

    def write_overview(self, path: Path, phase: str) -> None:
        press, angle, _, _ = self.mechanism_state()
        phases = [phase for _ in range(NUM_ENVS)]
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.make_mosaic(rgb_batch(self.env.table_rgb), phases, press, angle))

    def _record_and_step(
        self,
        positions_m: np.ndarray,
        yaw_rad: np.ndarray,
        gripper,
        phases: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(positions_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        yaws = np.asarray(yaw_rad, dtype=np.float32).reshape(NUM_ENVS)
        grippers = np.asarray(gripper, dtype=np.float32)
        if grippers.ndim == 0:
            grippers = np.full(NUM_ENVS, float(grippers), dtype=np.float32)
        actions = np.concatenate(
            (
                positions,
                np.full((NUM_ENVS, 1), math.pi, dtype=np.float32),
                np.zeros((NUM_ENVS, 1), dtype=np.float32),
                yaws[:, None],
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
        tcp = as_numpy(torch.cat(
            (self.env.fingertip_midpoint_pos, self.env.fingertip_midpoint_quat), dim=-1
        ))
        knob_pose, knob_joint_position, knob_joint_velocity = self.privileged_batch()
        press, angle, _, _ = self.mechanism_state()
        timestamp = (len(self.buffers[0]) + 1) * self.env.step_dt
        for env_id, buffer in enumerate(self.buffers):
            source_type = self.source_type(env_id, phases[env_id])
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
                    "objects/stove_knob/pose": knob_pose[env_id],
                    "objects/stove_knob/joint_position": knob_joint_position[env_id],
                    "objects/stove_knob/joint_velocity": knob_joint_velocity[env_id],
                    "objects/stove_knob/press_position": np.asarray([press[env_id]], dtype=np.float32),
                    "objects/stove_knob/turn_position": np.asarray([angle[env_id]], dtype=np.float32),
                },
                phase_id=PHASE_SCHEMA[phases[env_id]],
                control_source_type_id=source_type,
            )
            self.phase_traces[env_id].append(phases[env_id])
            self.source_type_traces[env_id].append(source_type)
        self.video.write(self.make_mosaic(main, phases, press, angle))
        self.env.set_cartesian_rpy_target(actions[:, :6], grippers)
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        for env_id, buffer in enumerate(self.buffers):
            buffer.set_last_transition_outcome(
                reward=float(reward[env_id]),
                success=False,
                terminated=bool(terminated[env_id]),
                truncated=bool(truncated[env_id]),
            )
        new_press, new_angle, new_press_velocity, new_angle_velocity = self.mechanism_state()
        new_angle_deg = np.degrees(new_angle)
        ema_after = as_numpy(self.env.force_sensor_joint_smooth)
        self.minimum_press_m = np.minimum(self.minimum_press_m, new_press)
        self.maximum_easy_progress_deg = np.maximum(
            self.maximum_easy_progress_deg, self.easy_sign * new_angle_deg
        )
        self.maximum_hard_progress_deg = np.maximum(
            self.maximum_hard_progress_deg, self.hard_sign * new_angle_deg
        )
        for env_id in range(NUM_ENVS):
            self.angle_traces[env_id].append(float(new_angle_deg[env_id]))
            self.press_traces[env_id].append(float(new_press[env_id]))
            self.ema_torque_traces[env_id].append(abs(float(ema_after[env_id, 5])))
            if phases[env_id] in {"probe_high", "switch_direction", "turn_selected", "settle"}:
                self.maximum_turn_press_m[env_id] = max(
                    self.maximum_turn_press_m[env_id], float(new_press[env_id])
                )
        tracking = as_numpy(torch.linalg.vector_norm(
            self.env.fingertip_midpoint_pos
            - torch.as_tensor(positions, dtype=torch.float32, device=self.env.device), dim=1
        ))
        self.maximum_tracking_error_m = np.maximum(self.maximum_tracking_error_m, tracking)
        self.env.update_cameras()
        return new_press, new_angle, new_press_velocity, new_angle_velocity

    def track_pose(self, positions, yaw, gripper, frames: int, phase: str) -> None:
        positions_batch = np.asarray(positions, dtype=np.float32).reshape(NUM_ENVS, 3)
        yaws = np.asarray(yaw, dtype=np.float32)
        if yaws.ndim == 0:
            yaws = np.full(NUM_ENVS, float(yaws), dtype=np.float32)
        phases = [phase for _ in range(NUM_ENVS)]
        for _ in range(frames):
            self._record_and_step(positions_batch, yaws, gripper, phases)

    def run(self, layouts) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        knob_positions = np.asarray(
            [layout["target_knob_position_m"] for layout in layouts], dtype=np.float32
        )
        grasp = knob_positions.copy(); grasp[:, 2] += float(args.grasp_height_offset_m)
        pressed_grasp = grasp.copy(); pressed_grasp[:, 2] -= PRESS_COMMAND_DEPTH_M
        approach = knob_positions.copy(); approach[:, 2] += 0.145
        lift = knob_positions.copy(); lift[:, 2] += 0.165
        retreat = knob_positions.copy(); retreat[:, 1] += np.asarray([final_move_sign(s) for s in SEEDS]) * 0.11; retreat[:, 2] += 0.185

        self.track_pose(approach, 0.0, 1.0, SPEED_PROFILE["approach_frames"], "approach")
        self.track_pose(grasp, 0.0, 1.0, SPEED_PROFILE["descend_frames"], "descend")
        self.track_pose(grasp, 0.0, 0.0, SPEED_PROFILE["close_frames"], "close")
        self.closed_finger_positions_m = as_numpy(self.env.robot.data.joint_pos.torch[:, 7:9])
        for frame in range(SPEED_PROFILE["press_ramp_frames"]):
            alpha = (frame + 1) / SPEED_PROFILE["press_ramp_frames"]
            positions = grasp + alpha * (pressed_grasp - grasp)
            self._record_and_step(
                positions, self.command_yaw, 0.0, ["press" for _ in range(NUM_ENVS)]
            )
        self.track_pose(
            pressed_grasp, self.command_yaw, 0.0, SPEED_PROFILE["press_hold_frames"], "press"
        )

        # Only scheduling changes: the existing stages, gates and trajectories
        # run independently per clone instead of sharing all-env barriers.
        release_states = [
            ReleaseState(spec.target_final_angle_deg, int(spec.easy_direction_sign),
                         spec.final_angle_tolerance_deg, SPEED_PROFILE)
            for spec in self.specs
        ]
        before_reports = [None for _ in range(NUM_ENVS)]
        completed_states = [None for _ in range(NUM_ENVS)]

        def event(env_id, old_stage, state, angle_deg):
            if old_stage == state.stage:
                return
            record = {
                "time_s": len(self.buffers[0]) / CONTROL_RATE_HZ,
                "env_id": env_id, "seed": SEEDS[env_id],
                "from": old_stage, "to": state.stage,
                "angle_deg": float(angle_deg),
                "target_error_deg": float(angle_deg-self.specs[env_id].target_final_angle_deg),
                "tcp_xyz_m": as_numpy(self.env.fingertip_midpoint_pos[env_id]).tolist(),
                "tcp_rpy_deg": np.degrees(as_numpy(self.env.policy_state[env_id, 3:6])).tolist(),
            }
            self.stage_events.append(record)
            if state.stage in {"open_ramp", "failed", "done"}:
                print("INDEPENDENT_STAGE=" + json.dumps(record), flush=True)

        budget = sum(SPEED_PROFILE[key] for key in (
            "maximum_turn_frames", "maximum_settle_frames",
            "release_stability_max_frames", "open_ramp_frames",
            "open_hold_frames", "release_press_frames", "lift_frames", "retreat_frames"
        )) + 4
        for tick in range(budget):
            _, angles, _, _ = self.mechanism_state()
            angles_deg = np.degrees(angles)
            positions = pressed_grasp.copy()
            grippers = np.zeros(NUM_ENVS, dtype=np.float32)
            phases = []
            for env_id, (state, spec) in enumerate(zip(release_states, self.specs)):
                old_stage = state.stage
                advance_deg = state.prepare(float(angles_deg[env_id]))
                self.command_yaw[env_id] += math.radians(advance_deg)
                self.settle_complete[env_id] = state.settle_complete
                event(env_id, old_stage, state, angles_deg[env_id])
                stage = state.stage
                if stage == "turn":
                    if self.modes[env_id] == "force_explore_switch" and not self.switched[env_id]:
                        self.command_yaw[env_id] += self.hard_sign[env_id] * math.radians(SPEED_PROFILE["command_step_deg"])
                        phase = "probe_high"
                    else:
                        self.command_yaw[env_id] += self.easy_sign[env_id] * math.radians(SPEED_PROFILE["command_step_deg"])
                        phase = "turn_selected" if self.easy_sign[env_id] * angles_deg[env_id] >= 0 else "switch_direction"
                elif stage == "open_ramp":
                    progress = (state.frames+1) / SPEED_PROFILE["open_ramp_frames"]
                    grippers[env_id] = progress*progress*(3-2*progress)
                    phase = "open"
                elif stage == "open_hold":
                    grippers[env_id] = 1
                    phase = "open"
                elif stage == "release_press":
                    alpha = (state.frames+1) / SPEED_PROFILE["release_press_frames"]
                    positions[env_id] = pressed_grasp[env_id]+alpha*(grasp[env_id]-pressed_grasp[env_id])
                    grippers[env_id] = 1
                    phase = "release_press"
                elif stage == "lift":
                    positions[env_id] = lift[env_id]
                    grippers[env_id] = 1
                    phase = "lift"
                elif stage in {"retreat", "done"}:
                    positions[env_id] = retreat[env_id]
                    grippers[env_id] = 1
                    phase = "retreat"
                else:
                    phase = "settle"
                phases.append(phase)
            self.independent_stage = [state.stage for state in release_states]
            press, angles, _, _ = self._record_and_step(positions, self.command_yaw, grippers, phases)
            angles_deg = np.degrees(angles)
            ema = as_numpy(self.env.force_sensor_joint_smooth)
            for env_id, (state, spec) in enumerate(zip(release_states, self.specs)):
                if state.terminal:
                    continue
                old_stage = state.stage
                if state.stage == "turn":
                    if self.modes[env_id] == "force_explore_switch" and not self.switched[env_id]:
                        progress = self.hard_sign[env_id] * angles_deg[env_id]
                        sample = abs(float(ema[env_id, 5]))
                        self.detector_windows[env_id].append(sample)
                        filtered = float(np.median(self.detector_windows[env_id]))
                        if progress >= float(args.probe_min_angle_deg) and filtered >= float(args.switch_torque_threshold_nm):
                            self.detector_streak[env_id] += 1
                        else:
                            self.detector_streak[env_id] = 0
                        if self.detector_streak[env_id] >= 4:
                            self.switched[env_id] = True
                            self.force_switch_triggered[env_id] = True
                            self.switch_frame[env_id] = len(self.buffers[env_id])-1
                            self.switch_angle_deg[env_id] = float(angles_deg[env_id])
                            self.switch_torque_nm[env_id] = filtered
                        elif progress >= float(args.probe_max_angle_deg):
                            self.probe_failed[env_id] = True
                            self.switched[env_id] = True
                    if self.easy_sign[env_id]*angles_deg[env_id] >= spec.target_angle_magnitude_deg-SPEED_PROFILE["target_approach_margin_deg"]:
                        self.turn_complete[env_id] = True
                state.observe(float(angles_deg[env_id]), turn_complete=bool(self.turn_complete[env_id]))
                event(env_id, old_stage, state, angles_deg[env_id])
                if old_stage == "release_check" and state.stage == "open_ramp":
                    self.release_stability_frames[env_id] = state.release_frames
                    self.release_stability_range_deg[env_id] = state.release_range_deg
                    before_reports[env_id] = self.task.evaluate_final_state_for(self.env, env_id)
                    self.angle_before_open_deg[env_id] = angles_deg[env_id]
                if old_stage == "open_hold" and state.stage == "release_press":
                    self.angle_after_open_deg[env_id] = angles_deg[env_id]
                    self.open_finger_positions_m[env_id] = as_numpy(self.env.robot.data.joint_pos.torch[env_id, 7:9])
                if state.stage == "done":
                    self.completed_frames[env_id] = len(self.buffers[env_id])
                    completed_states[env_id] = CompletionSnapshot.capture(
                        report=self.task.evaluate_final_state_for(self.env, env_id),
                        control_frames=self.completed_frames[env_id],
                        press_m=press[env_id],
                        turn_angle_rad=angles[env_id],
                        fingertip_xyz_m=as_numpy(self.env.fingertip_midpoint_pos[env_id]),
                        history_metrics={
                            name: getattr(self, name)[env_id] for name in QUALITY_HISTORY_FIELDS
                        },
                    )
                if state.stage == "failed":
                    before_reports[env_id] = self.task.evaluate_final_state_for(self.env, env_id)
            if tick % 150 == 0:
                counts = {stage: sum(s.stage == stage for s in release_states) for stage in sorted({s.stage for s in release_states})}
                print("INDEPENDENT_PROGRESS=" + json.dumps({"sim_time_s":len(self.buffers[0])/CONTROL_RATE_HZ,"stages":counts}), flush=True)
            if all(state.terminal for state in release_states):
                break
        if not all(state.terminal for state in release_states):
            raise RuntimeError("independent-stage scheduling exceeded the sum of unchanged stage budgets")
        for env_id in range(NUM_ENVS):
            if before_reports[env_id] is None:
                before_reports[env_id] = self.task.evaluate_final_state_for(self.env, env_id)
        self.independent_stage = [state.stage for state in release_states]
        self.release_states = release_states

        # Successful clones retain their own completion state. Failed clones
        # retain their whole buffer and are inspected at the shared batch end.
        batch_press, batch_angle, _, _ = self.mechanism_state()
        batch_fingertips = as_numpy(self.env.fingertip_midpoint_pos)
        final_states = [
            completed_states[i] or CompletionSnapshot.capture(
                report=self.task.evaluate_final_state_for(self.env, i),
                control_frames=len(self.buffers[i]),
                press_m=batch_press[i],
                turn_angle_rad=batch_angle[i],
                fingertip_xyz_m=batch_fingertips[i],
                history_metrics={name: getattr(self, name)[i] for name in QUALITY_HISTORY_FIELDS},
            )
            for i in range(NUM_ENVS)
        ]
        final_reports = [state.report for state in final_states]
        self.total_control_frames = len(self.buffers[0])
        for i in range(NUM_ENVS):
            n = final_states[i].control_frames
            trim_episode_buffer(self.buffers[i], n)
            for traces in (self.phase_traces, self.source_type_traces, self.angle_traces, self.press_traces, self.ema_torque_traces):
                del traces[i][n:]
        final_press = np.asarray([state.press_m for state in final_states])
        final_angle = np.asarray([state.turn_angle_rad for state in final_states])
        fingertips = np.asarray([state.fingertip_xyz_m for state in final_states])
        # Diagnostics and quality gates describe only the retained transitions.
        for name in QUALITY_HISTORY_FIELDS:
            setattr(self, name, np.asarray([state.history_metrics[name] for state in final_states]))
        gripped = (
            (np.min(self.closed_finger_positions_m, axis=1) > 0.010)
            & (np.max(self.closed_finger_positions_m, axis=1) < 0.0395)
        )
        opened = np.min(self.open_finger_positions_m, axis=1) >= 0.0395
        lifted = fingertips[:, 2] >= knob_positions[:, 2] + 0.12
        retreated = np.asarray([final_move_sign(s) for s in SEEDS]) * (fingertips[:, 1] - knob_positions[:, 1]) >= 0.075
        for env_id, report in enumerate(final_reports):
            direct_valid = (
                self.modes[env_id] != "direct_expert"
                or self.maximum_hard_progress_deg[env_id] <= 2.0
            )
            explore_valid = (
                self.modes[env_id] != "force_explore_switch"
                or (
                    self.force_switch_triggered[env_id]
                    and not self.probe_failed[env_id]
                    and args.probe_min_angle_deg
                    <= abs(self.switch_angle_deg[env_id])
                    <= args.probe_max_angle_deg
                )
            )
            spec = self.specs[env_id]
            collection_angle_checks = all(
                np.isfinite(angle) and abs(float(angle)-spec.target_final_angle_deg) <= spec.final_angle_tolerance_deg
                for angle in (self.angle_before_open_deg[env_id], self.angle_after_open_deg[env_id], report["knob_angle_deg"])
            )
            success = bool(
                release_states[env_id].stage == "done"
                and release_states[env_id].release_passed
                and collection_angle_checks
                and gripped[env_id]
                and opened[env_id]
                and lifted[env_id]
                and retreated[env_id]
                and self.minimum_press_m[env_id] <= self.specs[env_id].press_engage_position_m
                and self.maximum_turn_press_m[env_id] <= self.specs[env_id].press_engage_position_m
                and self.turn_complete[env_id]
                and self.settle_complete[env_id]
                and before_reports[env_id]["success"]
                and report["success"]
                and not report["wrong_direction_detent_crossed"]
                and final_press[env_id] > -0.001
                and direct_valid
                and explore_valid
            )
            report.update({
                "success": success,
                "expert_collection_target_band_passed": bool(collection_angle_checks),
                "open_rebound_is_success_gate": False,
                "open_rebound_deg": float(self.angle_after_open_deg[env_id]-self.angle_before_open_deg[env_id]),
                "termination_reason": "independent_release_completed" if success else (release_states[env_id].failure_reason or "expert_collection_checks_failed"),
                "independent_release": True,
                "collection_quality_state_timepoint": (
                    "per_clone_completion_post_action"
                    if completed_states[env_id] is not None else "batch_end_post_action"
                ),
                "collection_quality_control_frames": final_states[env_id].control_frames,
                "trajectory_mode": self.modes[env_id],
                "minimum_press_m": float(self.minimum_press_m[env_id]),
                "maximum_turn_press_m": float(self.maximum_turn_press_m[env_id]),
                "maximum_easy_progress_deg": float(self.maximum_easy_progress_deg[env_id]),
                "maximum_hard_progress_deg": float(self.maximum_hard_progress_deg[env_id]),
                "force_switch_triggered": bool(self.force_switch_triggered[env_id]),
                "switch_frame": int(self.switch_frame[env_id]),
                "switch_angle_deg": (
                    float(self.switch_angle_deg[env_id])
                    if np.isfinite(self.switch_angle_deg[env_id]) else None
                ),
                "switch_torque_nm": (
                    float(self.switch_torque_nm[env_id])
                    if np.isfinite(self.switch_torque_nm[env_id]) else None
                ),
                "gripped": bool(gripped[env_id]),
                "opened": bool(opened[env_id]),
                "lifted": bool(lifted[env_id]),
                "retreated": bool(retreated[env_id]),
                "final_press_m": float(final_press[env_id]),
                "final_angle_deg": float(math.degrees(final_angle[env_id])),
                "report_before_release": before_reports[env_id],
            })
            if success:
                self.buffers[env_id].set_last_transition_outcome(
                    reward=1.0, success=True, terminated=False, truncated=False
                )
        return before_reports, final_reports

    def phase_ranges(self, env_id: int) -> list[dict[str, int | str]]:
        trace = self.phase_traces[env_id]
        if not trace:
            return []
        result = []
        start = 0
        for index in range(1, len(trace) + 1):
            if index == len(trace) or trace[index] != trace[start]:
                result.append({"name": trace[start], "start": start, "end": index})
                start = index
        return result

    def force_drop_metrics(self, env_id: int) -> dict[str, object]:
        spec = self.specs[env_id]
        angles = np.asarray(self.angle_traces[env_id], dtype=np.float64)
        torques = np.asarray(self.ema_torque_traces[env_id], dtype=np.float64)
        progress = spec.easy_direction_sign * angles
        peak = spec.detent_center_magnitude_deg
        pre = torques[(progress >= peak - 5.0) & (progress <= peak - 1.0)]
        post = torques[(progress >= peak + 0.5) & (progress <= peak + 2.5)]
        pre_median = float(np.median(pre)) if len(pre) else float("nan")
        post_median = float(np.median(post)) if len(post) else float("nan")
        absolute_drop = pre_median - post_median
        relative_drop = absolute_drop / pre_median if pre_median > 0.0 else float("nan")
        passed = bool(
            np.isfinite(absolute_drop)
            and (absolute_drop >= 0.010 or relative_drop >= 0.15)
        )
        return {
            "pre_median_nm": pre_median if np.isfinite(pre_median) else None,
            "post_median_nm": post_median if np.isfinite(post_median) else None,
            "absolute_drop_nm": absolute_drop if np.isfinite(absolute_drop) else None,
            "relative_drop": relative_drop if np.isfinite(relative_drop) else None,
            "passed": passed,
        }

    def details(self, env_id: int) -> dict[str, object]:
        spec = self.specs[env_id]
        angles = np.asarray(self.angle_traces[env_id], dtype=np.float64)
        torques = np.asarray(self.ema_torque_traces[env_id], dtype=np.float64)
        if self.modes[env_id] == "direct_expert":
            progress = spec.easy_direction_sign * angles
        else:
            progress = spec.hard_direction_sign * angles
        early = torques[(progress >= 8.0) & (progress <= 20.0)]
        return {
            "trajectory_mode": self.modes[env_id],
            "independent_release": True,
            "episode_control_frames": len(self.buffers[env_id]),
            "stage_events": [event for event in self.stage_events if event["env_id"] == env_id],
            "phase_ranges_control_frames": self.phase_ranges(env_id),
            "minimum_press_m": float(self.minimum_press_m[env_id]),
            "maximum_turn_press_m": float(self.maximum_turn_press_m[env_id]),
            "maximum_easy_progress_deg": float(self.maximum_easy_progress_deg[env_id]),
            "maximum_hard_progress_deg": float(self.maximum_hard_progress_deg[env_id]),
            "force_switch_triggered": bool(self.force_switch_triggered[env_id]),
            "probe_failed": bool(self.probe_failed[env_id]),
            "switch_frame": int(self.switch_frame[env_id]),
            "switch_angle_deg": (
                float(self.switch_angle_deg[env_id])
                if np.isfinite(self.switch_angle_deg[env_id]) else None
            ),
            "switch_torque_nm": (
                float(self.switch_torque_nm[env_id])
                if np.isfinite(self.switch_torque_nm[env_id]) else None
            ),
            "early_direction_torque_median_nm": (
                float(np.median(early)) if len(early) else None
            ),
            "force_drop": self.force_drop_metrics(env_id),
            "maximum_tracking_error_m": float(self.maximum_tracking_error_m[env_id]),
            "closed_finger_positions_m": self.closed_finger_positions_m[env_id].tolist(),
            "open_finger_positions_m": self.open_finger_positions_m[env_id].tolist(),
            "release_stability_frames": int(self.release_stability_frames[env_id]),
            "release_stability_range_deg": float(self.release_stability_range_deg[env_id]),
            "angle_before_open_deg": float(self.angle_before_open_deg[env_id]),
            "angle_after_open_deg": float(self.angle_after_open_deg[env_id]),
            "open_rebound_deg": float(
                self.angle_after_open_deg[env_id] - self.angle_before_open_deg[env_id]
            ),
        }


def reset_visual_qc(env: ForceBenchEnv) -> list[dict[str, object]]:
    main = rgb_batch(env.table_rgb)
    wrist = rgb_batch(env.wrist_rgb)
    reports = []
    for env_id in range(NUM_ENVS):
        metrics = {
            "main_mean_intensity": float(np.mean(main[env_id])),
            "main_dark_fraction": float(np.mean(np.max(main[env_id], axis=2) < 25)),
            "wrist_mean_intensity": float(np.mean(wrist[env_id])),
        }
        if (
            metrics["main_mean_intensity"] < 25.0
            or metrics["main_dark_fraction"] > 0.55
            or metrics["wrist_mean_intensity"] < 12.0
        ):
            raise RuntimeError(f"env_{env_id} reset visual QC failed: {metrics}")
        reports.append({"env_id": env_id, "seed": SEEDS[env_id], "metrics": metrics})
    return reports


def load_model_metadata(spec) -> dict[str, object]:
    return {
        "type": "center_continuous_early_split_gradual_rise_sudden_drop_plus_damping",
        "physics_rate_hz": PHYSICS_RATE_HZ,
        "resistance_peak_magnitude_deg": spec.detent_center_magnitude_deg,
        "resistance_peak_angle_range_deg": list(spec.detent_center_range_deg),
        "direction_split_width_deg": spec.direction_split_width_deg,
        "detent_drop_width_deg": spec.detent_drop_width_deg,
        "easy_shoulder_torque_nm": spec.easy_shoulder_torque_nm,
        "hard_shoulder_torque_nm": spec.hard_shoulder_torque_nm,
        "easy_peak_torque_nm": spec.easy_peak_torque_nm,
        "hard_peak_torque_nm": spec.hard_peak_torque_nm,
        "base_friction_torque_nm": spec.base_friction_torque_nm,
        "post_drop_resistance_nm": spec.post_drop_resistance_nm,
        "damping_nm_s_rad": spec.damping_nm_s_rad,
        "pressed_max_effort_nm": spec.turn_max_effort_nm,
        "unpressed_lock_max_effort_nm": spec.unpressed_lock_max_effort_nm,
    }


task = load_task("stove_press_turn", ASSET_DIR)
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
expert: ParallelStovePressTurnExpert | None = None
args.artifact_dir.mkdir(parents=True, exist_ok=True)
args.output_dir.mkdir(parents=True, exist_ok=True)
video_path = args.artifact_dir / "stove_press_turn_32env_mosaic.mp4"
overview_path = args.artifact_dir / "stove_press_turn_32env_initial.png"
final_overview_path = args.artifact_dir / "stove_press_turn_32env_final.png"

try:
    env.set_episode_seeds(SEEDS)
    env.reset(seed=SEEDS[0])
    env.reset_cameras()
    for _ in range(3):
        env.sim.render()
    env.update_cameras()
    visual_qc = reset_visual_qc(env)
    layouts = env.layouts
    specs = [spec_from_layout(layout) for layout in layouts]
    env.start_wrench_capture()
    expert = ParallelStovePressTurnExpert(env, task, specs, video_path)
    expert.write_overview(overview_path, "initial_reset")
    _, reports = expert.run(layouts)
    high_rate_batch = env.stop_wrench_capture()
    expert.write_overview(final_overview_path, "final")
    expert.close()

    task_metadata = dict(task.episode_metadata())
    task_metadata.pop("load_model", None)
    asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
    asset_paths.update(env.scene_asset_paths)
    asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
    writer = HDF5EpisodeWriter(args.output_dir)
    physics_wrench = np.asarray(high_rate_batch["eef_wrench_joint_raw"], dtype=np.float32)
    written = []
    failed = []
    for env_id, seed in enumerate(SEEDS):
        if seed not in WRITE_SEEDS:
            continue
        if not reports[env_id]["success"]:
            failure = {"env_id": env_id, "seed": seed,
            "direction_assignment_version": DIRECTION_VERSION,
            "final_move_sign_y": final_move_sign(seed), "report": reports[env_id]}
            failed.append(failure)
            print(
                "FORCEBENCH_STOVE_PRESS_TURN_32ENV_FAILURE="
                + json.dumps(failure, separators=(",", ":")),
                flush=True,
            )
            continue
        existing = sorted(args.output_dir.glob(f"seed_{seed:03d}_episode_*.hdf5"))
        if existing:
            raise RuntimeError(
                f"refusing to duplicate successful seed {seed}: {[str(path) for path in existing]}"
            )
        spec = specs[env_id]
        mode = trajectory_mode(seed)
        metadata = {
            "task": "stove_press_turn",
            "prompt": task.manifest.prompt,
            "success": True,
            "end_reason": "press_force_disambiguated_turn_release_retract",
            "seed": seed,
            "direction_assignment_version": DIRECTION_VERSION,
            "final_move_sign_y": final_move_sign(seed),
            "initial_layout": layouts[env_id],
            "final_success_report": reports[env_id],
            "control_rate_hz": float(CONTROL_RATE_HZ),
            "physics_rate_hz": float(PHYSICS_RATE_HZ),
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "arm_controller_diagnostics": env.arm_controller_diagnostics,
            "scene_profile": env.scene_profile_name,
            "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "control_source": "scripted_32env_press_turn_force_teacher",
            "trajectory_type": "expert" if mode == "direct_expert" else "exploration",
            "control_source_type_schema": CONTROL_SOURCE_TYPE_SCHEMA,
            "wrench_profile": env.wrench_profile,
            "wrench_storage": "raw_and_ema_control_plus_raw_physics",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [args.main_image_size, args.main_image_size, 3],
                "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
            },
            "phase_schema": PHASE_SCHEMA,
            "dataset_group": mode,
            "scripted_expert_generator_version": "stove_press_turn_32env_press7mm_v10_open_delta_diagnostic_measured_tcp",
            "collection_quality_policy": {
                "target_tolerance_deg": spec.final_angle_tolerance_deg,
                "open_rebound_is_success_gate": False,
                "open_rebound_role": "diagnostic_only_user_approved_20260904",
            },
            "press_command_depth_m": PRESS_COMMAND_DEPTH_M,
            "scripted_expert_speed_profile": SPEED_PROFILE,
            "scripted_expert_details": expert.details(env_id),
            "force_switch_logic": {
                "initial_probe_side_allocated_from_hidden_hard_direction": mode == "force_explore_switch",
                "hidden_direction_used_for_online_switch": False,
                "measured_signal": "abs(forces/control/eef_wrench_joint_ema[Tz])",
                "median_window_control_frames": 11,
                "probe_min_angle_deg": args.probe_min_angle_deg,
                "probe_max_angle_deg": args.probe_max_angle_deg,
                "threshold_nm": args.switch_torque_threshold_nm,
                "required_consecutive_frames": 4,
            },
            "randomization_audit": {
                "easy_direction": spec.easy_direction_name,
                "easy_direction_sign": spec.easy_direction_sign,
                "detent_center_magnitude_deg": spec.detent_center_magnitude_deg,
                "detent_angle_bin_3deg": min(
                    9, int((spec.detent_center_magnitude_deg - 60.0) // 3.0)
                ),
            },
            "load_model": load_model_metadata(spec),
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
            },
            "scene_roles": list(env.entity_roles),
            "asset_usd_paths": asset_paths,
            **task_metadata,
        }
        high_rate = {
            "timestamp": high_rate_batch["timestamp"][:4*len(expert.buffers[env_id])],
            "eef_wrench_joint_raw": physics_wrench[:4*len(expert.buffers[env_id]), env_id],
        }
        path = writer.write(expert.buffers[env_id], metadata, high_rate=high_rate)[0]
        validation = validate_hdf5_episode(path)
        written.append({
            "env_id": env_id,
            "seed": seed,
            "direction_assignment_version": DIRECTION_VERSION,
            "final_move_sign_y": final_move_sign(seed),
            "mode": mode,
            "path": str(path),
            "validation": validation,
            "details": expert.details(env_id),
        })

    summary = {
        "schema": "forcebench_parallel_stove_press_turn_32env_v1",
        "task": "stove_press_turn",
        "trajectory_mode_argument": args.trajectory_mode,
        "seeds": SEEDS,
        "write_seeds": sorted(WRITE_SEEDS),
        "num_envs": NUM_ENVS,
        "env_spacing_m": args.env_spacing,
        "frames_per_episode": [len(buffer) for buffer in expert.buffers],
        "total_control_frames": expert.total_control_frames,
        "independent_release": True,
        "stage_events": expert.stage_events,
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
        "FORCEBENCH_PARALLEL_STOVE_PRESS_TURN="
        + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
    if failed:
        raise RuntimeError(f"{len(failed)} requested stove_press_turn episodes failed")
except BaseException as error:
    print(
        f"FORCEBENCH_PARALLEL_STOVE_PRESS_TURN_ERROR={type(error).__name__}:{error}",
        flush=True,
    )
    raise
finally:
    if expert is not None:
        expert.close()
    env.discard_wrench_capture()
    env.close()
    simulation_app.close()
