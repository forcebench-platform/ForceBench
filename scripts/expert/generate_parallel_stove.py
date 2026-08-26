#!/usr/bin/env python3
"""Collect 32 stove_turn_on episodes in one Isaac process with cloned workspaces."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--artifact_dir", type=Path, required=True)
parser.add_argument("--seeds", required=True, help="exactly 32 distinct comma-separated seeds")
parser.add_argument(
    "--write_seeds",
    required=True,
    help="subset of --seeds missing from the formal dataset and eligible to be written",
)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--video_tile_size", type=int, default=192)
parser.add_argument("--grasp_height_offset_m", type=float, default=0.002)
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
    parser.error("the default stove collector requires exactly 32 cloned workspaces")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env_spacing must be at least 5.5 m")
if args.grasp_height_offset_m <= 0.0:
    parser.error("grasp_height_offset_m must be positive")

os.environ.setdefault("CORIBENCH_EXPERT_COLLECTION_CONTRACT", "coribench_expert_collection")
os.environ["CORIBENCH_TASK"] = "stove_turn_on"
os.environ.setdefault("CORIBENCH_EXPERT_BACKEND", "scripted")

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import cv2
import numpy as np
import torch
import warp as wp

from coribench.env import CoRiBenchEnv
from coribench.env_cfg import ASSET_DIR, CoRiBenchEnvCfg, configure_runtime
from coribench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from coribench.registry import load_task
from coribench.tasks.stove_turn_on.spec import (
    SPEC,
    detent_effort_nm,
    detent_resistance_nm,
    evaluate_angle,
    spec_from_layout,
)


PHASES = ("approach", "descend", "close", "rotate", "hold_on", "release", "lift", "retreat")
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASES)}
SPEED_PROFILE = {
    "approach_frames": 75,
    "descend_frames": 55,
    "close_frames": 30,
    "maximum_forward_frames": 420,
    "maximum_backoff_frames": 180,
    "hold_on_frames": 20,
    "maximum_release_frames": 30,
    "lift_frames": 75,
    "retreat_frames": 45,
    "command_step_deg": 0.32,
    "backoff_tolerance_deg": 0.50,
    "backoff_stable_frames": 5,
}


def as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def rgb_batch(value) -> np.ndarray:
    return as_numpy(value)[..., :3].astype(np.uint8)


class ParallelStoveExpert:
    def __init__(self, env: CoRiBenchEnv, specs, video_path: Path):
        self.env = env
        self.specs = tuple(specs)
        self.buffers = [EpisodeBuffer() for _ in range(NUM_ENVS)]
        self.zero_action = torch.zeros((NUM_ENVS, 7), dtype=torch.float32, device=env.device)
        self.phase_ranges: list[dict[str, int | str]] = []
        self.angle_traces = [[] for _ in range(NUM_ENVS)]
        self.velocity_traces = [[] for _ in range(NUM_ENVS)]
        self.effort_traces = [[] for _ in range(NUM_ENVS)]
        self.maximum_tracking_error_m = np.zeros(NUM_ENVS, dtype=np.float64)
        self.maximum_angle_deg = np.zeros(NUM_ENVS, dtype=np.float64)
        self.crossing_frame = np.full(NUM_ENVS, -1, dtype=np.int64)
        self.transition_frame = np.full(NUM_ENVS, -1, dtype=np.int64)
        self.transition_angle_deg = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.transition_command_deg = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.backoff_frame = np.full(NUM_ENVS, -1, dtype=np.int64)
        self.backoff_final_angle_deg = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.transition_crossed = np.zeros(NUM_ENVS, dtype=bool)
        self.backoff_complete = np.zeros(NUM_ENVS, dtype=bool)
        self.release_frames = 0
        self.closed_finger_positions_m = np.zeros((NUM_ENVS, 2), dtype=np.float64)
        self.release_finger_positions_m = np.zeros((NUM_ENVS, 2), dtype=np.float64)
        size = int(args.video_tile_size)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_RATE_HZ, (8 * size, 4 * size)
        )
        if not self.video.isOpened():
            raise RuntimeError(f"failed to create video: {video_path}")

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    def knob_state(self) -> tuple[np.ndarray, np.ndarray]:
        entity = self.env.entity("stove_knob")
        joint_id = entity.joint_names.index(SPEC.knob_joint_name)
        positions = wp.to_torch(entity.root_view.get_dof_positions())
        velocities = wp.to_torch(entity.root_view.get_dof_velocities())
        return as_numpy(positions[:, joint_id]), as_numpy(velocities[:, joint_id])

    def privileged_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position, quaternion = self.env.entity_local_pose("stove_knob")
        entity = self.env.entity("stove_knob")
        pose = as_numpy(torch.cat((position, quaternion), dim=-1))
        return pose, as_numpy(entity.data.joint_pos.torch), as_numpy(entity.data.joint_vel.torch)

    def make_mosaic(self, images: np.ndarray, phase: str, angles_deg: np.ndarray) -> np.ndarray:
        size = int(args.video_tile_size)
        canvas = np.zeros((4 * size, 8 * size, 3), dtype=np.uint8)
        wrench = as_numpy(self.env.force_sensor_joint_raw)
        for env_id, image in enumerate(images):
            tile = cv2.resize(image[..., ::-1], (size, size), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (size, 39), (18, 18, 18), -1)
            mark = "REC" if SEEDS[env_id] in WRITE_SEEDS else "FILL"
            cv2.putText(
                tile,
                f"e{env_id:02d} s{SEEDS[env_id]:03d} {mark} A{angles_deg[env_id]:05.1f}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (245, 245, 245), 1, cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"{phase[:12]} Tz {wrench[env_id, 5]:+.3f}Nm",
                (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (225, 225, 225), 1, cv2.LINE_AA,
            )
            row, column = divmod(env_id, 8)
            canvas[row * size:(row + 1) * size, column * size:(column + 1) * size] = tile
        return canvas

    def write_overview(self, path: Path, phase: str) -> None:
        angles, _ = self.knob_state()
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(path),
            self.make_mosaic(rgb_batch(self.env.table_rgb), phase, np.degrees(angles)),
        )

    def _record_and_step(
        self, positions_m: np.ndarray, yaw_rad: np.ndarray, gripper, phase: str
    ) -> tuple[np.ndarray, np.ndarray]:
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
        tcp = as_numpy(
            torch.cat((self.env.fingertip_midpoint_pos, self.env.fingertip_midpoint_quat), dim=-1)
        )
        knob_pose, knob_joint_position, knob_joint_velocity = self.privileged_batch()
        angle, angular_velocity = self.knob_state()
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
                    "objects/stove_knob/pose": knob_pose[env_id],
                    "objects/stove_knob/joint_position": knob_joint_position[env_id],
                    "objects/stove_knob/joint_velocity": knob_joint_velocity[env_id],
                },
                phase_id=PHASE_SCHEMA[phase],
            )
        self.video.write(self.make_mosaic(main, phase, np.degrees(angle)))
        self.env.set_cartesian_rpy_target(actions[:, :6], grippers)
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        for env_id, buffer in enumerate(self.buffers):
            buffer.set_last_transition_outcome(
                reward=float(reward[env_id]), success=False,
                terminated=bool(terminated[env_id]), truncated=bool(truncated[env_id]),
            )
        new_angle, new_velocity = self.knob_state()
        new_angle_deg = np.degrees(new_angle)
        for env_id, spec in enumerate(self.specs):
            effort = detent_effort_nm(float(new_angle[env_id]), float(new_velocity[env_id]), spec)
            self.angle_traces[env_id].append(float(new_angle_deg[env_id]))
            self.velocity_traces[env_id].append(float(new_velocity[env_id]))
            self.effort_traces[env_id].append(float(effort))
        tracking = as_numpy(torch.linalg.vector_norm(
            self.env.fingertip_midpoint_pos
            - torch.as_tensor(positions, dtype=torch.float32, device=self.env.device), dim=1
        ))
        self.maximum_tracking_error_m = np.maximum(self.maximum_tracking_error_m, tracking)
        self.env.update_cameras()
        return new_angle, new_velocity

    def track_pose(self, positions, yaw, gripper, frames: int, phase: str) -> None:
        positions_batch = np.asarray(positions, dtype=np.float32).reshape(NUM_ENVS, 3)
        yaws = np.asarray(yaw, dtype=np.float32)
        if yaws.ndim == 0:
            yaws = np.full(NUM_ENVS, float(yaws), dtype=np.float32)
        start = len(self.buffers[0])
        for _ in range(frames):
            self._record_and_step(positions_batch, yaws, gripper, phase)
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})

    def report(self, angles: np.ndarray, velocities: np.ndarray) -> list[dict[str, object]]:
        reports = []
        for env_id, spec in enumerate(self.specs):
            report = evaluate_angle(float(angles[env_id]), spec)
            report["ignition_detent_crossed"] = bool(self.transition_crossed[env_id])
            report["success"] = bool(
                self.transition_crossed[env_id]
                and report["burner_lit"]
                and report["within_final_angle_tolerance"]
            )
            report["knob_angular_velocity_rad_s"] = float(velocities[env_id])
            report["load_effort_nm"] = float(
                detent_effort_nm(float(angles[env_id]), float(velocities[env_id]), spec)
            )
            report["load_resistance_nm"] = float(
                detent_resistance_nm(float(angles[env_id]), spec)
            )
            report["flame_visible"] = bool(report["burner_lit"])
            reports.append(report)
        return reports

    def run(self, layouts) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        knob_positions = np.asarray([layout["target_knob_position_m"] for layout in layouts], dtype=np.float32)
        grasp = knob_positions.copy(); grasp[:, 2] += float(args.grasp_height_offset_m)
        approach = knob_positions.copy(); approach[:, 2] += 0.145
        lift = knob_positions.copy(); lift[:, 2] += 0.165
        retreat = knob_positions.copy(); retreat[:, 1] -= 0.11; retreat[:, 2] += 0.185
        self.track_pose(approach, 0.0, 1.0, SPEED_PROFILE["approach_frames"], "approach")
        self.track_pose(grasp, 0.0, 1.0, SPEED_PROFILE["descend_frames"], "descend")
        self.track_pose(grasp, 0.0, 0.0, SPEED_PROFILE["close_frames"], "close")
        self.closed_finger_positions_m = as_numpy(self.env.robot.data.joint_pos.torch[:, 7:9])

        angles, _ = self.knob_state()
        self.maximum_angle_deg = np.degrees(angles)
        command_yaw = np.zeros(NUM_ENVS, dtype=np.float64)
        targets = np.asarray([spec.success_angle_deg for spec in self.specs], dtype=np.float64)
        overshoot = np.asarray(
            [min(spec.open_limit_deg - 5.0, spec.success_angle_deg + 4.0) for spec in self.specs],
            dtype=np.float64,
        )
        maximum_command = np.asarray([spec.open_limit_deg - 5.0 for spec in self.specs])
        start = len(self.buffers[0])
        for _ in range(SPEED_PROFILE["maximum_forward_frames"]):
            active = ~self.transition_crossed
            command_deg = np.degrees(command_yaw)
            command_deg[active] = np.minimum(
                maximum_command[active], command_deg[active] + SPEED_PROFILE["command_step_deg"]
            )
            command_yaw = np.radians(command_deg)
            angles, _ = self._record_and_step(grasp, command_yaw, 0.0, "rotate")
            angle_deg = np.degrees(angles)
            self.maximum_angle_deg = np.maximum(self.maximum_angle_deg, angle_deg)
            frame = len(self.buffers[0]) - 1
            just_crossed = (self.crossing_frame < 0) & (angle_deg >= targets)
            self.crossing_frame[just_crossed] = frame
            transitioned = (~self.transition_crossed) & (angle_deg >= overshoot)
            self.transition_crossed[transitioned] = True
            self.transition_frame[transitioned] = frame
            self.transition_angle_deg[transitioned] = angle_deg[transitioned]
            self.transition_command_deg[transitioned] = command_deg[transitioned]
            exhausted = (~self.transition_crossed) & (command_deg >= maximum_command)
            if np.all(self.transition_crossed | exhausted):
                break
        self.phase_ranges.append({"name": "rotate", "start": start, "end": len(self.buffers[0])})

        stable = np.zeros(NUM_ENVS, dtype=np.int64)
        start = len(self.buffers[0])
        for _ in range(SPEED_PROFILE["maximum_backoff_frames"]):
            angles, _ = self.knob_state()
            angle_deg = np.degrees(angles)
            error = targets - angle_deg
            within = np.abs(error) <= SPEED_PROFILE["backoff_tolerance_deg"]
            stable = np.where(within, stable + 1, 0)
            completed = (~self.backoff_complete) & (
                stable >= SPEED_PROFILE["backoff_stable_frames"]
            )
            self.backoff_complete[completed] = True
            self.backoff_frame[completed] = len(self.buffers[0]) - 1
            self.backoff_final_angle_deg[completed] = angle_deg[completed]
            if np.all(self.backoff_complete):
                break
            active = ~self.backoff_complete
            command_deg = np.degrees(command_yaw)
            command_deg[active] += np.clip(
                error[active], -SPEED_PROFILE["command_step_deg"], SPEED_PROFILE["command_step_deg"]
            )
            command_deg = np.clip(command_deg, 0.0, maximum_command)
            command_yaw = np.radians(command_deg)
            angles, _ = self._record_and_step(grasp, command_yaw, 0.0, "rotate")
            angle_deg = np.degrees(angles)
            self.maximum_angle_deg = np.maximum(self.maximum_angle_deg, angle_deg)
            just_crossed = (self.crossing_frame < 0) & (angle_deg >= targets)
            self.crossing_frame[just_crossed] = len(self.buffers[0]) - 1
        self.phase_ranges.append({"name": "rotate", "start": start, "end": len(self.buffers[0])})

        self.track_pose(grasp, command_yaw, 0.0, SPEED_PROFILE["hold_on_frames"], "hold_on")
        before_angles, before_velocities = self.knob_state()
        before_reports = self.report(before_angles, before_velocities)

        for frame in range(1, SPEED_PROFILE["maximum_release_frames"] + 1):
            self._record_and_step(grasp, command_yaw, 1.0, "release")
            self.release_frames = frame
            self.release_finger_positions_m = as_numpy(
                self.env.robot.data.joint_pos.torch[:, 7:9]
            )
            if frame >= 8 and np.all(np.min(self.release_finger_positions_m, axis=1) >= 0.0395):
                break
        self.track_pose(lift, command_yaw, 1.0, SPEED_PROFILE["lift_frames"], "lift")
        self.track_pose(retreat, 0.0, 1.0, SPEED_PROFILE["retreat_frames"], "retreat")

        final_angles, final_velocities = self.knob_state()
        final_reports = self.report(final_angles, final_velocities)
        gripper = as_numpy(self.env.gripper_open_fraction_tensor)
        fingertips = as_numpy(self.env.fingertip_midpoint_pos)
        root_position, _ = self.env.entity_local_pose("stove_knob")
        position_error = np.linalg.norm(as_numpy(root_position) - knob_positions, axis=1)
        gripped = (
            (np.min(self.closed_finger_positions_m, axis=1) > 0.010)
            & (np.max(self.closed_finger_positions_m, axis=1) < 0.0395)
        )
        released = gripper >= 0.90
        lifted_clear = fingertips[:, 2] >= knob_positions[:, 2] + 0.12
        retreated = fingertips[:, 1] <= knob_positions[:, 1] - 0.075
        release_open = np.min(self.release_finger_positions_m, axis=1) >= 0.0395
        for env_id, report in enumerate(final_reports):
            success = bool(
                gripped[env_id]
                and self.transition_crossed[env_id]
                and self.transition_frame[env_id] >= 0
                and self.backoff_complete[env_id]
                and self.backoff_frame[env_id] >= 0
                and self.crossing_frame[env_id] >= 0
                and before_reports[env_id]["success"]
                and report["success"]
                and released[env_id]
                and lifted_clear[env_id]
                and retreated[env_id]
                and release_open[env_id]
                and position_error[env_id] <= 1.0e-5
            )
            report.update(
                {
                    "success": success,
                    "gripped": bool(gripped[env_id]),
                    "released": bool(released[env_id]),
                    "lifted_clear": bool(lifted_clear[env_id]),
                    "retreated": bool(retreated[env_id]),
                    "knob_root_position_error_m": float(position_error[env_id]),
                    "report_before_release": before_reports[env_id],
                }
            )
            if success:
                self.buffers[env_id].set_last_transition_outcome(
                    reward=1.0, success=True, terminated=False, truncated=False
                )
        return before_reports, final_reports

    def details(self, env_id: int) -> dict[str, object]:
        values = self.effort_traces[env_id]
        return {
            "phase_ranges_control_frames": list(self.phase_ranges),
            "maximum_angle_deg": float(self.maximum_angle_deg[env_id]),
            "crossing_frame": int(self.crossing_frame[env_id]),
            "privileged_transition_frame": int(self.transition_frame[env_id]),
            "privileged_transition_angle_deg": float(self.transition_angle_deg[env_id]),
            "privileged_transition_command_deg": float(self.transition_command_deg[env_id]),
            "privileged_backoff_frame": int(self.backoff_frame[env_id]),
            "privileged_backoff_final_angle_deg": float(self.backoff_final_angle_deg[env_id]),
            "closed_finger_positions_m": self.closed_finger_positions_m[env_id].tolist(),
            "release_finger_positions_m": self.release_finger_positions_m[env_id].tolist(),
            "release_frames": self.release_frames,
            "minimum_load_effort_nm": float(min(values)),
            "maximum_load_effort_nm": float(max(values)),
            "maximum_tracking_error_m": float(self.maximum_tracking_error_m[env_id]),
        }


def reset_visual_qc(env: CoRiBenchEnv) -> list[dict[str, object]]:
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
        "type": "faucet_style_asymmetric_resistance_plus_damping",
        "physics_rate_hz": PHYSICS_RATE_HZ,
        "resistance_peak_angle_deg": spec.detent_center_deg,
        "resistance_peak_angle_range_deg": list(spec.detent_center_range_deg),
        "detent_rise_width_deg": spec.detent_rise_width_deg,
        "detent_drop_width_deg": spec.detent_drop_width_deg,
        "post_drop_low_width_deg": spec.post_drop_low_width_deg,
        "post_drop_rise_width_deg": spec.post_drop_rise_width_deg,
        "post_drop_resistance_nm": spec.post_drop_resistance_nm,
        "detent_peak_torque_nm": spec.detent_peak_torque_nm,
        "final_angle_tolerance_deg": spec.final_angle_tolerance_deg,
        "damping_nm_s_rad": spec.damping_nm_s_rad,
        "base_friction_torque_nm": spec.base_friction_torque_nm,
        "max_effort_nm": spec.max_effort_nm,
        "contact_friction": spec.knob_friction,
        "friction_combine_mode": spec.knob_friction_combine_mode,
    }


task = load_task("stove_turn_on", ASSET_DIR)
cfg = CoRiBenchEnvCfg()
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
env = CoRiBenchEnv(
    cfg=cfg,
    task_runtime=task,
    main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size,
    enable_cameras=True,
)
expert: ParallelStoveExpert | None = None
args.artifact_dir.mkdir(parents=True, exist_ok=True)
video_path = args.artifact_dir / "stove_turn_on_32env_mosaic.mp4"
overview_path = args.artifact_dir / "stove_turn_on_32env_initial.png"
final_overview_path = args.artifact_dir / "stove_turn_on_32env_final.png"

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
    expert = ParallelStoveExpert(env, specs, video_path)
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
            failure = {"env_id": env_id, "seed": seed, "report": reports[env_id]}
            failed.append(failure)
            print(
                "CORIBENCH_PARALLEL_STOVE_FINAL_FAILURE="
                + json.dumps(failure, separators=(",", ":")), flush=True,
            )
            continue
        existing = sorted(args.output_dir.glob(f"seed_{seed:03d}_episode_*.hdf5"))
        if existing:
            raise RuntimeError(
                f"refusing to duplicate successful seed {seed}: {[str(path) for path in existing]}"
            )
        spec = specs[env_id]
        metadata = {
            "task": "stove_turn_on",
            "prompt": task.manifest.prompt,
            "success": True,
            "end_reason": "scripted_parallel_stove_knob_turn_release_retract",
            "seed": seed,
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
            "control_source": "scripted_32env_single_process_stove_privileged_angle_oracle",
            "wrench_profile": env.wrench_profile,
            "wrench_storage": "raw_and_ema",
            "primary_wrist_wrench_dataset": "forces/control/eef_wrench_joint_raw",
            "recorded_wrist_wrench_datasets": [
                "forces/control/eef_wrench_joint_raw", "forces/control/eef_wrench_joint_ema"
            ],
            "success_check_timing": "after_privileged_detent_cross_backoff_release_and_retract",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [args.main_image_size, args.main_image_size, 3],
                "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
            },
            "phase_schema": PHASE_SCHEMA,
            "scripted_expert_generator_version": "stove_turn_on_32env_tiled_v2_physics60",
            "scripted_expert_speed_profile": SPEED_PROFILE,
            "scripted_expert_details": expert.details(env_id),
            "expert_uses_privileged_knob_angle_for_control": True,
            "expert_transition_signal": "privileged_stove_knob_joint_angle",
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
            "timestamp": high_rate_batch["timestamp"],
            "eef_wrench_joint_raw": physics_wrench[:, env_id],
        }
        path = writer.write(expert.buffers[env_id], metadata, high_rate=high_rate)[0]
        validation = validate_hdf5_episode(path)
        written.append({"env_id": env_id, "seed": seed, "path": str(path), "validation": validation})

    summary = {
        "schema": "coribench_parallel_stove_32env_v1",
        "task": "stove_turn_on",
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
    print("CORIBENCH_PARALLEL_STOVE=" + json.dumps(summary, separators=(",", ":")), flush=True)
except BaseException as error:
    print(f"CORIBENCH_PARALLEL_STOVE_ERROR={type(error).__name__}:{error}", flush=True)
    raise
finally:
    if expert is not None:
        expert.close()
    env.discard_wrench_capture()
    env.close()
    simulation_app.close()
