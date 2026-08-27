#!/usr/bin/env python3
"""Collect 16 physical faucet-turn demonstrations in one Isaac process."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
parser.add_argument("--max_refine_frames", type=int, default=360)
parser.add_argument("--hold_frames", type=int, default=30)
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
PHASE_SCHEMA = {
    name: index
    for index, name in enumerate(
        (
            "approach",
            "descend",
            "close",
            "cross_and_detect",
            "reverse_to_band",
            "hold_post_peak_band",
            "release",
            "lift_clear",
            "retreat_away_from_pipe",
        )
    )
}
if len(SEEDS) != NUM_ENVS:
    parser.error("the parallel faucet collector requires exactly 16 seeds")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env spacing must be at least 5.5 m")
if args.max_refine_frames < 1 or args.hold_frames < 1:
    parser.error("frame counts must be positive")

os.environ.setdefault("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
os.environ["FORCEBENCH_TASK"] = "faucet_turn"
os.environ.setdefault("FORCEBENCH_EXPERT_BACKEND", "scripted")

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import numpy as np
import torch
import warp as wp
from isaaclab.utils import math as math_utils

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task
from forcebench.tasks.faucet_turn.spec import SPEC, spec_from_layout


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rgb_batch(value) -> np.ndarray:
    return as_numpy(value)[..., :3].astype(np.uint8)


class ParallelFaucetExpert:
    def __init__(self, env, task):
        self.env = env
        self.task = task
        self.buffers = [EpisodeBuffer() for _ in range(NUM_ENVS)]
        self.zero_action = torch.zeros((NUM_ENVS, 7), dtype=torch.float32, device=env.device)
        self.specs = [spec_from_layout(layout) for layout in env.layouts]
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
            [layout["expert_detent_sequence"]["critical_peak_angle_deg"] for layout in env.layouts],
            dtype=np.float32,
        )
        self.drop_end = np.asarray(
            [layout["expert_detent_sequence"]["drop_end_angle_deg"] for layout in env.layouts],
            dtype=np.float32,
        )
        self.overshoot_target = np.asarray(
            [layout["expert_detent_sequence"]["overshoot_target_deg"] for layout in env.layouts],
            dtype=np.float32,
        )
        self.backoff_target = np.asarray(
            [layout["expert_detent_sequence"]["backoff_target_deg"] for layout in env.layouts],
            dtype=np.float32,
        )
        self.backoff_tolerance = np.asarray(
            [layout["expert_detent_sequence"]["backoff_tolerance_deg"] for layout in env.layouts],
            dtype=np.float32,
        )
        self.stop_speed = np.asarray(
            [layout["expert_detent_sequence"]["stop_speed_rad_s"] for layout in env.layouts],
            dtype=np.float32,
        )
        self.valve_y = np.asarray(
            [layout["valve_position_m"][1] for layout in env.layouts], dtype=np.float32
        )
        self.water_outlets = np.asarray(
            [layout["water_outlet_position_m"] for layout in env.layouts], dtype=np.float32
        )
        self.min_outlet_distance = np.full(NUM_ENVS, np.inf, dtype=np.float32)
        self.phase_ranges: list[dict[str, int | str]] = []

    def valve_state(self) -> tuple[np.ndarray, np.ndarray]:
        entity = self.env.entity("faucet_valve")
        angles = as_numpy(wp.to_torch(entity.root_view.get_dof_positions()))[:, 0]
        velocities = as_numpy(wp.to_torch(entity.root_view.get_dof_velocities()))[:, 0]
        return angles.astype(np.float32), velocities.astype(np.float32)

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

    def record_and_step(self, positions, orientation_angles, gripper: float, phase: str) -> None:
        positions = np.asarray(positions, dtype=np.float32).reshape(NUM_ENVS, 3)
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
        valve_joint_position, valve_joint_velocity = self.valve_state()
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
                    "objects/faucet_valve/joint_position": np.asarray(
                        [valve_joint_position[env_id]], dtype=np.float32
                    ),
                    "objects/faucet_valve/joint_velocity": np.asarray(
                        [valve_joint_velocity[env_id]], dtype=np.float32
                    ),
                },
                phase_id=PHASE_SCHEMA[phase],
            )
        self.env.set_cartesian_rpy_target(actions[:, :6], np.full(NUM_ENVS, gripper))
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        for env_id, buffer in enumerate(self.buffers):
            buffer.set_last_transition_outcome(
                reward=float(reward[env_id]),
                success=False,
                terminated=bool(terminated[env_id]),
                truncated=bool(truncated[env_id]),
            )
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

        peak_observed = np.zeros(NUM_ENVS, dtype=bool)
        transition = np.zeros(NUM_ENVS, dtype=bool)
        transition_streak = np.zeros(NUM_ENVS, dtype=np.int32)
        overshoot_observed = np.zeros(NUM_ENVS, dtype=np.float32)
        cross_command_deg = np.zeros(NUM_ENVS, dtype=np.float32)
        start = len(self.buffers[0])
        for frame in range(1, 241):
            nominal = self.overshoot_target + 0.80
            requested = nominal * min(frame / 100.0, 1.0) + min(max(frame - 100, 0) * 0.05, 10.0)
            cross_command_deg[~transition] = requested[~transition]
            positions = self.arc_points(np.deg2rad(cross_command_deg)) + grasp_offset
            live_angles, _ = self.valve_state()
            self.record_and_step(positions, live_angles, 0.0, "cross_and_detect")
            after_angles, _ = self.valve_state()
            after_deg = np.rad2deg(after_angles)
            peak_observed |= after_deg >= self.critical
            transition_streak = np.where(
                peak_observed & (after_deg >= self.drop_end), transition_streak + 1, 0
            )
            transition |= transition_streak >= 3
            overshoot_observed = np.maximum(overshoot_observed, after_deg)
            done = transition & (after_deg >= self.overshoot_target)
            if np.all(done):
                break
        self.phase_ranges.append(
            {"name": "cross_and_detect", "start": start, "end": len(self.buffers[0])}
        )

        live_angles, _ = self.valve_state()
        contact_offset = as_numpy(self.env.fingertip_midpoint_pos) - self.arc_points(live_angles)
        backoff_command_deg = np.rad2deg(live_angles).astype(np.float32)
        windows: list[np.ndarray] = []
        stable_count = np.zeros(NUM_ENVS, dtype=np.int32)
        converged = np.zeros(NUM_ENVS, dtype=bool)
        start = len(self.buffers[0])
        for _ in range(args.max_refine_frames):
            angles, _ = self.valve_state()
            errors = self.backoff_target - np.rad2deg(angles)
            backoff_command_deg[~converged] += np.clip(
                0.18 * errors[~converged], -0.30, 0.30
            )
            positions = self.arc_points(np.deg2rad(backoff_command_deg)) + contact_offset
            self.record_and_step(positions, angles, 0.0, "reverse_to_band")
            after_angles, _ = self.valve_state()
            windows.append(np.rad2deg(after_angles))
            windows = windows[-12:]
            if len(windows) == 12:
                window = np.stack(windows)
                within = np.max(np.abs(window - self.backoff_target), axis=0) <= (
                    0.5 * self.backoff_tolerance
                )
                quiet = np.ptp(window, axis=0) <= 0.08
                stable_count = np.where(within & quiet, stable_count + 1, 0)
                converged |= stable_count >= 12
            if np.all(converged):
                break
        self.phase_ranges.append(
            {"name": "reverse_to_band", "start": start, "end": len(self.buffers[0])}
        )

        start = len(self.buffers[0])
        for _ in range(args.hold_frames):
            angles, _ = self.valve_state()
            errors = self.backoff_target - np.rad2deg(angles)
            backoff_command_deg += np.clip(0.12 * errors, -0.10, 0.10)
            positions = self.arc_points(np.deg2rad(backoff_command_deg)) + contact_offset
            self.record_and_step(positions, angles, 0.0, "hold_post_peak_band")
            windows.append(np.rad2deg(self.valve_state()[0]))
            windows = windows[-12:]
        self.phase_ranges.append(
            {"name": "hold_post_peak_band", "start": start, "end": len(self.buffers[0])}
        )

        checkpoint_angles, checkpoint_velocities = self.valve_state()
        checkpoint_deg = np.rad2deg(checkpoint_angles)
        backoff_error = np.abs(checkpoint_deg - self.backoff_target)
        window = np.stack(windows) if windows else checkpoint_deg[None, :]
        window_range = np.ptp(window, axis=0)
        stopped = (window_range <= 0.08) & (np.abs(checkpoint_velocities) <= self.stop_speed)
        checkpoint_success = (backoff_error <= self.backoff_tolerance) & stopped

        release_positions = self.arc_points(np.deg2rad(self.backoff_target + 2.5)) + contact_offset
        self.track(release_positions, checkpoint_angles, 1.0, 24, "release")
        release_fingers = as_numpy(self.env.robot.data.joint_pos.torch)[:, 7:9].copy()
        lift = release_positions.copy()
        lift[:, 2] = grasp[:, 2] + 0.130
        self.track(lift, checkpoint_angles, 1.0, 60, "lift_clear")
        retreat = lift.copy()
        retreat[:, 1] = self.valve_y - 0.085
        retreat[:, 2] = grasp[:, 2] + 0.155
        self.track(retreat, checkpoint_angles, 1.0, 40, "retreat_away_from_pipe")

        final_angles, final_velocities = self.valve_state()
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
                and peak_observed[env_id]
                and transition[env_id]
                and overshoot_observed[env_id] >= self.overshoot_target[env_id]
                and checkpoint_success[env_id]
                and np.min(release_fingers[env_id]) >= 0.0395
                and fingertip[env_id, 2] >= grasp[env_id, 2] + 0.09
                and fingertip[env_id, 1] <= self.valve_y[env_id] - 0.055
                and report["success"]
                and root_error <= 1.0e-5
                and self.min_outlet_distance[env_id] >= 0.10
            )
            report["success"] = custom_success
            report["parallel_expert_checks"] = {
                "gripped": bool(np.max(closed_fingers[env_id]) < 0.0395),
                "peak_observed": bool(peak_observed[env_id]),
                "physical_transition_detected": bool(transition[env_id]),
                "overshoot_angle_deg": float(overshoot_observed[env_id]),
                "backoff_checkpoint_success": bool(checkpoint_success[env_id]),
                "backoff_error_deg": float(backoff_error[env_id]),
                "backoff_window_range_deg": float(window_range[env_id]),
                "released": bool(np.min(release_fingers[env_id]) >= 0.0395),
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
                    "overshoot_target_deg": float(self.overshoot_target[env_id]),
                    "backoff_target_deg": float(self.backoff_target[env_id]),
                    "final_valve_angle_deg": float(np.rad2deg(final_angles[env_id])),
                    "final_valve_velocity_rad_s": float(final_velocities[env_id]),
                }
            )
        return results


task = load_task("faucet_turn", ASSET_DIR)
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
cfg.num_rerenders_on_reset = 0
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
            failure = {"env_id": env_id, "seed": seed, "report": report}
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
                    "detent_center_deg": episode_spec.detent_center_deg,
                    "detent_peak_torque_nm": episode_spec.detent_peak_torque_nm,
                }
            )
        details = {key: value for key, value in results[env_id].items() if key != "report"}
        metadata = {
            "task": "faucet_turn",
            "prompt": task.manifest.prompt,
            "success": True,
            "end_reason": "parallel_faucet_post_peak_final_check",
            "seed": seed,
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
            "control_source": "parallel_physical_valve_joint_feedback_arc_controller",
            "success_check_timing": "after_detent_cross_backoff_release_and_retract",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [args.main_image_size, args.main_image_size, 3],
                "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
            },
            "phase_schema": PHASE_SCHEMA,
            "scripted_expert_details": details,
            "expert_transition_signal": "valve_joint_angle_rad",
            "expert_uses_analytic_resistance_for_transition": False,
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
            "scripted_expert_generator_version": "parallel_faucet_physical_detent_v1_physics60",
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
        "schema": "forcebench_parallel_faucet_16env_v1",
        "task": "faucet_turn",
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
        "FORCEBENCH_PARALLEL_FAUCET=" + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
finally:
    if capture_active:
        env.discard_wrench_capture()
    env.close()
    simulation_app.close()
