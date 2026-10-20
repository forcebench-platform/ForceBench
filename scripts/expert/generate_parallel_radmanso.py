#!/usr/bin/env python3
"""Collect 32 RADMANSO resistance-stop episodes in one Isaac process."""

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
parser.add_argument("--seeds", required=True)
parser.add_argument("--write_seeds", required=True)
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
NUM_ENVS = 32
PHYSICS_RATE_HZ = 60
CONTROL_RATE_HZ = 15
SOLVER_POSITION_ITERATIONS = 64
if len(SEEDS) != NUM_ENVS:
    parser.error("the default RADMANSO collector requires exactly 32 cloned workspaces")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env_spacing must be at least 5.5 m")

os.environ.setdefault("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
os.environ["FORCEBENCH_TASK"] = "radmanso_drawer_sort_table"
os.environ.setdefault("FORCEBENCH_EXPERT_BACKEND", "scripted")

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import cv2
import numpy as np
import torch

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task
from forcebench.tasks.radmanso_drawer_sort_table.spec import BOTTOM_JOINT, DRAWER_JOINTS, SPEC


GRASP_RPY = np.asarray((math.pi / 2.0, 0.0, 0.0), dtype=np.float32)
APPROACH_STANDOFF_M = 0.100
PULL_RATE_M = 0.00045
MAX_PULL_M = 0.120
STABLE_STEPS = 6
# Formal RADMANSO demonstrations use only continuous pull profiles.  Assigning
# by seed parity keeps the 0--199 formal set deterministic and exactly balanced.
PROFILE_ORDER = ("perfect", "uniform")
PHASES = (
    "observe", "approach", "descend", "grasp", "bias", "pull_free",
    "intentional_slowdown", "intentional_pause", "resume_pull",
    "resistance_rise", "stop_and_hold",
)
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASES)}


def as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def rgb_batch(value) -> np.ndarray:
    return as_numpy(value)[..., :3].astype(np.uint8)


class ParallelRadmansoExpert:
    def __init__(self, env: ForceBenchEnv, task, layouts, video_path: Path):
        self.env = env
        self.task = task
        self.layouts = tuple(dict(layout) for layout in layouts)
        self.handles = np.asarray([layout["handle_center_m"] for layout in layouts], dtype=np.float32)
        self.axes = np.asarray([layout["pull_direction_world"] for layout in layouts], dtype=np.float32)
        self.profiles = tuple(PROFILE_ORDER[seed % len(PROFILE_ORDER)] for seed in SEEDS)
        self.buffers = [EpisodeBuffer() for _ in range(NUM_ENVS)]
        self.zero_action = torch.zeros((NUM_ENVS, 7), dtype=torch.float32, device=env.device)
        self.phase_ranges: list[dict[str, object]] = []
        self.trigger_frames = np.full(NUM_ENVS, -1, dtype=np.int64)
        self.trigger_opening_m = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.trigger_resistance_n = np.full(NUM_ENVS, np.nan, dtype=np.float64)
        self.commanded_pull_m = np.zeros(NUM_ENVS, dtype=np.float64)
        self.maximum_tracking_error_m = np.zeros(NUM_ENVS, dtype=np.float64)
        self.closed_finger_positions_m = np.zeros((NUM_ENVS, 2), dtype=np.float64)
        self.opening_traces = [[] for _ in range(NUM_ENVS)]
        self.resistance_traces = [[] for _ in range(NUM_ENVS)]
        size = int(args.video_tile_size)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            CONTROL_RATE_HZ, (8 * size, 4 * size),
        )
        if not self.video.isOpened():
            raise RuntimeError(f"failed to create video: {video_path}")

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    def openings(self) -> np.ndarray:
        return np.asarray(
            [self.task.drawer_opening_world_m(self.env, env_id) for env_id in range(NUM_ENVS)],
            dtype=np.float64,
        )

    def resistances(self) -> np.ndarray:
        return np.asarray(
            [self.task.designed_resistance_n_for(env_id) for env_id in range(NUM_ENVS)],
            dtype=np.float64,
        )

    def make_mosaic(self, images: np.ndarray, phase_names) -> np.ndarray:
        size = int(args.video_tile_size)
        canvas = np.zeros((4 * size, 8 * size, 3), dtype=np.uint8)
        openings = self.openings()
        resistances = self.resistances()
        for env_id, image in enumerate(images):
            tile = cv2.resize(image[..., ::-1], (size, size), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (size, 40), (18, 18, 18), -1)
            mark = "REC" if SEEDS[env_id] in WRITE_SEEDS else "FILL"
            cv2.putText(
                tile,
                f"e{env_id:02d} s{SEEDS[env_id]:03d} {mark} {self.profiles[env_id][:4]}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (245, 245, 245), 1, cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"{str(phase_names[env_id])[:9]} {1000*openings[env_id]:4.1f}mm {resistances[env_id]:.2f}N",
                (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (225, 225, 225), 1, cv2.LINE_AA,
            )
            row, column = divmod(env_id, 8)
            canvas[row * size:(row + 1) * size, column * size:(column + 1) * size] = tile
        return canvas

    def write_overview(self, path: Path, phase: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.make_mosaic(rgb_batch(self.env.table_rgb), [phase] * NUM_ENVS))

    def privileged_batch(self) -> dict[str, np.ndarray]:
        position, quaternion = self.env.entity_local_pose("cabinet")
        pose = as_numpy(torch.cat((position, quaternion), dim=-1))
        joints = as_numpy(self.env.entity("cabinet").data.joint_pos.torch)
        state = np.asarray(
            [
                [
                    self.task.drawer_opening_world_m(self.env, env_id),
                    self.task.designed_resistance_n_for(env_id),
                    self.task.axial_pull_n(self.env, env_id),
                    1.0 if self.task.latch_broken_for(env_id) else 0.0,
                    float(self.layouts[env_id]["success_open_m"]),
                ]
                for env_id in range(NUM_ENVS)
            ], dtype=np.float32,
        )
        return {
            "objects/cabinet/pose": pose,
            "mechanism/joint_position": joints,
            "mechanism/resistance_state": state,
        }

    def step(self, positions, grippers, phase_names) -> None:
        positions = np.asarray(positions, dtype=np.float32).reshape(NUM_ENVS, 3)
        grippers = np.asarray(grippers, dtype=np.float32).reshape(NUM_ENVS)
        actions = np.concatenate(
            (
                positions,
                np.repeat(GRASP_RPY[None, :], NUM_ENVS, axis=0),
                (2.0 * grippers - 1.0)[:, None],
            ), axis=1,
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
        privileged = self.privileged_batch()
        timestamp = (len(self.buffers[0]) + 1) * self.env.step_dt
        for env_id, buffer in enumerate(self.buffers):
            buffer.append(
                state=states[env_id], action=actions[env_id],
                wrench_raw=wrench_raw[env_id], wrench_ema=wrench_ema[env_id],
                timestamp=timestamp, image=main[env_id], wrist_image=wrist[env_id],
                joint_position=joints[env_id], joint_velocity=velocities[env_id],
                tcp_pose=tcp[env_id],
                privileged={key: value[env_id] for key, value in privileged.items()},
                phase_id=PHASE_SCHEMA[str(phase_names[env_id])],
            )
        self.video.write(self.make_mosaic(main, phase_names))
        self.env.set_cartesian_rpy_target(actions[:, :6], grippers)
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        for env_id, buffer in enumerate(self.buffers):
            buffer.set_last_transition_outcome(
                reward=float(reward[env_id]), success=False,
                terminated=bool(terminated[env_id]), truncated=bool(truncated[env_id]),
            )
        tracking = as_numpy(torch.linalg.vector_norm(
            self.env.fingertip_midpoint_pos
            - torch.as_tensor(positions, dtype=torch.float32, device=self.env.device), dim=1,
        ))
        self.maximum_tracking_error_m = np.maximum(self.maximum_tracking_error_m, tracking)
        openings = self.openings()
        resistances = self.resistances()
        for env_id in range(NUM_ENVS):
            self.opening_traces[env_id].append(float(openings[env_id]))
            self.resistance_traces[env_id].append(float(resistances[env_id]))
        self.env.update_cameras()

    def hold(self, target, gripper: float, phase: str, frames: int) -> None:
        positions = np.asarray(target, dtype=np.float32).reshape(NUM_ENVS, 3)
        grippers = np.full(NUM_ENVS, float(gripper), dtype=np.float32)
        start = len(self.buffers[0])
        for _ in range(int(frames)):
            self.step(positions, grippers, [phase] * NUM_ENVS)
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})

    def move(self, target, gripper: float, phase: str, max_frames: int = 240) -> None:
        positions = np.asarray(target, dtype=np.float32).reshape(NUM_ENVS, 3)
        grippers = np.full(NUM_ENVS, float(gripper), dtype=np.float32)
        stable = np.zeros(NUM_ENVS, dtype=np.int64)
        start = len(self.buffers[0])
        for _ in range(max_frames):
            self.step(positions, grippers, [phase] * NUM_ENVS)
            error = as_numpy(torch.linalg.vector_norm(
                self.env.fingertip_midpoint_pos
                - torch.as_tensor(positions, dtype=torch.float32, device=self.env.device), dim=1,
            ))
            stable = np.where(error <= 0.005, stable + 1, 0)
            if np.all(stable >= STABLE_STEPS):
                break
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})
        if not np.all(stable >= STABLE_STEPS):
            failed = np.flatnonzero(stable < STABLE_STEPS).tolist()
            raise RuntimeError(f"phase {phase} did not converge for envs {failed}")

    def run(self) -> list[dict[str, object]]:
        current = as_numpy(self.env.fingertip_midpoint_pos)
        self.hold(current, 1.0, "observe", 15)
        approach = self.handles + APPROACH_STANDOFF_M * self.axes
        self.move(approach, 1.0, "approach")
        self.move(self.handles, 1.0, "descend")
        self.hold(self.handles, 0.0, "grasp", 45)
        self.closed_finger_positions_m = as_numpy(self.env.robot.data.joint_pos.torch[:, 7:9])

        bias_samples = []
        for _ in range(10):
            self.step(self.handles, np.zeros(NUM_ENVS), ["bias"] * NUM_ENVS)
            bias_samples.append(self.env.force_sensor_world.detach().clone())
        bias = torch.stack(bias_samples).mean(dim=0)
        for env_id in range(NUM_ENVS):
            self.task.arm_after_franka_pregrasp(self.env, bias[env_id], env_id)

        command = self.handles.copy()
        pull_origin = self.handles.copy()
        pulled = np.zeros(NUM_ENVS, dtype=np.float64)
        desired = np.zeros(NUM_ENVS, dtype=np.float64)
        triggered = np.zeros(NUM_ENVS, dtype=bool)
        failed = np.zeros(NUM_ENVS, dtype=bool)
        start = len(self.buffers[0])
        for _ in range(420):
            openings = self.openings()
            resistances = self.resistances()
            phases = ["stop_and_hold" if triggered[i] or failed[i] else "pull_free" for i in range(NUM_ENVS)]
            for env_id in range(NUM_ENVS):
                if triggered[env_id] or failed[env_id]:
                    continue
                if self.task.latch_broken_for(env_id):
                    failed[env_id] = True
                    continue
                desired[env_id] = min(MAX_PULL_M, desired[env_id] + PULL_RATE_M)
                if self.profiles[env_id] == "uniform":
                    lead = max(0.0, resistances[env_id] - SPEC.free_resistance_n) / 300.0
                    pulled[env_id] = min(MAX_PULL_M, desired[env_id] + lead)
                else:
                    pulled[env_id] = min(MAX_PULL_M, pulled[env_id] + PULL_RATE_M)
                command[env_id] = pull_origin[env_id] + pulled[env_id] * self.axes[env_id]
                if openings[env_id] >= float(self.layouts[env_id]["resistance_start_open_m"]):
                    phases[env_id] = "resistance_rise"
            self.step(command, np.zeros(NUM_ENVS), phases)
            openings = self.openings()
            resistances = self.resistances()
            frame = len(self.buffers[0]) - 1
            for env_id in range(NUM_ENVS):
                if triggered[env_id] or failed[env_id]:
                    continue
                if self.task.latch_broken_for(env_id):
                    failed[env_id] = True
                elif resistances[env_id] >= float(self.layouts[env_id]["stop_resistance_n"]):
                    triggered[env_id] = True
                    self.trigger_frames[env_id] = frame
                    self.trigger_opening_m[env_id] = openings[env_id]
                    self.trigger_resistance_n[env_id] = resistances[env_id]
                    self.commanded_pull_m[env_id] = pulled[env_id]
            if np.all(triggered | failed):
                break
        self.phase_ranges.append({"name": "pull", "start": start, "end": len(self.buffers[0])})
        self.hold(command, 0.0, "stop_and_hold", 90)

        reports = []
        for env_id in range(NUM_ENVS):
            report = self.task.evaluate_final_state_for(self.env, env_id)
            gripped = bool(
                np.min(self.closed_finger_positions_m[env_id]) > 0.010
                and np.max(self.closed_finger_positions_m[env_id]) < 0.0395
            )
            # The RADMANSO knob is small enough that a valid pinch can close
            # below 10 mm.  Task success is therefore based on the observed
            # drawer motion, resistance entry, stable target hold, and intact
            # latch; finger aperture remains a diagnostic rather than a gate.
            success = bool(report["success"] and triggered[env_id] and not failed[env_id])
            report.update({
                "success": success,
                "gripper_aperture_diagnostic_passed": gripped,
                "closed_finger_positions_m": self.closed_finger_positions_m[env_id].tolist(),
                "triggered_resistance_stop": bool(triggered[env_id]),
                "trajectory_profile": self.profiles[env_id],
            })
            if success:
                self.buffers[env_id].set_last_transition_outcome(
                    reward=1.0, success=True, terminated=False, truncated=False
                )
            reports.append(report)
        return reports

    def details(self, env_id: int) -> dict[str, object]:
        return {
            "trajectory_profile": self.profiles[env_id],
            "phase_ranges_control_frames": list(self.phase_ranges),
            "trigger_frame": int(self.trigger_frames[env_id]),
            "trigger_opening_m": float(self.trigger_opening_m[env_id]),
            "trigger_resistance_n": float(self.trigger_resistance_n[env_id]),
            "commanded_pull_m": float(self.commanded_pull_m[env_id]),
            "maximum_tracking_error_m": float(self.maximum_tracking_error_m[env_id]),
            "closed_finger_positions_m": self.closed_finger_positions_m[env_id].tolist(),
            "maximum_opening_m": float(max(self.opening_traces[env_id])),
            "maximum_designed_resistance_n": float(max(self.resistance_traces[env_id])),
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
        if metrics["main_mean_intensity"] < 25.0 or metrics["main_dark_fraction"] > 0.55 or metrics["wrist_mean_intensity"] < 12.0:
            raise RuntimeError(f"env_{env_id} reset visual QC failed: {metrics}")
        reports.append({"env_id": env_id, "seed": SEEDS[env_id], "metrics": metrics})
    return reports


task = load_task("radmanso_drawer_sort_table", ASSET_DIR)
cfg = ForceBenchEnvCfg()
cfg.seed = SEEDS[0]
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(
    cfg, device=args.device, physics_rate_hz=PHYSICS_RATE_HZ,
    solver_iterations=SOLVER_POSITION_ITERATIONS, num_envs=NUM_ENVS,
    env_spacing=args.env_spacing,
)
cfg.num_rerenders_on_reset = 1
env = ForceBenchEnv(
    cfg=cfg, task_runtime=task, main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size, enable_cameras=True,
)
if not math.isclose(1.0 / env.physics_dt, PHYSICS_RATE_HZ, abs_tol=1.0e-9):
    raise RuntimeError("RADMANSO physics rate is not 60 Hz")
if not math.isclose(1.0 / env.step_dt, CONTROL_RATE_HZ, abs_tol=1.0e-9):
    raise RuntimeError("RADMANSO control rate is not 15 Hz")

expert = None
args.artifact_dir.mkdir(parents=True, exist_ok=True)
video_path = args.artifact_dir / "radmanso_drawer_sort_table_32env_mosaic.mp4"
overview_path = args.artifact_dir / "radmanso_drawer_sort_table_32env_initial.png"
final_overview_path = args.artifact_dir / "radmanso_drawer_sort_table_32env_final.png"

try:
    env.set_episode_seeds(SEEDS)
    env.reset(seed=SEEDS[0])
    env.reset_cameras()
    for _ in range(3):
        env.sim.render()
    env.update_cameras()
    visual_qc = reset_visual_qc(env)
    layouts = env.layouts
    env.start_wrench_capture()
    expert = ParallelRadmansoExpert(env, task, layouts, video_path)
    expert.write_overview(overview_path, "initial")
    reports = expert.run()
    high_rate_batch = env.stop_wrench_capture()
    expert.write_overview(final_overview_path, "final")
    expert.close()

    writer = HDF5EpisodeWriter(args.output_dir)
    physics_wrench = np.asarray(high_rate_batch["eef_wrench_joint_raw"], dtype=np.float32)
    timestamps = np.asarray(high_rate_batch["timestamp"], dtype=np.float64)
    expected_highrate = len(expert.buffers[0]) * (PHYSICS_RATE_HZ // CONTROL_RATE_HZ)
    if len(timestamps) != expected_highrate or physics_wrench.shape[:2] != (expected_highrate, NUM_ENVS):
        raise RuntimeError(
            f"60 Hz wrench capture mismatch: timestamps={len(timestamps)}, wrench={physics_wrench.shape}, expected={expected_highrate}"
        )
    if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 1.0 / PHYSICS_RATE_HZ, rtol=1e-5, atol=1e-8):
        raise RuntimeError("physics timestamps are not uniformly 60 Hz")
    asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd"), **env.scene_asset_paths}
    written = []
    failed = []
    for env_id, seed in enumerate(SEEDS):
        if seed not in WRITE_SEEDS:
            continue
        if not reports[env_id]["success"]:
            failed.append({"env_id": env_id, "seed": seed, "report": reports[env_id]})
            continue
        existing = sorted(args.output_dir.glob(f"seed_{seed:03d}_episode_*.hdf5"))
        if existing:
            raise RuntimeError(f"refusing to duplicate successful seed {seed}: {existing}")
        layout = layouts[env_id]
        metadata = {
            "task": "radmanso_drawer_sort_table",
            "prompt": task.instruction(layout),
            "success": True,
            "end_reason": "scripted_parallel_resistance_rise_stop_and_hold",
            "seed": seed,
            "initial_layout": layout,
            "final_success_report": reports[env_id],
            "control_rate_hz": float(CONTROL_RATE_HZ),
            "physics_rate_hz": float(PHYSICS_RATE_HZ),
            "physics_steps_per_control": PHYSICS_RATE_HZ // CONTROL_RATE_HZ,
            "solver_position_iterations": SOLVER_POSITION_ITERATIONS,
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "arm_controller_diagnostics": env.arm_controller_diagnostics,
            "scene_profile": env.scene_profile_name,
            "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "control_source": "scripted_32env_single_process_privileged_resistance_stop_oracle",
            "wrench_profile": env.wrench_profile,
            "wrench_storage": "raw_and_ema_control_plus_raw_physics",
            "primary_wrist_wrench_dataset": "forces/control/eef_wrench_joint_raw",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [args.main_image_size, args.main_image_size, 3],
                "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
            },
            "phase_schema": PHASE_SCHEMA,
            "scripted_expert_generator_version": "radmanso_resistance_stop_32env_tiled_v2_continuous_physics60",
            "scripted_expert_details": expert.details(env_id),
            "trajectory_profile": expert.profiles[env_id],
            "resistance_profile": {
                "type": "free_then_cubic_rise_then_irreversible_release",
                "free_resistance_n": SPEC.free_resistance_n,
                "resistance_start_open_m": layout["resistance_start_open_m"],
                "resistance_band_length_m": layout["resistance_band_length_m"],
                "success_open_m": layout["success_open_m"],
                "success_tolerance_m": SPEC.success_tolerance_m,
                "breakaway_open_m": layout["breakaway_open_m"],
                "breakaway_resistance_n": layout["breakaway_resistance_n"],
                "stop_resistance_n": SPEC.stop_resistance_n,
                "post_breakaway_resistance_n": SPEC.post_breakaway_resistance_n,
                "max_safe_open_m": SPEC.max_safe_open_m,
            },
            "parallel_environment": {
                "enabled": True, "single_isaac_process": True,
                "num_envs": NUM_ENVS, "env_id": env_id,
                "env_spacing_m": args.env_spacing, "all_seeds": SEEDS,
                "camera_backend": "TiledCamera", "collision_filtering": True,
                "replicate_physics": True, "reset_visual_qc": visual_qc[env_id],
            },
            "scene_roles": list(env.entity_roles),
            "asset_usd_paths": asset_paths,
            "selected_drawer": layout["selected_drawer"],
            "drawer_joint": layout["drawer_joint"],
            "always_locked_joint": BOTTOM_JOINT,
            "selectable_drawers": {"middle": DRAWER_JOINTS["middle"]},
        }
        high_rate = {"timestamp": timestamps, "eef_wrench_joint_raw": physics_wrench[:, env_id]}
        path = writer.write(expert.buffers[env_id], metadata, high_rate=high_rate)[0]
        validation = validate_hdf5_episode(path)
        written.append({"env_id": env_id, "seed": seed, "path": str(path), "validation": validation})

    summary = {
        "schema": "forcebench_parallel_radmanso_32env_v1",
        "task": "radmanso_drawer_sort_table",
        "seeds": SEEDS, "write_seeds": sorted(WRITE_SEEDS),
        "num_envs": NUM_ENVS, "physics_rate_hz": PHYSICS_RATE_HZ,
        "control_rate_hz": CONTROL_RATE_HZ, "env_spacing_m": args.env_spacing,
        "frames_per_episode": len(expert.buffers[0]),
        "video": str(video_path.resolve()), "overview": str(overview_path.resolve()),
        "final_overview": str(final_overview_path.resolve()),
        "episodes": written, "failed_episodes": failed,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("FORCEBENCH_PARALLEL_RADMANSO=" + json.dumps(summary, separators=(",", ":")), flush=True)
except BaseException as error:
    print(f"FORCEBENCH_PARALLEL_RADMANSO_ERROR={type(error).__name__}:{error}", flush=True)
    raise
finally:
    if expert is not None:
        expert.close()
    env.discard_wrench_capture()
    env.close()
    simulation_app.close()
