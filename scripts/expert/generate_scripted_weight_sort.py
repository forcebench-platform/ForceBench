#!/usr/bin/env python3
"""Generate one validated single-box weight_sort expert trajectory."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from coribench.isaac_session import shared_simulation_app


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument(
    "--video",
    type=Path,
    help="optional diagnostic MP4; omit during high-throughput formal collection",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument(
    "--lift_box_height_min_m",
    type=float,
    default=float(os.environ.get("CORIBENCH_WEIGHT_SORT_LIFT_BOX_MIN_M", "0.12")),
)
parser.add_argument(
    "--lift_box_height_max_m",
    type=float,
    default=float(os.environ.get("CORIBENCH_WEIGHT_SORT_LIFT_BOX_MAX_M", "0.18")),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0.075 <= args.lift_box_height_min_m < args.lift_box_height_max_m <= 0.22:
    parser.error("lift-box height range must satisfy 0.075 <= min < max <= 0.22 metres")
# One process owns one GPU.  Isaac Sim's headless experience otherwise
# auto-enables renderer multi-GPU and makes independent dataset workers fight
# over every device on an eight-GPU host.
simulation_app = shared_simulation_app()
owns_simulation_app = simulation_app is None
if owns_simulation_app:
    launcher = AppLauncher(args, multi_gpu=False)
    simulation_app = launcher.app

import cv2
import numpy as np
import torch
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import compute_pose_error

from coribench.env import CoRiBenchEnv
from coribench.env_cfg import ASSET_DIR, CoRiBenchEnvCfg, configure_runtime
from coribench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from coribench.registry import load_task
from coribench.tasks.weight_sort.spec import SPEC


DOWN_RPY = np.asarray([math.pi, 0.0, 0.0], dtype=np.float32)
PHASES = (
    "observe_layout",
    "approach_box",
    "grasp_box",
    "lift_and_weigh",
    "transport_to_container",
    "release_box",
    "final_retract_and_settle",
)
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASES)}
SPEED_PROFILE = {
    "convergence_stable_frames": 6,
    "observe_frames": 12,
    "grasp_close_frames": 40,
    "weigh_hold_frames": 35,
    "pre_release_hold_frames": 12,
    "release_open_frames": 40,
    "post_release_hold_frames": 24,
    "final_settle_frames": 70,
}


def vector(value) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1).astype(np.float32)


def rgb(value) -> np.ndarray:
    return value.detach().cpu().numpy()[..., :3].astype(np.uint8)


def current_state(env: CoRiBenchEnv) -> np.ndarray:
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(env.fingertip_midpoint_quat)
    return np.asarray(
        [
            *vector(env.fingertip_midpoint_pos[0]),
            float(roll[0]),
            float(pitch[0]),
            float(yaw[0]),
            2.0 * env.gripper_open_fraction - 1.0,
        ],
        dtype=np.float32,
    )


class WeightSortExpert:
    def __init__(self, env: CoRiBenchEnv, task, video_path: Path | None):
        self.env = env
        self.task = task
        self.buffer = EpisodeBuffer()
        self.zero_action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
        self.down = math_utils.quat_from_euler_xyz(
            torch.tensor([torch.pi], device=env.device),
            torch.zeros(1, device=env.device),
            torch.zeros(1, device=env.device),
        )
        self.phase_ranges: list[dict[str, int | str]] = []
        self.load_signals_n: dict[str, float] = {}
        lift_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x4C494654]))
        self.sampled_lift_box_height_m = float(
            lift_rng.uniform(args.lift_box_height_min_m, args.lift_box_height_max_m)
        )
        self.lift_tcp_target_z_m = float("nan")
        self.lifted_box_root_z_m: float | None = None
        self.video = None
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            self.video = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                15.0,
                (args.main_image_size, args.main_image_size),
            )
            if not self.video.isOpened():
                raise RuntimeError(f"failed to open video writer: {video_path}")

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    def _write_video_frame(self, phase: str) -> None:
        if self.video is None:
            return
        image = rgb(self.env.table_rgb[0])[..., ::-1].copy()
        height, width = image.shape[:2]
        wrench = vector(self.env.force_sensor_joint_raw[0])
        cv2.rectangle(image, (10, 10), (width - 10, 106), (18, 18, 18), -1)
        cv2.putText(
            image,
            phase.replace("_", " ").upper(),
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"RAW F  {wrench[0]:+6.1f} {wrench[1]:+6.1f} {wrench[2]:+6.1f} N",
            (24, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"RAW T  {wrench[3]:+6.2f} {wrench[4]:+6.2f} {wrench[5]:+6.2f} Nm",
            (24, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        self.video.write(image)

    def _record_and_step(self, target_m: np.ndarray, gripper: float, phase: str) -> None:
        action = np.asarray([*target_m, *DOWN_RPY, 2.0 * gripper - 1.0], dtype=np.float32)
        self.buffer.append(
            state=current_state(self.env),
            action=action,
            wrench_raw=vector(self.env.force_sensor_joint_raw[0]),
            wrench_ema=vector(self.env.force_sensor_joint_smooth[0]),
            timestamp=(len(self.buffer) + 1) * self.env.step_dt,
            image=rgb(self.env.table_rgb[0]),
            wrist_image=rgb(self.env.wrist_rgb[0]),
            joint_position=vector(self.env.robot.data.joint_pos.torch[0]),
            joint_velocity=vector(self.env.robot.data.joint_vel.torch[0]),
            tcp_pose=vector(
                torch.cat((self.env.fingertip_midpoint_pos[0], self.env.fingertip_midpoint_quat[0]))
            ),
            privileged=dict(self.task.privileged_observations(self.env)),
            phase_id=PHASE_SCHEMA[phase],
        )
        self._write_video_frame(phase)
        self.env.set_cartesian_rpy_target(action[:6], gripper)
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        self.buffer.set_last_transition_outcome(
            reward=float(reward[0]),
            success=False,
            terminated=bool(terminated[0]),
            truncated=bool(truncated[0]),
        )
        self.env.update_cameras()

    def move(
        self,
        target_m,
        gripper: float,
        phase: str,
        *,
        max_frames: int = 220,
        tolerance_m: float = 0.004,
    ) -> None:
        target = np.asarray(target_m, dtype=np.float32)
        start = len(self.buffer)
        stable = 0
        position_error = float("inf")
        rotation_error = float("inf")
        for _ in range(max_frames):
            self._record_and_step(target, gripper, phase)
            position_error = float(
                torch.linalg.vector_norm(
                    self.env.fingertip_midpoint_pos[0]
                    - torch.as_tensor(target, dtype=torch.float32, device=self.env.device)
                ).item()
            )
            _, rotation = compute_pose_error(
                self.env.fingertip_midpoint_pos,
                self.env.fingertip_midpoint_quat,
                torch.as_tensor(target, dtype=torch.float32, device=self.env.device).unsqueeze(0),
                self.down,
                rot_error_type="axis_angle",
            )
            rotation_error = float(torch.linalg.vector_norm(rotation[0]).item())
            # The Panda can retain a small harmless wrist-yaw residual near its
            # workspace limits. Physical box placement depends on TCP position;
            # each grasp and destination is validated separately below.
            stable = stable + 1 if position_error <= tolerance_m else 0
            if stable >= SPEED_PROFILE["convergence_stable_frames"]:
                break
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffer)})
        if stable < SPEED_PROFILE["convergence_stable_frames"]:
            raise RuntimeError(
                f"phase {phase} did not converge: position={position_error:.4f}, rotation={rotation_error:.4f}"
            )
        print(
            f"CORIBENCH_WEIGHT_SORT_PHASE={phase},frames={len(self.buffer) - start},"
            f"position_error_m={position_error:.5f},rotation_error_rad={rotation_error:.5f}",
            flush=True,
        )

    def hold(self, frames: int, gripper: float, phase: str, *, measure_load: str | None = None) -> None:
        target = vector(self.env.fingertip_midpoint_pos[0])
        start = len(self.buffer)
        samples: list[float] = []
        for _ in range(frames):
            self._record_and_step(target, gripper, phase)
            if measure_load is not None:
                samples.append(
                    float(
                        torch.linalg.vector_norm(
                            self.env.force_sensor_joint_raw[0, :3]
                        ).item()
                    )
                )
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffer)})
        if measure_load is not None:
            tail = samples[len(samples) // 2 :]
            self.load_signals_n[measure_load] = float(np.mean(tail))

    def pickup_and_weigh(self) -> None:
        position, _ = self.env.entity_local_pose("box")
        xyz = vector(position[0])
        self.move((xyz[0], xyz[1], 0.12), 1.0, "approach_box")
        self.move((xyz[0], xyz[1], 0.018), 1.0, "approach_box")
        self.hold(SPEED_PROFILE["grasp_close_frames"], 0.0, "grasp_box")
        grasped, _ = self.env.entity_local_pose("box")
        tcp_box_z_offset_m = float(
            self.env.fingertip_midpoint_pos[0, 2].item() - grasped[0, 2].item()
        )
        self.lift_tcp_target_z_m = self.sampled_lift_box_height_m + tcp_box_z_offset_m
        self.move(
            (xyz[0], xyz[1], self.lift_tcp_target_z_m),
            0.0,
            "lift_and_weigh",
            tolerance_m=0.025,
        )
        self.hold(
            SPEED_PROFILE["weigh_hold_frames"],
            0.0,
            "lift_and_weigh",
            measure_load="box",
        )
        lifted, _ = self.env.entity_local_pose("box")
        self.lifted_box_root_z_m = float(lifted[0, 2])
        if float(lifted[0, 2]) < 0.075:
            raise RuntimeError("box was not lifted during weighing")

    def place_held(self, destination: str) -> None:
        phase = "transport_to_container"
        release_phase = "release_box"
        if destination == "black_container":
            # Both receptacles now sit on the positive-Y (image-right) side.
            # Approach them through central, then near-side waypoints to avoid
            # sweeping a held box across either receptacle wall.
            waypoints = [(0.50, 0.10, 0.22), (0.58, 0.24, 0.21)]
            target = (*SPEC.black_container_center_xy_m, 0.20)
        elif destination == "blue_container":
            waypoints = [(0.44, 0.10, 0.20), (0.36, 0.24, 0.20)]
            target = (*SPEC.blue_container_center_xy_m, 0.20)
        else:
            raise ValueError(destination)
        for waypoint in waypoints:
            # These are collision-clear transit points, not placement targets.
            # A 40 mm loaded-TCP residual is harmless here; the destination
            # pose and post-release object state remain independently checked.
            self.move(waypoint, 0.0, phase, max_frames=170, tolerance_m=0.040)
        self.move(target, 0.0, phase, max_frames=300, tolerance_m=0.030)
        self.hold(SPEED_PROFILE["pre_release_hold_frames"], 0.0, phase)
        self.hold(SPEED_PROFILE["release_open_frames"], 1.0, release_phase)
        if destination == "black_container":
            retract = (0.56, 0.22, 0.25)
        else:
            retract = (0.36, 0.20, 0.25)
        self.move(retract, 1.0, release_phase, max_frames=180)
        self.hold(SPEED_PROFILE["post_release_hold_frames"], 1.0, release_phase)
        position, _ = self.env.entity_local_pose("box")
        center = (
            SPEC.black_container_center_xy_m
            if destination == "black_container"
            else SPEC.blue_container_center_xy_m
        )
        if float(np.linalg.norm(vector(position[0])[:2] - np.asarray(center))) > 0.075:
            raise RuntimeError(f"box missed {destination}: {vector(position[0]).tolist()}")

    def run(self) -> dict[str, object]:
        box_kind = str(self.env.layout["box_kind"])
        destination = str(self.env.layout["destination"])
        self.hold(SPEED_PROFILE["observe_frames"], 1.0, "observe_layout")
        self.pickup_and_weigh()
        self.place_held(destination)
        self.hold(SPEED_PROFILE["final_settle_frames"], 1.0, "final_retract_and_settle")

        report = self.env.evaluate_final_success()
        if not bool(report["success"]):
            raise RuntimeError("final success check failed: " + json.dumps(report, separators=(",", ":")))
        self.buffer.set_last_transition_outcome(
            reward=1.0,
            success=True,
            terminated=False,
            truncated=False,
        )
        return {
            "report": report,
            "box_kind": box_kind,
            "box_mass_kg": float(self.env.layout["box_mass_kg"]),
            "lift_box_height_range_m": [
                args.lift_box_height_min_m,
                args.lift_box_height_max_m,
            ],
            "sampled_lift_box_height_m": self.sampled_lift_box_height_m,
            "lift_tcp_target_z_m": self.lift_tcp_target_z_m,
            "lifted_box_root_z_m": self.lifted_box_root_z_m,
            "destination": destination,
            "load_signals_n": self.load_signals_n,
            "phase_ranges_control_frames": self.phase_ranges,
        }


task = load_task("weight_sort", ASSET_DIR)
cfg = CoRiBenchEnvCfg()
cfg.seed = args.seed
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(cfg, device=args.device, physics_rate_hz=120, solver_iterations=64)
cfg.num_rerenders_on_reset = 1
env = CoRiBenchEnv(
    cfg=cfg,
    task_runtime=task,
    main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size,
    enable_cameras=True,
)
expert: WeightSortExpert | None = None

try:
    env.set_episode_seed(args.seed)
    env.reset(seed=args.seed)
    env.reset_cameras()
    for _ in range(3):
        env.sim.render()
    env.update_cameras()
    env.start_wrench_capture()
    expert = WeightSortExpert(env, task, args.video)
    result = expert.run()
    high_rate = env.stop_wrench_capture()
    recorded_buffer = expert.buffer
    expert.close()
    expert = None

    task_metadata = dict(task.episode_metadata())
    asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
    asset_paths.update(env.scene_asset_paths)
    asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
    metadata = {
        "task": "weight_sort",
        "prompt": task.manifest.prompt,
        "success": True,
        "end_reason": "scripted_weight_sort_final_check",
        "seed": args.seed,
        "initial_layout": env.layout,
        "final_success_report": result["report"],
        "control_rate_hz": 15.0,
        "physics_rate_hz": 120.0,
        "arm_control_mode": env.arm_control_mode,
        "arm_controller_profile": env.arm_controller_profile,
        "wrench_profile": env.wrench_profile,
        "wrench_storage": "raw_and_ema",
        "primary_wrist_wrench_dataset": "forces/control/eef_wrench_joint_raw",
        "recorded_wrist_wrench_datasets": [
            "forces/control/eef_wrench_joint_raw",
            "forces/control/eef_wrench_joint_ema",
        ],
        "arm_controller_diagnostics": env.arm_controller_diagnostics,
        "wrench_semantics": (
            "panda_hand_joint incoming constraint reaction wrench, parent-on-child, "
            "joint frame, Fx_Fy_Fz_Tx_Ty_Tz; raw is primary and EMA(alpha=0.25) "
            "is stored as an auxiliary channel"
        ),
        "scene_profile": env.scene_profile_name,
        "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
        "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
        "control_source": f"scripted_single_box_weighing_pose_oracle_{env.arm_control_mode}",
        "success_check_timing": "only_after_box_released_and_settled",
        "images_present": True,
        "image_shapes_hwc": {
            "main": [args.main_image_size, args.main_image_size, 3],
            "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
        },
        "phase_schema": PHASE_SCHEMA,
        "scripted_expert_generator_version": "weight_sort_random_lift_raw_ema",
        "scripted_expert_speed_profile": SPEED_PROFILE,
        "scripted_expert_details": {key: value for key, value in result.items() if key != "report"},
        "scene_roles": list(env.entity_roles),
        "asset_usd_paths": asset_paths,
        **task_metadata,
    }
    path = HDF5EpisodeWriter(args.output_dir).write(
        recorded_buffer,
        metadata,
        high_rate=high_rate,
    )[0]
    validation = validate_hdf5_episode(path)
    summary = {
        "episode": str(path.resolve()),
        "video": str(args.video.resolve()) if args.video is not None else None,
        "frames": len(recorded_buffer),
        "duration_s": len(recorded_buffer) / 15.0,
        "validation": validation,
        **{key: value for key, value in result.items() if key != "phase_ranges_control_frames"},
    }
    if args.video is not None:
        args.video.with_suffix(".json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("CORIBENCH_WEIGHT_SORT_EXPERT=" + json.dumps(summary, separators=(",", ":")), flush=True)
except Exception as exc:
    print(f"CORIBENCH_WEIGHT_SORT_EXPERT_ERROR={type(exc).__name__}:{exc}", flush=True)
    raise
finally:
    if expert is not None:
        expert.close()
    env.close()
    if owns_simulation_app:
        simulation_app.close()
