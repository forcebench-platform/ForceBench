#!/usr/bin/env python3
"""Generate one validated static equal-mass balance-sort expert trajectory."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from forcebench.isaac_session import shared_simulation_app


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
    default=float(os.environ.get("FORCEBENCH_RATTLE_LIFT_BOX_MIN_M", "0.12")),
)
parser.add_argument(
    "--lift_box_height_max_m",
    type=float,
    default=float(os.environ.get("FORCEBENCH_RATTLE_LIFT_BOX_MAX_M", "0.18")),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0.075 <= args.lift_box_height_min_m < args.lift_box_height_max_m <= 0.22:
    parser.error(
        "lift-box height range must satisfy "
        "0.075 <= min < max <= 0.22 metres"
    )
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

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task
from forcebench.tasks.rattle_sort.spec import SPEC


DOWN_RPY = np.asarray([math.pi, 0.0, 0.0], dtype=np.float32)
PHASES = (
    "observe_layout",
    "approach_box",
    "grasp_box",
    "lift_box",
    "static_inspect",
    "transport_to_destination",
    "release_box",
    "final_retract_and_settle",
)
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASES)}
SPEED_PROFILE = {
    "convergence_stable_frames": 6,
    "observe_frames": 12,
    "grasp_close_frames": 40,
    # Match weight_sort: the formal inspection segment is 35 control frames.
    "static_inspection_hold_frames": SPEC.inspection_expert_hold_frames,
    "pre_release_hold_frames": 12,
    "release_open_frames": 40,
    "post_release_hold_frames": 24,
    "final_settle_frames": 70,
}


def vector(value) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1).astype(np.float32)


def rgb(value) -> np.ndarray:
    return value.detach().cpu().numpy()[..., :3].astype(np.uint8)


def current_state(env: ForceBenchEnv) -> np.ndarray:
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


class RattleSortExpert:
    def __init__(self, env: ForceBenchEnv, task, video_path: Path | None):
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
        self.static_force_samples_n: list[float] = []
        self.static_torque_samples_nm: list[float] = []
        lift_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, 0x4C494654])
        )
        self.sampled_lift_box_height_m = float(
            lift_rng.uniform(
                args.lift_box_height_min_m,
                args.lift_box_height_max_m,
            )
        )
        self.lift_tcp_target_z_m = float("nan")
        self.lifted_box_height_m = float("nan")
        self.inspection_box_height_m = float("nan")
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
        if phase == "static_inspect":
            self.static_force_samples_n.append(
                float(torch.linalg.vector_norm(self.env.force_sensor_joint_raw[0, :3]).item())
            )
            self.static_torque_samples_nm.append(
                float(torch.linalg.vector_norm(self.env.force_sensor_joint_raw[0, 3:]).item())
            )
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
            f"FORCEBENCH_RATTLE_SORT_PHASE={phase},frames={len(self.buffer) - start},"
            f"position_error_m={position_error:.5f},rotation_error_rad={rotation_error:.5f}",
            flush=True,
        )

    def hold(self, frames: int, gripper: float, phase: str) -> None:
        target = vector(self.env.fingertip_midpoint_pos[0])
        start = len(self.buffer)
        for _ in range(frames):
            self._record_and_step(target, gripper, phase)
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffer)})

    def pickup_and_inspect(self) -> None:
        position, _ = self.env.entity_local_pose("box")
        xyz = vector(position[0])
        self.move((xyz[0], xyz[1], 0.12), 1.0, "approach_box")
        self.move((xyz[0], xyz[1], 0.014), 1.0, "approach_box")
        self.hold(SPEED_PROFILE["grasp_close_frames"], 0.0, "grasp_box")
        grasped, _ = self.env.entity_local_pose("box")
        tcp_box_z_offset_m = float(
            self.env.fingertip_midpoint_pos[0, 2].item() - grasped[0, 2].item()
        )
        self.lift_tcp_target_z_m = self.sampled_lift_box_height_m + tcp_box_z_offset_m
        self.move(
            (xyz[0], xyz[1], self.lift_tcp_target_z_m),
            0.0,
            "lift_box",
            tolerance_m=0.025,
        )
        lifted, _ = self.env.entity_local_pose("box")
        self.lifted_box_height_m = float(lifted[0, 2].item())
        if self.lifted_box_height_m < 0.075:
            raise RuntimeError("box was not lifted before static inspection")

        self.hold(
            SPEED_PROFILE["static_inspection_hold_frames"],
            0.0,
            "static_inspect",
        )
        inspected, _ = self.env.entity_local_pose("box")
        self.inspection_box_height_m = float(inspected[0, 2].item())

    def place_held(self, destination: str) -> None:
        phase = "transport_to_destination"
        release_phase = "release_box"
        if destination == "black_basket":
            waypoints = [(0.44, -0.12, 0.24), (0.40, -0.23, 0.22)]
            target = (*SPEC.black_basket_center_xy_m, 0.20)
        elif destination == "conveyor":
            waypoints = [(0.50, 0.14, 0.24), (0.54, 0.26, 0.22)]
            target = (*SPEC.conveyor_center_xy_m, 0.20)
        else:
            raise ValueError(destination)
        for waypoint in waypoints:
            # These are collision-clear transit points, not placement targets.
            # A 40 mm held-TCP residual is harmless here; the destination
            # pose and post-release object state remain independently checked.
            self.move(waypoint, 0.0, phase, max_frames=170, tolerance_m=0.040)
        self.move(target, 0.0, phase, max_frames=300, tolerance_m=0.040)
        self.hold(SPEED_PROFILE["pre_release_hold_frames"], 0.0, phase)
        self.hold(SPEED_PROFILE["release_open_frames"], 1.0, release_phase)
        if destination == "black_basket":
            retract = (0.40, -0.20, 0.27)
        else:
            retract = (0.54, 0.23, 0.27)
        self.move(retract, 1.0, release_phase, max_frames=180)
        self.hold(SPEED_PROFILE["post_release_hold_frames"], 1.0, release_phase)
        position, _ = self.env.entity_local_pose("box")
        center = (
            SPEC.black_basket_center_xy_m
            if destination == "black_basket"
            else SPEC.conveyor_center_xy_m
        )
        tolerance = 0.10 if destination == "black_basket" else 0.14
        if float(np.linalg.norm(vector(position[0])[:2] - np.asarray(center))) > tolerance:
            raise RuntimeError(f"box missed {destination}: {vector(position[0]).tolist()}")

    def run(self) -> dict[str, object]:
        box_kind = str(self.env.layout["box_kind"])
        destination = str(self.env.layout["destination"])
        self.hold(SPEED_PROFILE["observe_frames"], 1.0, "observe_layout")
        self.pickup_and_inspect()
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
            "total_mass_kg": float(self.env.layout["total_mass_kg"]),
            "center_of_mass_offset_local_m": self.env.layout[
                "center_of_mass_offset_local_m"
            ],
            "heavy_end": self.env.layout["heavy_end"],
            "destination": destination,
            "lift_box_height_range_m": [
                args.lift_box_height_min_m,
                args.lift_box_height_max_m,
            ],
            "sampled_lift_box_height_m": self.sampled_lift_box_height_m,
            "lift_tcp_target_z_m": self.lift_tcp_target_z_m,
            "lifted_box_height_m": self.lifted_box_height_m,
            "inspection_box_height_m": self.inspection_box_height_m,
            "static_force_mean_n": (
                float(np.mean(self.static_force_samples_n))
                if self.static_force_samples_n
                else 0.0
            ),
            "static_torque_mean_nm": (
                float(np.mean(self.static_torque_samples_nm))
                if self.static_torque_samples_nm
                else 0.0
            ),
            "inspection_state": self.task.debug_state(self.env),
            "phase_ranges_control_frames": self.phase_ranges,
        }


task = load_task("rattle_sort", ASSET_DIR)
cfg = ForceBenchEnvCfg()
cfg.seed = args.seed
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(cfg, device=args.device, physics_rate_hz=120, solver_iterations=64)
cfg.num_rerenders_on_reset = 1
env = ForceBenchEnv(
    cfg=cfg,
    task_runtime=task,
    main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size,
    enable_cameras=True,
)
expert: RattleSortExpert | None = None

try:
    env.set_episode_seed(args.seed)
    env.reset(seed=args.seed)
    env.reset_cameras()
    for _ in range(3):
        env.sim.render()
    env.update_cameras()
    env.start_wrench_capture()
    expert = RattleSortExpert(env, task, args.video)
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
        "task": "rattle_sort",
        "prompt": task.manifest.prompt,
        "success": True,
        "end_reason": "scripted_rattle_sort_final_check",
        "seed": args.seed,
        "initial_layout": env.layout,
        "final_success_report": result["report"],
        "control_rate_hz": 15.0,
        "physics_rate_hz": 120.0,
        "arm_control_mode": env.arm_control_mode,
        "arm_controller_profile": env.arm_controller_profile,
        "scene_profile": env.scene_profile_name,
        "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
        "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
        "control_source": (
            f"scripted_equal_mass_center_of_mass_sort_pose_oracle_{env.arm_control_mode}"
        ),
        "wrench_profile": env.wrench_profile,
        "wrench_storage": "raw_and_ema",
        "primary_wrist_wrench_dataset": "forces/control/eef_wrench_joint_raw",
        "recorded_wrist_wrench_datasets": [
            "forces/control/eef_wrench_joint_raw",
            "forces/control/eef_wrench_joint_ema",
        ],
        "wrist_wrench_semantics": (
            "incoming panda_hand_joint reaction [Fx,Fy,Fz,Tx,Ty,Tz]; raw is primary, "
            "EMA(alpha=0.25) is auxiliary, with no coordinate transform or gravity compensation"
        ),
        "success_check_timing": "only_after_box_released_and_settled",
        "images_present": True,
        "image_shapes_hwc": {
            "main": [args.main_image_size, args.main_image_size, 3],
            "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
        },
        "phase_schema": PHASE_SCHEMA,
        "scripted_expert_generator_version": "rattle_sort_random_lift_raw_ema",
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
    print("FORCEBENCH_RATTLE_SORT_EXPERT=" + json.dumps(summary, separators=(",", ":")), flush=True)
except Exception as exc:
    print(f"FORCEBENCH_RATTLE_SORT_EXPERT_ERROR={type(exc).__name__}:{exc}", flush=True)
    raise
finally:
    if expert is not None:
        expert.close()
    env.close()
    if owns_simulation_app:
        simulation_app.close()
