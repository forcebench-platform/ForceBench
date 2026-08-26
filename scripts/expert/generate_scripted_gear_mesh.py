#!/usr/bin/env python3
"""Generate one physical, validated gear-meshing expert episode and video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

from coribench.isaac_session import shared_simulation_app


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--video", type=Path)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument(
    "--variant",
    choices=("direct", "contact_recovery"),
    required=True,
    help="Direct placement or a small offset/contact followed by slow search.",
)
parser.add_argument(
    "--offset_radius_min_m",
    "--offset_distance_min_m",
    dest="offset_radius_min_m",
    type=float,
    default=0.0025,
    help="Minimum contact-recovery offset radius in metres (default: 0.0025).",
)
parser.add_argument(
    "--offset_radius_max_m",
    "--offset_distance_max_m",
    dest="offset_radius_max_m",
    type=float,
    default=0.0045,
    help="Maximum contact-recovery offset radius in metres (default: 0.0045).",
)
parser.add_argument(
    "--offset_relative_angle_min_deg",
    "--offset_angle_min_deg",
    dest="offset_relative_angle_min_deg",
    type=float,
    default=-60.0,
    help=(
        "Minimum offset angle in degrees relative to the pickup-centre-to-target-centre "
        "direction (default: -60)."
    ),
)
parser.add_argument(
    "--offset_relative_angle_max_deg",
    "--offset_angle_max_deg",
    dest="offset_relative_angle_max_deg",
    type=float,
    default=60.0,
    help=(
        "Maximum offset angle in degrees relative to the pickup-centre-to-target-centre "
        "direction (default: 60)."
    ),
)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--physics_rate_hz", type=int, choices=(60,), default=60)
parser.add_argument("--solver_position_iterations", type=int, choices=(64,), default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seed < 0:
    parser.error("--seed must be non-negative")
if not all(
    math.isfinite(value)
    for value in (
        args.offset_radius_min_m,
        args.offset_radius_max_m,
        args.offset_relative_angle_min_deg,
        args.offset_relative_angle_max_deg,
    )
):
    parser.error("contact-offset range values must be finite")
if not 0.0 < args.offset_radius_min_m <= args.offset_radius_max_m <= 0.010:
    parser.error("offset radius range must satisfy 0 < min <= max <= 0.010 m")
if not (
    -180.0
    <= args.offset_relative_angle_min_deg
    <= args.offset_relative_angle_max_deg
    <= 180.0
):
    parser.error("relative angle range must satisfy -180 <= min <= max <= 180 degrees")

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
from coribench.tasks.gear_mesh.spec import SPEC


DOWN_ROLL = math.pi
PHASES = (
    "observe_layout",
    "approach_pickup",
    "grasp_gear",
    "lift_gear",
    "transport_above_fixture",
    "direct_alignment",
    "offset_contact",
    "slow_search",
    "final_seating",
    "release_gear",
    "retract_and_settle",
)
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASES)}
PROFILE = {
    "stable_frames": 5,
    "observe_frames": 18,
    "grasp_frames": 38,
    "post_grasp_hold_frames": 10,
    "transport_height_m": 0.19,
    "preinsert_height_m": 0.105,
    "direct_preseat_height_m": 0.060,
    "contact_offset_sampling": {
        "radius_range_m": [args.offset_radius_min_m, args.offset_radius_max_m],
        "relative_angle_range_deg": [
            args.offset_relative_angle_min_deg,
            args.offset_relative_angle_max_deg,
        ],
        "angle_reference": "pickup_gear_center_to_target_shaft_center",
    },
    "contact_command_frames": 20,
    "search_waypoint_frames": 9,
    "twist_search_offsets_deg": [0.0, 1.5, -1.5, 3.0, -3.0, 4.5, -4.5],
    "seat_hold_frames": 18,
    "release_frames": 42,
    "settle_frames": 55,
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


def wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


class GearMeshExpert:
    def __init__(self, env: CoRiBenchEnv, task, video_path: Path | None, variant: str):
        self.env = env
        self.task = task
        self.variant = variant
        self.buffer = EpisodeBuffer()
        self.zero_action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
        self.phase_ranges: list[dict[str, int | str]] = []
        self.phase_force_peaks_n: dict[str, float] = {}
        self.phase_force_samples_n: dict[str, list[float]] = {}
        self.collision_offset_xy_m = np.zeros(2, dtype=np.float32)
        self.collision_offset_radius_m: float | None = None
        self.collision_offset_reference_angle_deg: float | None = None
        self.collision_offset_relative_angle_deg: float | None = None
        self.collision_offset_world_angle_deg: float | None = None
        self.precontact_force_n = float("nan")
        self.collision_force_peak_n = float("nan")
        self.placement_tcp_command_xy_m = np.zeros(2, dtype=np.float32)
        self.preinsert_gear_center_error_m = float("nan")
        self.released_during_seating = False
        self.recovery_recentered = False
        self.gear_tcp_z_offset_m = SPEC.pickup_grasp_tcp_z_m - SPEC.pickup_gear_root_z_m
        self.pickup_tcp_yaw_rad = float(env.layout["target_base_yaw_rad"])
        desired_gear_yaw = float(env.layout["target_base_yaw_rad"]) + math.radians(
            SPEC.expert_twist_alignment_offset_deg
        )
        self.mesh_tcp_yaw_rad = wrap_angle(
            self.pickup_tcp_yaw_rad
            + desired_gear_yaw
            - float(env.layout["medium_gear_yaw_rad"])
        )
        self.seat_tcp_yaw_rad = self.mesh_tcp_yaw_rad
        if self.variant == "contact_recovery":
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x4D455348]))
            self.collision_offset_radius_m = float(
                rng.uniform(args.offset_radius_min_m, args.offset_radius_max_m)
            )
            self.collision_offset_relative_angle_deg = float(
                rng.uniform(
                    args.offset_relative_angle_min_deg,
                    args.offset_relative_angle_max_deg,
                )
            )
            pickup_center_xy = np.asarray(
                env.layout["pickup_gear_center_position_m"][:2], dtype=np.float64
            )
            target_center_xy = np.asarray(
                env.layout["target_medium_shaft_position_m"][:2], dtype=np.float64
            )
            center_direction_xy = target_center_xy - pickup_center_xy
            if float(np.linalg.norm(center_direction_xy)) <= 1.0e-9:
                raise RuntimeError("pickup and target centres cannot coincide")
            reference_angle_rad = math.atan2(
                float(center_direction_xy[1]), float(center_direction_xy[0])
            )
            world_angle_rad = wrap_angle(
                reference_angle_rad
                + math.radians(self.collision_offset_relative_angle_deg)
            )
            self.collision_offset_reference_angle_deg = math.degrees(reference_angle_rad)
            self.collision_offset_world_angle_deg = math.degrees(world_angle_rad)
            self.collision_offset_xy_m = np.asarray(
                [
                    self.collision_offset_radius_m * math.cos(world_angle_rad),
                    self.collision_offset_radius_m * math.sin(world_angle_rad),
                ],
                dtype=np.float32,
            )
            print(
                "CORIBENCH_GEAR_MESH_CONTACT_OFFSET="
                + json.dumps(
                    {
                        "radius_m": self.collision_offset_radius_m,
                        "reference_angle_deg": self.collision_offset_reference_angle_deg,
                        "relative_angle_deg": self.collision_offset_relative_angle_deg,
                        "world_angle_deg": self.collision_offset_world_angle_deg,
                        "offset_xy_m": self.collision_offset_xy_m.tolist(),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

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

    def _target_quaternion(self, yaw_rad: float) -> torch.Tensor:
        return math_utils.quat_from_euler_xyz(
            torch.tensor([DOWN_ROLL], dtype=torch.float32, device=self.env.device),
            torch.zeros(1, dtype=torch.float32, device=self.env.device),
            torch.tensor([yaw_rad], dtype=torch.float32, device=self.env.device),
        )

    def _force_norm_n(self) -> float:
        return float(torch.linalg.vector_norm(self.env.force_sensor_joint_raw[0, :3]).item())

    def _write_video_frame(self, phase: str, target_xyz: np.ndarray) -> None:
        if self.video is None:
            return
        image = rgb(self.env.table_rgb[0])[..., ::-1].copy()
        height, width = image.shape[:2]
        force = vector(self.env.force_sensor_joint_raw[0])
        actual = vector(self.env.fingertip_midpoint_pos[0])
        cv2.rectangle(image, (8, 8), (width - 8, 128), (18, 18, 18), -1)
        cv2.putText(
            image,
            f"GEAR MESH  {self.variant.upper()}  SEED {args.seed}",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (242, 242, 242),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            phase.replace("_", " ").upper(),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 210, 255) if "contact" in phase or "search" in phase else (120, 235, 150),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"RAW F [{force[0]:+5.1f} {force[1]:+5.1f} {force[2]:+5.1f}] N",
            (20, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (242, 242, 242),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"TCP z {actual[2]:.3f} -> {target_xyz[2]:.3f} m",
            (20, 111),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (242, 242, 242),
            1,
            cv2.LINE_AA,
        )
        self.video.write(image)

    def _record_and_step(
        self,
        target_xyz,
        yaw_rad: float,
        gripper: float,
        phase: str,
    ) -> None:
        xyz = np.asarray(target_xyz, dtype=np.float32)
        action = np.asarray(
            [*xyz, DOWN_ROLL, 0.0, yaw_rad, 2.0 * gripper - 1.0], dtype=np.float32
        )
        force_norm = self._force_norm_n()
        self.phase_force_samples_n.setdefault(phase, []).append(force_norm)
        self.phase_force_peaks_n[phase] = max(
            force_norm, self.phase_force_peaks_n.get(phase, 0.0)
        )
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
                torch.cat(
                    (self.env.fingertip_midpoint_pos[0], self.env.fingertip_midpoint_quat[0])
                )
            ),
            privileged=dict(self.task.privileged_observations(self.env)),
            phase_id=PHASE_SCHEMA[phase],
        )
        self._write_video_frame(phase, xyz)
        self.env.set_cartesian_rpy_target(action[:6], gripper)
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        self.buffer.set_last_transition_outcome(
            reward=float(reward[0]),
            success=False,
            terminated=bool(terminated[0]),
            truncated=bool(truncated[0]),
        )
        self.env.update_cameras()

    def _append_phase_range(self, phase: str, start: int) -> None:
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffer)})

    def move(
        self,
        target_xyz,
        yaw_rad: float,
        gripper: float,
        phase: str,
        *,
        max_frames: int = 240,
        position_tolerance_m: float = 0.004,
        rotation_tolerance_rad: float = 0.12,
        require_convergence: bool = True,
        minimum_frames: int = 0,
    ) -> None:
        target = np.asarray(target_xyz, dtype=np.float32)
        target_tensor = torch.as_tensor(
            target, dtype=torch.float32, device=self.env.device
        ).unsqueeze(0)
        target_quaternion = self._target_quaternion(yaw_rad)
        start = len(self.buffer)
        stable = 0
        position_error = float("inf")
        rotation_error = float("inf")
        for frame in range(max_frames):
            self._record_and_step(target, yaw_rad, gripper, phase)
            position_error = float(
                torch.linalg.vector_norm(
                    self.env.fingertip_midpoint_pos[0] - target_tensor[0]
                ).item()
            )
            _, rotation = compute_pose_error(
                self.env.fingertip_midpoint_pos,
                self.env.fingertip_midpoint_quat,
                target_tensor,
                target_quaternion,
                rot_error_type="axis_angle",
            )
            rotation_error = float(torch.linalg.vector_norm(rotation[0]).item())
            converged = (
                position_error <= position_tolerance_m
                and rotation_error <= rotation_tolerance_rad
            )
            stable = stable + 1 if converged else 0
            if frame + 1 >= minimum_frames and stable >= PROFILE["stable_frames"]:
                break
        self._append_phase_range(phase, start)
        if require_convergence and stable < PROFILE["stable_frames"]:
            raise RuntimeError(
                f"phase {phase} did not converge: position={position_error:.4f}, "
                f"rotation={rotation_error:.4f}"
            )
        print(
            f"CORIBENCH_GEAR_MESH_PHASE={phase},frames={len(self.buffer) - start},"
            f"position_error_m={position_error:.5f},rotation_error_rad={rotation_error:.5f}",
            flush=True,
        )

    def hold(self, frames: int, target_xyz, yaw_rad: float, gripper: float, phase: str) -> None:
        start = len(self.buffer)
        for _ in range(frames):
            self._record_and_step(target_xyz, yaw_rad, gripper, phase)
        self._append_phase_range(phase, start)

    def _gear_position(self) -> np.ndarray:
        position, _ = self.env.entity_local_pose("medium_gear")
        return vector(position[0])

    def _gear_yaw_rad(self) -> float:
        _, quaternion = self.env.entity_local_pose("medium_gear")
        _, _, yaw = math_utils.euler_xyz_from_quat(quaternion)
        return float(yaw[0].item())

    def pickup(self) -> None:
        pickup = np.asarray(self.env.layout["pickup_gear_center_position_m"], dtype=np.float32)
        grasp_xyz = np.asarray(
            [pickup[0], pickup[1], SPEC.pickup_grasp_tcp_z_m], dtype=np.float32
        )
        self.move(
            (pickup[0], pickup[1], 0.145),
            self.pickup_tcp_yaw_rad,
            1.0,
            "approach_pickup",
        )
        self.move(
            (pickup[0], pickup[1], 0.065),
            self.pickup_tcp_yaw_rad,
            1.0,
            "approach_pickup",
        )
        self.move(
            grasp_xyz,
            self.pickup_tcp_yaw_rad,
            1.0,
            "approach_pickup",
            position_tolerance_m=0.003,
        )
        self.hold(
            PROFILE["grasp_frames"],
            grasp_xyz,
            self.pickup_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "grasp_gear",
        )
        grasped = self._gear_position()
        self.gear_tcp_z_offset_m = float(
            self.env.fingertip_midpoint_pos[0, 2].item() - grasped[2]
        )
        self.hold(
            PROFILE["post_grasp_hold_frames"],
            grasp_xyz,
            self.pickup_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "grasp_gear",
        )
        self.move(
            (pickup[0], pickup[1], PROFILE["transport_height_m"]),
            self.pickup_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "lift_gear",
            position_tolerance_m=0.006,
        )
        lifted = self._gear_position()
        if lifted[2] < 0.085:
            raise RuntimeError(f"medium gear was not lifted: {lifted.tolist()}")

    def transport(self) -> tuple[np.ndarray, float]:
        pickup = np.asarray(self.env.layout["pickup_gear_center_position_m"], dtype=np.float32)
        target = np.asarray(self.env.layout["target_medium_shaft_position_m"], dtype=np.float32)
        midpoint = 0.5 * (pickup[:2] + target[:2])
        self.move(
            (pickup[0], pickup[1], PROFILE["transport_height_m"]),
            self.mesh_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "transport_above_fixture",
            position_tolerance_m=0.008,
        )
        self.move(
            (midpoint[0], midpoint[1], PROFILE["transport_height_m"] + 0.025),
            self.mesh_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "transport_above_fixture",
            position_tolerance_m=0.008,
        )
        self.move(
            (target[0], target[1], PROFILE["preinsert_height_m"]),
            self.mesh_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "transport_above_fixture",
            # Loaded differential IK retains a harmless 8--10 mm residual at
            # this collision-free staging pose.  Physical seating below is
            # checked from the gear state, not inferred from TCP convergence.
            position_tolerance_m=0.012,
        )
        seated_tcp_z = float(target[2] + self.gear_tcp_z_offset_m)
        return target, seated_tcp_z

    def center_gear_above_shaft(self, target: np.ndarray) -> np.ndarray:
        """Cancel loaded-TCP bias using physical gear pose at a collision-free height."""

        command_xy = target[:2].astype(np.float32).copy()
        center_error = float("inf")
        for iteration in range(8):
            gear = self._gear_position()
            error = target[:2] - gear[:2]
            center_error = float(np.linalg.norm(error))
            print(
                f"CORIBENCH_GEAR_MESH_CENTER_ITERATION={iteration},"
                f"gear_center_error_m={center_error:.6f},"
                f"command_xy_m={command_xy.tolist()}",
                flush=True,
            )
            if center_error <= 0.0015:
                break
            correction_norm = float(np.linalg.norm(error))
            if correction_norm > 0.015:
                error = error * (0.015 / correction_norm)
            command_xy += error
            self.move(
                (command_xy[0], command_xy[1], PROFILE["preinsert_height_m"]),
                self.mesh_tcp_yaw_rad,
                SPEC.grasp_gripper_open_fraction,
                "direct_alignment",
                max_frames=12,
                position_tolerance_m=0.004,
                rotation_tolerance_rad=0.12,
                require_convergence=False,
                minimum_frames=12,
            )
        gear = self._gear_position()
        center_error = float(np.linalg.norm(gear[:2] - target[:2]))
        if center_error > 0.0025:
            raise RuntimeError(
                f"gear center oracle correction did not converge: {center_error:.6f} m"
            )
        self.placement_tcp_command_xy_m = command_xy.copy()
        self.preinsert_gear_center_error_m = center_error
        return command_xy

    def recover_gear_pose_above_shaft(
        self, target: np.ndarray, command_xy: np.ndarray
    ) -> np.ndarray:
        """Re-establish XY and tooth phase after deliberate contact/slip."""

        command_xy = command_xy.astype(np.float32).copy()
        command_yaw = self.mesh_tcp_yaw_rad
        desired_gear_yaw = float(self.env.layout["target_base_yaw_rad"]) + math.radians(
            SPEC.expert_twist_alignment_offset_deg
        )
        for iteration in range(8):
            gear = self._gear_position()
            xy_error = target[:2] - gear[:2]
            center_error = float(np.linalg.norm(xy_error))
            yaw_error = wrap_angle(desired_gear_yaw - self._gear_yaw_rad())
            print(
                f"CORIBENCH_GEAR_MESH_RECOVER_POSE_ITERATION={iteration},"
                f"center_error_m={center_error:.6f},"
                f"yaw_error_deg={math.degrees(yaw_error):.4f}",
                flush=True,
            )
            if center_error <= 0.0020 and abs(yaw_error) <= math.radians(0.5):
                break
            correction_norm = float(np.linalg.norm(xy_error))
            if correction_norm > 0.012:
                xy_error = xy_error * (0.012 / correction_norm)
            command_xy += xy_error
            yaw_step = float(np.clip(yaw_error, -math.radians(3.0), math.radians(3.0)))
            command_yaw = wrap_angle(command_yaw + yaw_step)
            self.move(
                (command_xy[0], command_xy[1], PROFILE["preinsert_height_m"]),
                command_yaw,
                SPEC.grasp_gripper_open_fraction,
                "slow_search",
                max_frames=12,
                position_tolerance_m=0.004,
                rotation_tolerance_rad=0.12,
                require_convergence=False,
                minimum_frames=12,
            )
        gear = self._gear_position()
        center_error = float(np.linalg.norm(gear[:2] - target[:2]))
        yaw_error = wrap_angle(desired_gear_yaw - self._gear_yaw_rad())
        if center_error > 0.003 or abs(yaw_error) > math.radians(1.0):
            raise RuntimeError(
                "post-contact pose recovery did not converge: "
                f"center={center_error:.6f} m, yaw={math.degrees(yaw_error):.3f} deg"
            )
        self.mesh_tcp_yaw_rad = command_yaw
        self.seat_tcp_yaw_rad = command_yaw
        self.recovery_recentered = True
        return command_xy

    def direct_insert(
        self, target: np.ndarray, command_xy: np.ndarray, seated_tcp_z: float
    ) -> np.ndarray:
        return self.feedback_seat(target, command_xy, seated_tcp_z)

    def feedback_seat(
        self, target: np.ndarray, command_xy: np.ndarray, seated_tcp_z: float
    ) -> np.ndarray:
        """Descend while cancelling lateral deflection from shaft/tooth contact."""

        command_xy = command_xy.astype(np.float32).copy()
        command_z = float(PROFILE["direct_preseat_height_m"] + 0.010)
        command_yaw = self.mesh_tcp_yaw_rad
        twist_index = 0
        for iteration in range(40):
            gear = self._gear_position()
            xy_error = target[:2] - gear[:2]
            center_error = float(np.linalg.norm(xy_error))
            height_error = float(gear[2] - target[2])
            print(
                f"CORIBENCH_GEAR_MESH_SEAT_ITERATION={iteration},"
                f"gear_center_error_m={center_error:.6f},"
                f"gear_height_error_m={height_error:.6f},"
                f"command_z_m={command_z:.6f}",
                flush=True,
            )
            if center_error <= 0.0035 and abs(height_error) <= SPEC.max_height_error_m:
                break
            release_iteration = 8 if (self.variant == "direct" or self.recovery_recentered) else 15
            if (
                iteration >= release_iteration
                and command_z <= float(seated_tcp_z) + 1.0e-6
                and center_error <= 0.0025
                and height_error > 0.012
            ):
                # The Panda fingertips become the lowest obstruction once the
                # hole is centered over the shaft.  Open them and let gravity
                # complete the last few centimetres of the physical mesh.
                self.hold(
                    36,
                    (command_xy[0], command_xy[1], command_z),
                    command_yaw,
                    1.0,
                    "release_gear",
                )
                self.released_during_seating = True
                released_gear = self._gear_position()
                released_center_error = float(
                    np.linalg.norm(released_gear[:2] - target[:2])
                )
                released_height_error = float(released_gear[2] - target[2])
                if (
                    released_center_error <= SPEC.max_center_error_m
                    and abs(released_height_error) <= SPEC.max_height_error_m
                ):
                    break
                raise RuntimeError(
                    "aligned gravity release did not seat gear: "
                    f"gear={released_gear.tolist()}, target={target.tolist()}"
                )
            correction_norm = float(np.linalg.norm(xy_error))
            if correction_norm > 0.012:
                xy_error = xy_error * (0.012 / correction_norm)
            command_xy += xy_error
            # Only push farther down after the hole is laterally close to the
            # shaft.  Otherwise hold height and use the contact as a guide.
            if center_error <= 0.0045:
                command_z = max(float(seated_tcp_z), command_z - 0.005)
            if (
                command_z <= float(seated_tcp_z) + 1.0e-6
                and center_error <= 0.0045
                and height_error > 0.012
            ):
                twist_offsets = PROFILE["twist_search_offsets_deg"]
                command_yaw = wrap_angle(
                    self.mesh_tcp_yaw_rad
                    + math.radians(twist_offsets[twist_index % len(twist_offsets)])
                )
                twist_index += 1
            self.move(
                (command_xy[0], command_xy[1], command_z),
                command_yaw,
                SPEC.grasp_gripper_open_fraction,
                "final_seating",
                max_frames=10,
                position_tolerance_m=0.003,
                rotation_tolerance_rad=0.12,
                require_convergence=False,
                minimum_frames=10,
            )
        gear = self._gear_position()
        center_error = float(np.linalg.norm(gear[:2] - target[:2]))
        height_error = float(gear[2] - target[2])
        if center_error > 0.006 or abs(height_error) > 0.012:
            raise RuntimeError(
                "feedback seating did not converge: "
                f"gear={gear.tolist()}, target={target.tolist()}"
            )
        self.placement_tcp_command_xy_m = command_xy.copy()
        self.seat_tcp_yaw_rad = command_yaw
        return command_xy

    def contact_recovery_insert(
        self, target: np.ndarray, command_xy: np.ndarray, seated_tcp_z: float
    ) -> np.ndarray:
        offset_xy = command_xy + self.collision_offset_xy_m
        self.move(
            (offset_xy[0], offset_xy[1], 0.090),
            self.mesh_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "offset_contact",
            position_tolerance_m=0.012,
        )
        lift_samples = self.phase_force_samples_n.get("lift_gear", [])
        self.precontact_force_n = float(np.median(lift_samples[-12:])) if lift_samples else 0.0
        contact_tcp_z = float(seated_tcp_z + 0.025)
        self.move(
            (offset_xy[0], offset_xy[1], contact_tcp_z),
            self.mesh_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "offset_contact",
            max_frames=PROFILE["contact_command_frames"],
            position_tolerance_m=0.002,
            rotation_tolerance_rad=0.10,
            require_convergence=False,
            minimum_frames=PROFILE["contact_command_frames"],
        )
        self.collision_force_peak_n = self.phase_force_peaks_n.get("offset_contact", 0.0)

        unit = self.collision_offset_xy_m / np.linalg.norm(self.collision_offset_xy_m)
        perpendicular = np.asarray([-unit[1], unit[0]], dtype=np.float32)
        search_offsets = (
            0.90 * self.collision_offset_xy_m,
            0.70 * np.linalg.norm(self.collision_offset_xy_m) * perpendicular,
            -0.55 * self.collision_offset_xy_m,
            -0.38 * np.linalg.norm(self.collision_offset_xy_m) * perpendicular,
            0.24 * self.collision_offset_xy_m,
            np.zeros(2, dtype=np.float32),
        )
        for index, offset in enumerate(search_offsets):
            progress = index / max(len(search_offsets) - 1, 1)
            z = contact_tcp_z - progress * 0.008
            self.move(
                (command_xy[0] + offset[0], command_xy[1] + offset[1], z),
                self.mesh_tcp_yaw_rad,
                SPEC.grasp_gripper_open_fraction,
                "slow_search",
                max_frames=PROFILE["search_waypoint_frames"],
                position_tolerance_m=0.003,
                rotation_tolerance_rad=0.12,
                require_convergence=False,
                minimum_frames=PROFILE["search_waypoint_frames"],
            )
        self.move(
            (command_xy[0], command_xy[1], PROFILE["preinsert_height_m"]),
            self.mesh_tcp_yaw_rad,
            SPEC.grasp_gripper_open_fraction,
            "slow_search",
            position_tolerance_m=0.012,
        )
        command_xy = self.recover_gear_pose_above_shaft(target, command_xy)
        return self.feedback_seat(target, command_xy, seated_tcp_z)

    def release_and_retract(
        self, target: np.ndarray, command_xy: np.ndarray, seated_tcp_z: float
    ) -> None:
        seat_target = (command_xy[0], command_xy[1], seated_tcp_z)
        seat_gripper = 1.0 if self.released_during_seating else SPEC.grasp_gripper_open_fraction
        self.hold(
            PROFILE["seat_hold_frames"],
            seat_target,
            self.seat_tcp_yaw_rad,
            seat_gripper,
            "final_seating",
        )
        gear_before_release = self._gear_position()
        center_error = float(np.linalg.norm(gear_before_release[:2] - target[:2]))
        if center_error > 0.008 or abs(float(gear_before_release[2] - target[2])) > 0.014:
            raise RuntimeError(
                "gear not plausibly seated before release: "
                f"position={gear_before_release.tolist()}, target={target.tolist()}"
            )
        self.hold(
            PROFILE["release_frames"],
            seat_target,
            self.seat_tcp_yaw_rad,
            1.0,
            "release_gear",
        )
        self.move(
            (target[0], target[1], 0.165),
            self.seat_tcp_yaw_rad,
            1.0,
            "retract_and_settle",
            position_tolerance_m=0.006,
        )
        self.hold(
            PROFILE["settle_frames"],
            (target[0], target[1], 0.165),
            self.seat_tcp_yaw_rad,
            1.0,
            "retract_and_settle",
        )

    def run(self) -> dict[str, object]:
        start_xyz = vector(self.env.fingertip_midpoint_pos[0])
        start_rpy = current_state(self.env)[3:6]
        self.hold(
            PROFILE["observe_frames"],
            start_xyz,
            float(start_rpy[2]),
            1.0,
            "observe_layout",
        )
        self.pickup()
        target, seated_tcp_z = self.transport()
        command_xy = self.center_gear_above_shaft(target)
        if self.variant == "direct":
            command_xy = self.direct_insert(target, command_xy, seated_tcp_z)
        else:
            command_xy = self.contact_recovery_insert(target, command_xy, seated_tcp_z)
        self.release_and_retract(target, command_xy, seated_tcp_z)

        report = self.env.evaluate_final_success()
        if not bool(report["success"]):
            raise RuntimeError(
                "final success check failed: " + json.dumps(report, separators=(",", ":"))
            )
        self.buffer.set_last_transition_outcome(
            reward=1.0,
            success=True,
            terminated=False,
            truncated=False,
        )
        return {
            "report": report,
            "variant": self.variant,
            "pickup_position_m": self.env.layout["pickup_gear_center_position_m"],
            "fixture_position_m": self.env.layout["target_base_position_m"],
            "target_shaft_position_m": target.tolist(),
            "gear_tcp_z_offset_m": self.gear_tcp_z_offset_m,
            "pickup_tcp_yaw_rad": self.pickup_tcp_yaw_rad,
            "mesh_tcp_yaw_rad": self.mesh_tcp_yaw_rad,
            "seat_tcp_yaw_rad": self.seat_tcp_yaw_rad,
            "seat_yaw_correction_deg": math.degrees(
                wrap_angle(self.seat_tcp_yaw_rad - self.mesh_tcp_yaw_rad)
            ),
            "placement_tcp_command_xy_m": self.placement_tcp_command_xy_m.tolist(),
            "preinsert_gear_center_error_m": self.preinsert_gear_center_error_m,
            "released_during_seating": self.released_during_seating,
            "recovery_recentered": self.recovery_recentered,
            "collision_command_offset_xy_m": self.collision_offset_xy_m.tolist()
            if self.variant == "contact_recovery"
            else [0.0, 0.0],
            "collision_offset_radius_m": self.collision_offset_radius_m,
            "collision_offset_reference_angle_deg": (
                self.collision_offset_reference_angle_deg
            ),
            "collision_offset_relative_angle_deg": (
                self.collision_offset_relative_angle_deg
            ),
            "collision_offset_world_angle_deg": self.collision_offset_world_angle_deg,
            "precontact_force_norm_n": self.precontact_force_n
            if self.variant == "contact_recovery"
            else None,
            "collision_force_peak_norm_n": self.collision_force_peak_n
            if self.variant == "contact_recovery"
            else None,
            "phase_force_peaks_n": self.phase_force_peaks_n,
            "phase_ranges_control_frames": self.phase_ranges,
        }


task = load_task("gear_mesh", ASSET_DIR)
cfg = CoRiBenchEnvCfg()
cfg.seed = args.seed
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(
    cfg,
    device=args.device,
    physics_rate_hz=args.physics_rate_hz,
    solver_iterations=args.solver_position_iterations,
)
cfg.num_rerenders_on_reset = 1
env = CoRiBenchEnv(
    cfg=cfg,
    task_runtime=task,
    main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size,
    enable_cameras=True,
)
expert: GearMeshExpert | None = None

try:
    env.set_episode_seed(args.seed)
    env.reset(seed=args.seed)
    env.reset_cameras()
    for _ in range(3):
        env.sim.render()
    env.update_cameras()
    env.start_wrench_capture()
    expert = GearMeshExpert(env, task, args.video, args.variant)
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
        "task": "gear_mesh",
        "prompt": task.manifest.prompt,
        "success": True,
        "end_reason": "scripted_gear_mesh_final_check",
        "seed": args.seed,
        "initial_layout": env.layout,
        "final_success_report": result["report"],
        "control_rate_hz": 15.0,
        "physics_rate_hz": float(args.physics_rate_hz),
        "solver_position_iterations": args.solver_position_iterations,
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
        "scene_profile": env.scene_profile_name,
        "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
        "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
        "control_source": "scripted_physical_gear_mesh_smooth_position",
        "success_check_timing": "only_after_release_retract_and_settle",
        "object_state_overwrite": False,
        "robot_state_overwrite": False,
        "images_present": True,
        "image_shapes_hwc": {
            "main": [args.main_image_size, args.main_image_size, 3],
            "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
        },
        "phase_schema": PHASE_SCHEMA,
        "scripted_expert_variant": args.variant,
        "scripted_expert_generator_version": "gear_mesh_random_both_polar_offset_v2",
        "scripted_expert_profile": PROFILE,
        "scripted_expert_details": {
            key: value for key, value in result.items() if key != "report"
        },
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
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        "CORIBENCH_GEAR_MESH_EXPERT=" + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
except Exception as exc:
    print(f"CORIBENCH_GEAR_MESH_EXPERT_ERROR={type(exc).__name__}:{exc}", flush=True)
    raise
finally:
    if expert is not None:
        expert.close()
    env.close()
    if owns_simulation_app:
        simulation_app.close()
