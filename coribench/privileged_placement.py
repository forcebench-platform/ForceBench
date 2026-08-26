"""Reusable recorder for task-owned privileged-state assembly experts.

The teacher may read object and fixture poses, but all motion still goes
through the benchmark's SmoothPosition arm controller and PhysX contacts.  It
never teleports an object or overwrites robot state.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from isaaclab.utils import math as math_utils

from coribench.recording import EpisodeBuffer


DOWN_ROLL_RAD = math.pi


def _rgb(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().numpy()[..., :3].astype(np.uint8)


def _vector(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _wrap_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _yaw_from_quaternion(quaternion: torch.Tensor) -> float:
    _, _, yaw = math_utils.euler_xyz_from_quat(quaternion.reshape(1, 4))
    return float(yaw[0].item())


@dataclass(frozen=True)
class PrivilegedPlacementPlan:
    object_role: str
    pickup_tcp_z_m: float
    lift_tcp_z_m: float
    grasp_open_fraction: float
    desired_object_yaw_rad: float
    placement_clearances_m: tuple[float, ...]
    release_frames: int = 14
    retract_height_m: float = 0.11
    translation_speed_m_s: float = 0.09
    object_position_tolerance_m: float = 0.0015
    object_yaw_tolerance_rad: float = math.radians(1.0)

    def validate(self) -> None:
        if not self.object_role:
            raise ValueError("privileged placement object role cannot be empty")
        if self.pickup_tcp_z_m <= 0.0 or self.lift_tcp_z_m <= self.pickup_tcp_z_m:
            raise ValueError("lift TCP height must exceed positive pickup TCP height")
        if not 0.0 <= self.grasp_open_fraction <= 1.0:
            raise ValueError("grasp opening must lie in [0, 1]")
        if not self.placement_clearances_m or self.placement_clearances_m[-1] != 0.0:
            raise ValueError("placement clearances must end at the exact target")
        if any(value < 0.0 for value in self.placement_clearances_m):
            raise ValueError("placement clearances cannot be negative")


class PrivilegedPlacementEpisode:
    """Closed-loop object-pose teacher with canonical raw/EMA recording."""

    def __init__(self, env: Any, task: Any, plan: PrivilegedPlacementPlan):
        plan.validate()
        self.env = env
        self.task = task
        self.plan = plan
        self.buffer = EpisodeBuffer()
        self.zero_action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
        self.translation_step_m = plan.translation_speed_m_s / 15.0
        self.command_yaw_rad = 0.0
        self.phase_ranges: list[dict[str, int | str]] = []
        self.max_object_position_error_m: dict[str, float] = {}

    def _target_position(self) -> np.ndarray:
        target = self.task.wrench_reference_position(self.env)
        if isinstance(target, tuple):
            target = target[0]
        return _vector(target[0])

    def _object_pose(self) -> tuple[np.ndarray, float]:
        position, quaternion = self.env.entity_local_pose(self.plan.object_role)
        return _vector(position[0]), _yaw_from_quaternion(quaternion[0])

    def _slew_position(self, requested: np.ndarray) -> np.ndarray:
        current = _vector(self.env.fingertip_midpoint_pos[0])
        delta = np.asarray(requested, dtype=np.float32) - current
        distance = float(np.linalg.vector_norm(delta))
        if distance <= self.translation_step_m:
            return np.asarray(requested, dtype=np.float32)
        return current + delta * np.float32(self.translation_step_m / distance)

    def _record_and_step(self, requested_position_m, gripper: float) -> None:
        command_position = self._slew_position(np.asarray(requested_position_m, dtype=np.float32))
        action = np.asarray(
            [
                *command_position,
                DOWN_ROLL_RAD,
                0.0,
                self.command_yaw_rad,
                2.0 * float(gripper) - 1.0,
            ],
            dtype=np.float32,
        )
        self.buffer.append(
            state=_vector(self.env.policy_state[0]),
            action=action,
            wrench_raw=_vector(self.env.force_sensor_joint_raw[0]),
            wrench_ema=_vector(self.env.force_sensor_joint_smooth[0]),
            timestamp=(len(self.buffer) + 1) * self.env.step_dt,
            image=_rgb(self.env.table_rgb[0]),
            wrist_image=_rgb(self.env.wrist_rgb[0]),
            joint_position=_vector(self.env.joint_pos[0]),
            joint_velocity=_vector(self.env.robot.data.joint_vel.torch[0]),
            tcp_pose=_vector(
                torch.cat(
                    (
                        self.env.fingertip_midpoint_pos[0],
                        self.env.fingertip_midpoint_quat[0],
                    )
                )
            ),
            privileged=dict(self.task.privileged_observations(self.env)),
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

    def _phase(self, name: str, function) -> None:
        start = len(self.buffer)
        function()
        self.phase_ranges.append({"name": name, "start": start, "end": len(self.buffer)})

    def move_tcp(
        self,
        target_position_m,
        gripper: float,
        phase: str,
        max_frames: int = 220,
        tolerance_m: float = 0.0025,
    ) -> None:
        target = np.asarray(target_position_m, dtype=np.float32)
        if tolerance_m <= 0.0:
            raise ValueError("TCP position tolerance must be positive")

        def execute() -> None:
            last_position = _vector(self.env.fingertip_midpoint_pos[0])
            last_error = math.inf
            for frame in range(max_frames):
                self._record_and_step(target, gripper)
                last_position = _vector(self.env.fingertip_midpoint_pos[0])
                last_error = float(
                    torch.linalg.vector_norm(
                        self.env.fingertip_midpoint_pos[0]
                        - torch.as_tensor(target, dtype=torch.float32, device=self.env.device)
                    ).item()
                )
                if frame >= 4 and last_error <= tolerance_m:
                    return
            raise RuntimeError(
                f"phase {phase!r} did not converge to TCP target: "
                f"error={last_error:.4f} m target={target.tolist()} "
                f"actual={last_position.tolist()}"
            )

        self._phase(phase, execute)

    def hold(self, frames: int, gripper: float, phase: str) -> None:
        target = _vector(self.env.fingertip_midpoint_pos[0])
        self._phase(
            phase,
            lambda: [self._record_and_step(target, gripper) for _ in range(frames)],
        )

    def servo_object(self, target_position_m, phase: str, max_frames: int = 180) -> None:
        desired = np.asarray(target_position_m, dtype=np.float32)

        def execute() -> None:
            last_position_error = math.inf
            last_yaw_error = math.inf
            for frame in range(max_frames):
                object_position, object_yaw = self._object_pose()
                position_error = desired - object_position
                yaw_error = _wrap_pi(self.plan.desired_object_yaw_rad - object_yaw)
                self.command_yaw_rad += float(np.clip(yaw_error, -0.04, 0.04))
                target_eef = _vector(self.env.fingertip_midpoint_pos[0]) + position_error
                self._record_and_step(target_eef, self.plan.grasp_open_fraction)
                last_position_error = float(np.linalg.vector_norm(position_error))
                last_yaw_error = abs(yaw_error)
                self.max_object_position_error_m[phase] = max(
                    self.max_object_position_error_m.get(phase, 0.0),
                    last_position_error,
                )
                if (
                    frame >= 5
                    and last_position_error <= self.plan.object_position_tolerance_m
                    and last_yaw_error <= self.plan.object_yaw_tolerance_rad
                ):
                    return
            raise RuntimeError(
                f"phase {phase!r} failed privileged object servo: "
                f"position={last_position_error:.4f} m yaw={math.degrees(last_yaw_error):.2f} deg"
            )

        self._phase(phase, execute)

    def run(self) -> dict[str, object]:
        object_position, _ = self._object_pose()
        pickup_xy = object_position[:2]
        self.move_tcp([*pickup_xy, self.plan.pickup_tcp_z_m + 0.09], 1.0, "approach_pickup")
        self.move_tcp([*pickup_xy, self.plan.pickup_tcp_z_m], 1.0, "descend_to_grasp")
        self.hold(16, self.plan.grasp_open_fraction, "close_gripper")
        # Contact loads create a benign TCP tracking residual while an object
        # is pinched.  Gate this transit on the same broad controller tolerance
        # used by the sorting experts, then validate the measured object pose
        # below; placement itself remains millimetre-accurate object feedback.
        self.move_tcp(
            [*pickup_xy, self.plan.lift_tcp_z_m],
            self.plan.grasp_open_fraction,
            "lift_object",
            tolerance_m=0.050,
        )

        lifted_object, _ = self._object_pose()
        if lifted_object[2] <= object_position[2] + 0.035:
            raise RuntimeError(
                f"{self.plan.object_role} was not lifted: z={lifted_object[2]:.4f}"
            )
        eef_minus_object = _vector(self.env.fingertip_midpoint_pos[0]) - lifted_object
        target_object = self._target_position()
        transit = target_object + eef_minus_object
        transit[2] = self.plan.lift_tcp_z_m
        self.move_tcp(
            transit,
            self.plan.grasp_open_fraction,
            "transport_above_target",
            tolerance_m=0.050,
        )

        for clearance in self.plan.placement_clearances_m:
            desired = target_object.copy()
            desired[2] += clearance
            phase = "privileged_exact_place" if clearance == 0.0 else f"privileged_clearance_{clearance:.3f}m"
            self.servo_object(desired, phase)

        self.hold(8, self.plan.grasp_open_fraction, "placement_settle")
        self.hold(self.plan.release_frames, 1.0, "release")
        retract = _vector(self.env.fingertip_midpoint_pos[0])
        retract[2] += self.plan.retract_height_m
        self.move_tcp(retract, 1.0, "retract")
        self.hold(12, 1.0, "final_settle")

        report = self.env.evaluate_final_success()
        if not bool(report["success"]):
            raise RuntimeError("final success check failed: " + json.dumps(report, separators=(",", ":")))
        self.buffer.set_last_transition_outcome(
            reward=1.0,
            success=True,
            terminated=False,
            truncated=False,
        )
        final_object, final_yaw = self._object_pose()
        return {
            "report": report,
            "phase_ranges_control_frames": self.phase_ranges,
            "eef_minus_object_after_grasp_m": eef_minus_object.tolist(),
            "target_object_position_m": target_object.tolist(),
            "final_object_position_m": final_object.tolist(),
            "desired_object_yaw_rad": self.plan.desired_object_yaw_rad,
            "final_object_yaw_rad": final_yaw,
            "oracle_pose_feedback": True,
            "physics_intervention": False,
            "object_state_overwrite": False,
            "max_object_position_error_m": self.max_object_position_error_m,
        }
