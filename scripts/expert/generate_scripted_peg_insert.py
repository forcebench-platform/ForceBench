#!/usr/bin/env python3
"""Generate validated scripted-oracle peg-insertion demonstrations."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument(
    "--artifact_dir",
    type=Path,
    default=None,
    help="write summaries outside the formal HDF5 dataset directory",
)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--start_seed", type=int, default=1000)
parser.add_argument("--max_attempts_per_episode", type=int, default=3)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--physics_rate_hz", type=int, choices=(120,), default=120)
parser.add_argument("--solver_position_iterations", type=int, choices=(64,), default=64)
parser.add_argument("--final_settle_frames", type=int, default=8)
parser.add_argument(
    "--translation_speed_m_s",
    type=float,
    default=0.075,
    help="Maximum Cartesian target slew speed; 0.075 m/s is 5 mm at 15 Hz.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1:
    parser.error("--episodes must be positive")
if args.start_seed < 0 or args.max_attempts_per_episode < 1:
    parser.error("start_seed must be non-negative and max_attempts_per_episode must be positive")
if args.physics_rate_hz % 15:
    parser.error("physics rate must be divisible by the 15 Hz control rate")
if args.main_image_size < 16 or args.wrist_image_size < 16:
    parser.error("camera image sizes must be at least 16")
if not 0.0 < args.translation_speed_m_s <= 0.20:
    parser.error("translation speed must be in (0, 0.20] m/s")

launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import compute_pose_error

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task
from forcebench.tasks.peg_insert.spec import SPEC


DOWN_RPY = np.asarray([math.pi, 0.0, 0.0], dtype=np.float32)
FIXED_LIFT_TCP_Z_M = 0.18
# The current contact geometry settles with a repeatable ~0.676 mm lateral
# residual late in insertion. This remains far inside the task's formal 3 mm
# success bound; the stage gate only decides when to continue servoing.
INSERT_STAGE_XY_TOLERANCE_M = 0.00075


def vector(value) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1).astype(np.float32)


def rgb(value) -> np.ndarray:
    image = value.detach().cpu().numpy()
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"invalid RGB buffer: {image.shape}")
    return image[..., :3].astype(np.uint8)


def current_state(env: ForceBenchEnv) -> np.ndarray:
    return vector(env.policy_state[0])


class RuleEpisode:
    def __init__(self, env: ForceBenchEnv, task):
        self.env = env
        self.task = task
        self.buffer = EpisodeBuffer()
        self.zero_action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
        self.down_quaternion = math_utils.quat_from_euler_xyz(
            torch.as_tensor([DOWN_RPY[0]], device=env.device),
            torch.as_tensor([DOWN_RPY[1]], device=env.device),
            torch.as_tensor([DOWN_RPY[2]], device=env.device),
        )
        self.phase_ranges: list[dict[str, int | str]] = []
        self.phase_force_peaks: dict[str, float] = {}
        self.privileged_terminal_teacher = False
        self.insertion_stopped_on_success_bounds = False
        self.translation_step_m = float(args.translation_speed_m_s / 15.0)

    def _slew_limited_target(self, requested_position_m: np.ndarray) -> np.ndarray:
        requested = np.asarray(requested_position_m, dtype=np.float32)
        current = vector(self.env.fingertip_midpoint_pos[0])
        delta = requested - current
        distance = float(np.linalg.vector_norm(delta))
        if distance <= self.translation_step_m:
            return requested
        return current + delta * np.float32(self.translation_step_m / distance)

    def _pose_error(self, position_m: np.ndarray) -> tuple[float, float]:
        target_position = torch.as_tensor(position_m, dtype=torch.float32, device=self.env.device).unsqueeze(0)
        position_error, rotation_error = compute_pose_error(
            self.env.fingertip_midpoint_pos,
            self.env.fingertip_midpoint_quat,
            target_position,
            self.down_quaternion,
            rot_error_type="axis_angle",
        )
        position_norm = float(torch.linalg.vector_norm(position_error[0]).item())
        rotation_norm = float(torch.linalg.vector_norm(rotation_error[0]).item())
        return position_norm, rotation_norm

    def _record_and_step(
        self,
        target_position_m: np.ndarray,
        gripper: float,
        phase: str,
    ) -> float:
        requested_position = np.asarray(target_position_m, dtype=np.float32)
        command_position = self._slew_limited_target(requested_position)
        position_error, rotation_error = self._pose_error(requested_position)
        state = current_state(self.env)
        action = np.asarray(
            [*command_position, *DOWN_RPY, 2.0 * gripper - 1.0],
            dtype=np.float32,
        )
        self.buffer.append(
            state=state,
            action=action,
            wrench_raw=vector(self.env.force_sensor_joint_raw[0]),
            wrench_ema=vector(self.env.force_sensor_joint_smooth[0]),
            timestamp=(len(self.buffer) + 1) * self.env.step_dt,
            image=rgb(self.env.table_rgb[0]),
            wrist_image=rgb(self.env.wrist_rgb[0]),
            joint_position=vector(self.env.joint_pos[0]),
            joint_velocity=vector(self.env.robot.data.joint_vel.torch[0]),
            tcp_pose=vector(
                torch.cat((self.env.fingertip_midpoint_pos[0], self.env.fingertip_midpoint_quat[0]))
            ),
            privileged=dict(self.task.privileged_observations(self.env)),
        )
        self.env.set_cartesian_rpy_target(action[:6], gripper)
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        self.buffer.set_last_transition_outcome(
            reward=float(reward[0]), success=False, terminated=bool(terminated[0]), truncated=bool(truncated[0])
        )
        self.env.update_cameras()
        force = float(torch.linalg.vector_norm(self.env.force_sensor_smooth[0, :3]).item())
        self.phase_force_peaks[phase] = max(self.phase_force_peaks.get(phase, 0.0), force)
        return max(position_error, rotation_error)

    def move_pose(
        self,
        position_m,
        gripper: float,
        phase: str,
        *,
        max_frames: int = 240,
        min_frames: int = 4,
        position_tolerance_m: float = 0.0025,
        rotation_tolerance_rad: float = 0.04,
    ) -> None:
        target = np.asarray(position_m, dtype=np.float32)
        start = len(self.buffer)
        converged = False
        for frame in range(max_frames):
            self._record_and_step(target, gripper, phase)
            position_error = float(
                torch.linalg.vector_norm(
                    self.env.fingertip_midpoint_pos[0]
                    - torch.as_tensor(target, dtype=torch.float32, device=self.env.device)
                ).item()
            )
            _, rotation_error = compute_pose_error(
                self.env.fingertip_midpoint_pos,
                self.env.fingertip_midpoint_quat,
                torch.as_tensor(target, dtype=torch.float32, device=self.env.device).unsqueeze(0),
                self.down_quaternion,
                rot_error_type="axis_angle",
            )
            rotation_error_norm = float(torch.linalg.vector_norm(rotation_error[0]).item())
            if frame + 1 >= min_frames and position_error <= position_tolerance_m and rotation_error_norm <= rotation_tolerance_rad:
                converged = True
                break
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffer)})
        if not converged:
            raise RuntimeError(
                f"phase {phase!r} did not converge: position={position_error:.4f} m, "
                f"rotation={rotation_error_norm:.4f} rad"
            )

    def hold(self, frames: int, gripper: float, phase: str, position_m=None) -> None:
        target = (
            vector(self.env.fingertip_midpoint_pos[0])
            if position_m is None
            else np.asarray(position_m, dtype=np.float32)
        )
        start = len(self.buffer)
        for _ in range(frames):
            self._record_and_step(target, gripper, phase)
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffer)})

    def servo_peg_pose(
        self,
        hole_position_m,
        peg_clearance_m: float,
        phase: str,
        *,
        max_frames: int = 100,
        xy_tolerance_m: float = 0.0012,
        z_tolerance_m: float = 0.0020,
    ) -> None:
        """Close the loop on the peg pose, compensating for slip inside the grasp."""
        hole = np.asarray(hole_position_m, dtype=np.float32)
        desired_peg = hole.copy()
        desired_peg[2] = hole[2] + float(peg_clearance_m)
        start = len(self.buffer)
        converged = False
        peg_error = np.full(3, np.inf, dtype=np.float32)
        for frame in range(max_frames):
            peg_position, _ = self.env.entity_local_pose("peg")
            peg = vector(peg_position[0])
            eef = vector(self.env.fingertip_midpoint_pos[0])
            peg_error = desired_peg - peg
            target_eef = eef + peg_error
            self._record_and_step(target_eef, 0.0, phase)
            if (
                frame >= 5
                and float(np.linalg.vector_norm(peg_error[:2])) <= xy_tolerance_m
                and abs(float(peg_error[2])) <= z_tolerance_m
            ):
                converged = True
                break
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffer)})
        if not converged:
            peg_position, _ = self.env.entity_local_pose("peg")
            target_position, _ = self.env.entity_local_pose("target_hole")
            detail = {
                "phase": phase,
                "desired_peg_m": desired_peg.tolist(),
                "peg_m": vector(peg_position[0]).tolist(),
                "target_hole_m": vector(target_position[0]).tolist(),
                "eef_m": vector(self.env.fingertip_midpoint_pos[0]).tolist(),
                "peg_error_m": peg_error.tolist(),
                "force_peak_n": self.phase_force_peaks.get(phase, 0.0),
            }
            raise RuntimeError("peg pose servo did not converge: " + json.dumps(detail, separators=(",", ":")))

    def progressive_insert(self, hole_position_m, start_clearance_m: float) -> None:
        """Insert in 2 mm stages, re-centering the physical peg before every descent."""
        hole = np.asarray(hole_position_m, dtype=np.float32)
        self.servo_peg_pose(
            hole,
            start_clearance_m,
            "precision_centering",
            max_frames=100,
            xy_tolerance_m=0.0005,
            z_tolerance_m=0.001,
        )
        clearances = np.arange(start_clearance_m - 0.002, -0.001, -0.002, dtype=np.float32)
        if len(clearances) == 0 or float(clearances[-1]) > 0.0001:
            clearances = np.append(clearances, np.float32(0.0))
        for index, clearance in enumerate(clearances, start=1):
            self.servo_peg_pose(
                hole,
                max(float(clearance), 0.0),
                f"insert_stage_{index:02d}",
                max_frames=35,
                xy_tolerance_m=INSERT_STAGE_XY_TOLERANCE_M,
                z_tolerance_m=0.001,
            )
            peg_position, _ = self.env.entity_local_pose("peg")
            peg = vector(peg_position[0])
            center_error_m = float(np.linalg.vector_norm(peg[:2] - hole[:2]))
            mate_z_m = float(hole[2] + SPEC.hole_mate_height_m)
            height_error_m = abs(float(peg[2]) - mate_z_m)
            if (
                center_error_m <= SPEC.max_center_error_m
                and height_error_m <= SPEC.max_base_height_error_m
            ):
                self.insertion_stopped_on_success_bounds = True
                break


    def run(self) -> dict[str, object]:
        pickup, _ = self.env.entity_local_pose("peg")
        pickup_xy = vector(pickup[0])[:2]

        self.move_pose([pickup_xy[0], pickup_xy[1], 0.12], 1.0, "approach_pickup")
        self.move_pose([pickup_xy[0], pickup_xy[1], 0.045], 1.0, "descend_to_grasp")
        self.hold(14, 0.0, "close_gripper")
        self.move_pose(
            [pickup_xy[0], pickup_xy[1], FIXED_LIFT_TCP_Z_M],
            0.0,
            "lift_peg",
        )

        peg_position, _ = self.env.entity_local_pose("peg")
        peg_after_lift = vector(peg_position[0])
        if peg_after_lift[2] < 0.07:
            raise RuntimeError(f"peg was not lifted: z={peg_after_lift[2]:.4f}")
        eef_minus_peg = vector(self.env.fingertip_midpoint_pos[0]) - peg_after_lift
        target_position, _ = self.env.entity_local_pose("target_hole")
        target_hole = vector(target_position[0])

        transit = target_hole + eef_minus_peg
        transit[2] = FIXED_LIFT_TCP_Z_M
        self.move_pose(transit, 0.0, "transport_above_target")
        preinsert = target_hole + eef_minus_peg
        preinsert[2] = target_hole[2] + eef_minus_peg[2] + 0.038
        self.move_pose(preinsert, 0.0, "prealign")
        self.servo_peg_pose(
            target_hole,
            0.038,
            "peg_centering",
            xy_tolerance_m=0.0006,
            z_tolerance_m=0.0012,
        )

        final_hole = target_hole

        # The controller may read simulator peg/hole poses as an oracle, but it
        # never teleports the peg or overwrites arm state.  The complete
        # insertion therefore runs through smooth arm drives and PhysX contact.
        self.progressive_insert(final_hole, 0.032)
        inserted = vector(self.env.fingertip_midpoint_pos[0])
        self.hold(8, 0.0, "insertion_settle", inserted)
        self.hold(12, 1.0, "release", inserted)
        retract = inserted.copy()
        retract[2] += 0.10
        self.move_pose(retract, 1.0, "final_retract", max_frames=180)
        self.hold(args.final_settle_frames, 1.0, "final_settle", retract)

        report = self.env.evaluate_final_success()
        if not bool(report["success"]):
            raise RuntimeError("final success check failed: " + json.dumps(report, separators=(",", ":")))
        self.buffer.set_last_transition_outcome(
            reward=1.0, success=True, terminated=False, truncated=False
        )
        return {
            "report": report,
            "eef_minus_peg_after_grasp_m": eef_minus_peg.tolist(),
            "phase_ranges_control_frames": self.phase_ranges,
            "phase_force_peaks_n": self.phase_force_peaks,
            "privileged_terminal_teacher": self.privileged_terminal_teacher,
            "oracle_pose_feedback": True,
            "physics_intervention": False,
            "insertion_stopped_on_success_bounds": self.insertion_stopped_on_success_bounds,
        }


task = load_task("peg_insert", ASSET_DIR)
cfg = ForceBenchEnvCfg()
cfg.seed = args.start_seed
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(
    cfg,
    device=args.device,
    physics_rate_hz=args.physics_rate_hz,
    solver_iterations=args.solver_position_iterations,
)
cfg.num_rerenders_on_reset = 3

env = ForceBenchEnv(
    cfg=cfg,
    task_runtime=task,
    main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size,
    enable_cameras=True,
)
writer = HDF5EpisodeWriter(args.output_dir)
artifact_dir = args.artifact_dir or args.output_dir
artifact_dir.mkdir(parents=True, exist_ok=True)
saved: list[dict[str, object]] = []
failures: list[dict[str, object]] = []

print("FORCEBENCH_SCRIPTED_EXPERT_READY=1", flush=True)
print(f"FORCEBENCH_SCRIPTED_EXPERT_OUTPUT={args.output_dir.resolve()}", flush=True)
print(f"FORCEBENCH_SCRIPTED_EXPERT_PLAN=episodes:{args.episodes},trajectory:privileged_exact", flush=True)

try:
    for episode_slot in range(args.episodes):
        seed = args.start_seed + episode_slot
        completed = False
        for attempt in range(args.max_attempts_per_episode):
            env.set_episode_seed(seed)
            env.reset(seed=seed)
            fixed_arm = env.joint_pos[0, :7].clone()
            env.restore_fixed_arm_pose(fixed_arm)
            env.reset_cameras()
            for _ in range(3):
                env.sim.render()
            env.update_cameras()
            env.start_wrench_capture()
            episode = RuleEpisode(env, task)
            try:
                result = episode.run()
                high_rate = env.stop_wrench_capture()
                task_metadata = dict(task.episode_metadata())
                asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
                asset_paths.update(env.scene_asset_paths)
                asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
                metadata = {
                    "task": "peg_insert",
                    "prompt": task.manifest.prompt,
                    "success": True,
                    "end_reason": "scripted_expert_final_check",
                    "seed": seed,
                    "initial_layout": env.layout,
                    "final_success_report": result["report"],
                    "control_rate_hz": 15.0,
                    "physics_rate_hz": float(args.physics_rate_hz),
                    "arm_control_mode": env.arm_control_mode,
                    "arm_controller_profile": env.arm_controller_profile,
                    "scene_profile": env.scene_profile_name,
                    "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
                    "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
                    "success_check_timing": "only_after_scripted_trajectory_complete",
                    "control_source": "scripted_pose_oracle",
                    "images_present": True,
                    "image_shapes_hwc": {
                        "main": [args.main_image_size, args.main_image_size, 3],
                        "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
                    },
                    "scripted_expert_generator_version": "peg_privileged_pose_closed_loop",
                    "fixed_lift_tcp_z_m": FIXED_LIFT_TCP_Z_M,
                    "translation_speed_limit_m_s": float(args.translation_speed_m_s),
                    "translation_target_step_m_per_control_frame": float(
                        args.translation_speed_m_s / 15.0
                    ),
                    "layout_randomization_contract": {
                        "randomized": ["target_hole_position_m.x", "target_hole_position_m.y"],
                        "fixed": [
                            "pickup_hole_position_m",
                            "peg_position_m",
                            "target_hole_position_m.z",
                            "target_hole_yaw_rad",
                            "peg_upright_yaw_rad",
                        ],
                    },
                    "scripted_expert_attempt": attempt + 1,
                    "privileged_terminal_teacher": bool(result["privileged_terminal_teacher"]),
                    "physics_intervention": bool(result["physics_intervention"]),
                    "scripted_expert_details": {
                        key: value for key, value in result.items() if key != "report"
                    },
                    "scene_roles": list(env.entity_roles),
                    "asset_usd_paths": asset_paths,
                    **task_metadata,
                }
                path = writer.write(episode.buffer, metadata, high_rate=high_rate)[0]
                validation = validate_hdf5_episode(path)
                record = {
                    "path": str(path.resolve()),
                    "seed": seed,
                    "frames": len(episode.buffer),
                    "attempt": attempt + 1,
                    "validation": validation,
                }
                saved.append(record)
                print(
                    "FORCEBENCH_SCRIPTED_EXPERT_SAVED="
                    + json.dumps(record, separators=(",", ":")),
                    flush=True,
                )
                completed = True
                break
            except Exception as exc:
                failure = {
                    "seed": seed,
                    "attempt": attempt + 1,
                    "error": str(exc),
                }
                env.discard_wrench_capture()
                failures.append(failure)
                print(
                    "FORCEBENCH_SCRIPTED_EXPERT_RETRY="
                    + json.dumps(failure, separators=(",", ":")),
                    flush=True,
                )
        if not completed:
            raise RuntimeError(
                f"failed to generate seed {seed} after {args.max_attempts_per_episode} attempts"
            )

    summary = {
        "requested_episodes": args.episodes,
        "saved_episodes": len(saved),
        "start_seed": args.start_seed,
        "output_dir": str(args.output_dir.resolve()),
        "episodes": saved,
        "failed_attempts": failures,
    }
    summary_name = f"generation_seed_{args.start_seed:03d}.json"
    summary_path = artifact_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "FORCEBENCH_SCRIPTED_EXPERT_SUMMARY="
        + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
finally:
    env.close()
    simulation_app.close()

