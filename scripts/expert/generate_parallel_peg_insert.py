#!/usr/bin/env python3
"""Collect 32 complete peg_insert demonstrations in one cloned Isaac process."""

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
parser.add_argument("--final_settle_frames", type=int, default=8)
parser.add_argument("--translation_speed_m_s", type=float, default=0.075)
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
    parser.error("the default peg_insert collector requires exactly 32 cloned workspaces")
if not WRITE_SEEDS.issubset(SEEDS):
    parser.error("--write_seeds must be a subset of --seeds")
if args.env_spacing < 5.5:
    parser.error("env_spacing must be at least 5.5 m")
if not 0.0 < args.translation_speed_m_s <= 0.20:
    parser.error("translation speed must be in (0, 0.20] m/s")

os.environ.setdefault("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
os.environ["FORCEBENCH_TASK"] = "peg_insert"
os.environ.setdefault("FORCEBENCH_EXPERT_BACKEND", "scripted")

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
from forcebench.tasks.peg_insert.spec import (
    SPEC,
    evaluate_final_state,
    evaluate_release_stability_batch,
)


DOWN_RPY = np.asarray([math.pi, 0.0, 0.0], dtype=np.float32)
FIXED_LIFT_TCP_Z_M = 0.18
INSERT_STAGE_XY_TOLERANCE_M = 0.00075
FINAL_SEAT_COMMAND_CLEARANCE_M = 0.006
PRE_RELEASE_XY_TOLERANCE_M = 0.0008
# With the fingers closed, the gripper/fixture collision pair holds the peg root
# about 4.0--4.7 mm above its released mate height.  The peg is already engaged
# in the hole at this mechanical seat and drops the remaining distance on release.
PRE_RELEASE_HEIGHT_TOLERANCE_M = 0.0055
POST_RELEASE_HEIGHT_TOLERANCE_M = 0.0015
PRE_RELEASE_TILT_TOLERANCE_DEG = 5.0
PRE_RELEASE_MAX_LATERAL_FORCE_N = 0.15
PRE_RELEASE_SETTLE_FRAMES = 20
CLEAN_RELEASE_STEPS = 20
RELEASE_SETTLE_FRAMES = 20
SOURCE_TYPE_SCHEMA = {"expert": 0, "exploration": 1, "expert_recovery": 2}
EXPERT_SOURCE = SOURCE_TYPE_SCHEMA["expert"]
EXPLORATION_SOURCE = SOURCE_TYPE_SCHEMA["exploration"]
EXPERT_RECOVERY_SOURCE = SOURCE_TYPE_SCHEMA["expert_recovery"]
EXPLORATION_SAMPLER_VERSION = "seed_task_attempt_polar_v1"
EXPLORATION_ATTEMPTS = 1
# The asset pair is nominally 8 mm.  Keep the probe inside 75% of the
# benchmark's 3 mm center-error envelope while still producing visible rim
# contact at the 8 mm vertical clearance used below.
EXPLORATION_RADIUS_RANGE_M = (
    0.0008,
    min(0.0022, 0.75 * float(SPEC.max_center_error_m)),
)
EXPLORATION_CONTACT_CLEARANCE_M = 0.008
EXPLORATION_VERTICAL_DISENGAGE_M = 0.004
EXPLORATION_MASK = np.zeros(NUM_ENVS, dtype=bool)
PHASES = (
    "approach_pickup", "descend_to_grasp", "close_gripper", "lift_peg",
    "transport_above_target", "prealign", "peg_centering", "precision_centering",
    "exploration_probe", "expert_recovery_center", "insert_stage",
    "insertion_settle", "pre_release_centering", "pre_release_neutralize",
    "release", "release_settle", "final_retract", "final_settle",
)
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASES)}
L20_MIGRATION_SOURCE = {
    "host": "ubuntu22-l20a-08-zhangpei",
    "collector": "/home/ubuntu/forcevla_peg/code/tasks/blind_peg_insertion/isaac_collect_dataset.py",
    "collector_sha256": "ffd1113770d9fc930e67be322f56f8a70c417dc957da2b92f69db6b4d9dcbfda",
    "isaac_env": "/home/ubuntu/forcevla_peg/code/tasks/blind_peg_insertion/isaac_peg_env.py",
    "isaac_env_sha256": "98b7586b37ed9b0128dd84b30bb226a9c535b1f8feef7459de0bebe87e8f59a0",
    "policy_semantics_retained": "privileged peg/socket pose feedback after grasp and during contact",
    "schema_semantics_retained": "ForceBench complete pickup_transport_insert_release_retract",
}


def as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def rgb_batch(value) -> np.ndarray:
    return as_numpy(value)[..., :3].astype(np.uint8)


class ParallelPegExpert:
    def __init__(self, env: ForceBenchEnv, task, video_path: Path):
        self.env = env
        self.task = task
        self.buffers = [EpisodeBuffer() for _ in range(NUM_ENVS)]
        self.zero_action = torch.zeros((NUM_ENVS, 7), dtype=torch.float32, device=env.device)
        self.down_quaternion = math_utils.quat_from_euler_xyz(
            torch.full((NUM_ENVS,), math.pi, device=env.device),
            torch.zeros(NUM_ENVS, device=env.device),
            torch.zeros(NUM_ENVS, device=env.device),
        )
        self.translation_step_m = float(args.translation_speed_m_s / CONTROL_RATE_HZ)
        self.phase_ranges: list[dict[str, int | str]] = []
        self.phase_force_peaks = [dict() for _ in range(NUM_ENVS)]
        self.failures = [[] for _ in range(NUM_ENVS)]
        self.warnings = [[] for _ in range(NUM_ENVS)]
        self.insertion_stopped_on_success_bounds = np.zeros(NUM_ENVS, dtype=bool)
        self.pre_release_valid = np.zeros(NUM_ENVS, dtype=bool)
        self.pre_release_lateral_force_n = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.seat_clearance_m = np.full(NUM_ENVS, np.nan, dtype=np.float32)
        self.eef_minus_peg_after_grasp = np.zeros((NUM_ENVS, 3), dtype=np.float32)
        self.exploration_mask = EXPLORATION_MASK.copy()
        self.exploration_probe_samples: list[list[dict[str, float | int]]] = [
            [] for _ in range(NUM_ENVS)
        ]
        for env_id, seed in enumerate(SEEDS):
            for attempt in range(EXPLORATION_ATTEMPTS):
                generator = np.random.default_rng(
                    np.random.SeedSequence([seed, 0x50454758, attempt])
                )
                radius = float(generator.uniform(*EXPLORATION_RADIUS_RANGE_M))
                theta = float(generator.uniform(-math.pi, math.pi))
                self.exploration_probe_samples[env_id].append(
                    {
                        "attempt": attempt,
                        "radius_m": radius,
                        "theta_rad": theta,
                        "dx_m": radius * math.cos(theta),
                        "dy_m": radius * math.sin(theta),
                    }
                )
        size = int(args.video_tile_size)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_RATE_HZ,
            (8 * size, 4 * size),
        )
        if not self.video.isOpened():
            raise RuntimeError(f"failed to create video: {video_path}")

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
            self.video = None

    def make_mosaic(self, images: np.ndarray, phase: str) -> np.ndarray:
        size = int(args.video_tile_size)
        canvas = np.zeros((4 * size, 8 * size, 3), dtype=np.uint8)
        force = as_numpy(torch.linalg.vector_norm(self.env.force_sensor_joint_raw[:, :3], dim=1))
        for env_id, image in enumerate(images):
            tile = cv2.resize(image[..., ::-1], (size, size), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (size, 39), (18, 18, 18), -1)
            mark = "REC" if SEEDS[env_id] in WRITE_SEEDS else "FILL"
            trajectory = "EXP" if self.exploration_mask[env_id] else "EXPERT"
            cv2.putText(
                tile, f"e{env_id:02d} s{SEEDS[env_id]:03d} {mark} {trajectory}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.31, (245, 245, 245), 1, cv2.LINE_AA,
            )
            cv2.putText(
                tile, f"{phase[:13]} F {force[env_id]:.1f}N", (4, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (225, 225, 225), 1, cv2.LINE_AA,
            )
            row, column = divmod(env_id, 8)
            canvas[row * size:(row + 1) * size, column * size:(column + 1) * size] = tile
        return canvas

    def write_overview(self, path: Path, phase: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.make_mosaic(rgb_batch(self.env.table_rgb), phase))

    def _slew_limited(self, requested: np.ndarray) -> np.ndarray:
        current = as_numpy(self.env.fingertip_midpoint_pos).astype(np.float32)
        delta = requested - current
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.translation_step_m / np.maximum(distance, 1.0e-9))
        return current + delta * scale.astype(np.float32)

    def source_types(self, exploration: bool = False, recovery: bool = False) -> np.ndarray:
        values = np.full(NUM_ENVS, EXPERT_SOURCE, dtype=np.uint8)
        if exploration:
            values[self.exploration_mask] = EXPLORATION_SOURCE
        if recovery:
            values[self.exploration_mask] = EXPERT_RECOVERY_SOURCE
        return values

    def _record_and_step(
        self, targets_m, gripper, phase: str, source_type_ids=None
    ) -> None:
        requested = np.asarray(targets_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        command = self._slew_limited(requested)
        grippers = np.asarray(gripper, dtype=np.float32)
        if grippers.ndim == 0:
            grippers = np.full(NUM_ENVS, float(grippers), dtype=np.float32)
        actions = np.concatenate((
            command,
            np.repeat(DOWN_RPY[None, :], NUM_ENVS, axis=0),
            (2.0 * grippers - 1.0)[:, None],
        ), axis=1).astype(np.float32)
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
        privileged = {
            key: as_numpy(value) for key, value in self.task.privileged_observations(self.env).items()
        }
        for key, values in privileged.items():
            bad = np.flatnonzero(~np.isfinite(values).all(axis=1))
            if len(bad):
                details = {int(env_id): values[env_id].tolist() for env_id in bad}
                raise RuntimeError(
                    f"non-finite privileged field {key!r} before phase {phase!r}: {details}"
                )
        timestamp = (len(self.buffers[0]) + 1) * self.env.step_dt
        phase_key = "insert_stage" if phase.startswith("insert_stage_") else phase
        source_types = np.asarray(
            EXPERT_SOURCE if source_type_ids is None else source_type_ids,
            dtype=np.uint8,
        )
        if source_types.ndim == 0:
            source_types = np.full(NUM_ENVS, int(source_types), dtype=np.uint8)
        if source_types.shape != (NUM_ENVS,):
            raise ValueError("source_type_ids must be scalar or one value per environment")
        for env_id, buffer in enumerate(self.buffers):
            buffer.append(
                state=states[env_id], action=actions[env_id],
                wrench_raw=wrench_raw[env_id], wrench_ema=wrench_ema[env_id],
                timestamp=timestamp, image=main[env_id], wrist_image=wrist[env_id],
                joint_position=joints[env_id], joint_velocity=velocities[env_id],
                tcp_pose=tcp[env_id],
                privileged={key: value[env_id] for key, value in privileged.items()},
                phase_id=PHASE_SCHEMA[phase_key],
                control_source_type_id=int(source_types[env_id]),
            )
        self.video.write(self.make_mosaic(main, phase))
        self.env.set_cartesian_rpy_target(actions[:, :6], grippers)
        _, reward, terminated, truncated, _ = self.env.step(self.zero_action)
        for env_id, buffer in enumerate(self.buffers):
            buffer.set_last_transition_outcome(
                reward=float(reward[env_id]), success=False,
                terminated=bool(terminated[env_id]), truncated=bool(truncated[env_id]),
            )
            peak = float(torch.linalg.vector_norm(self.env.force_sensor_smooth[env_id, :3]).item())
            previous = float(self.phase_force_peaks[env_id].get(phase_key, 0.0))
            self.phase_force_peaks[env_id][phase_key] = max(previous, peak)
        self.env.update_cameras()

    def _pose_errors(self, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = torch.as_tensor(targets, dtype=torch.float32, device=self.env.device)
        position_error, rotation_error = compute_pose_error(
            self.env.fingertip_midpoint_pos, self.env.fingertip_midpoint_quat,
            target, self.down_quaternion, rot_error_type="axis_angle",
        )
        return (
            as_numpy(torch.linalg.vector_norm(position_error, dim=1)),
            as_numpy(torch.linalg.vector_norm(rotation_error, dim=1)),
        )

    def move_pose(
        self, targets_m, gripper, phase: str, *, max_frames: int = 240,
        min_frames: int = 4, position_tolerance_m: float = 0.0025,
        rotation_tolerance_rad: float = 0.04,
        source_type_ids=None,
    ) -> np.ndarray:
        targets = np.asarray(targets_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        start = len(self.buffers[0])
        converged = np.zeros(NUM_ENVS, dtype=bool)
        pos_error = np.full(NUM_ENVS, np.inf)
        rot_error = np.full(NUM_ENVS, np.inf)
        for frame in range(max_frames):
            self._record_and_step(targets, gripper, phase, source_type_ids)
            pos_error, rot_error = self._pose_errors(targets)
            if frame + 1 >= min_frames:
                converged |= (pos_error <= position_tolerance_m) & (rot_error <= rotation_tolerance_rad)
            if np.all(converged):
                break
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})
        for env_id in np.flatnonzero(~converged):
            self.failures[env_id].append(
                f"phase {phase} did not converge: position={pos_error[env_id]:.4f}m "
                f"rotation={rot_error[env_id]:.4f}rad"
            )
        return converged

    def hold(
        self, frames: int, gripper, phase: str, targets_m=None, source_type_ids=None
    ) -> None:
        targets = (
            as_numpy(self.env.fingertip_midpoint_pos)
            if targets_m is None else np.asarray(targets_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        )
        start = len(self.buffers[0])
        for _ in range(frames):
            self._record_and_step(targets, gripper, phase, source_type_ids)
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})

    def servo_peg_pose(
        self, holes_m, clearance_m: float, phase: str, *, max_frames: int = 100,
        xy_tolerance_m: float = 0.0012, z_tolerance_m: float = 0.0020,
        enabled=None, gripper=0.0, fatal_nonconvergence: bool = True,
        source_type_ids=None,
    ) -> np.ndarray:
        holes = np.asarray(holes_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        desired = holes.copy()
        clearance = np.asarray(clearance_m, dtype=np.float32)
        if clearance.ndim == 0:
            clearance = np.full(NUM_ENVS, float(clearance), dtype=np.float32)
        if clearance.shape != (NUM_ENVS,):
            raise ValueError("peg clearance must be scalar or one value per environment")
        desired[:, 2] = holes[:, 2] + clearance
        enabled_mask = (
            np.ones(NUM_ENVS, dtype=bool) if enabled is None else np.asarray(enabled, dtype=bool)
        )
        converged = ~enabled_mask.copy()
        stable_frames = np.where(enabled_mask, 0, 6).astype(np.int32)
        start = len(self.buffers[0])
        peg_error = np.full((NUM_ENVS, 3), np.inf, dtype=np.float32)
        for frame in range(max_frames):
            peg, _ = self.env.entity_local_pose("peg")
            peg_np = as_numpy(peg)
            eef = as_numpy(self.env.fingertip_midpoint_pos)
            peg_error = desired - peg_np
            targets = eef + peg_error
            targets[converged] = eef[converged]
            self._record_and_step(targets, gripper, phase, source_type_ids)
            peg_after, _ = self.env.entity_local_pose("peg")
            peg_error = desired - as_numpy(peg_after)
            within = (
                (np.linalg.norm(peg_error[:, :2], axis=1) <= xy_tolerance_m)
                & (np.abs(peg_error[:, 2]) <= z_tolerance_m)
            )
            stable_frames = np.where(
                enabled_mask & within,
                stable_frames + 1,
                np.where(enabled_mask, 0, stable_frames),
            )
            converged = ~enabled_mask | (stable_frames >= 6)
            if np.all(converged | ~enabled_mask):
                break
        self.phase_ranges.append({"name": phase, "start": start, "end": len(self.buffers[0])})
        safe_finish = (
            np.linalg.norm(peg_error[:, :2], axis=1)
            <= max(float(xy_tolerance_m), 0.0012)
        ) & (
            np.abs(peg_error[:, 2])
            <= max(float(z_tolerance_m), 0.0020)
        )
        converged |= enabled_mask & safe_finish
        failed = enabled_mask & ~converged
        for env_id in np.flatnonzero(failed):
            destination = self.failures if fatal_nonconvergence else self.warnings
            destination[env_id].append(
                f"phase {phase} peg servo did not converge: error={peg_error[env_id].tolist()}"
            )
        return converged

    def controlled_exploration(self, holes_m) -> None:
        """Make one seeded rim contact, disengage vertically, then recenter."""

        holes = np.asarray(holes_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        explore_sources = self.source_types(exploration=True)
        recovery_sources = self.source_types(recovery=True)
        samples = [values[0] for values in self.exploration_probe_samples]
        probe_holes = holes.copy()
        probe_holes[:, 0] += np.asarray(
            [sample["dx_m"] for sample in samples], dtype=np.float32
        )
        probe_holes[:, 1] += np.asarray(
            [sample["dy_m"] for sample in samples], dtype=np.float32
        )
        peg_before, _ = self.env.entity_local_pose("peg")
        approach_clearance = as_numpy(peg_before)[:, 2] - holes[:, 2]
        # Move laterally at the measured pre-contact height, make one short
        # downward contact, then lift exactly 4 mm from the measured contact
        # pose before returning to center.  This avoids both sustained rubbing
        # and an arbitrary absolute recovery height.
        self.servo_peg_pose(
            probe_holes,
            approach_clearance,
            "exploration_probe",
            max_frames=60,
            xy_tolerance_m=0.0015,
            z_tolerance_m=0.0020,
            enabled=self.exploration_mask,
            fatal_nonconvergence=False,
            source_type_ids=explore_sources,
        )
        self.servo_peg_pose(
            probe_holes,
            EXPLORATION_CONTACT_CLEARANCE_M,
            "exploration_probe",
            max_frames=8,
            xy_tolerance_m=0.0015,
            z_tolerance_m=0.0020,
            enabled=self.exploration_mask,
            fatal_nonconvergence=False,
            source_type_ids=explore_sources,
        )
        peg_after_contact, _ = self.env.entity_local_pose("peg")
        retreat_clearance = (
            as_numpy(peg_after_contact)[:, 2]
            - holes[:, 2]
            + np.float32(EXPLORATION_VERTICAL_DISENGAGE_M)
        )
        self.servo_peg_pose(
            probe_holes,
            retreat_clearance,
            "expert_recovery_center",
            max_frames=30,
            xy_tolerance_m=0.0015,
            z_tolerance_m=0.0020,
            enabled=self.exploration_mask,
            fatal_nonconvergence=False,
            source_type_ids=recovery_sources,
        )
        self.servo_peg_pose(
            holes,
            retreat_clearance,
            "expert_recovery_center",
            max_frames=60,
            xy_tolerance_m=0.0006,
            z_tolerance_m=0.0012,
            enabled=self.exploration_mask,
            fatal_nonconvergence=True,
            source_type_ids=recovery_sources,
        )

    def progressive_insert(self, holes_m, source_type_ids=None) -> None:
        holes = np.asarray(holes_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        peg_before, _ = self.env.entity_local_pose("peg")
        commanded_clearance = as_numpy(peg_before)[:, 2] - holes[:, 2]
        self.servo_peg_pose(
            holes, commanded_clearance, "precision_centering", max_frames=100,
            xy_tolerance_m=0.0005, z_tolerance_m=0.001,
            fatal_nonconvergence=False,
            source_type_ids=source_type_ids,
        )
        # The physical convex-hull pair bottoms out at a repeatable peg-root
        # height of about 7.3 mm.  A 6 mm command reaches that stable contact
        # seat and satisfies the formal mate-height bound.  Driving on toward
        # the nominal 3 mm visual mate height only stores lateral contact
        # energy and can eject the peg when the fingers open.
        stage_count = int(np.clip(
            math.ceil(
                (float(np.max(commanded_clearance)) - FINAL_SEAT_COMMAND_CLEARANCE_M)
                / 0.002
            ),
            1,
            24,
        ))
        seated = np.zeros(NUM_ENVS, dtype=bool)
        for index in range(1, stage_count + 1):
            commanded_clearance = np.maximum(
                np.float32(FINAL_SEAT_COMMAND_CLEARANCE_M),
                commanded_clearance - np.float32(0.002),
            )
            self.servo_peg_pose(
                holes, commanded_clearance, f"insert_stage_{index:02d}",
                max_frames=55, xy_tolerance_m=INSERT_STAGE_XY_TOLERANCE_M,
                z_tolerance_m=0.001, enabled=~seated, fatal_nonconvergence=False,
                source_type_ids=source_type_ids,
            )
            center_error, height_error, tilt_deg, peg_np = self.insertion_geometry(holes)
            newly_seated = (~seated) & evaluate_release_stability_batch(
                center_error, height_error, tilt_deg
            )
            seated |= newly_seated
            self.seat_clearance_m[newly_seated] = (
                peg_np[newly_seated, 2] - holes[newly_seated, 2]
            )
            if np.all(seated):
                break
        center_error, height_error, tilt_deg, peg_np = self.insertion_geometry(holes)
        still_valid = evaluate_release_stability_batch(center_error, height_error, tilt_deg)
        seated |= still_valid
        missing_clearance = seated & ~np.isfinite(self.seat_clearance_m)
        self.seat_clearance_m[missing_clearance] = (
            peg_np[missing_clearance, 2] - holes[missing_clearance, 2]
        )
        self.insertion_stopped_on_success_bounds = seated
        for env_id in np.flatnonzero(~seated):
            self.failures[env_id].append(
                "no stable insertion seat before release: "
                f"center={center_error[env_id]:.4f}m height={height_error[env_id]:.4f}m "
                f"tilt={tilt_deg[env_id]:.2f}deg"
            )

    def insertion_geometry(self, holes_m) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        holes = np.asarray(holes_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        peg_position, peg_quaternion = self.env.entity_local_pose("peg")
        local_z = torch.tensor([[0.0, 0.0, 1.0]], device=self.env.device).repeat(NUM_ENVS, 1)
        peg_axis = as_numpy(math_utils.quat_apply(peg_quaternion, local_z))
        peg_np = as_numpy(peg_position)
        center_error = np.linalg.norm(peg_np[:, :2] - holes[:, :2], axis=1)
        height_error = peg_np[:, 2] - (holes[:, 2] + SPEC.hole_mate_height_m)
        axis_norm = np.maximum(np.linalg.norm(peg_axis, axis=1), 1.0e-9)
        tilt_deg = np.degrees(np.arccos(np.clip(np.abs(peg_axis[:, 2]) / axis_norm, 0.0, 1.0)))
        return center_error, height_error, tilt_deg, peg_np

    def prepare_clean_release(self, holes_m, source_type_ids=None) -> np.ndarray:
        """Center the seated peg and remove the grasp's lateral position offset."""

        holes = np.asarray(holes_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        eligible = self.insertion_stopped_on_success_bounds.copy()
        seat = self.seat_clearance_m.copy()
        invalid_seat = ~np.isfinite(seat)
        seat[invalid_seat] = FINAL_SEAT_COMMAND_CLEARANCE_M

        # First put the held peg back on the measured stable seat with tighter
        # XY/Z tolerances than the final task success gate.
        self.servo_peg_pose(
            holes,
            seat,
            "pre_release_centering",
            max_frames=90,
            xy_tolerance_m=0.0005,
            z_tolerance_m=0.0008,
            enabled=eligible,
            fatal_nonconvergence=True,
            source_type_ids=source_type_ids,
        )

        # The old implementation preserved the lateral EEF-to-peg offset that
        # was measured immediately after pickup.  Center the two fingers on the
        # current peg XY instead, while preserving the current vertical grasp.
        peg_position, _ = self.env.entity_local_pose("peg")
        peg_np = as_numpy(peg_position)
        neutral_targets = as_numpy(self.env.fingertip_midpoint_pos)
        neutral_targets[eligible, :2] = peg_np[eligible, :2]
        self.hold(
            12,
            0.0,
            "pre_release_neutralize",
            neutral_targets,
            source_type_ids,
        )

        # The centering move above is intentionally small, but it can move the
        # grasped peg by the same sub-millimetre amount.  Re-center the peg once
        # more without changing its measured seat depth, then let contact settle.
        self.servo_peg_pose(
            holes,
            seat,
            "pre_release_centering",
            max_frames=90,
            xy_tolerance_m=0.0005,
            z_tolerance_m=0.0008,
            enabled=eligible,
            fatal_nonconvergence=True,
            source_type_ids=source_type_ids,
        )
        release_targets = as_numpy(self.env.fingertip_midpoint_pos)
        self.hold(
            PRE_RELEASE_SETTLE_FRAMES,
            0.0,
            "pre_release_neutralize",
            release_targets,
            source_type_ids,
        )

        center_error, height_error, tilt_deg, _ = self.insertion_geometry(holes)
        lateral_force = as_numpy(
            torch.linalg.vector_norm(self.env.force_sensor_joint_smooth[:, :2], dim=1)
        )
        self.pre_release_lateral_force_n = lateral_force.astype(np.float32)
        self.pre_release_valid = eligible & (
            (center_error <= PRE_RELEASE_XY_TOLERANCE_M)
            & (np.abs(height_error) <= PRE_RELEASE_HEIGHT_TOLERANCE_M)
            & (tilt_deg <= PRE_RELEASE_TILT_TOLERANCE_DEG)
            & (lateral_force <= PRE_RELEASE_MAX_LATERAL_FORCE_N)
        )
        for env_id in np.flatnonzero(eligible & ~self.pre_release_valid):
            self.failures[env_id].append(
                "clean pre-release gate failed: "
                f"center={center_error[env_id]:.4f}m "
                f"height={height_error[env_id]:.4f}m "
                f"tilt={tilt_deg[env_id]:.2f}deg "
                f"lateral_force={lateral_force[env_id]:.3f}N"
            )
        return release_targets

    def clean_release(self, holes_m, release_targets, source_type_ids=None) -> None:
        """Release once, never recover, and reject any episode that slips."""

        holes = np.asarray(holes_m, dtype=np.float32).reshape(NUM_ENVS, 3)
        targets = np.asarray(release_targets, dtype=np.float32).reshape(NUM_ENVS, 3)
        # Open both fingers symmetrically and smoothly.  Unlike the old guarded
        # release, this is one monotonic release: it never re-closes, re-seats,
        # or turns a failed release into an accepted expert trajectory.
        for step in range(1, CLEAN_RELEASE_STEPS + 1):
            self.hold(
                1,
                step / CLEAN_RELEASE_STEPS,
                "release",
                targets,
                source_type_ids,
            )
        self.hold(RELEASE_SETTLE_FRAMES, 1.0, "release_settle", targets, source_type_ids)

        center_error, height_error, tilt_deg, _ = self.insertion_geometry(holes)
        stable_after_release = self.pre_release_valid & (
            (center_error <= PRE_RELEASE_XY_TOLERANCE_M)
            & (np.abs(height_error) <= POST_RELEASE_HEIGHT_TOLERANCE_M)
            & (tilt_deg <= PRE_RELEASE_TILT_TOLERANCE_DEG)
        )
        for env_id in np.flatnonzero(self.pre_release_valid & ~stable_after_release):
            self.failures[env_id].append(
                "clean release slipped; no recovery trajectory recorded: "
                f"center={center_error[env_id]:.4f}m "
                f"height={height_error[env_id]:.4f}m "
                f"tilt={tilt_deg[env_id]:.2f}deg"
            )

    def final_reports(self) -> list[dict[str, object]]:
        peg_position, peg_quaternion = self.env.entity_local_pose("peg")
        hole_position, _ = self.env.entity_local_pose("target_hole")
        local_z = torch.tensor([[0.0, 0.0, 1.0]], device=self.env.device).repeat(NUM_ENVS, 1)
        peg_axis = math_utils.quat_apply(peg_quaternion, local_z)
        gripper = as_numpy(self.env.gripper_open_fraction_tensor)
        eef = as_numpy(self.env.fingertip_midpoint_pos)
        peg_np, target_np, axis_np = as_numpy(peg_position), as_numpy(hole_position), as_numpy(peg_axis)
        target_np[:, 2] += SPEC.hole_mate_height_m
        reports = []
        for env_id in range(NUM_ENVS):
            report = evaluate_final_state(
                peg_position_m=peg_np[env_id], peg_axis_world=axis_np[env_id],
                target_hole_position_m=target_np[env_id],
                gripper_open_fraction=float(gripper[env_id]), eef_position_m=eef[env_id],
            ).to_dict()
            report["trajectory_failures"] = list(self.failures[env_id])
            report["trajectory_warnings"] = list(self.warnings[env_id])
            report["success"] = bool(report["success"] and not self.failures[env_id])
            reports.append(report)
            if report["success"]:
                self.buffers[env_id].set_last_transition_outcome(
                    reward=1.0, success=True, terminated=False, truncated=False
                )
        return reports

    def run(self) -> list[dict[str, object]]:
        pickup, _ = self.env.entity_local_pose("peg")
        pickup_np = as_numpy(pickup)
        target = pickup_np.copy(); target[:, 2] = 0.12
        self.move_pose(target, 1.0, "approach_pickup")
        target[:, 2] = 0.045
        self.move_pose(target, 1.0, "descend_to_grasp")
        self.hold(14, 0.0, "close_gripper")
        target[:, 2] = FIXED_LIFT_TCP_Z_M
        self.move_pose(target, 0.0, "lift_peg")

        peg_after_lift, _ = self.env.entity_local_pose("peg")
        peg_after_lift_np = as_numpy(peg_after_lift)
        for env_id in np.flatnonzero(peg_after_lift_np[:, 2] < 0.07):
            self.failures[env_id].append(f"peg was not lifted: z={peg_after_lift_np[env_id, 2]:.4f}")
        self.eef_minus_peg_after_grasp = (
            as_numpy(self.env.fingertip_midpoint_pos) - peg_after_lift_np
        )
        hole, _ = self.env.entity_local_pose("target_hole")
        holes = as_numpy(hole)
        transit = holes + self.eef_minus_peg_after_grasp
        transit[:, 2] = FIXED_LIFT_TCP_Z_M
        self.move_pose(transit, 0.0, "transport_above_target")
        preinsert = holes + self.eef_minus_peg_after_grasp
        preinsert[:, 2] = holes[:, 2] + self.eef_minus_peg_after_grasp[:, 2] + 0.038
        self.move_pose(preinsert, 0.0, "prealign")
        self.servo_peg_pose(
            holes, 0.038, "peg_centering", xy_tolerance_m=0.0006, z_tolerance_m=0.0012
        )
        # Peg collection is intentionally expert-only.  Continue directly from
        # privileged centering into insertion with expert source labels.
        insertion_sources = self.source_types()
        self.progressive_insert(holes, insertion_sources)
        inserted = as_numpy(self.env.fingertip_midpoint_pos)
        self.hold(20, 0.0, "insertion_settle", inserted, insertion_sources)
        release_targets = self.prepare_clean_release(holes, insertion_sources)
        self.clean_release(holes, release_targets, insertion_sources)
        retract = release_targets.copy(); retract[:, 2] += 0.10
        self.move_pose(
            retract,
            1.0,
            "final_retract",
            max_frames=180,
            source_type_ids=insertion_sources,
        )
        self.hold(
            args.final_settle_frames,
            1.0,
            "final_settle",
            retract,
            insertion_sources,
        )
        return self.final_reports()

    def details(self, env_id: int) -> dict[str, object]:
        return {
            "eef_minus_peg_after_grasp_m": self.eef_minus_peg_after_grasp[env_id].tolist(),
            "phase_ranges_control_frames": list(self.phase_ranges),
            "phase_force_peaks_n": dict(self.phase_force_peaks[env_id]),
            "trajectory_warnings": list(self.warnings[env_id]),
            "release_recovery_count": 0,
            "pre_release_valid": bool(self.pre_release_valid[env_id]),
            "pre_release_lateral_force_n": float(
                self.pre_release_lateral_force_n[env_id]
            ),
            "stable_contact_seat_clearance_m": float(self.seat_clearance_m[env_id]),
            "oracle_pose_feedback": True,
            "physics_intervention": False,
            "insertion_stopped_on_success_bounds": bool(
                self.insertion_stopped_on_success_bounds[env_id]
            ),
            "trajectory_type": "exploration" if self.exploration_mask[env_id] else "expert",
            "exploration_sampler_version": None,
            "exploration_probe_samples": list(self.exploration_probe_samples[env_id])
            if self.exploration_mask[env_id]
            else [],
        }


def reset_visual_qc(env: ForceBenchEnv) -> list[dict[str, object]]:
    main, wrist = rgb_batch(env.table_rgb), rgb_batch(env.wrist_rgb)
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


task = load_task("peg_insert", ASSET_DIR)
cfg = ForceBenchEnvCfg()
cfg.seed = SEEDS[0]
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(
    cfg, device=args.device, physics_rate_hz=PHYSICS_RATE_HZ,
    solver_iterations=64, num_envs=NUM_ENVS, env_spacing=args.env_spacing,
)
cfg.num_rerenders_on_reset = 1
env = ForceBenchEnv(
    cfg=cfg, task_runtime=task, main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size, enable_cameras=True,
)
expert: ParallelPegExpert | None = None
args.artifact_dir.mkdir(parents=True, exist_ok=True)
video_path = args.artifact_dir / "peg_insert_32env_mosaic.mp4"
overview_path = args.artifact_dir / "peg_insert_32env_initial.png"
final_overview_path = args.artifact_dir / "peg_insert_32env_final.png"

try:
    env.set_episode_seeds(SEEDS)
    env.reset(seed=SEEDS[0])
    env.restore_fixed_arm_pose(env.joint_pos[0, :7].clone())
    env.reset_cameras()
    for _ in range(3):
        env.sim.render()
    env.update_cameras()
    visual_qc = reset_visual_qc(env)
    layouts = env.layouts
    env.start_wrench_capture()
    expert = ParallelPegExpert(env, task, video_path)
    expert.write_overview(overview_path, "initial_reset")
    reports = expert.run()
    high_rate_batch = env.stop_wrench_capture()
    expert.write_overview(final_overview_path, "final")
    expert.close()

    task_metadata = dict(task.episode_metadata())
    asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
    asset_paths.update(env.scene_asset_paths)
    asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
    writer = HDF5EpisodeWriter(args.output_dir)
    physics_wrench = np.asarray(high_rate_batch["eef_wrench_joint_raw"], dtype=np.float32)
    physics_timestamps = np.asarray(high_rate_batch["timestamp"], dtype=np.float64)
    if physics_timestamps.ndim != 1 or len(physics_timestamps) < 2:
        raise RuntimeError("high-rate wrench timestamps are missing or malformed")
    physics_dt = np.diff(physics_timestamps)
    if not np.isfinite(physics_dt).all() or not np.allclose(
        physics_dt, 1.0 / PHYSICS_RATE_HZ, rtol=0.0, atol=2.0e-5
    ):
        raise RuntimeError(
            f"high-rate wrench timestamps violate {PHYSICS_RATE_HZ} Hz contract: "
            f"min={physics_dt.min():.9f}, max={physics_dt.max():.9f}"
        )
    expected_ratio = PHYSICS_RATE_HZ // CONTROL_RATE_HZ
    if abs(len(physics_timestamps) - len(expert.buffers[0]) * expected_ratio) > expected_ratio:
        raise RuntimeError(
            "high-rate wrench/control sample ratio violates 60 Hz physics / 15 Hz control contract"
        )
    written, failed = [], []
    for env_id, seed in enumerate(SEEDS):
        if seed not in WRITE_SEEDS:
            continue
        if not reports[env_id]["success"]:
            failure = {"env_id": env_id, "seed": seed, "report": reports[env_id]}
            failed.append(failure)
            print(
                "FORCEBENCH_PARALLEL_PEG_FINAL_FAILURE="
                + json.dumps(failure, separators=(",", ":")), flush=True,
            )
            continue
        existing = sorted(args.output_dir.glob(f"seed_{seed:03d}_episode_*.hdf5"))
        if existing:
            raise RuntimeError(
                f"refusing to duplicate successful seed {seed}: {[str(path) for path in existing]}"
            )
        trajectory_type = "expert"
        metadata = {
            "task": "peg_insert", "prompt": task.manifest.prompt, "success": True,
            "end_reason": "scripted_32env_complete_insert_release_retract",
            "seed": seed, "initial_layout": layouts[env_id],
            "final_success_report": reports[env_id],
            "control_rate_hz": float(CONTROL_RATE_HZ),
            "physics_rate_hz": float(PHYSICS_RATE_HZ),
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "arm_controller_diagnostics": env.arm_controller_diagnostics,
            "scene_profile": env.scene_profile_name,
            "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "control_source": "scripted_32env_all_expert_smooth_position",
            "trajectory_type": trajectory_type,
            "control_source_type_schema": SOURCE_TYPE_SCHEMA,
            "control_source_type_dataset": "control/source_type",
            "control_source_type_semantics": "all frames use expert=0 scripted teacher",
            "wrench_profile": env.wrench_profile,
            "wrench_storage": "raw_and_ema",
            "primary_wrist_wrench_dataset": "forces/control/eef_wrench_joint_raw",
            "success_check_timing": "after_physical_insert_release_and_retract",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [args.main_image_size, args.main_image_size, 3],
                "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
            },
            "phase_schema": PHASE_SCHEMA,
            "scripted_expert_generator_version": (
                "peg_insert_32env_v2_all_expert_clean_release_physics60"
            ),
            "batch_trajectory_split": {
                "rule": "all_seeds_expert",
                "expert_count": NUM_ENVS,
                "exploration_count": 0,
            },
            "controlled_exploration_profile": {
                "enabled": False,
                "probe_attempts": 0,
                "reason": "peg_insert_all_expert_collection",
            },
            "stable_contact_seat_command_clearance_m": (
                FINAL_SEAT_COMMAND_CLEARANCE_M
            ),
            "clean_release_profile": {
                "command": "single_monotonic_symmetric_open_ramp",
                "release_steps": CLEAN_RELEASE_STEPS,
                "settle_frames": RELEASE_SETTLE_FRAMES,
                "pre_release_settle_frames": PRE_RELEASE_SETTLE_FRAMES,
                "pre_release_max_center_error_m": PRE_RELEASE_XY_TOLERANCE_M,
                "pre_release_max_height_error_m": PRE_RELEASE_HEIGHT_TOLERANCE_M,
                "post_release_max_height_error_m": POST_RELEASE_HEIGHT_TOLERANCE_M,
                "pre_release_max_tilt_deg": PRE_RELEASE_TILT_TOLERANCE_DEG,
                "pre_release_max_lateral_force_n": PRE_RELEASE_MAX_LATERAL_FORCE_N,
                "recovery": "none; any slip rejects the episode",
                "accepted_trajectory_constraint": (
                    "no release_reclose or release_reseat phases"
                ),
            },
            "translation_speed_limit_m_s": float(args.translation_speed_m_s),
            "scripted_expert_details": expert.details(env_id),
            "expert_uses_privileged_peg_hole_pose_for_control": True,
            "l20_migration_source": L20_MIGRATION_SOURCE,
            "semantic_migration_decision": (
                "retained L20 post-contact privileged pose correction while preserving the "
                "newer ForceBench full pickup-transport-insert-release task and per-seed HDF5 schema"
            ),
            "parallel_environment": {
                "enabled": True, "single_isaac_process": True, "num_envs": NUM_ENVS,
                "env_id": env_id, "env_spacing_m": args.env_spacing, "all_seeds": SEEDS,
                "camera_backend": "TiledCamera", "collision_filtering": True,
                "replicate_physics": True, "reset_visual_qc": visual_qc[env_id],
            },
            "scene_roles": list(env.entity_roles), "asset_usd_paths": asset_paths,
            **task_metadata,
        }
        high_rate = {
            "timestamp": physics_timestamps,
            "eef_wrench_joint_raw": physics_wrench[:, env_id],
        }
        path = writer.write(expert.buffers[env_id], metadata, high_rate=high_rate)[0]
        validation = validate_hdf5_episode(path)
        written.append({
            "env_id": env_id, "seed": seed, "path": str(path), "validation": validation
        })

    summary = {
        "schema": "forcebench_parallel_peg_insert_32env_v3_all_expert",
        "task": "peg_insert",
        "seeds": SEEDS, "write_seeds": sorted(WRITE_SEEDS), "num_envs": NUM_ENVS,
        "env_spacing_m": args.env_spacing, "frames_per_episode": len(expert.buffers[0]),
        "video": str(video_path.resolve()), "overview": str(overview_path.resolve()),
        "final_overview": str(final_overview_path.resolve()), "episodes": written,
        "failed_episodes": failed, "environment_reports": reports,
        "l20_migration_source": L20_MIGRATION_SOURCE,
        "trajectory_type_counts": {"expert": NUM_ENVS, "exploration": 0},
        "control_source_type_schema": SOURCE_TYPE_SCHEMA,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("FORCEBENCH_PARALLEL_PEG=" + json.dumps(summary, separators=(",", ":")), flush=True)
except BaseException as error:
    print(f"FORCEBENCH_PARALLEL_PEG_ERROR={type(error).__name__}:{error}", flush=True)
    raise
finally:
    if expert is not None:
        expert.close()
    env.discard_wrench_capture()
    env.close()
    simulation_app.close()
