#!/usr/bin/env python3
"""Collect one scripted stove-knob episode with a live angle/torque video."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

from forcebench.isaac_session import shared_simulation_app


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--dataset_dir", type=Path, required=True)
parser.add_argument("--physics_rate_hz", type=int, choices=(120,), default=120)
parser.add_argument("--solver_position_iterations", type=int, choices=(64,), default=64)
parser.add_argument("--fps", type=float, default=15.0)
parser.add_argument("--grasp_height_offset_m", type=float, default=0.002)
parser.add_argument(
    "--record_video",
    action="store_true",
    help="write diagnostic MP4/PNG/NPZ artifacts; disabled for formal batch collection",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seed < 0 or args.fps <= 0.0 or args.grasp_height_offset_m <= 0.0:
    parser.error("seed must be non-negative; fps and grasp height must be positive")
simulation_app = shared_simulation_app()
owns_simulation_app = simulation_app is None
if owns_simulation_app:
    launcher = AppLauncher(args, multi_gpu=False)
    simulation_app = launcher.app

import cv2
import numpy as np
import torch
import warp as wp

from isaaclab.utils import math as math_utils

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task
from forcebench.tasks.stove_turn_on.spec import detent_effort_nm, spec_from_layout


def rgb(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().numpy()[..., :3].astype(np.uint8)


def vector(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1).astype(np.float32)


def current_state(environment: ForceBenchEnv) -> np.ndarray:
    return vector(environment.policy_state[0])


task = load_task("stove_turn_on", ASSET_DIR)
cfg = ForceBenchEnvCfg()
cfg.seed = args.seed
cfg.scene_profile = task.manifest.scene_profile
configure_runtime(
    cfg,
    device=args.device,
    physics_rate_hz=args.physics_rate_hz,
    solver_iterations=args.solver_position_iterations,
)
cfg.num_rerenders_on_reset = 0
env = ForceBenchEnv(
    cfg=cfg,
    task_runtime=task,
    main_image_size=224,
    wrist_image_size=224,
    enable_cameras=True,
)

dataset_buffer = EpisodeBuffer()
writer = None
wrench_capture_active = False
joint_targets: list[np.ndarray] = []
cartesian_actions: list[np.ndarray] = []
angle_trace_deg: list[float] = []
velocity_trace_rad_s: list[float] = []
effort_trace_nm: list[float] = []
eef_wrench_trace: list[np.ndarray] = []
eef_wrench_raw_trace: list[np.ndarray] = []
eef_torque_z_trace_nm: list[float] = []
phase_trace: list[str] = []
tracking_errors_m: list[float] = []
fingertip_position_trace_m: list[np.ndarray] = []
plot_reference_angle_deg = np.empty(0, dtype=np.float32)
plot_reference_effort_nm = np.empty(0, dtype=np.float32)
plot_angle_max_deg = 120.0
plot_effort_limit_nm = 0.35
plot_detent_center_deg = 35.0
plot_success_angle_deg = 45.0
privileged_transition_crossed = False
privileged_backoff_complete = False
privileged_target_angle_deg = 0.0
privileged_overshoot_target_deg = 0.0

CANVAS_SIZE = (960, 512)
PANEL_BG = (25, 28, 33)
PLOT_BG = (17, 20, 24)
TEXT = (238, 241, 245)
MUTED = (156, 165, 176)
GRID = (58, 64, 72)
REFERENCE = (130, 137, 147)
LIVE = (255, 205, 78)
DETENT = (48, 177, 245)
SUCCESS = (98, 205, 116)

PHASE_LABELS = {
    "approach": "APPROACH",
    "descend": "DESCEND",
    "close": "GRASP",
    "rotate": "TURN THROUGH DETENT",
    "hold_on": "HOLD ON",
    "release": "OPEN GRIPPER",
    "lift": "LIFT CLEAR",
    "retreat": "RETREAT",
}
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASE_LABELS)}


def knob_state() -> tuple[float, float]:
    entity = env.entity("stove_knob")
    positions = wp.to_torch(entity.root_view.get_dof_positions())
    velocities = wp.to_torch(entity.root_view.get_dof_velocities())
    return float(positions[0, 0].item()), float(velocities[0, 0].item())


def target_quaternion(yaw_rad: float) -> torch.Tensor:
    return math_utils.quat_from_euler_xyz(
        torch.tensor([math.pi], device=env.device),
        torch.zeros(1, device=env.device),
        torch.tensor([yaw_rad], device=env.device),
    )


def draw_load_plot(canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, width, height = rect
    cv2.putText(
        canvas,
        "MEASURED EEF Tz vs KNOB ANGLE",
        (x, y + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT,
        1,
        cv2.LINE_AA,
    )
    left, right = x + 48, x + width - 10
    top, bottom = y + 34, y + height - 43
    cv2.rectangle(canvas, (left, top), (right, bottom), PLOT_BG, -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), GRID, 1)

    def point(angle_deg: float, effort_nm: float) -> tuple[int, int]:
        angle_fraction = np.clip(angle_deg / plot_angle_max_deg, 0.0, 1.0)
        effort_fraction = np.clip(
            (effort_nm + plot_effort_limit_nm) / (2.0 * plot_effort_limit_nm),
            0.0,
            1.0,
        )
        return (
            round(left + angle_fraction * (right - left)),
            round(bottom - effort_fraction * (bottom - top)),
        )

    for angle_tick in np.linspace(0.0, plot_angle_max_deg, 5):
        xx, _ = point(float(angle_tick), 0.0)
        cv2.line(canvas, (xx, top), (xx, bottom), GRID, 1)
        cv2.putText(
            canvas,
            f"{int(angle_tick)}",
            (xx - 8, bottom + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            MUTED,
            1,
            cv2.LINE_AA,
        )
    for effort_tick in (-plot_effort_limit_nm, 0.0, plot_effort_limit_nm):
        _, yy = point(0.0, effort_tick)
        cv2.line(canvas, (left, yy), (right, yy), GRID, 1)
        cv2.putText(
            canvas,
            f"{effort_tick:+.2f}",
            (x, yy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            MUTED,
            1,
            cv2.LINE_AA,
        )
    for angle_deg, color, label in (
        (plot_detent_center_deg, DETENT, "D"),
        (plot_success_angle_deg, SUCCESS, "S"),
    ):
        xx, _ = point(angle_deg, 0.0)
        cv2.line(canvas, (xx, top), (xx, bottom), color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (xx - 4, top + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    if len(plot_reference_angle_deg) >= 2:
        reference_points = np.asarray(
            [
                point(float(angle), float(effort))
                for angle, effort in zip(
                    plot_reference_angle_deg, plot_reference_effort_nm
                )
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [reference_points], False, REFERENCE, 2, cv2.LINE_AA)
    if len(angle_trace_deg) >= 2:
        live_points = np.asarray(
            [
                point(float(angle), float(effort))
                for angle, effort in zip(angle_trace_deg, eef_torque_z_trace_nm)
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [live_points], False, LIVE, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(live_points[-1]), 4, LIVE, -1, cv2.LINE_AA)
    legend_y = y + height - 8
    cv2.line(canvas, (x + 50, legend_y - 4), (x + 70, legend_y - 4), REFERENCE, 2)
    cv2.putText(
        canvas,
        "zero-speed model",
        (x + 76, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (x + 215, legend_y - 4), (x + 235, legend_y - 4), LIVE, 2)
    cv2.putText(
        canvas,
        "measured EEF Tz",
        (x + 241, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "angle (deg)",
        (right - 71, bottom + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        MUTED,
        1,
        cv2.LINE_AA,
    )


def composite_frame(phase: str, episode_spec) -> np.ndarray:
    main_bgr = cv2.resize(
        rgb(env.table_rgb[0])[..., ::-1],
        (512, 512),
        interpolation=cv2.INTER_AREA,
    )
    wrist_bgr = cv2.resize(
        rgb(env.wrist_rgb[0])[..., ::-1],
        (216, 216),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), PANEL_BG, dtype=np.uint8)
    canvas[:, :512] = main_bgr
    canvas[12:228, 524:740] = wrist_bgr
    cv2.rectangle(canvas, (523, 11), (741, 229), GRID, 1)
    cv2.rectangle(canvas, (524, 12), (594, 38), (18, 21, 25), -1)
    cv2.putText(
        canvas,
        "WRIST",
        (534, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT,
        1,
        cv2.LINE_AA,
    )
    state = task.debug_state(env)
    angle_deg = math.degrees(float(state["angle_rad"]))
    wrench = (
        eef_wrench_raw_trace[-1]
        if eef_wrench_raw_trace
        else np.zeros(6, dtype=np.float32)
    )
    lines = (
        f"SEED    {args.seed:03d}",
        "PHASE",
        PHASE_LABELS.get(phase, phase.upper()),
        f"ANGLE   {angle_deg:6.2f} deg",
        f"RAW F  {wrench[0]:+4.1f} {wrench[1]:+4.1f} {wrench[2]:+4.1f} N",
        f"RAW T  {wrench[3]:+4.2f} {wrench[4]:+4.2f} {wrench[5]:+4.2f} Nm",
        f"TARGET  {privileged_target_angle_deg:6.2f} deg",
        f"PASS    {privileged_overshoot_target_deg:6.2f} deg",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (754, 28 + 25 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37 if index not in (2, 4, 5) else 0.32,
            MUTED if index == 1 else TEXT,
            1,
            cv2.LINE_AA,
        )
    if privileged_backoff_complete:
        transition_status = "TARGET RECOVERED"
    elif privileged_transition_crossed:
        transition_status = "DETENT CROSSED"
    else:
        transition_status = "PRIVILEGED TURN"
    flame_status = "IGNITED" if bool(state["flame_visible"]) else "FLAME OFF"
    cv2.putText(
        canvas,
        f"{transition_status} | {flame_status}",
        (754, 224),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        SUCCESS if privileged_transition_crossed else MUTED,
        1,
        cv2.LINE_AA,
    )
    draw_load_plot(canvas, (524, 238, 424, 266))
    return canvas


def command_pose(
    position_m: tuple[float, float, float],
    yaw_rad: float,
    gripper: float,
    phase: str,
    episode_spec,
) -> tuple[float, float, float, np.ndarray]:
    target_position = torch.tensor([position_m], dtype=torch.float32, device=env.device)
    action = np.asarray(
        [*position_m, math.pi, 0.0, yaw_rad, 2.0 * gripper - 1.0],
        dtype=np.float32,
    )
    dataset_buffer.append(
        state=current_state(env),
        action=action,
        wrench_raw=vector(env.force_sensor_joint_raw[0]),
        wrench_ema=vector(env.force_sensor_joint_smooth[0]),
        timestamp=(len(dataset_buffer) + 1) * env.step_dt,
        image=rgb(env.table_rgb[0]),
        wrist_image=rgb(env.wrist_rgb[0]),
        joint_position=vector(env.joint_pos[0]),
        joint_velocity=vector(env.robot.data.joint_vel.torch[0]),
        tcp_pose=vector(
            torch.cat(
                (env.fingertip_midpoint_pos[0], env.fingertip_midpoint_quat[0])
            )
        ),
        privileged=dict(task.privileged_observations(env)),
        phase_id=PHASE_SCHEMA[phase],
    )
    env.set_cartesian_rpy_target(action[:6], gripper)
    env.step(torch.zeros((1, 7), dtype=torch.float32, device=env.device))
    dataset_buffer.set_last_transition_outcome(
        reward=0.0,
        success=False,
        terminated=False,
        truncated=False,
    )
    angle_rad, velocity_rad_s = knob_state()
    state = task.debug_state(env)
    load_effort_nm = float(state["load_effort_nm"])
    raw_wrench = vector(env.force_sensor_joint_raw[0])
    smoothed_wrench = vector(env.force_sensor_joint_smooth[0])
    joint_targets.append(vector(env.joint_pos[0, :7]))
    cartesian_actions.append(action)
    angle_trace_deg.append(math.degrees(angle_rad))
    velocity_trace_rad_s.append(velocity_rad_s)
    effort_trace_nm.append(load_effort_nm)
    eef_wrench_raw_trace.append(raw_wrench)
    eef_wrench_trace.append(smoothed_wrench)
    eef_torque_z_trace_nm.append(float(smoothed_wrench[5]))
    phase_trace.append(phase)
    tracking_errors_m.append(
        float(
            torch.linalg.vector_norm(
                env.fingertip_midpoint_pos[0] - target_position[0]
            ).item()
        )
    )
    fingertip_position_trace_m.append(
        env.fingertip_midpoint_pos[0].detach().cpu().numpy().copy()
    )
    env.update_cameras()
    if writer is not None:
        writer.write(composite_frame(phase, episode_spec))
    return angle_rad, velocity_rad_s, load_effort_nm, smoothed_wrench


def track_pose(
    position_m: tuple[float, float, float],
    yaw_rad: float,
    gripper: float,
    frames: int,
    phase: str,
    episode_spec,
) -> None:
    for _ in range(frames):
        command_pose(position_m, yaw_rad, gripper, phase, episode_spec)


try:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    env.set_episode_seed(args.seed)
    env.reset(seed=args.seed)
    layout = env.layout
    episode_spec = spec_from_layout(layout)
    knob_x, knob_y, knob_z = (
        float(value) for value in layout["target_knob_position_m"]
    )
    grasp_position = (
        knob_x,
        knob_y,
        knob_z + args.grasp_height_offset_m,
    )
    approach_position = (knob_x, knob_y, knob_z + 0.145)
    lift_position = (knob_x, knob_y, knob_z + 0.165)
    retreat_position = (knob_x, knob_y - 0.11, knob_z + 0.185)
    # This is an expert-data oracle: use the seed-specific physical knob angle
    # to prove the detent was crossed, then close the loop back to the task's
    # post-detent success angle. Wrench remains recorded but never controls the
    # trajectory.
    privileged_target_angle_deg = float(episode_spec.success_angle_deg)
    privileged_overshoot_target_deg = min(
        float(episode_spec.open_limit_deg) - 5.0,
        privileged_target_angle_deg + 4.0,
    )
    privileged_backoff_tolerance_deg = 0.50
    privileged_backoff_stable_frames = 5
    command_step_deg = 0.32
    maximum_forward_command_deg = episode_spec.open_limit_deg - 5.0

    plot_detent_center_deg = float(episode_spec.detent_center_deg)
    plot_success_angle_deg = float(episode_spec.success_angle_deg)
    plot_angle_max_deg = float(episode_spec.open_limit_deg)
    plot_effort_limit_nm = float(episode_spec.max_effort_nm)
    plot_reference_angle_deg = np.linspace(
        0.0, plot_angle_max_deg, 481, dtype=np.float32
    )
    plot_reference_effort_nm = np.asarray(
        [
            detent_effort_nm(math.radians(float(angle)), 0.0, episode_spec)
            for angle in plot_reference_angle_deg
        ],
        dtype=np.float32,
    )

    env.reset_cameras()
    env.update_cameras()
    raw_video_path = None
    video_path = None
    if args.record_video:
        cv2.imwrite(
            str(args.output_dir / "main_initial.png"),
            rgb(env.table_rgb[0])[..., ::-1],
        )
        raw_video_path = args.output_dir / "stove_knob_expert_raw.mp4"
        writer = cv2.VideoWriter(
            str(raw_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            CANVAS_SIZE,
        )
        if not writer.isOpened():
            raise OSError(f"failed to open video writer: {raw_video_path}")
    env.start_wrench_capture()
    wrench_capture_active = True

    track_pose(approach_position, 0.0, 1.0, 75, "approach", episode_spec)
    track_pose(grasp_position, 0.0, 1.0, 55, "descend", episode_spec)
    track_pose(grasp_position, 0.0, 0.0, 30, "close", episode_spec)
    closed_finger_positions_m = (
        env.robot.data.joint_pos.torch[0, 7:9].detach().cpu().tolist()
    )

    maximum_angle_deg = math.degrees(knob_state()[0])
    crossing_frame = None
    privileged_transition_frame = None
    privileged_transition_angle_deg = None
    privileged_transition_command_deg = None
    privileged_backoff_frame = None
    privileged_backoff_final_angle_deg = None
    command_yaw = 0.0
    for _ in range(420):
        command_yaw = min(
            math.radians(maximum_forward_command_deg),
            command_yaw + math.radians(command_step_deg),
        )
        angle_rad, _, _, _ = command_pose(
            grasp_position,
            command_yaw,
            0.0,
            "rotate",
            episode_spec,
        )
        angle_deg = math.degrees(angle_rad)
        maximum_angle_deg = max(maximum_angle_deg, angle_deg)
        if crossing_frame is None and angle_deg >= episode_spec.success_angle_deg:
            crossing_frame = len(angle_trace_deg) - 1
        command_deg = math.degrees(command_yaw)
        if angle_deg >= privileged_overshoot_target_deg:
            privileged_transition_crossed = True
            privileged_transition_frame = len(angle_trace_deg) - 1
            privileged_transition_angle_deg = angle_deg
            privileged_transition_command_deg = command_deg
            break
        if command_deg >= maximum_forward_command_deg:
            break

    backoff_stable = 0
    for _ in range(180):
        angle_rad, _ = knob_state()
        angle_deg = math.degrees(angle_rad)
        angle_error_deg = privileged_target_angle_deg - angle_deg
        if abs(angle_error_deg) <= privileged_backoff_tolerance_deg:
            backoff_stable += 1
        else:
            backoff_stable = 0
        if backoff_stable >= privileged_backoff_stable_frames:
            privileged_backoff_complete = True
            privileged_backoff_frame = len(angle_trace_deg) - 1
            privileged_backoff_final_angle_deg = angle_deg
            break
        command_yaw += math.radians(
            float(np.clip(angle_error_deg, -command_step_deg, command_step_deg))
        )
        command_yaw = float(
            np.clip(
                command_yaw,
                math.radians(episode_spec.closed_angle_deg),
                math.radians(maximum_forward_command_deg),
            )
        )
        angle_rad, _, _, _ = command_pose(
            grasp_position,
            command_yaw,
            0.0,
            "rotate",
            episode_spec,
        )
        angle_deg = math.degrees(angle_rad)
        maximum_angle_deg = max(maximum_angle_deg, angle_deg)
        if crossing_frame is None and angle_deg >= episode_spec.success_angle_deg:
            crossing_frame = len(angle_trace_deg) - 1

    command_target_yaw_rad = command_yaw
    track_pose(
        grasp_position,
        command_target_yaw_rad,
        0.0,
        20,
        "hold_on",
        episode_spec,
    )
    report_before_release = env.evaluate_final_success()

    release_frames = 0
    release_finger_positions_m = [0.0, 0.0]
    for release_frames in range(1, 31):
        command_pose(
            grasp_position,
            command_target_yaw_rad,
            1.0,
            "release",
            episode_spec,
        )
        release_finger_positions_m = (
            env.robot.data.joint_pos.torch[0, 7:9].detach().cpu().tolist()
        )
        if release_frames >= 8 and min(release_finger_positions_m) >= 0.0395:
            break
    track_pose(
        lift_position,
        command_target_yaw_rad,
        1.0,
        75,
        "lift",
        episode_spec,
    )
    track_pose(
        retreat_position,
        0.0,
        1.0,
        45,
        "retreat",
        episode_spec,
    )

    if writer is not None:
        writer.release()
        writer = None
        video_path = args.output_dir / "stove_knob_expert.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            video_path = raw_video_path
        else:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_video_path),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video_path),
                ],
                check=True,
            )
    env.update_cameras()
    if args.record_video:
        cv2.imwrite(
            str(args.output_dir / "main_final.png"),
            rgb(env.table_rgb[0])[..., ::-1],
        )
        cv2.imwrite(
            str(args.output_dir / "wrist_final.png"),
            rgb(env.wrist_rgb[0])[..., ::-1],
        )

    final_angle_rad, final_velocity_rad_s = knob_state()
    final_angle_deg = math.degrees(final_angle_rad)
    report_after_release = env.evaluate_final_success()
    gripped = (
        min(closed_finger_positions_m) > 0.010
        and max(closed_finger_positions_m) < 0.0395
    )
    crossed = maximum_angle_deg >= episode_spec.success_angle_deg
    released = env.gripper_open_fraction >= 0.90
    fingertip_position = env.fingertip_midpoint_pos[0].detach().cpu().numpy()
    lifted_clear = float(fingertip_position[2]) >= knob_z + 0.12
    retreated = float(fingertip_position[1]) <= knob_y - 0.075
    knob_root_position = (
        env.entity_local_pose("stove_knob")[0][0].detach().cpu().numpy().tolist()
    )
    position_error_m = float(
        np.linalg.norm(
            np.asarray(knob_root_position)
            - np.asarray(layout["target_knob_position_m"])
        )
    )
    success = bool(
        gripped
        and privileged_transition_crossed
        and privileged_transition_frame is not None
        and privileged_backoff_complete
        and privileged_backoff_frame is not None
        and crossed
        and crossing_frame is not None
        and report_before_release["success"]
        and report_after_release["success"]
        and released
        and lifted_clear
        and retreated
        and min(release_finger_positions_m) >= 0.0395
        and position_error_m <= 1.0e-5
    )

    if args.record_video:
        np.savez_compressed(
            args.output_dir / "trajectory.npz",
            joint_targets=np.asarray(joint_targets, dtype=np.float32),
            cartesian_actions=np.asarray(cartesian_actions, dtype=np.float32),
            knob_angle_deg=np.asarray(angle_trace_deg, dtype=np.float32),
            knob_velocity_rad_s=np.asarray(velocity_trace_rad_s, dtype=np.float32),
            knob_load_effort_nm=np.asarray(effort_trace_nm, dtype=np.float32),
            eef_wrench_joint_raw=np.asarray(eef_wrench_raw_trace, dtype=np.float32),
            eef_wrench_joint_smoothed=np.asarray(eef_wrench_trace, dtype=np.float32),
            wrist_torque_z_nm=np.asarray(eef_torque_z_trace_nm, dtype=np.float32),
            fingertip_position_m=np.asarray(fingertip_position_trace_m, dtype=np.float32),
            phase=np.asarray(phase_trace),
        )
    high_rate_wrench = env.stop_wrench_capture()
    wrench_capture_active = False
    if success:
        dataset_buffer.set_last_transition_outcome(
            reward=1.0,
            success=True,
            terminated=False,
            truncated=False,
        )

    dataset_episode_path = None
    dataset_validation = None
    if success:
        task_metadata = dict(task.episode_metadata())
        asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
        asset_paths.update(env.scene_asset_paths)
        asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
        metadata = {
            "task": "stove_turn_on",
            "prompt": task.manifest.prompt,
            "success": True,
            "end_reason": "scripted_stove_knob_turn_release_retract",
            "seed": args.seed,
            "initial_layout": env.layout,
            "final_success_report": report_after_release,
            "control_rate_hz": 15.0,
            "physics_rate_hz": float(args.physics_rate_hz),
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "scene_profile": env.scene_profile_name,
            "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "control_source": (
                f"scripted_stove_knob_privileged_angle_{env.arm_control_mode}"
            ),
            "success_check_timing": (
                "after_privileged_detent_cross_backoff_release_and_retract"
            ),
            "images_present": True,
            "image_shapes_hwc": {"main": [224, 224, 3], "wrist": [224, 224, 3]},
            "phase_schema": PHASE_SCHEMA,
            "scripted_expert_generator_version": "stove_knob_privileged_angle_v1",
            "expert_uses_privileged_knob_angle_for_control": True,
            "expert_transition_signal": "privileged_stove_knob_joint_angle",
            "privileged_angle_controller": {
                "overshoot_target_deg": privileged_overshoot_target_deg,
                "backoff_target_deg": privileged_target_angle_deg,
                "backoff_tolerance_deg": privileged_backoff_tolerance_deg,
                "backoff_stable_frames": privileged_backoff_stable_frames,
            },
            "fixed_damping_nm_s_rad": episode_spec.damping_nm_s_rad,
            "scene_roles": list(env.entity_roles),
            "asset_usd_paths": asset_paths,
            **task_metadata,
        }
        dataset_episode_path = HDF5EpisodeWriter(args.dataset_dir).write(
            dataset_buffer,
            metadata,
            high_rate=high_rate_wrench,
        )[0]
        dataset_validation = validate_hdf5_episode(dataset_episode_path)

    phase_durations_s = {
        phase_name: phase_trace.count(phase_name) / args.fps
        for phase_name in dict.fromkeys(phase_trace)
    }
    summary = {
        "task": "stove_turn_on",
        "trajectory_type": "grasp_privileged_cross_backoff_release_retreat",
        "seed": args.seed,
        "success": success,
        "episode": str(dataset_episode_path.resolve()) if dataset_episode_path else None,
        "dataset_validation": dataset_validation,
        "layout": layout,
        "frames": len(joint_targets),
        "fps": args.fps,
        "phase_durations_s": phase_durations_s,
        "grasp_height_offset_m": args.grasp_height_offset_m,
        "detent_center_deg": episode_spec.detent_center_deg,
        "success_angle_deg": episode_spec.success_angle_deg,
        "damping_nm_s_rad": episode_spec.damping_nm_s_rad,
        "expert_uses_privileged_knob_angle_for_control": True,
        "privileged_transition_crossed": privileged_transition_crossed,
        "privileged_transition_frame": privileged_transition_frame,
        "privileged_transition_angle_deg": privileged_transition_angle_deg,
        "privileged_transition_command_deg": privileged_transition_command_deg,
        "privileged_overshoot_target_deg": privileged_overshoot_target_deg,
        "privileged_backoff_complete": privileged_backoff_complete,
        "privileged_backoff_frame": privileged_backoff_frame,
        "privileged_backoff_target_deg": privileged_target_angle_deg,
        "privileged_backoff_final_angle_deg": privileged_backoff_final_angle_deg,
        "privileged_backoff_tolerance_deg": privileged_backoff_tolerance_deg,
        "final_command_yaw_deg": math.degrees(command_target_yaw_rad),
        "maximum_angle_deg": maximum_angle_deg,
        "crossing_frame": crossing_frame,
        "final_angle_deg": final_angle_deg,
        "final_velocity_rad_s": final_velocity_rad_s,
        "minimum_load_effort_nm": min(effort_trace_nm),
        "maximum_load_effort_nm": max(effort_trace_nm),
        "maximum_abs_wrist_torque_z_nm": max(abs(value) for value in eef_torque_z_trace_nm),
        "closed_finger_positions_m": closed_finger_positions_m,
        "release_finger_positions_m": release_finger_positions_m,
        "gripped": gripped,
        "crossed": crossed,
        "released": released,
        "lifted_clear": lifted_clear,
        "retreated": retreated,
        "knob_root_position_error_m": position_error_m,
        "maximum_tracking_error_m": max(tracking_errors_m),
        "report_before_release": report_before_release,
        "report_after_release": report_after_release,
        "raw_video_path": str(raw_video_path.resolve()) if raw_video_path else None,
        "video_path": str(video_path.resolve()) if video_path else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"FORCEBENCH_STOVE_EXPERT_SUCCESS={int(success)}", flush=True)
    print(
        f"FORCEBENCH_STOVE_EXPERT_EPISODE={dataset_episode_path.resolve() if dataset_episode_path else ''}",
        flush=True,
    )
    print(
        f"FORCEBENCH_STOVE_EXPERT_VIDEO={video_path.resolve() if video_path else ''}",
        flush=True,
    )
    print(
        "FORCEBENCH_STOVE_EXPERT_SUMMARY="
        + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
    if not success:
        raise RuntimeError("stove knob expert trajectory did not satisfy all checks")
except Exception as exc:
    print(
        f"FORCEBENCH_STOVE_EXPERT_ERROR={type(exc).__name__}:{exc}",
        flush=True,
    )
    raise
finally:
    if wrench_capture_active:
        try:
            env.stop_wrench_capture()
        except Exception:
            pass
    if writer is not None:
        writer.release()
    env.close()
    if owns_simulation_app:
        simulation_app.close()
