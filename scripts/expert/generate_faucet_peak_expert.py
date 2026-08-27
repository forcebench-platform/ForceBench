#!/usr/bin/env python3
"""Generate a faucet trajectory using physical valve state, then back off and release."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher


PHYSICS_RATE_HZ = 60
CONTROL_RATE_HZ = 15.0


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument(
    "--dataset_dir",
    type=Path,
    default=None,
    help="write a validated ForceBench HDF5 episode to this directory",
)
parser.add_argument(
    "--physics_rate_hz",
    type=int,
    choices=(PHYSICS_RATE_HZ,),
    default=PHYSICS_RATE_HZ,
)
parser.add_argument("--solver_position_iterations", type=int, choices=(64,), default=64)
parser.add_argument("--fps", type=float, default=15.0)
parser.add_argument("--max_refine_frames", type=int, default=360)
parser.add_argument("--hold_frames", type=int, default=30)
parser.add_argument(
    "--motion_profile",
    choices=("perfect", "low_resistance_pause_continue", "uniform_speed"),
    default="perfect",
    help="select the lever-pull timing profile while keeping the same successful task sequence",
)
parser.add_argument("--forearm_effort_limit", type=float, default=30.0)
parser.add_argument("--gripper_contact_friction", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (
    args.seed < 0
    or args.fps <= 0.0
    or args.max_refine_frames < 1
    or args.hold_frames < 1
    or args.forearm_effort_limit <= 0.0
    or args.gripper_contact_friction <= 0.0
):
    parser.error("seed and frame counts must be non-negative; fps must be positive")
launcher = AppLauncher(args)
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
from forcebench.tasks.faucet_turn.spec import SPEC, detent_resistance_nm, spec_from_layout


def rgb(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().numpy()[..., :3].astype(np.uint8)


def vector(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1).astype(np.float32)


def current_state(environment: ForceBenchEnv) -> np.ndarray:
    return vector(environment.policy_state[0])


task = load_task("faucet_turn", ASSET_DIR)
cfg = ForceBenchEnvCfg()
cfg.seed = args.seed
cfg.scene_profile = task.manifest.scene_profile
cfg.robot.actuators["arm_forearm"].effort_limit_sim = args.forearm_effort_limit
cfg.robot.spawn.physics_material.static_friction = args.gripper_contact_friction
cfg.robot.spawn.physics_material.dynamic_friction = args.gripper_contact_friction
cfg.robot.spawn.physics_material.friction_combine_mode = "max"
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
if not math.isclose(1.0 / env.physics_dt, args.physics_rate_hz, rel_tol=0.0, abs_tol=1.0e-9):
    raise RuntimeError("faucet physics rate does not match the requested 60 Hz contract")
if not math.isclose(1.0 / env.step_dt, CONTROL_RATE_HZ, rel_tol=0.0, abs_tol=1.0e-9):
    raise RuntimeError("faucet control rate does not match the 15 Hz contract")
writer = None
dataset_buffer = EpisodeBuffer()
wrench_capture_active = False
high_rate_wrench = None
dataset_episode_path = None
dataset_validation = None
joint_targets: list[np.ndarray] = []
cartesian_actions: list[np.ndarray] = []
angle_trace_deg: list[float] = []
velocity_trace_rad_s: list[float] = []
effort_trace_nm: list[float] = []
resistance_trace_nm: list[float] = []
phase_trace: list[str] = []
tracking_errors_m: list[float] = []
fingertip_position_trace_m: list[np.ndarray] = []
physical_transition_detected = False
physical_transition_angle_deg: float | None = None
physical_transition_effort_nm: float | None = None
physical_transition_resistance_nm: float | None = None
physical_transition_resistance_drop_nm: float | None = None
plot_reference_angle_deg = np.empty(0, dtype=np.float32)
plot_reference_resistance_nm = np.empty(0, dtype=np.float32)
plot_critical_peak_deg = 0.0
plot_backoff_target_deg = 0.0
plot_overshoot_target_deg = 0.0
plot_angle_max_deg = 110.0
plot_effort_limit_nm = 0.55

CANVAS_SIZE = (960, 512)
PANEL_BG = (25, 28, 33)
PLOT_BG = (17, 20, 24)
TEXT = (238, 241, 245)
MUTED = (156, 165, 176)
GRID = (58, 64, 72)
REFERENCE = (130, 137, 147)
LIVE = (255, 205, 78)
PEAK = (48, 177, 245)
BACKOFF = (98, 205, 116)
OVERSHOOT = (221, 144, 82)
TRANSITION = (75, 225, 132)

PHASE_LABELS = {
    "approach": "APPROACH",
    "descend": "DESCEND",
    "close": "GRASP",
    "cross_and_detect": "CROSS + DETECT",
    "uniform_speed_pull": "UNIFORM-SPEED PULL",
    "low_resistance_entry": "ENTER LOW RESISTANCE",
    "low_resistance_pause": "PAUSE (NOT STOP)",
    "continue_after_slowdown": "CONTINUE PULLING",
    "reverse_to_band": "REVERSE TO BAND",
    "hold_post_peak_band": "HOLD POST-PEAK",
    "release": "OPEN GRIPPER",
    "reorient_after_release": "REORIENT CLEAR",
    "lift_clear": "LIFT CLEAR",
    "retreat_away_from_pipe": "RETREAT OUTBOARD",
}
PHASE_SCHEMA = {name: index for index, name in enumerate(PHASE_LABELS)}

PROFILE_TRAINING_INSTRUCTION_ZH = {
    "perfect": "完美轨迹：跨过峰值、检测掉阻、回退到峰后目标并稳定释放。",
    "low_resistance_pause_continue": (
        "减速不等于停止；在低阻区减速或短暂停顿后，必须继续拉动并完成任务。"
    ),
    "uniform_speed": "拉动阶段保持比较匀速，连续跨过阻力峰值并完成任务。",
}

PROFILE_TRAINING_INSTRUCTION_EN = {
    "perfect": "Perfect trajectory: cross the peak, detect the load drop, back off, stabilize, and release.",
    "low_resistance_pause_continue": (
        "Slowdown is not stop: after slowing down or briefly pausing in the low-resistance zone, "
        "continue pulling and finish the task."
    ),
    "uniform_speed": "Keep the lever-pull phase comparatively uniform in speed and finish the task.",
}


def valve_state() -> tuple[float, float]:
    entity = env.entity("faucet_valve")
    angle = float(wp.to_torch(entity.root_view.get_dof_positions())[0, 0].item())
    velocity = float(wp.to_torch(entity.root_view.get_dof_velocities())[0, 0].item())
    return angle, velocity


def target_quaternion(lever_angle_rad: float) -> torch.Tensor:
    initial = math_utils.quat_from_euler_xyz(
        torch.tensor([math.pi], device=env.device),
        torch.zeros(1, device=env.device),
        torch.zeros(1, device=env.device),
    )
    swing = math_utils.quat_from_angle_axis(
        torch.tensor([SPEC.lever_rotation_sign * lever_angle_rad], device=env.device),
        torch.tensor([[0.0, 1.0, 0.0]], device=env.device),
    )
    return math_utils.quat_mul(swing, initial)


def lever_point(
    pivot_m: tuple[float, float, float],
    radius_m: float,
    angle_rad: float,
    *,
    centerline_y_offset_m: float,
    centerline_x_offset_m: float,
) -> tuple[float, float, float]:
    """Return the grasp point on the +Y-axis lever's vertical X-Z arc."""

    pivot_x, pivot_y, pivot_z = pivot_m
    physical_angle_rad = SPEC.lever_rotation_sign * angle_rad
    cosine = math.cos(physical_angle_rad)
    sine = math.sin(physical_angle_rad)
    return (
        pivot_x + centerline_x_offset_m * cosine + radius_m * sine,
        pivot_y + centerline_y_offset_m,
        pivot_z - centerline_x_offset_m * sine + radius_m * cosine,
    )


def draw_load_plot(canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, width, height = rect
    cv2.putText(
        canvas,
        "RESISTANCE TORQUE vs ANGLE",
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
            effort_nm / plot_effort_limit_nm,
            0.0,
            1.0,
        )
        return (
            round(left + angle_fraction * (right - left)),
            round(bottom - effort_fraction * (bottom - top)),
        )

    for angle_tick in np.linspace(0.0, plot_angle_max_deg, 5):
        xx, _ = point(angle_tick, 0.0)
        cv2.line(canvas, (xx, top), (xx, bottom), GRID, 1)
        label = f"{int(angle_tick)}"
        cv2.putText(
            canvas,
            label,
            (xx - 8, bottom + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            MUTED,
            1,
            cv2.LINE_AA,
        )
    for effort_tick in (0.0, 0.5 * plot_effort_limit_nm, plot_effort_limit_nm):
        _, yy = point(0.0, effort_tick)
        cv2.line(canvas, (left, yy), (right, yy), GRID, 1)
        cv2.putText(
            canvas,
            f"{effort_tick:.2f}",
            (x + 1, yy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            MUTED,
            1,
            cv2.LINE_AA,
        )

    for angle_deg, color, label in (
        (plot_critical_peak_deg, PEAK, "P"),
        (plot_backoff_target_deg, BACKOFF, "B"),
        (plot_overshoot_target_deg, OVERSHOOT, "O"),
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
                    plot_reference_angle_deg, plot_reference_resistance_nm
                )
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            canvas, [reference_points], False, REFERENCE, 2, cv2.LINE_AA
        )
    if len(angle_trace_deg) >= 2:
        live_points = np.asarray(
            [
                point(float(angle), float(effort))
                for angle, effort in zip(angle_trace_deg, resistance_trace_nm)
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [live_points], False, LIVE, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(live_points[-1]), 4, LIVE, -1, cv2.LINE_AA)
    if (
        physical_transition_angle_deg is not None
        and physical_transition_resistance_nm is not None
    ):
        transition_point = point(
            physical_transition_angle_deg, physical_transition_resistance_nm
        )
        cv2.circle(canvas, transition_point, 6, TRANSITION, 2, cv2.LINE_AA)

    legend_y = y + height - 8
    cv2.line(canvas, (x + 50, legend_y - 4), (x + 70, legend_y - 4), REFERENCE, 2)
    cv2.putText(
        canvas,
        "target curve",
        (x + 76, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (x + 184, legend_y - 4), (x + 204, legend_y - 4), LIVE, 2)
    cv2.putText(
        canvas,
        "runtime R(angle)",
        (x + 210, legend_y),
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


def composite_frame(phase: str, critical_peak_deg: float) -> np.ndarray:
    main_bgr = cv2.resize(
        rgb(env.table_rgb[0])[..., ::-1],
        (512, 512),
        interpolation=cv2.INTER_AREA,
    )
    wrist_bgr = cv2.resize(
        rgb(env.wrist_rgb[0])[..., ::-1], (216, 216), interpolation=cv2.INTER_AREA
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
    load_effort_nm = float(state["load_effort_nm"])
    angle_resistance_nm = float(state["resistance_torque_nm"])
    lines = (
        f"SEED  {args.seed:03d}",
        "PHASE",
        PHASE_LABELS.get(phase, phase.upper()),
        f"PEAK    {critical_peak_deg:6.2f} deg",
        f"ANGLE   {angle_deg:6.2f} deg",
        f"R(ANGLE){angle_resistance_nm:6.3f} Nm",
        f"EFFORT  {load_effort_nm:+6.3f} Nm",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (754, 30 + 26 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43 if index != 2 else 0.40,
            MUTED if index == 1 else TEXT,
            1,
            cv2.LINE_AA,
        )
    transition_text = (
        "PHYSICAL DETENT: CROSSED"
        if physical_transition_detected
        else "PHYSICAL DETENT: WAITING"
    )
    cv2.putText(
        canvas,
        transition_text,
        (754, 216),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        TRANSITION if physical_transition_detected else MUTED,
        1,
        cv2.LINE_AA,
    )
    draw_load_plot(canvas, (524, 238, 424, 266))
    return canvas


def command_pose(
    position_m: tuple[float, float, float],
    lever_angle_rad: float,
    gripper: float,
    phase: str,
    critical_peak_deg: float,
) -> tuple[float, float, float, float]:
    target_position = torch.tensor([position_m], dtype=torch.float32, device=env.device)
    target_rotation = target_quaternion(lever_angle_rad)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(target_rotation)
    action = np.asarray(
        [
            *position_m,
            float(roll[0].item()),
            float(pitch[0].item()),
            float(yaw[0].item()),
            2.0 * gripper - 1.0,
        ],
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
                (
                    env.fingertip_midpoint_pos[0],
                    env.fingertip_midpoint_quat[0],
                )
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

    angle_rad, velocity_rad_s = valve_state()
    state = task.debug_state(env)
    joint_targets.append(vector(env.joint_pos[0, :7]))
    cartesian_actions.append(action)
    angle_trace_deg.append(math.degrees(angle_rad))
    velocity_trace_rad_s.append(velocity_rad_s)
    effort_trace_nm.append(float(state["load_effort_nm"]))
    resistance_trace_nm.append(float(state["resistance_torque_nm"]))
    phase_trace.append(phase)
    tracking_errors_m.append(
        float(torch.linalg.vector_norm(env.fingertip_midpoint_pos[0] - target_position[0]).item())
    )
    fingertip_position_trace_m.append(
        env.fingertip_midpoint_pos[0].detach().cpu().numpy().copy()
    )
    env.update_cameras()
    writer.write(composite_frame(phase, critical_peak_deg))
    return (
        angle_rad,
        velocity_rad_s,
        float(state["load_effort_nm"]),
        float(state["resistance_torque_nm"]),
    )


def track_pose(
    position_m: tuple[float, float, float],
    lever_angle_rad: float,
    gripper: float,
    frames: int,
    phase: str,
    critical_peak_deg: float,
) -> None:
    for _ in range(frames):
        command_pose(
            position_m,
            lever_angle_rad,
            gripper,
            phase,
            critical_peak_deg,
        )


try:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env.set_episode_seed(args.seed)
    env.reset(seed=args.seed)
    layout = env.layout
    valve_x, valve_y, valve_z = (float(value) for value in layout["valve_position_m"])
    sequence = layout["expert_detent_sequence"]
    rise_start_deg = float(sequence["rise_start_angle_deg"])
    critical_peak_deg = float(sequence["critical_peak_angle_deg"])
    drop_end_deg = float(sequence["drop_end_angle_deg"])
    low_load_end_deg = float(sequence["low_load_end_angle_deg"])
    overshoot_target_deg = float(sequence["overshoot_target_deg"])
    backoff_target_deg = float(sequence["backoff_target_deg"])
    backoff_tolerance_deg = float(sequence["backoff_tolerance_deg"])
    stop_speed_rad_s = float(sequence["stop_speed_rad_s"])
    detent_center_deg = float(layout["valve_load_profile"]["detent_center_deg"])
    sampled_peak_torque_nm = float(
        layout["valve_load_profile"]["detent_peak_torque_nm"]
    )
    episode_spec = spec_from_layout(layout)
    plot_critical_peak_deg = critical_peak_deg
    plot_backoff_target_deg = backoff_target_deg
    plot_overshoot_target_deg = overshoot_target_deg
    plot_angle_max_deg = min(
        float(episode_spec.open_limit_deg),
        critical_peak_deg + 18.0,
    )
    plot_effort_limit_nm = 1.15 * sampled_peak_torque_nm
    plot_reference_angle_deg = np.linspace(
        0.0, plot_angle_max_deg, 361, dtype=np.float32
    )
    plot_reference_resistance_nm = np.asarray(
        [
            detent_resistance_nm(math.radians(float(angle)), episode_spec)
            for angle in plot_reference_angle_deg
        ],
        dtype=np.float32,
    )
    pivot_position = tuple(
        base + offset
        for base, offset in zip(
            (valve_x, valve_y, valve_z),
            episode_spec.lever_pivot_offset_m,
        )
    )
    # Grasp the upper shaft rather than its rounded tip so both fingers remain
    # wrapped around the lever while it crosses the low-angle detent.
    grasp_radius_m = episode_spec.expert_grasp_radius_m

    def grasp_arc_point(radius_m: float, angle_rad: float) -> tuple[float, float, float]:
        return lever_point(
            pivot_position,
            radius_m,
            angle_rad,
            centerline_y_offset_m=episode_spec.lever_rod_centerline_y_offset_m,
            centerline_x_offset_m=episode_spec.lever_rod_centerline_x_offset_m,
        )

    grasp_position = grasp_arc_point(grasp_radius_m, 0.0)
    approach_position = (
        grasp_position[0],
        grasp_position[1],
        grasp_position[2] + 0.090,
    )

    env.reset_cameras()
    env.update_cameras()
    cv2.imwrite(str(args.output_dir / "main_initial.png"), rgb(env.table_rgb[0])[..., ::-1])
    raw_video_path = args.output_dir / "faucet_post_peak_expert_raw.mp4"
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

    track_pose(approach_position, 0.0, 1.0, 60, "approach", critical_peak_deg)
    track_pose(grasp_position, 0.0, 1.0, 40, "descend", critical_peak_deg)
    track_pose(grasp_position, 0.0, 0.0, 20, "close", critical_peak_deg)
    closed_finger_positions_m = (
        env.robot.data.joint_pos.torch[0, 7:9].detach().cpu().tolist()
    )
    grasp_contact_offset_m = (
        env.fingertip_midpoint_pos[0].detach().cpu().numpy()
        - np.asarray(grasp_position, dtype=np.float32)
    )

    peak_observed = False
    minimum_load_effort_nm = 0.0
    maximum_resistance_nm = 0.0
    physical_transition_frame = None
    overshoot_angle_deg = 0.0
    cross_frames = 0
    command_position = grasp_position
    # The expert transition is determined only from the simulated valve's
    # physical joint state.  Wrist wrench channels are still recorded in the
    # standard dataset, but they never control the scripted trajectory.
    physical_transition_sustain_frames = 3
    physical_transition_streak = 0
    # Follow the same circular arc for all profiles so timing is the only
    # intentional difference.  The pause profile freezes the commanded angle
    # inside the post-drop low-resistance zone, then explicitly resumes.
    cross_ramp_frames = 100
    nominal_cross_command_deg = overshoot_target_deg + 0.80
    uniform_command_step_deg = nominal_cross_command_deg / cross_ramp_frames
    pause_requested_frames = 18
    pause_started = False
    pause_completed = False
    pause_frames_recorded = 0
    pause_start_frame: int | None = None
    pause_end_frame: int | None = None
    pause_command_angle_deg: float | None = None
    pause_start_angle_deg: float | None = None
    pause_end_angle_deg: float | None = None
    resume_frames_recorded = 0
    resume_target_deg = min(
        low_load_end_deg - 0.25,
        overshoot_target_deg + 0.60,
    )
    for cross_frames in range(1, 241):
        if args.motion_profile == "uniform_speed":
            command_angle_deg = min(
                cross_frames * uniform_command_step_deg,
                nominal_cross_command_deg + 10.0,
            )
            pull_phase = "uniform_speed_pull"
        elif args.motion_profile == "low_resistance_pause_continue" and pause_started:
            if pause_frames_recorded < pause_requested_frames:
                command_angle_deg = float(pause_command_angle_deg)
                pause_frames_recorded += 1
                pull_phase = "low_resistance_pause"
            else:
                if not pause_completed:
                    pause_completed = True
                    pause_end_frame = len(joint_targets)
                    pause_end_angle_deg = math.degrees(valve_state()[0])
                resume_frames_recorded += 1
                command_angle_deg = float(pause_command_angle_deg) + (
                    resume_frames_recorded * uniform_command_step_deg
                )
                pull_phase = "continue_after_slowdown"
        else:
            command_angle_deg = (
                nominal_cross_command_deg
                * min(cross_frames / cross_ramp_frames, 1.0)
                + min(max(cross_frames - cross_ramp_frames, 0) * 0.05, 10.0)
            )
            pull_phase = (
                "low_resistance_entry"
                if args.motion_profile == "low_resistance_pause_continue"
                else "cross_and_detect"
            )
        command_angle_rad = math.radians(command_angle_deg)
        command_position = tuple(
            float(value)
            for value in np.asarray(
                grasp_arc_point(grasp_radius_m, command_angle_rad),
                dtype=np.float32,
            )
            + grasp_contact_offset_m
        )
        live_angle_rad, _ = valve_state()
        angle_rad, _, load_effort_nm, measured_resistance_nm = command_pose(
            command_position,
            live_angle_rad,
            0.0,
            pull_phase,
            critical_peak_deg,
        )
        angle_deg = math.degrees(angle_rad)
        minimum_load_effort_nm = min(minimum_load_effort_nm, load_effort_nm)
        maximum_resistance_nm = max(maximum_resistance_nm, measured_resistance_nm)
        peak_observed = peak_observed or angle_deg >= critical_peak_deg
        if peak_observed and angle_deg >= drop_end_deg:
            physical_transition_streak += 1
        else:
            physical_transition_streak = 0
        if (
            not physical_transition_detected
            and physical_transition_streak >= physical_transition_sustain_frames
        ):
            physical_transition_detected = True
            physical_transition_frame = len(joint_targets) - 1
            physical_transition_angle_deg = angle_deg
            physical_transition_effort_nm = load_effort_nm
            physical_transition_resistance_nm = measured_resistance_nm
            physical_transition_resistance_drop_nm = max(
                0.0, maximum_resistance_nm - measured_resistance_nm
            )
        overshoot_angle_deg = max(overshoot_angle_deg, angle_deg)
        if (
            args.motion_profile == "low_resistance_pause_continue"
            and not pause_started
            and physical_transition_detected
            and angle_deg >= drop_end_deg + 0.08
        ):
            pause_started = True
            pause_start_frame = len(joint_targets)
            pause_start_angle_deg = angle_deg
            pause_command_angle_deg = angle_deg
        completion_target_deg = (
            resume_target_deg
            if args.motion_profile == "low_resistance_pause_continue"
            else overshoot_target_deg
        )
        profile_motion_complete = (
            args.motion_profile != "low_resistance_pause_continue"
            or (pause_completed and resume_frames_recorded >= 4)
        )
        if (
            physical_transition_detected
            and profile_motion_complete
            and angle_deg >= completion_target_deg
        ):
            break

    overshoot_live_angle_rad, _ = valve_state()
    overshoot_ideal_position = np.asarray(
        grasp_arc_point(grasp_radius_m, overshoot_live_angle_rad),
        dtype=np.float32,
    )
    contact_offset_m = (
        env.fingertip_midpoint_pos[0].detach().cpu().numpy()
        - overshoot_ideal_position
    )

    stable_frames = 0
    refinement_frames = 0
    backoff_angle_window_deg: list[float] = []
    backoff_command_angle_deg = math.degrees(overshoot_live_angle_rad)
    for refinement_frames in range(1, args.max_refine_frames + 1):
        angle_rad, _ = valve_state()
        angle_deg = math.degrees(angle_rad)
        angle_error_deg = backoff_target_deg - angle_deg
        backoff_command_angle_deg += max(
            -0.30,
            min(0.30, 0.18 * angle_error_deg),
        )
        ideal_position = grasp_arc_point(
            grasp_radius_m,
            math.radians(backoff_command_angle_deg),
        )
        command_position = tuple(
            float(value)
            for value in np.asarray(ideal_position) + contact_offset_m
        )
        command_pose(
            command_position,
            angle_rad,
            0.0,
            "reverse_to_band",
            critical_peak_deg,
        )
        angle_rad, _ = valve_state()
        backoff_angle_window_deg.append(math.degrees(angle_rad))
        backoff_angle_window_deg = backoff_angle_window_deg[-12:]
        if (
            len(backoff_angle_window_deg) == 12
            and max(
                abs(value - backoff_target_deg)
                for value in backoff_angle_window_deg
            )
            <= 0.5 * backoff_tolerance_deg
            and max(backoff_angle_window_deg) - min(backoff_angle_window_deg)
            <= 0.08
        ):
            stable_frames += 1
        else:
            stable_frames = 0
        if stable_frames >= 12:
            break

    for _ in range(args.hold_frames):
        hold_angle_rad, _ = valve_state()
        hold_error_deg = backoff_target_deg - math.degrees(hold_angle_rad)
        backoff_command_angle_deg += max(
            -0.10,
            min(0.10, 0.12 * hold_error_deg),
        )
        command_position = tuple(
            float(value)
            for value in np.asarray(
                grasp_arc_point(
                    grasp_radius_m,
                    math.radians(backoff_command_angle_deg),
                )
            )
            + contact_offset_m
        )
        command_pose(
            command_position,
            hold_angle_rad,
            0.0,
            "hold_post_peak_band",
            critical_peak_deg,
        )
        backoff_angle_window_deg.append(math.degrees(valve_state()[0]))
        backoff_angle_window_deg = backoff_angle_window_deg[-12:]

    backoff_angle_rad, backoff_velocity_rad_s = valve_state()
    backoff_angle_deg = math.degrees(backoff_angle_rad)
    backoff_error_deg = abs(backoff_angle_deg - backoff_target_deg)
    backoff_report = env.evaluate_final_success()
    release_frames = 0
    release_finger_positions_m = [0.0, 0.0]
    # Opening the fingers unloads their elastic pinch and otherwise nudges the
    # reverse-opening lever back toward the peak.  A small opening-direction
    # positional bias holds the post-peak angle only while contact remains;
    # once the gap clears the rod, the open gripper no longer drives it.
    release_position = tuple(
        float(value)
        for value in np.asarray(
            grasp_arc_point(
                grasp_radius_m,
                math.radians(backoff_target_deg + 2.5),
            )
        )
        + contact_offset_m
    )
    command_position = release_position
    for release_frames in range(1, 25):
        release_angle_rad, _ = valve_state()
        command_pose(
            command_position,
            release_angle_rad,
            1.0,
            "release",
            critical_peak_deg,
        )
        release_finger_positions_m = (
            env.robot.data.joint_pos.torch[0, 7:9].detach().cpu().tolist()
        )
        if release_frames >= 8 and min(release_finger_positions_m) >= 0.0395:
            break
    lift_position = (
        command_position[0],
        command_position[1],
        grasp_position[2] + 0.130,
    )
    outboard_position = (
        command_position[0],
        valve_y - 0.085,
        grasp_position[2] + 0.155,
    )
    # Open first, then lift before moving laterally away from the pipe.  Keep
    # the post-peak wrist orientation while clearing the lever: rotating back
    # to zero in place would let an open finger sweep the valve closed again.
    track_pose(
        lift_position,
        backoff_angle_rad,
        1.0,
        60,
        "lift_clear",
        critical_peak_deg,
    )
    track_pose(
        outboard_position,
        backoff_angle_rad,
        1.0,
        40,
        "retreat_away_from_pipe",
        critical_peak_deg,
    )

    writer.release()
    writer = None
    video_path = args.output_dir / "faucet_post_peak_expert.mp4"
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
    cv2.imwrite(str(args.output_dir / "main_final.png"), rgb(env.table_rgb[0])[..., ::-1])
    cv2.imwrite(str(args.output_dir / "wrist_final.png"), rgb(env.wrist_rgb[0])[..., ::-1])

    final_angle_rad, final_velocity_rad_s = valve_state()
    final_angle_deg = math.degrees(final_angle_rad)
    final_backoff_error_deg = abs(final_angle_deg - backoff_target_deg)
    final_stopped_at_backoff = bool(
        final_backoff_error_deg <= backoff_tolerance_deg
        and abs(final_velocity_rad_s) <= stop_speed_rad_s
    )
    gripped = max(closed_finger_positions_m) < 0.0395
    backoff_window_range_deg = (
        max(backoff_angle_window_deg) - min(backoff_angle_window_deg)
        if backoff_angle_window_deg
        else math.inf
    )
    backoff_drift_rate_deg_s = (
        abs(backoff_angle_window_deg[-1] - backoff_angle_window_deg[0])
        / env.step_dt
        / max(1, len(backoff_angle_window_deg) - 1)
        if len(backoff_angle_window_deg) >= 2
        else math.inf
    )
    stopped_at_backoff = bool(
        backoff_window_range_deg <= 0.08
        and backoff_drift_rate_deg_s <= 0.10
    )
    released = env.gripper_open_fraction >= 0.90
    fingertip_position = env.fingertip_midpoint_pos[0].detach().cpu().numpy()
    lifted_clear = float(fingertip_position[2]) >= grasp_position[2] + 0.09
    retreated_outboard = float(fingertip_position[1]) <= valve_y - 0.055
    post_release_report = env.evaluate_final_success()
    backoff_checkpoint_success = bool(
        backoff_error_deg <= backoff_tolerance_deg
        and stopped_at_backoff
    )
    success = bool(
        gripped
        and peak_observed
        and physical_transition_detected
        and physical_transition_angle_deg is not None
        and physical_transition_angle_deg > critical_peak_deg + 0.15
        and overshoot_angle_deg >= overshoot_target_deg
        and backoff_target_deg > detent_center_deg
        and backoff_checkpoint_success
        and released
        and lifted_clear
        and retreated_outboard
        and min(release_finger_positions_m) >= 0.0395
        and post_release_report["success"]
    )
    valve_root_position = (
        env.entity_local_pose("faucet_valve")[0][0].detach().cpu().numpy().tolist()
    )
    position_error_m = float(
        np.linalg.norm(
            np.asarray(valve_root_position) - np.asarray(layout["valve_position_m"])
        )
    )
    success = success and position_error_m <= 1.0e-5
    water_outlet_position = np.asarray(layout["water_outlet_position_m"], dtype=np.float32)
    minimum_fingertip_outlet_distance_m = min(
        float(np.linalg.norm(position - water_outlet_position))
        for position in fingertip_position_trace_m
    )
    pipe_clearance_safe = minimum_fingertip_outlet_distance_m >= 0.10
    success = success and pipe_clearance_safe
    phase_durations_s = {
        phase_name: phase_trace.count(phase_name) * env.step_dt
        for phase_name in dict.fromkeys(phase_trace)
    }
    pause_angle_change_deg = (
        abs(float(pause_end_angle_deg) - float(pause_start_angle_deg))
        if pause_start_angle_deg is not None and pause_end_angle_deg is not None
        else None
    )
    resume_angle_gain_deg = (
        overshoot_angle_deg - float(pause_end_angle_deg)
        if pause_end_angle_deg is not None
        else None
    )
    profile_requirement_satisfied = True
    if args.motion_profile == "low_resistance_pause_continue":
        profile_requirement_satisfied = bool(
            pause_started
            and pause_completed
            and pause_frames_recorded == pause_requested_frames
            and resume_frames_recorded >= 4
            and resume_angle_gain_deg is not None
            and resume_angle_gain_deg >= 0.35
        )
    success = success and profile_requirement_satisfied

    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        joint_targets=np.asarray(joint_targets, dtype=np.float32),
        cartesian_actions=np.asarray(cartesian_actions, dtype=np.float32),
        valve_angle_deg=np.asarray(angle_trace_deg, dtype=np.float32),
        valve_velocity_rad_s=np.asarray(velocity_trace_rad_s, dtype=np.float32),
        valve_load_effort_nm=np.asarray(effort_trace_nm, dtype=np.float32),
        valve_resistance_torque_nm=np.asarray(resistance_trace_nm, dtype=np.float32),
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
    if success and args.dataset_dir is not None:
        task_metadata = dict(task.episode_metadata())
        asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
        asset_paths.update(env.scene_asset_paths)
        asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
        metadata = {
            "task": "faucet_turn",
            "prompt": task.manifest.prompt,
            "success": True,
            "end_reason": "scripted_faucet_post_peak_final_check",
            "seed": args.seed,
            "initial_layout": env.layout,
            "final_success_report": post_release_report,
            "control_rate_hz": CONTROL_RATE_HZ,
            "video_fps": args.fps,
            "physics_rate_hz": float(args.physics_rate_hz),
            "physics_steps_per_control": int(args.physics_rate_hz // CONTROL_RATE_HZ),
            "solver_position_iterations": int(args.solver_position_iterations),
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "scene_profile": env.scene_profile_name,
            "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "action_semantics": "world_target_tcp_xyz_rpy_gripper_open_minus1_plus1",
            "control_source": "scripted_reverse_vertical_lever_physical_detent_backoff",
            "success_check_timing": "after_detent_cross_backoff_release_and_retract",
            "images_present": True,
            "image_shapes_hwc": {
                "main": [224, 224, 3],
                "wrist": [224, 224, 3],
            },
            "phase_schema": PHASE_SCHEMA,
            "motion_profile": args.motion_profile,
            "training_instruction_zh": PROFILE_TRAINING_INSTRUCTION_ZH[args.motion_profile],
            "training_instruction_en": PROFILE_TRAINING_INSTRUCTION_EN[args.motion_profile],
            "deceleration_is_not_stop": (
                args.motion_profile == "low_resistance_pause_continue"
            ),
            "uniform_speed_scope": (
                "lever_pull_phase" if args.motion_profile == "uniform_speed" else None
            ),
            "scripted_expert_generator_version": (
                "faucet_physical_state_transition_fixed_resistance_physics60"
            ),
            "fixed_detent_peak_torque_nm": sampled_peak_torque_nm,
            "expert_transition_signal": "valve_joint_angle_rad",
            "expert_uses_analytic_resistance_for_transition": False,
            "physical_transition_condition": {
                "peak_crossing_angle_deg": critical_peak_deg,
                "detent_drop_end_angle_deg": drop_end_deg,
                "sustain_frames": physical_transition_sustain_frames,
            },
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
    summary = {
        "task": "faucet_turn",
        "trajectory_type": f"{args.motion_profile}_cross_peak_backoff_release",
        "motion_profile": args.motion_profile,
        "training_instruction_zh": PROFILE_TRAINING_INSTRUCTION_ZH[args.motion_profile],
        "training_instruction_en": PROFILE_TRAINING_INSTRUCTION_EN[args.motion_profile],
        "profile_requirement_satisfied": profile_requirement_satisfied,
        "seed": args.seed,
        "success": success,
        "episode": (
            str(dataset_episode_path.resolve())
            if dataset_episode_path is not None
            else None
        ),
        "dataset_validation": dataset_validation,
        "layout": layout,
        "frames": len(joint_targets),
        "fps": args.fps,
        "phase_durations_s": phase_durations_s,
        "physics_rate_hz": args.physics_rate_hz,
        "arm_control_mode": env.arm_control_mode,
        "arm_controller_diagnostics": env.arm_controller_diagnostics,
        "rise_start_angle_deg": rise_start_deg,
        "critical_peak_angle_deg": critical_peak_deg,
        "drop_end_angle_deg": drop_end_deg,
        "low_load_end_angle_deg": low_load_end_deg,
        "cross_frames": cross_frames,
        "uniform_command_step_deg_per_frame": uniform_command_step_deg,
        "pause_requested_frames": pause_requested_frames,
        "pause_frames_recorded": pause_frames_recorded,
        "pause_start_frame": pause_start_frame,
        "pause_end_frame": pause_end_frame,
        "pause_start_angle_deg": pause_start_angle_deg,
        "pause_end_angle_deg": pause_end_angle_deg,
        "pause_angle_change_deg": pause_angle_change_deg,
        "resume_frames_recorded": resume_frames_recorded,
        "resume_target_deg": resume_target_deg,
        "resume_angle_gain_deg": resume_angle_gain_deg,
        "grasp_radius_m": grasp_radius_m,
        "contact_offset_m": contact_offset_m.tolist(),
        "peak_observed": peak_observed,
        "minimum_load_effort_nm": minimum_load_effort_nm,
        "maximum_resistance_nm": maximum_resistance_nm,
        "physical_transition_detected": physical_transition_detected,
        "physical_transition_frame": physical_transition_frame,
        "physical_transition_angle_deg": physical_transition_angle_deg,
        "physical_transition_effort_nm": physical_transition_effort_nm,
        "physical_transition_resistance_nm": physical_transition_resistance_nm,
        "physical_transition_resistance_drop_nm": (
            physical_transition_resistance_drop_nm
        ),
        "physical_transition_sustain_frames": physical_transition_sustain_frames,
        "overshoot_target_deg": overshoot_target_deg,
        "overshoot_angle_deg": overshoot_angle_deg,
        "backoff_target_deg": backoff_target_deg,
        "backoff_tolerance_deg": backoff_tolerance_deg,
        "backoff_angle_deg": backoff_angle_deg,
        "backoff_error_deg": backoff_error_deg,
        "backoff_velocity_rad_s": backoff_velocity_rad_s,
        "backoff_window_range_deg": backoff_window_range_deg,
        "backoff_drift_rate_deg_s": backoff_drift_rate_deg_s,
        "backoff_checkpoint_success": backoff_checkpoint_success,
        "backoff_task_report": backoff_report,
        "final_valve_angle_deg": final_angle_deg,
        "final_backoff_error_deg": final_backoff_error_deg,
        "final_valve_velocity_rad_s": final_velocity_rad_s,
        "final_load_effort_nm": float(task.debug_state(env)["load_effort_nm"]),
        "closed_finger_positions_m": closed_finger_positions_m,
        "gripped": gripped,
        "stopped_at_backoff": stopped_at_backoff,
        "final_stopped_at_backoff": final_stopped_at_backoff,
        "released": released,
        "release_frames": release_frames,
        "release_finger_positions_m": release_finger_positions_m,
        "lifted_clear": lifted_clear,
        "retreated_outboard": retreated_outboard,
        "minimum_fingertip_outlet_distance_m": minimum_fingertip_outlet_distance_m,
        "pipe_clearance_safe": pipe_clearance_safe,
        "final_fingertip_position_m": fingertip_position.tolist(),
        "post_release_task_report": post_release_report,
        "refinement_frames": refinement_frames,
        "maximum_tracking_error_m": max(tracking_errors_m),
        "valve_root_position_error_m": position_error_m,
        "raw_video_path": str(raw_video_path.resolve()),
        "video_path": str(video_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"FORCEBENCH_FAUCET_PEAK_EXPERT_SUCCESS={int(success)}", flush=True)
    print(f"FORCEBENCH_FAUCET_PEAK_EXPERT_VIDEO={video_path.resolve()}", flush=True)
    print("FORCEBENCH_FAUCET_PEAK_EXPERT_SUMMARY=" + json.dumps(summary, separators=(",", ":")), flush=True)
    if not success:
        raise RuntimeError("faucet post-peak expert trajectory did not satisfy all checks")
except Exception as exc:
    print(
        f"FORCEBENCH_FAUCET_PEAK_EXPERT_ERROR={type(exc).__name__}:{exc}",
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
    simulation_app.close()
