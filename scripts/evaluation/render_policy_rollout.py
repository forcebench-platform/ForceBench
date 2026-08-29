#!/usr/bin/env python3
"""Physics-replay a saved policy rollout and render main/wrist views to MP4."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

from forcebench.control import ARM_CONTROL_MODE
from forcebench.recording import CANONICAL_RUNTIME_PROFILE


CANONICAL_PHYSICS_RATE_HZ = int(CANONICAL_RUNTIME_PROFILE["physics_rate_hz"])
CANONICAL_SOLVER_POSITION_ITERATIONS = int(
    CANONICAL_RUNTIME_PROFILE["solver_position_iterations"]
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rollout_dir", type=Path, required=True)
parser.add_argument("--video_path", type=Path, required=True)
parser.add_argument("--task", default=None)
parser.add_argument("--fps", type=float, default=15.0)
parser.add_argument("--frame_stride", type=int, default=1)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument(
    "--physics_rate_hz",
    type=int,
    choices=(CANONICAL_PHYSICS_RATE_HZ,),
    default=CANONICAL_PHYSICS_RATE_HZ,
)
parser.add_argument(
    "--solver_position_iterations",
    type=int,
    choices=(CANONICAL_SOLVER_POSITION_ITERATIONS,),
    default=CANONICAL_SOLVER_POSITION_ITERATIONS,
)
parser.add_argument("--final_settle_frames", type=int, default=8)
parser.add_argument("--progress_interval", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.fps <= 0 or args.frame_stride < 1 or args.final_settle_frames < 0 or args.progress_interval < 0:
    parser.error("fps and frame stride must be positive; frame counts must be non-negative")
if args.video_path.exists():
    parser.error(f"video already exists: {args.video_path}")

summary_path = args.rollout_dir / "summary.json"
rollout_path = args.rollout_dir / "rollout.npz"
if not summary_path.is_file() or not rollout_path.is_file():
    parser.error(f"rollout directory is incomplete: {args.rollout_dir}")
saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
if saved_summary.get("arm_control_mode") != ARM_CONTROL_MODE:
    parser.error(
        f"rollout controller {saved_summary.get('arm_control_mode')!r} is not the canonical "
        f"{ARM_CONTROL_MODE!r} controller"
    )
task_name = args.task or str(saved_summary["task"])
seed = int(saved_summary["seed"])
physics_rate_hz = args.physics_rate_hz
if int(saved_summary.get("physics_rate_hz", physics_rate_hz)) != physics_rate_hz:
    parser.error(
        "rollout physics rate does not match the canonical collection/evaluation "
        f"contract ({physics_rate_hz} Hz)"
    )

launcher = AppLauncher(args)
simulation_app = launcher.app

import cv2
import numpy as np
import torch
from isaaclab.utils import math as math_utils

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.registry import load_task


def rgb(value) -> np.ndarray:
    image = value.detach().cpu().numpy()
    return image[..., :3].astype(np.uint8)


def current_xyz_rpy(env: ForceBenchEnv) -> np.ndarray:
    return env.policy_state[0, :6].detach().cpu().numpy().astype(np.float32)


def composite_frame(env: ForceBenchEnv, frame: int, total: int, gripper: float) -> np.ndarray:
    main_bgr = rgb(env.table_rgb[0])[..., ::-1]
    wrist_bgr = rgb(env.wrist_rgb[0])[..., ::-1]
    main_bgr = cv2.resize(main_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    wrist_bgr = cv2.resize(wrist_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    canvas = np.full((512, 768, 3), 242, dtype=np.uint8)
    canvas[:, :512] = main_bgr
    canvas[:256, 512:] = wrist_bgr
    cv2.putText(canvas, "WRIST", (520, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    lines = (
        f"seed: {seed}",
        f"frame: {frame}/{total}",
        f"time: {frame / args.fps:.1f}s",
        f"gripper: {gripper:.2f}",
        f"control: {ARM_CONTROL_MODE}",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (530, 300 + 38 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    return canvas


env = None
writer = None
try:
    data = np.load(rollout_path)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"rollout actions must be finite [T, 7], got {actions.shape}")

    task = load_task(task_name, ASSET_DIR)
    cfg = ForceBenchEnvCfg()
    cfg.seed = seed
    recorded_scene = str(saved_summary["scene_profile"])
    if recorded_scene != task.manifest.scene_profile:
        raise ValueError(
            f"rollout scene {recorded_scene!r} does not match the task manifest "
            f"{task.manifest.scene_profile!r}"
        )
    recorded_camera = saved_summary.get("camera_profile")
    expected_camera = json.loads(json.dumps(asdict(task.camera_profile)))
    if isinstance(recorded_camera, dict) and recorded_camera != expected_camera:
        raise ValueError(
            "rollout camera profile does not match the current task camera profile"
        )
    cfg.scene_profile = task.manifest.scene_profile
    configure_runtime(
        cfg,
        device=args.device,
        physics_rate_hz=physics_rate_hz,
        solver_iterations=args.solver_position_iterations,
    )
    cfg.num_rerenders_on_reset = 3
    # Each recorded frame is rendered explicitly after its physics step.
    cfg.sim.render_interval = cfg.decimation * (
        len(actions) + args.final_settle_frames + 2
    )
    cfg.sim.render.antialiasing_mode = "DLAA"
    cfg.sim.render.enable_translucency = False
    cfg.sim.render.enable_reflections = False
    cfg.sim.render.enable_global_illumination = False
    cfg.sim.render.enable_dlssg = False
    cfg.sim.render.enable_dl_denoiser = True
    cfg.sim.render.samples_per_pixel = 1
    cfg.sim.render.enable_shadows = False
    cfg.sim.render.enable_ambient_occlusion = False

    env = ForceBenchEnv(
        cfg=cfg,
        task_runtime=task,
        main_image_size=args.main_image_size,
        wrist_image_size=args.wrist_image_size,
    )
    env.set_episode_seed(seed)
    env.reset(seed=seed)
    initial_arm = env.robot.data.joint_pos.torch[0, :7].clone()
    env.restore_fixed_arm_pose(initial_arm)
    env.reset_cameras()
    for _ in range(3):
        env.sim.render()
    env.update_cameras()

    args.video_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = args.fps / args.frame_stride
    writer = cv2.VideoWriter(
        str(args.video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (768, 512),
    )
    if not writer.isOpened():
        raise OSError(f"failed to open video writer: {args.video_path}")

    zero_action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    total = len(actions) + args.final_settle_frames
    written_frames = 0
    for frame, action in enumerate(actions, start=1):
        gripper = float(np.clip((action[6] + 1.0) * 0.5, 0.0, 1.0))
        env.set_cartesian_rpy_target(action[:6], gripper)
        env.step(zero_action)
        if args.progress_interval and (frame % args.progress_interval == 0 or frame == len(actions)):
            print(f"FORCEBENCH_POLICY_REPLAY_PROGRESS={frame}/{len(actions)}", flush=True)
        if frame % args.frame_stride == 0 or frame == len(actions):
            env.update_cameras()
            writer.write(composite_frame(env, frame, total, gripper))
            written_frames += 1

    settle_gripper = env.gripper_open_fraction
    settle_pose = current_xyz_rpy(env)
    for offset in range(1, args.final_settle_frames + 1):
        env.set_cartesian_rpy_target(settle_pose, settle_gripper)
        env.step(zero_action)
        frame = len(actions) + offset
        if frame % args.frame_stride == 0 or offset == args.final_settle_frames:
            env.update_cameras()
            writer.write(composite_frame(env, frame, total, settle_gripper))
            written_frames += 1

    writer.release()
    writer = None
    replay_report = env.evaluate_final_success()
    result = {
        "seed": seed,
        "frames": written_frames,
        "source_frames": total,
        "fps": output_fps,
        "frame_stride": args.frame_stride,
        "arm_control_mode": env.arm_control_mode,
        "arm_controller_profile": env.arm_controller_profile,
        "scene_profile": env.scene_profile_name,
        "camera_profile": expected_camera,
        "physics_rate_hz": physics_rate_hz,
        "duration_s": total / args.fps,
        "video_path": str(args.video_path.resolve()),
        "source_success_report": saved_summary.get("success_report"),
        "replay_success_report": replay_report,
    }
    result_path = args.video_path.with_suffix(".json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FORCEBENCH_POLICY_REPLAY_RESULT=" + json.dumps(result, separators=(",", ":")), flush=True)
except BaseException:
    print("FORCEBENCH_POLICY_REPLAY_ERROR=python_exception", flush=True)
    traceback.print_exc()
    raise
finally:
    if writer is not None:
        writer.release()
    if env is not None:
        env.close()
    simulation_app.close()
