#!/usr/bin/env python3
"""Replay a saved CoRiBench demonstration through the physics environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from coribench.control import ARM_CONTROL_MODE
from coribench.registry import task_names


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("episode", type=Path)
parser.add_argument("--task", choices=task_names(), default=None)
parser.add_argument("--preview_dir", type=Path, default=None)
parser.add_argument("--final_settle_frames", type=int, default=None)
parser.add_argument("--progress_interval", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.final_settle_frames is not None and args.final_settle_frames < 0:
    parser.error("final_settle_frames must be non-negative")
if args.progress_interval < 0:
    parser.error("progress_interval must be non-negative")

launcher = AppLauncher(args)
simulation_app = launcher.app

import cv2
import h5py
import numpy as np
import torch
from isaaclab.utils import math as math_utils

from coribench.env import CoRiBenchEnv
from coribench.env_cfg import ASSET_DIR, CoRiBenchEnvCfg, configure_runtime
from coribench.recording import validate_hdf5_episode
from coribench.registry import load_task


def current_xyz_rpy(env: CoRiBenchEnv) -> torch.Tensor:
    return env.policy_state[0, :6].clone()


env = None
try:
    validation = validate_hdf5_episode(args.episode)
    with h5py.File(args.episode, "r") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        recorded_joints = np.asarray(handle["observations/joint_position"], dtype=np.float32)
        commands = np.asarray(handle["actions/canonical"], dtype=np.float32)

    task_name = args.task or str(metadata["task"])
    if task_name != str(metadata["task"]):
        raise ValueError(f"episode task is {metadata['task']!r}, not {task_name!r}")
    if commands.shape != (int(validation["control_frames"]), 7) or not np.isfinite(commands).all():
        raise ValueError("canonical Cartesian action stream is invalid")

    task = load_task(task_name, ASSET_DIR)
    forbidden_prefixes = (
        "isaaclab_tasks.direct." + "factory.factory",
        "isaaclab_tasks.direct." + "forge.forge",
    )
    forbidden = sorted(name for name in sys.modules if name.startswith(forbidden_prefixes))
    if forbidden:
        raise RuntimeError(f"forbidden Factory/FORGE runtime modules were loaded: {forbidden}")

    cfg = CoRiBenchEnvCfg()
    seed = int(metadata["seed"])
    cfg.seed = seed
    recorded_scene = str(metadata["scene_profile"])
    if recorded_scene != task.manifest.scene_profile:
        raise ValueError(
            f"episode scene {recorded_scene!r} does not match the task manifest "
            f"{task.manifest.scene_profile!r}"
        )
    cfg.scene_profile = task.manifest.scene_profile
    if metadata.get("arm_control_mode") != ARM_CONTROL_MODE:
        raise ValueError(
            f"episode controller {metadata.get('arm_control_mode')!r} is not the canonical "
            f"{ARM_CONTROL_MODE!r} controller"
        )
    physics_rate_hz = int(round(float(metadata["physics_rate_hz"])))
    configure_runtime(cfg, device=args.device, physics_rate_hz=physics_rate_hz, solver_iterations=64)
    cfg.num_rerenders_on_reset = 3
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.render.antialiasing_mode = "DLAA"
    cfg.sim.render.enable_translucency = False
    cfg.sim.render.enable_reflections = False
    cfg.sim.render.enable_global_illumination = False
    cfg.sim.render.enable_dlssg = False
    cfg.sim.render.enable_dl_denoiser = True
    cfg.sim.render.samples_per_pixel = 1
    cfg.sim.render.enable_shadows = False
    cfg.sim.render.enable_ambient_occlusion = False

    image_shapes = metadata.get("image_shapes_hwc", {})
    main_size = int(image_shapes.get("main", [224])[0])
    wrist_size = int(image_shapes.get("wrist", [224])[0])
    env = CoRiBenchEnv(
        cfg=cfg,
        task_runtime=task,
        main_image_size=main_size,
        wrist_image_size=wrist_size,
        camera_data_type="rgb",
    )
    env.set_episode_seed(seed)
    env.reset(seed=seed)
    initial_arm = env.robot.data.joint_pos.torch[0, :7].detach().cpu().numpy().astype(np.float32)
    env.restore_fixed_arm_pose(initial_arm)

    print(
        "CORIBENCH_REPLAY_STARTED="
        + json.dumps(
            {
                "episode": str(args.episode.resolve()),
                "seed": seed,
                "frames": len(commands),
                "recorded_success": bool(metadata.get("success", False)),
                "arm_control_mode": env.arm_control_mode,
                "arm_controller_profile": env.arm_controller_profile,
                "initial_joint_error_rad": float(np.max(np.abs(recorded_joints[0, :7] - initial_arm))),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    zero_action = torch.zeros((1, 7), dtype=torch.float32, device=env.device)
    for frame, command in enumerate(commands, start=1):
        gripper = float(np.clip((command[6] + 1.0) * 0.5, 0.0, 1.0))
        env.set_cartesian_rpy_target(command[:6], gripper)
        env.step(zero_action)
        if args.progress_interval and (frame % args.progress_interval == 0 or frame == len(commands)):
            print(f"CORIBENCH_REPLAY_PROGRESS={frame}/{len(commands)}", flush=True)

    settle_frames = args.final_settle_frames
    if settle_frames is None:
        settle_seconds = float(metadata.get("success_evaluation_settle_s", 0.0))
        settle_frames = int(round(settle_seconds / env.step_dt))
    settle_gripper = env.gripper_open_fraction
    settle_pose = current_xyz_rpy(env)
    for _ in range(settle_frames):
        env.set_cartesian_rpy_target(settle_pose, settle_gripper)
        env.step(zero_action)

    report = env.evaluate_final_success()
    recorded_report = metadata.get("final_success_report", {})
    comparison = {
        "center_error_delta_m": float(report["center_error_m"]) - float(recorded_report["center_error_m"]),
        "base_height_error_delta_m": float(report["base_height_error_m"])
        - float(recorded_report["base_height_error_m"]),
        "tilt_delta_deg": float(report["tilt_deg"]) - float(recorded_report["tilt_deg"]),
    }

    preview_paths = {}
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        env.reset_cameras()
        env.update_cameras()
        stem = args.episode.stem + "_physics_replay_final"
        main_path = args.preview_dir / f"{stem}_main.png"
        wrist_path = args.preview_dir / f"{stem}_wrist.png"
        main_rgb = env.table_rgb[0].detach().cpu().numpy()[..., :3].astype(np.uint8)
        wrist_rgb = env.wrist_rgb[0].detach().cpu().numpy()[..., :3].astype(np.uint8)
        if not cv2.imwrite(str(main_path), cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to save {main_path}")
        if not cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to save {wrist_path}")
        preview_paths = {"main": str(main_path.resolve()), "wrist": str(wrist_path.resolve())}

    result = {
        "success": bool(report["success"]),
        "seed": seed,
        "frames": len(commands),
        "settle_frames": settle_frames,
        "report": report,
        "recorded_report": recorded_report,
        "replay_minus_recorded": comparison,
        "previews": preview_paths,
    }
    print("CORIBENCH_REPLAY_RESULT=" + json.dumps(result, separators=(",", ":")), flush=True)
    if not bool(report["success"]):
        raise SystemExit(2)
finally:
    if env is not None:
        env.close()
    simulation_app.close()
