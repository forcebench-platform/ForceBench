"""Collection loop shared by task-specific privileged assembly scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from forcebench.env import ForceBenchEnv
from forcebench.env_cfg import ASSET_DIR, ForceBenchEnvCfg, configure_runtime
from forcebench.privileged_placement import PrivilegedPlacementEpisode, PrivilegedPlacementPlan
from forcebench.recording import HDF5EpisodeWriter, validate_hdf5_episode
from forcebench.registry import load_task


def collect_privileged_placements(
    *,
    args: Any,
    task_name: str,
    plan_factory: Callable[[ForceBenchEnv], PrivilegedPlacementPlan],
) -> dict[str, object]:
    """Collect successful episodes; artifacts never share the formal data root."""

    task = load_task(task_name, ASSET_DIR)
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
    print(f"FORCEBENCH_PRIVILEGED_EXPERT_READY={task_name}", flush=True)
    try:
        for slot in range(args.episodes):
            seed = args.start_seed + slot
            completed = False
            for attempt in range(1, args.max_attempts_per_episode + 1):
                env.set_episode_seed(seed)
                env.reset(seed=seed)
                env.restore_fixed_arm_pose(env.joint_pos[0, :7].clone())
                env.reset_cameras()
                for _ in range(3):
                    env.sim.render()
                env.update_cameras()
                env.start_wrench_capture()
                episode = PrivilegedPlacementEpisode(env, task, plan_factory(env))
                try:
                    result = episode.run()
                    high_rate = env.stop_wrench_capture()
                    task_metadata = dict(task.episode_metadata())
                    asset_paths = {"robot": str(ASSET_DIR / "franka_mimic.usd")}
                    asset_paths.update(env.scene_asset_paths)
                    asset_paths.update(task_metadata.pop("asset_usd_paths", {}))
                    metadata = {
                        "task": task_name,
                        "prompt": task.manifest.prompt,
                        "success": True,
                        "end_reason": "privileged_pose_teacher_final_check",
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
                        "control_source": "privileged_object_pose_closed_loop_smooth_position",
                        "success_check_timing": "after_place_release_and_retract",
                        "images_present": True,
                        "image_shapes_hwc": {
                            "main": [args.main_image_size, args.main_image_size, 3],
                            "wrist": [args.wrist_image_size, args.wrist_image_size, 3],
                        },
                        "privileged_expert_contract": {
                            "reads_object_and_fixture_pose": True,
                            "object_state_overwrite": False,
                            "robot_state_overwrite": False,
                            "commands_through_environment_controller": True,
                        },
                        "scripted_expert_details": {
                            key: value for key, value in result.items() if key != "report"
                        },
                        "scene_roles": list(env.entity_roles),
                        "asset_usd_paths": asset_paths,
                        **task_metadata,
                    }
                    path = writer.write(episode.buffer, metadata, high_rate=high_rate)[0]
                    record = {
                        "path": str(path.resolve()),
                        "seed": seed,
                        "frames": len(episode.buffer),
                        "attempt": attempt,
                        "validation": validate_hdf5_episode(path),
                    }
                    saved.append(record)
                    print(
                        "FORCEBENCH_PRIVILEGED_EXPERT_SAVED="
                        + json.dumps(record, separators=(",", ":")),
                        flush=True,
                    )
                    completed = True
                    break
                except Exception as error:
                    env.discard_wrench_capture()
                    failure = {"seed": seed, "attempt": attempt, "error": str(error)}
                    failures.append(failure)
                    print(
                        "FORCEBENCH_PRIVILEGED_EXPERT_RETRY="
                        + json.dumps(failure, separators=(",", ":")),
                        flush=True,
                    )
            if not completed:
                raise RuntimeError(
                    f"failed to generate {task_name} seed {seed} after "
                    f"{args.max_attempts_per_episode} attempts"
                )
    finally:
        env.close()

    summary = {
        "task": task_name,
        "requested_episodes": args.episodes,
        "saved_episodes": len(saved),
        "start_seed": args.start_seed,
        "output_dir": str(Path(args.output_dir).resolve()),
        "episodes": saved,
        "failed_attempts": failures,
    }
    summary_path = artifact_dir / f"generation_seed_{args.start_seed:03d}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "FORCEBENCH_PRIVILEGED_EXPERT_SUMMARY="
        + json.dumps(summary, separators=(",", ":")),
        flush=True,
    )
    return summary
