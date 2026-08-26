#!/usr/bin/env python3
"""Evaluate one policy in one Isaac process with many cloned CoRiBench environments."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher

from coribench.policy_registry import policy_adapter_names
from coribench.registry import get_task_manifest, task_names


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="peg_insert", choices=task_names())
parser.add_argument("--seed_start", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument("--policy_host", default="127.0.0.1")
parser.add_argument("--policy_port", type=int, default=8002)
parser.add_argument(
    "--policy_ports",
    default=os.environ.get("CORIBENCH_POLICY_PORTS", ""),
    help="Optional comma-separated H200 policy ports assigned round-robin to clones.",
)
parser.add_argument("--prompt", default=None)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--execute_chunk", type=int, default=50)
parser.add_argument(
    "--policy_adapter",
    choices=policy_adapter_names(),
    default=os.environ.get("CORIBENCH_POLICY_ADAPTER", "openpi_ws"),
)
parser.add_argument(
    "--policy_adapter_options",
    default=os.environ.get("CORIBENCH_POLICY_ADAPTER_OPTIONS", "{}"),
)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--physics_rate_hz", type=int, choices=(60,), default=60)
parser.add_argument("--solver_position_iterations", type=int, choices=(64,), default=64)
parser.add_argument(
    "--terminate_on_success", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument("--success_stable_frames", type=int, default=1)
parser.add_argument("--final_settle_frames", type=int, default=8)
parser.add_argument(
    "--smoke_hold",
    action="store_true",
    help="Do not connect to a policy; hold the measured TCP pose to benchmark cloned simulation.",
)
parser.add_argument("--save_images", action="store_true")
parser.add_argument("--save_rollouts", action="store_true")
parser.add_argument("--aggregate_filename", default="aggregate.json")
parser.add_argument("--output_dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

FORCEFLOW_ACTION_HORIZON = 64
FORCEFLOW_EXECUTE_CHUNK = 16
FORCEFLOW_OBSERVATION_HORIZON = 2
FORCEFLOW_FORCE_HISTORY = 10
forceflow_interface = args.policy_adapter == "coribench_ws"
effective_execute_chunk = FORCEFLOW_EXECUTE_CHUNK if forceflow_interface else args.execute_chunk

try:
    policy_adapter_options = json.loads(args.policy_adapter_options)
except json.JSONDecodeError as exc:
    parser.error(f"policy_adapter_options must be valid JSON: {exc}")
if not isinstance(policy_adapter_options, dict):
    parser.error("policy_adapter_options must decode to a JSON object")
try:
    policy_ports = (
        [int(value.strip()) for value in args.policy_ports.split(",") if value.strip()]
        if args.policy_ports
        else [args.policy_port]
    )
except ValueError:
    parser.error("policy_ports must be a comma-separated list of integers")
if not policy_ports or any(not 1 <= port <= 65535 for port in policy_ports):
    parser.error("policy ports must be in [1, 65535]")
if min(args.seed_start, args.num_envs, args.max_steps, args.execute_chunk) < 0:
    parser.error("seed_start must be non-negative and count/step values must be positive")
if args.num_envs < 1 or args.max_steps < 1 or args.execute_chunk < 1:
    parser.error("num_envs, max_steps, and execute_chunk must be positive")
if args.num_envs > 1 and args.env_spacing < 5.5:
    parser.error("parallel evaluation requires env_spacing >= 5.5 m for visual isolation")
if args.success_stable_frames < 1 or args.final_settle_frames < 0:
    parser.error("success_stable_frames must be positive and final_settle_frames non-negative")
if not args.aggregate_filename or Path(args.aggregate_filename).name != args.aggregate_filename:
    parser.error("aggregate_filename must be a plain filename")
seeds = list(range(args.seed_start, args.seed_start + args.num_envs))
seed_dirs = [args.output_dir / f"seed_{seed:03d}" for seed in seeds]
collisions = [path for path in seed_dirs if path.exists()]
aggregate_path = args.output_dir / args.aggregate_filename
if args.output_dir.exists() and not args.output_dir.is_dir():
    parser.error(f"output path is not a directory: {args.output_dir}")
if collisions:
    parser.error(f"seed output already exists: {collisions[0]}")
if aggregate_path.exists():
    parser.error(f"aggregate output already exists: {aggregate_path}")

manifest = get_task_manifest(args.task)
launcher = AppLauncher(args)
simulation_app = launcher.app

import cv2
import numpy as np
import torch

from coribench.env import CoRiBenchEnv
from coribench.env_cfg import ASSET_DIR, CoRiBenchEnvCfg, configure_runtime
from coribench.policy_api import POLICY_CONTRACT, PolicyAdapter, PolicyObservation
from coribench.policy_registry import load_policy_adapter
from coribench.registry import load_task


def as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def rgb_at(value: torch.Tensor, env_id: int) -> np.ndarray:
    image = value[env_id].detach().cpu().numpy()
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"invalid RGB buffer: {image.shape}")
    return image[..., :3].astype(np.uint8)


def padded_history(values: deque[np.ndarray], count: int) -> np.ndarray:
    if not values:
        raise ValueError("cannot build policy history from an empty window")
    frames = list(values)
    frames = [frames[0]] * (count - len(frames)) + frames
    return np.stack(frames)


def canonical_action(action: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    result = np.asarray(action, dtype=np.float32).copy()
    original = result.copy()
    result[:3] = np.clip(result[:3], [0.30, -0.35, -0.01], [0.80, 0.35, 0.55])
    result[6] = np.clip(result[6], -1.0, 1.0)
    clamped = not np.allclose(result[[0, 1, 2, 6]], original[[0, 1, 2, 6]], atol=1e-6)
    result[3:6] = state[3:6] + (result[3:6] - state[3:6] + np.pi) % (2 * np.pi) - np.pi
    canonicalized = not np.allclose(result[3:6], original[3:6], atol=1e-6)
    return result, clamped, canonicalized


def evaluate_one(task, env: CoRiBenchEnv, env_id: int) -> dict[str, object]:
    evaluate = getattr(task, "evaluate_final_state_for", None)
    if evaluate is None:
        raise RuntimeError(f"task {args.task} does not implement vectorized outcome evaluation")
    return dict(evaluate(env, env_id))


task = load_task(args.task, ASSET_DIR)
cfg = CoRiBenchEnvCfg()
cfg.seed = seeds[0]
cfg.scene_profile = manifest.scene_profile
configure_runtime(
    cfg,
    device=args.device,
    physics_rate_hz=args.physics_rate_hz,
    solver_iterations=args.solver_position_iterations,
    num_envs=args.num_envs,
    env_spacing=args.env_spacing,
)
cfg.sim.render_interval = cfg.decimation * (args.max_steps + args.final_settle_frames + 2)
cfg.num_rerenders_on_reset = 1

env = CoRiBenchEnv(
    cfg=cfg,
    task_runtime=task,
    main_image_size=args.main_image_size,
    wrist_image_size=args.wrist_image_size,
)
env.set_episode_seeds(seeds)
env.reset(seed=seeds[0])
env.restore_fixed_arm_pose(env.robot.data.joint_pos.torch[0, :7].clone())
env.reset_cameras()
for _ in range(3):
    env.sim.render()
env.update_cameras()

# Some task runtimes gate trajectory-dependent outcome bookkeeping until their
# force baseline has been initialized.  Expert collection performs this after
# grasping; policy evaluation has no scripted pre-grasp phase, so initialize
# every clone from the settled reset state before the rollout begins.
arm_task_runtime = getattr(task, "arm_after_franka_pregrasp", None)
task_runtime_armed = False
if callable(arm_task_runtime):
    wrench_bias_world = env.force_sensor_world.detach().clone()
    if wrench_bias_world.shape != (args.num_envs, 6) or not torch.isfinite(
        wrench_bias_world
    ).all():
        raise ValueError(
            f"invalid task-runtime wrench baseline: {tuple(wrench_bias_world.shape)}"
        )
    for env_id in range(args.num_envs):
        arm_task_runtime(env, wrench_bias_world[env_id], env_id)
    task_runtime_armed = True

prompts = [args.prompt or task.instruction(layout) for layout in env.layouts]
policies: list[PolicyAdapter] = []
queues = [deque() for _ in seeds]
latencies: list[list[float]] = [[] for _ in seeds]
steps_executed = [0 for _ in seeds]
first_success_steps: list[int | None] = [None for _ in seeds]
success_streaks = [0 for _ in seeds]
online_reports: list[dict[str, object] | None] = [None for _ in seeds]
termination_reasons = ["max_steps" for _ in seeds]
clamped_frames = [0 for _ in seeds]
canonicalized_frames = [0 for _ in seeds]
state_histories: list[list[np.ndarray]] = [[] for _ in seeds]
action_histories: list[list[np.ndarray]] = [[] for _ in seeds]
joint_histories: list[list[np.ndarray]] = [[] for _ in seeds]
wrench_histories: list[list[np.ndarray]] = [[] for _ in seeds]
policy_state_windows = [deque(maxlen=FORCEFLOW_OBSERVATION_HORIZON) for _ in seeds]
policy_main_windows = [deque(maxlen=FORCEFLOW_OBSERVATION_HORIZON) for _ in seeds]
policy_wrist_windows = [deque(maxlen=FORCEFLOW_OBSERVATION_HORIZON) for _ in seeds]
policy_wrench_windows = [deque(maxlen=FORCEFLOW_FORCE_HISTORY) for _ in seeds]
active = np.ones(args.num_envs, dtype=bool)
args.output_dir.mkdir(parents=True, exist_ok=True)
started_rollout = time.perf_counter()

print(f"CORIBENCH_VECTOR_READY=1", flush=True)
print(f"CORIBENCH_VECTOR_TASK={args.task}", flush=True)
print(f"CORIBENCH_VECTOR_NUM_ENVS={args.num_envs}", flush=True)
print(f"CORIBENCH_VECTOR_SEEDS={seeds[0]}:{seeds[-1]}", flush=True)
print(f"CORIBENCH_VECTOR_POLICY_PORTS={','.join(map(str, policy_ports))}", flush=True)
print(f"CORIBENCH_VECTOR_SMOKE_HOLD={int(args.smoke_hold)}", flush=True)
print(f"CORIBENCH_VECTOR_TASK_ARMED={int(task_runtime_armed)}", flush=True)

try:
    if not args.smoke_hold:
        def connect_policy(env_id: int) -> PolicyAdapter:
            return load_policy_adapter(
                args.policy_adapter,
                host=args.policy_host,
                port=policy_ports[env_id % len(policy_ports)],
                options=policy_adapter_options,
            )

        with ThreadPoolExecutor(max_workers=args.num_envs) as executor:
            policies = list(executor.map(connect_policy, range(args.num_envs)))

    if args.save_images:
        for env_id, seed_dir in enumerate(seed_dirs):
            seed_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(seed_dir / "main_initial.png"),
                rgb_at(env.table_rgb, env_id)[..., ::-1],
            )
            cv2.imwrite(
                str(seed_dir / "wrist_initial.png"),
                rgb_at(env.wrist_rgb, env_id)[..., ::-1],
            )

    zero_action = torch.zeros((args.num_envs, 7), dtype=torch.float32, device=env.device)
    for step in range(args.max_steps):
        state_batch = as_numpy(env.policy_state)
        wrench_batch = as_numpy(env.force_sensor_joint_raw)
        if wrench_batch.shape != (args.num_envs, 6) or not np.isfinite(wrench_batch).all():
            raise ValueError(f"invalid batched wrench: {wrench_batch.shape}")

        if forceflow_interface and step:
            env.update_cameras()
        if forceflow_interface:
            for env_id in np.flatnonzero(active):
                policy_state_windows[env_id].append(state_batch[env_id].copy())
                policy_main_windows[env_id].append(rgb_at(env.table_rgb, int(env_id)))
                policy_wrist_windows[env_id].append(rgb_at(env.wrist_rgb, int(env_id)))
                policy_wrench_windows[env_id].append(wrench_batch[env_id].copy())

        need = [env_id for env_id in range(args.num_envs) if active[env_id] and not queues[env_id]]
        if need and not args.smoke_hold:
            if step and not forceflow_interface:
                env.update_cameras()
            observations = [
                PolicyObservation(
                    step=step,
                    rollout_id=f"{args.task}:seed={seeds[env_id]}",
                    state=state_batch[env_id],
                    main_image=rgb_at(env.table_rgb, env_id),
                    wrist_image=rgb_at(env.wrist_rgb, env_id),
                    eef_wrench=wrench_batch[env_id],
                    prompt=prompts[env_id],
                    state_history=(
                        padded_history(policy_state_windows[env_id], FORCEFLOW_OBSERVATION_HORIZON)
                        if forceflow_interface
                        else None
                    ),
                    main_image_history=(
                        padded_history(policy_main_windows[env_id], FORCEFLOW_OBSERVATION_HORIZON)
                        if forceflow_interface
                        else None
                    ),
                    wrist_image_history=(
                        padded_history(policy_wrist_windows[env_id], FORCEFLOW_OBSERVATION_HORIZON)
                        if forceflow_interface
                        else None
                    ),
                    eef_wrench_history=(
                        padded_history(policy_wrench_windows[env_id], FORCEFLOW_FORCE_HISTORY)
                        if forceflow_interface
                        else None
                    ),
                )
                for env_id in need
            ]

            def infer_one(pair):
                env_id, observation = pair
                started = time.perf_counter()
                chunk = policies[env_id].infer(observation)
                return env_id, chunk.actions, time.perf_counter() - started

            with ThreadPoolExecutor(max_workers=len(need)) as executor:
                results = list(executor.map(infer_one, zip(need, observations, strict=True)))
            for env_id, predicted, latency in results:
                if forceflow_interface and predicted.shape != (FORCEFLOW_EXECUTE_CHUNK, 7):
                    raise RuntimeError(
                        "ForceFlow interface requires exactly "
                        f"[{FORCEFLOW_EXECUTE_CHUNK}, 7] actions, got {predicted.shape}"
                    )
                queues[env_id].extend(predicted[:effective_execute_chunk])
                latencies[env_id].append(latency)

        targets = state_batch[:, :6].copy()
        grippers = (state_batch[:, 6] + 1.0) * 0.5
        for env_id in range(args.num_envs):
            if not active[env_id] or args.smoke_hold:
                continue
            model_action, was_clamped, was_canonicalized = canonical_action(
                queues[env_id].popleft(), state_batch[env_id]
            )
            targets[env_id] = model_action[:6]
            grippers[env_id] = (model_action[6] + 1.0) * 0.5
            clamped_frames[env_id] += int(was_clamped)
            canonicalized_frames[env_id] += int(was_canonicalized)
            steps_executed[env_id] += 1

        action_batch = np.concatenate(
            (targets, (2.0 * grippers - 1.0)[:, None]), axis=1
        ).astype(np.float32)
        joint_batch = as_numpy(env.joint_pos[:, :7])
        for env_id in np.flatnonzero(active):
            state_histories[env_id].append(state_batch[env_id].copy())
            action_histories[env_id].append(action_batch[env_id].copy())
            joint_histories[env_id].append(joint_batch[env_id].copy())
            wrench_histories[env_id].append(wrench_batch[env_id].copy())

        env.set_cartesian_rpy_target(targets, grippers)
        env.step(zero_action)
        if args.smoke_hold:
            steps_executed = [value + 1 for value in steps_executed]
            continue

        for env_id in np.flatnonzero(active):
            report = evaluate_one(task, env, int(env_id))
            if bool(report["success"]):
                success_streaks[env_id] += 1
                if first_success_steps[env_id] is None:
                    first_success_steps[env_id] = step + 1
            else:
                success_streaks[env_id] = 0
            if args.terminate_on_success and success_streaks[env_id] >= args.success_stable_frames:
                active[env_id] = False
                online_reports[env_id] = report
                termination_reasons[env_id] = "success"
        if not active.any():
            break
        if (step + 1) % 50 == 0:
            print(
                f"CORIBENCH_VECTOR_PROGRESS=step:{step + 1},active:{int(active.sum())},"
                f"success:{args.num_envs - int(active.sum())}",
                flush=True,
            )

    settle_states = as_numpy(env.policy_state)
    settle_targets = settle_states[:, :6].copy()
    settle_grippers = (settle_states[:, 6] + 1.0) * 0.5
    for _ in range(args.final_settle_frames):
        env.set_cartesian_rpy_target(settle_targets, settle_grippers)
        env.step(zero_action)
    env.update_cameras()

    final_reports = [evaluate_one(task, env, env_id) for env_id in range(args.num_envs)]
    final_states = as_numpy(env.policy_state)
    elapsed = time.perf_counter() - started_rollout
    successes = 0
    for env_id, seed in enumerate(seeds):
        report = online_reports[env_id] or final_reports[env_id]
        success = online_reports[env_id] is not None or bool(final_reports[env_id]["success"])
        successes += int(success)
        seed_dir = seed_dirs[env_id]
        seed_dir.mkdir(parents=True, exist_ok=True)
        if args.save_images:
            cv2.imwrite(str(seed_dir / "main_final.png"), rgb_at(env.table_rgb, env_id)[..., ::-1])
            cv2.imwrite(str(seed_dir / "wrist_final.png"), rgb_at(env.wrist_rgb, env_id)[..., ::-1])
        if args.save_rollouts:
            np.savez_compressed(
                seed_dir / "rollout.npz",
                states=np.asarray(state_histories[env_id], dtype=np.float32),
                actions=np.asarray(action_histories[env_id], dtype=np.float32),
                joint_positions=np.asarray(joint_histories[env_id], dtype=np.float32),
                eef_wrenches=np.asarray(wrench_histories[env_id], dtype=np.float32),
                inference_latencies_s=np.asarray(latencies[env_id], dtype=np.float32),
            )
        summary = {
            "task": args.task,
            "seed": seed,
            "prompt": prompts[env_id],
            "steps": steps_executed[env_id],
            "execute_chunk": effective_execute_chunk,
            "action_horizon": FORCEFLOW_ACTION_HORIZON if forceflow_interface else None,
            "force_history": FORCEFLOW_FORCE_HISTORY if forceflow_interface else None,
            "policy_contract": POLICY_CONTRACT,
            "policy_adapter": "smoke_hold" if args.smoke_hold else args.policy_adapter,
            "policy_port": policy_ports[env_id % len(policy_ports)],
            "policy_adapter_metadata": (
                {} if args.smoke_hold else dict(policies[env_id].metadata)
            ),
            "arm_control_mode": env.arm_control_mode,
            "arm_controller_profile": env.arm_controller_profile,
            "wrench_profile": env.wrench_profile,
            "arm_controller_diagnostics": env.arm_controller_diagnostics,
            "num_envs": args.num_envs,
            "env_spacing_m": args.env_spacing,
            "vectorized": True,
            "terminate_on_success": args.terminate_on_success,
            "success_stable_frames": args.success_stable_frames,
            "termination_reason": termination_reasons[env_id],
            "first_success_step": first_success_steps[env_id],
            "scene_profile": env.scene_profile_name,
            "camera_profile": "canonical",
            "camera_profile_is_default": True,
            "policy_wrist_image_rotation_deg_clockwise": 0,
            "physics_rate_hz": args.physics_rate_hz,
            "solver_position_iterations": args.solver_position_iterations,
            "success": success,
            "success_report": report,
            "final_success_report": final_reports[env_id],
            "final_state": final_states[env_id].tolist(),
            "clamped_action_frames": clamped_frames[env_id],
            "orientation_canonicalized_frames": canonicalized_frames[env_id],
            "mean_inference_latency_s": float(np.mean(latencies[env_id])) if latencies[env_id] else 0.0,
            "max_inference_latency_s": float(np.max(latencies[env_id])) if latencies[env_id] else 0.0,
        }
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    aggregate = {
        "schema": "coribench_vectorized_policy_batch_v1",
        "task": args.task,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "num_envs": args.num_envs,
        "env_spacing_m": args.env_spacing,
        "policy_ports": policy_ports,
        "successes": successes,
        "success_rate": successes / args.num_envs,
        "elapsed_s": elapsed,
        "environment_steps": sum(steps_executed),
        "environment_steps_per_second": sum(steps_executed) / elapsed,
        "smoke_hold": args.smoke_hold,
        "execute_chunk": effective_execute_chunk,
        "action_horizon": FORCEFLOW_ACTION_HORIZON if forceflow_interface else None,
        "force_history": FORCEFLOW_FORCE_HISTORY if forceflow_interface else None,
        "physics_rate_hz": args.physics_rate_hz,
        "main_image_size": args.main_image_size,
        "wrist_image_size": args.wrist_image_size,
        "arm_control_mode": env.arm_control_mode,
        "arm_controller_profile": env.arm_controller_profile,
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print("CORIBENCH_VECTOR_RESULT=" + json.dumps(aggregate, separators=(",", ":")), flush=True)
except BaseException:
    print("CORIBENCH_VECTOR_ERROR=python_exception", flush=True)
    traceback.print_exc()
    raise
finally:
    for policy in policies:
        policy.close()
    env.close()
    simulation_app.close()
