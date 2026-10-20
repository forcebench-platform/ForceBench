"""Validate real ForceFlow data, checkpoint inference, transport and short training.

Run in the separate ForceFlow environment. Inputs are read-only; all generated
configurations, predictions, checkpoints and receipts go to a new output folder.
This is an execution check, not a reproduction of published success rates.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from coribench_forceflow import data as data_module
from coribench_forceflow import model as model_module
from coribench_forceflow import server as server_module
from coribench_forceflow.data import CoRiBenchLeRobotDataset
from coribench_forceflow.server import ForceFlowPolicy, ObservationHistory, serve
from coribench_forceflow.video_decode import decode_video_frames


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trainable_sha256(policy) -> str:
    digest = hashlib.sha256()
    for name, value in policy.named_parameters():
        if value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GradientChecks(pl.Callback):
    def __init__(self):
        self.steps = []

    def on_train_batch_start(self, trainer, module, batch, batch_index):
        if not module.model.training:
            raise RuntimeError("the actual ForceFlow training network is still in eval mode")

    def on_after_backward(self, trainer, module):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients:
            raise RuntimeError("training produced no gradients")
        finite = bool(torch.stack([g.isfinite().all() for g in gradients]).all().item())
        norm = float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt().item())
        if not finite or not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"invalid backward gradients: finite={finite}, norm={norm}")
        self.steps.append({"finite": finite, "l2_norm": norm})

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_index):
        loss = float(outputs["loss"].detach().item())
        if not np.isfinite(loss):
            raise RuntimeError("training loss is not finite")
        self.steps[-1].update({"loss": loss, "global_step": trainer.global_step})


def websocket_roundtrip(policy, observation):
    # Import the actual public adapter; it owns the MessagePack codec and client.
    from forcebench.policy_adapters.forceflow_ws import ForceFlowWebSocketAdapter

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    stop = threading.Event()
    failures = []

    async def run_server():
        args = SimpleNamespace(host="127.0.0.1", port=port, max_batch_size=2, batch_window_ms=1.0)
        task = asyncio.create_task(serve(policy, args))
        try:
            while not stop.is_set():
                if task.done():
                    await task
                await asyncio.sleep(0.02)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def target():
        try:
            asyncio.run(run_server())
        except BaseException as exc:
            failures.append(repr(exc))

    thread = threading.Thread(target=target, name="native-forceflow-server", daemon=True)
    thread.start()
    adapter = None
    try:
        adapter = ForceFlowWebSocketAdapter(
            host="127.0.0.1", port=port, options={"max_wait_seconds": 60}
        )
        result = adapter.infer(observation)
        return result.actions, dict(adapter.metadata)
    finally:
        if adapter is not None:
            adapter.close()
        stop.set()
        thread.join(timeout=20)
        if thread.is_alive():
            raise RuntimeError("native websocket server did not stop")
        if failures:
            raise RuntimeError(f"native websocket server failed: {failures}")


def run(args, receipt, save):
    from forcebench.policy_api import PolicyObservation
    from forcebench.registry import get_task_manifest

    source = Path(__file__).resolve().parent
    for module in (data_module, model_module, server_module):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError(f"import resolved outside candidate source: {module.__file__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is not a GPU validation")
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    receipt["runtime"] = {
        "python": sys.version, "torch": torch.__version__, "lightning": pl.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "source_sha256": {str(p.relative_to(source)): file_sha256(p)
                          for p in sorted((source / "coribench_forceflow").glob("*.py"))},
        "validation_tool_sha256": file_sha256(Path(__file__)),
    }
    receipt["inputs"] = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
        "config": str(args.config), "config_sha256": file_sha256(args.config),
        "normalizer": str(args.normalizer), "normalizer_sha256": file_sha256(args.normalizer),
        "dataset": str(args.dataset_root),
        "dataset_info_sha256": file_sha256(args.dataset_root / "meta/info.json"),
    }
    save("input_hashes")
    config = yaml.safe_load(args.config.read_text())
    config["data"]["dataset_root"] = str(args.dataset_root)
    config["output_dir"] = str(args.output / "unused_inference_output")
    config["normalizer_path"] = str(args.normalizer)
    config_path = args.output / "inference_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    temporal = config["temporal"]
    dataset = CoRiBenchLeRobotDataset(
        args.dataset_root,
        observation_horizon=int(temporal["observation_horizon"]),
        action_horizon=int(temporal["action_horizon"]),
        force_history=int(temporal["force_history"]),
        control_frequency_hz=float(temporal["control_frequency_hz"]),
        image_size=int(config["image"]["size"]),
    )
    expected_norm = json.loads(args.normalizer.read_text())
    actual_norm = dataset.normalizer_dict()
    norm_differences = {
        f"{field}.{bound}": float(np.max(np.abs(
            np.asarray(expected_norm[field][bound]) - np.asarray(actual_norm[field][bound])
        )))
        for field in ("pos", "action", "force") for bound in ("min", "max")
    }
    receipt["data"] = {
        "episodes": len(dataset.episodes), "frames": len(dataset),
        "normalizer_max_abs_differences": norm_differences,
        "normalizer_matches_checkpoint": all(v < 1e-6 for v in norm_differences.values()),
    }
    # A changed dataset is allowed to expose an error, but must never silently
    # be presented as validation of the checkpoint's authentic input pair.
    if not receipt["data"]["normalizer_matches_checkpoint"]:
        raise RuntimeError("dataset normalization differs from authentic checkpoint normalization")
    frame = min(32, dataset.episodes[0].length - 65)
    if frame < 10:
        raise RuntimeError("first real episode is too short for dense history and action window")
    episode = dataset.episodes[0]
    obs_indices = dataset._history_indices(frame, 2, episode.length)
    force_indices = dataset._history_indices(frame, 10, episode.length)
    timestamps = (obs_indices / episode.fps).tolist()

    def images(path):
        frames = decode_video_frames(path, timestamps, 0.5 / episode.fps + 1e-4, backend="pyav")
        return frames.mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    main, wrist = images(episode.main_video), images(episode.wrist_video)
    observation = PolicyObservation(
        step=frame, rollout_id="gpu-validation:real-weight-sort-episode-0",
        state=episode.state[frame], main_image=main[-1], wrist_image=wrist[-1],
        eef_wrench=episode.force[frame], prompt=get_task_manifest(config["task"]).prompt,
        state_history=episode.state[obs_indices], main_image_history=main,
        wrist_image_history=wrist, eef_wrench_history=episode.force[force_indices],
    )
    receipt["data"].update({
        "sample_episode": 0, "sample_frame": frame,
        "used_video_sha256": {str(p): file_sha256(p)
                              for p in (episode.main_video, episode.wrist_video)},
        "real_images_shape": list(main.shape),
        "image_std": {"main": float(main.std()), "wrist": float(wrist.std())},
    })
    save("real_data_loaded")
    policy = ForceFlowPolicy(config_path, args.checkpoint, "cuda:0")
    request = observation.canonical_request()
    request["contract"] = "coribench_policy"
    item = ObservationHistory(2, 10).append(request, require_explicit_history=True)
    direct = policy.infer_batch([item])[0]["actions"]
    if direct.shape != (32, 7) or not np.isfinite(direct).all():
        raise RuntimeError("native checkpoint inference did not produce finite [32,7] actions")
    wire, metadata = websocket_roundtrip(policy, observation)
    difference = float(np.max(np.abs(wire - direct)))
    if not np.allclose(wire, direct, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"native adapter round trip changed model output: {difference}")
    np.savez_compressed(args.output / "native_predictions.npz", direct=direct, websocket=wire)
    receipt["inference"] = {
        "strict_checkpoint_load": True, "prediction_action_horizon": 64,
        "returned_action_shape": list(wire.shape), "ode_steps": policy.sample_steps,
        "use_ema": policy.use_ema, "all_finite": True,
        "native_websocket_adapter": metadata, "direct_vs_wire_max_abs_diff": difference,
        "wire_payload": "Real decoded simulation images, recorded state and ten contiguous wrench frames.",
    }
    save("native_inference_and_transport")
    if args.train_steps:
        # ForceFlowPolicy intentionally enters eval mode for serving. Restore
        # the native training network mode before checking actual optimization;
        # retain the non-trainable EMA copy's evaluation mode.
        policy.policy.train()
        policy.policy.model_ema.eval()
        before = trainable_sha256(policy.policy)
        selected = [frame + i * 8 for i in range(args.train_steps * 2)]
        if max(selected) >= episode.length - 64:
            raise RuntimeError("not enough real frames for requested training validation")
        loader = DataLoader(Subset(dataset, selected), batch_size=2, num_workers=0, shuffle=False)
        gradients = GradientChecks()
        trainer = pl.Trainer(
            accelerator="gpu", devices=1, max_steps=args.train_steps,
            precision="bf16-mixed", callbacks=[gradients], logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, default_root_dir=str(args.output / "training"),
        )
        trainer.fit(policy.policy, loader)
        after = trainable_sha256(policy.policy)
        if trainer.global_step != args.train_steps or before == after:
            raise RuntimeError("short training did not update the actual model")
        fresh_checkpoint = args.output / "validation_steps.ckpt"
        trainer.save_checkpoint(fresh_checkpoint, weights_only=True)
        reloaded = ForceFlowPolicy(config_path, fresh_checkpoint, "cuda:0")
        actions = reloaded.infer_batch([item])[0]["actions"]
        if not np.isfinite(actions).all():
            raise RuntimeError("fresh validation checkpoint inference is not finite")
        receipt["training"] = {
            "optimizer_steps": trainer.global_step, "batch_size": 2,
            "training_network_mode_checked_each_batch": True,
            "real_dataset_frame_indices": selected, "gradient_checks": gradients.steps,
            "optimizer": config["optimizer"], "precision": "bf16-mixed",
            "trainable_sha256_before": before, "trainable_sha256_after": after,
            "parameters_changed": True, "saved_checkpoint": str(fresh_checkpoint),
            "saved_checkpoint_sha256": file_sha256(fresh_checkpoint),
            "saved_checkpoint_strict_reload_and_infer": True,
            "scope": "Small-batch execution validation; not a full training run or paper result.",
        }
        save("short_training_and_checkpoint_reload")
    if file_sha256(args.checkpoint) != receipt["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("input checkpoint changed during validation")
    receipt["input_checkpoint_unchanged"] = True
    receipt["max_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "normalizer", "dataset-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.train_steps <= 10:
        parser.error("training validation accepts zero to ten steps")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "forcebench_forceflow_gpu_validation_v1", "status": "running"}

    def save(stage):
        receipt["stage"] = stage
        receipt["updated_unix"] = time.time()
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"FORCEFLOW_VALIDATION_STAGE={stage}", flush=True)

    try:
        run(args, receipt, save)
        receipt["status"] = "passed"
        save("complete")
    except BaseException:
        receipt["status"] = "failed"
        receipt["error"] = traceback.format_exc()
        save(receipt.get("stage", "startup"))
        raise


if __name__ == "__main__":
    main()
