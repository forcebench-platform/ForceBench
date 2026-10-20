from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from .data import CoRiBenchLeRobotDataset
from .model import build_policy
from .run_output import initialize_run_output


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--print-effective-config", action="store_true",
                        help="Print resolved training configuration without loading data or training.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    training = config.setdefault("training", {})
    # configs/xarm.yaml plus upstream DataLoader/Trainer/PyTorch defaults.
    defaults = {
        "devices": 1, "max_steps": 100000, "batch_size": 128,
        "num_workers": 8, "grad_accumulation": 1, "gradient_clip": None,
        "precision": "bf16-mixed", "float32_matmul_precision": "highest",
        "seed_workers": False, "shuffle": True, "persistent_workers": True,
        "pin_memory": False, "drop_last": False,
        "checkpoint_every": 5000, "log_every": 50,
    }
    for key, value in defaults.items():
        training.setdefault(key, value)
    if not config.get("optimizer"):
        config["optimizer"] = {"type": "adam", "lr": 3e-4, "weight_decay": None}
    elif "type" not in config["optimizer"]:
        # Preserve the interpretation of historical saved run configurations.
        config["optimizer"]["type"] = "adamw"
    print(json.dumps({"task": config.get("task"),
                      "target_total_steps": int(training["max_steps"]),
                      "optimizer": config["optimizer"], "training": training},
                     indent=2), flush=True)
    if args.print_effective_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    output = Path(config["output_dir"])
    resume = args.resume_from or training.get("resume_from")
    run = initialize_run_output(output, config, resume=resume)
    output = run.output
    destination = output / "config.yaml"
    if run.is_primary and args.config.resolve() != destination.resolve():
        destination.write_text(yaml.safe_dump(config, sort_keys=False))

    pl.seed_everything(int(config.get("seed", 0)), workers=bool(training["seed_workers"]))
    torch.set_float32_matmul_precision(str(training["float32_matmul_precision"]))
    temporal = config["temporal"]
    interface = {
        "control_frequency_hz": float(temporal["control_frequency_hz"]),
        "action_horizon": int(temporal["action_horizon"]),
        "execute_chunk": int(temporal["evaluation_execute_chunk"]),
        "force_history": int(temporal["force_history"]),
        "observation_horizon": int(temporal["observation_horizon"]),
    }
    required = {
        "control_frequency_hz": 15.0,
        "action_horizon": 64,
        # 32 matches server.py EXECUTE_CHUNK and CoRiBench FORCEFLOW_EXECUTE_CHUNK.
        # Assertion + checkpoint metadata only; the training math never reads it.
        "execute_chunk": 32,
        "force_history": 10,
    }
    for name, expected_value in required.items():
        if not np.isclose(interface[name], expected_value):
            raise ValueError(
                f"ForceFlow CoRiBench interface requires {name}={expected_value:g}, "
                f"got {interface[name]:g}"
            )
    if interface["observation_horizon"] < 1:
        raise ValueError("observation_horizon must be positive")
    dataset = CoRiBenchLeRobotDataset(
        config["data"]["dataset_root"],
        observation_horizon=int(temporal["observation_horizon"]),
        action_horizon=int(temporal["action_horizon"]),
        force_history=int(temporal["force_history"]),
        control_frequency_hz=float(temporal["control_frequency_hz"]),
        image_size=int(config["image"]["size"]),
        episode_limit=config["data"].get("episode_limit"),
    )
    if run.is_primary:
        (output / "norm_stats.json").write_text(
            json.dumps(dataset.normalizer_dict(), indent=2) + "\n"
        )
    sample = dataset[0]
    expected = (int(temporal["action_horizon"]), 13)
    if tuple(sample["x0"].shape) != expected:
        raise RuntimeError(f"sample action shape {tuple(sample['x0'].shape)}, expected {expected}")

    training = config["training"]
    policy = build_policy(config)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=bool(training["shuffle"]),
        num_workers=int(training["num_workers"]),
        persistent_workers=bool(training["persistent_workers"]) and int(training["num_workers"]) > 0,
        pin_memory=bool(training["pin_memory"]),
        drop_last=bool(training["drop_last"]),
    )
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="step-{step:06d}",
        every_n_train_steps=int(training["checkpoint_every"]),
        save_top_k=-1,
        save_last=True,
    )
    # `gradient_clip: null` (or 0) disables clipping, matching upstream ForceFlow,
    # whose Trainer passes no gradient_clip_val at all. New default is disabled.
    _grad_clip = training.get("gradient_clip")
    _grad_clip = None if _grad_clip in (None, 0, 0.0) else float(_grad_clip)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=int(training.get("devices", 1)),
        max_steps=int(training["max_steps"]),
        precision=str(training["precision"]),
        accumulate_grad_batches=int(training.get("grad_accumulation", 1)),
        gradient_clip_val=_grad_clip,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=CSVLogger(save_dir=output / "logs", name="train"),
        log_every_n_steps=int(training.get("log_every", 50)),
        default_root_dir=output,
        num_sanity_val_steps=0,
    )
    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "forceflow_upstream_commit": "544bb35c8363521c19b7425c529364ae4b80032d",
        "cleandiffuser_upstream_commit": "b6df692788cd03127c6dcfc515c2638444e71e9d",
        "packaging": "ForceBench portable adapter; see SOURCE_MANIFEST.json",
        "adapter_sha256": {
            str(path.relative_to(repo)): file_sha256(path)
            for path in sorted((repo / "coribench_forceflow").glob("*.py"))
        },
        "torch": torch.__version__,
        "lightning": pl.__version__,
        "sample_shapes": {
            "action": list(sample["x0"].shape),
            **{key: list(value.shape) for key, value in sample["condition_cfg"].items()},
        },
        "interface": interface,
        "effective_training": training,
        "effective_optimizer": config["optimizer"],
    }
    if run.is_primary:
        manifest["run_identity"] = {"token": run.token, "config_sha256": run.config_sha256}
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    resume = args.resume_from or training.get("resume_from")
    trainer.fit(policy, loader, ckpt_path=None if resume is None else str(resume))


if __name__ == "__main__":
    main()
