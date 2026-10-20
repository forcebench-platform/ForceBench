from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a ForceFlow CoRiBench task config.")
    parser.add_argument("--task", required=True, choices=["weight_sort", "com_sort",
        "faucet_turn", "faucet_direction", "stove_turn", "stove_press", "drawer_pull", "nut_thread"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()
    if args.devices < 1 or 128 % args.devices:
        parser.error("devices must be a positive divisor of the global batch size 128")
    info = args.dataset_root / "meta" / "info.json"
    if not info.is_file():
        raise FileNotFoundError(info)
    config = {
        "task": args.task,
        "seed": 0,
        "output_dir": str(args.output_dir.resolve()),
        "data": {"dataset_root": str(args.dataset_root.resolve()), "episode_limit": None},
        "temporal": {
            "control_frequency_hz": 15,
            "observation_horizon": 2,
            "force_history": 10,
            "action_horizon": 64,
            "evaluation_execute_chunk": 32,
        },
        "image": {"size": 224},
        "optimizer": {"type": "adam", "lr": 3e-4, "weight_decay": None},
        "training": {
            "devices": args.devices,
            "batch_size": 128 // args.devices,
            "num_workers": 4,
            "max_steps": args.max_steps,
            "grad_accumulation": 1,
            "gradient_clip": None,
            "precision": "bf16-mixed",
            "float32_matmul_precision": "highest",
            "seed_workers": False,
            "shuffle": True,
            "persistent_workers": True,
            "pin_memory": False,
            "drop_last": False,
            "checkpoint_every": 5000,
            "log_every": 50,
        },
        "evaluation": {"ode_steps": 20, "use_ema": False},
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"ForceFlow target total training steps: {args.max_steps}")
    print(args.config.resolve())


if __name__ == "__main__":
    main()
