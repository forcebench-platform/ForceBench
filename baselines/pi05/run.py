"""Inspect explicit inputs or train the controlled pi0.5 candidate recipe."""
import argparse
import json
import os
from pathlib import Path
import shutil

from portable_common import activate, add_inputs, inspect_inputs, sha256, source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train"))
    add_inputs(parser)
    parser.add_argument("--preflight", action="store_true", help="With train: run the preserved three-update GPU check")
    parser.add_argument("--validate-inference", action="store_true",
                        help="With train --preflight: test the in-memory model and a loopback websocket after three updates")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "check" and (args.resume or args.preflight):
        parser.error("--resume and --preflight apply only to train")
    if args.validate_inference and not (args.mode == "train" and args.preflight):
        parser.error("--validate-inference requires train --preflight")
    paths = inspect_inputs(args)
    print(json.dumps({"input_identity": paths["identity"], "source_sha256": source_identity(),
                      "status": "input_metadata_checked_only", "weights_loaded": False,
                      "dataset_payload_checksums_checked": False}, indent=2))
    if args.mode == "check":
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly before training")
    args.base_params = paths["base"]
    args.norm_assets = paths["run"] / "assets/imported"
    target = args.norm_assets / args.repo_id / "norm_stats.json"
    if target.exists() and sha256(target) != sha256(paths["norm"]):
        raise ValueError("Run directory contains different normalization assets")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(paths["norm"], target)
    args.input_identity = paths["identity"]
    args.steps, args.batch_size = 10000, 32
    args.new_lr_multiplier, args.gate_initial = 10.0, 0.1
    activate(paths)
    # Cache paths and the preserved XLA memory setting must precede JAX import
    # and backend initialization, as in the historical training entry point.
    from runtime_env import configure
    configure()
    import jax
    if any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("JAX GPU is unavailable")
    import train_forcebench
    train_forcebench.run(args)


if __name__ == "__main__":
    main()
