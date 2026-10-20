"""Check explicit inputs or train a recorded ForceVLA candidate recipe."""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from portable_common import activate, add_inputs, inspect_inputs, sha256, source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train"))
    add_inputs(parser)
    args = parser.parse_args()
    paths = inspect_inputs(args)
    print(json.dumps({"status": "metadata_only", "inputs": paths["identity"],
                      "source_sha256": source_identity(args.source_snapshot),
                      "weights_loaded": False, "dataset_payload_checksums_checked": False}, indent=2))
    if args.mode == "check":
        return
    if sys.platform != "linux":
        raise RuntimeError("The preserved training stack requires Linux")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly")
    checkpoint_root = paths["run"] / "checkpoints"
    if checkpoint_root.exists():
        raise FileExistsError("Existing checkpoint root; use a fresh run directory. Resume is not exposed by this candidate.")
    assets = paths["run"] / "assets/imported"
    target = assets / args.repo_id / "norm_stats.json"
    if target.exists() and sha256(target) != sha256(paths["norm"]):
        raise ValueError("Run assets differ from selected normalization")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(paths["norm"], target)
    activate(args, paths)
    os.environ["FORCEVLA_IMPORTED_NORM"] = str(target)
    import jax
    if any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("JAX GPU is unavailable")
    from candidate_recipes import make_config
    import train
    config = make_config(args, paths, assets)
    train.main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
