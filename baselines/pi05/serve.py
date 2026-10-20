"""Serve an explicit portable-candidate checkpoint with mandatory identity checks."""
import argparse
import dataclasses
import logging
import os
from pathlib import Path

from portable_common import activate, add_inputs, inspect_inputs, validate_checkpoint_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    paths = inspect_inputs(args)
    checkpoint = args.checkpoint.resolve(strict=True)
    # Reject wrong source before loading model modules or creating cache folders.
    schema = "forcebench_portable_pi05_candidate_20261014"
    identity = validate_checkpoint_identity(checkpoint, args, paths, schema)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly before serving")
    activate(paths)
    from runtime_env import configure
    configure()
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared import normalize
    from openpi.training.forcebench_config import make_config, SCHEMA
    if SCHEMA != schema:
        raise ValueError("Portable schema changed; recheck the source package")
    config = make_config(args.task, args.method, paths["run"], repo_id=args.repo_id,
                         base_params=paths["base"], norm_assets=checkpoint / "assets",
                         input_identity=paths["identity"], checkpoint=checkpoint,
                         steps=identity["training_steps"],
                         new_lr_multiplier=identity["new_module_lr_multiplier"],
                         gate_initial=identity["f3_effective_gate_initial"])
    stats = normalize.load(checkpoint / "assets" / args.repo_id)
    config = dataclasses.replace(config, policy_metadata={**config.policy_metadata,
                                 "diagnostic_only": False, "source_sha256": identity["source_sha256"]})
    policy = create_trained_policy(config, checkpoint,
                                   norm_stats={key: stats[key] for key in ("state", "actions")})
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
