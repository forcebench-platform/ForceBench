"""Serve an explicitly selected new candidate checkpoint after identity checks."""
import argparse
import asyncio
import dataclasses
import logging
import os
from pathlib import Path
from portable_common import activate, add_inputs, inspect_inputs, validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--server-mode", choices=("single", "batched"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--batch-window-ms", type=float)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1,65535]")
    if args.server_mode == "batched":
        if args.host != "0.0.0.0":
            parser.error("Preserved batched server binds 0.0.0.0; specify --host 0.0.0.0 explicitly")
        if args.max_batch_size is None or args.max_batch_size < 1 or args.batch_window_ms is None or args.batch_window_ms < 0:
            parser.error("Batched mode requires explicit positive --max-batch-size and non-negative --batch-window-ms")
    elif args.max_batch_size is not None or args.batch_window_ms is not None:
        parser.error("Batch parameters apply only to batched mode")
    paths = inspect_inputs(args)
    checkpoint = args.checkpoint.resolve(strict=True)
    validate_checkpoint(checkpoint, args, paths)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly")
    activate(args, paths)
    from candidate_recipes import make_config
    from openpi.policies.policy_config import create_trained_policy
    config = make_config(args, paths, checkpoint / "assets")
    policy = create_trained_policy(config, checkpoint)
    if args.server_mode == "single":
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()
    else:
        from serve_policy_batched import Args, BatchedWebsocketPolicyServer
        server_args = Args(config=config.name, checkpoint_dir=str(checkpoint), port=args.port,
                           max_batch_size=args.max_batch_size, batch_window_ms=args.batch_window_ms)
        asyncio.run(BatchedWebsocketPolicyServer(policy, server_args).run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
