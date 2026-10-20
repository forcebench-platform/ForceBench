"""Real-model GPU diagnostics for a three-update preflight, never a paper result.

Called only by run.py train --preflight --validate-inference. The production
serve.py checkpoint budget and identity checks are intentionally unchanged.
"""
import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
import time


def validate_preflight_policy(config, state, identity, run_root):
    import flax.nnx as nnx
    import jax
    import numpy as np
    from openpi import transforms
    from openpi.policies.policy import Policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import data_loader
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client as ws_client
    import websockets.asyncio.server as ws_server

    if int(state.step) != 3:
        raise ValueError("The GPU diagnostic requires exactly three completed updates")
    inference_config = dataclasses.replace(config,
        data=dataclasses.replace(config.data, training=False))
    data_config = inference_config.data.create(config.assets_dirs, config.model)
    # Read an actual recorded observation, including both decoded RGB views.
    dataset = data_loader.create_torch_dataset(data_config, 50, config.model)
    item = dataset[0]
    observation = {
        "observation/image": np.asarray(item["image"]),
        "observation/wrist_image": np.asarray(item["wrist_image"]),
        "observation/state": np.asarray(item["state"]),
        "prompt": item["prompt"],
    }
    if identity["method"] != "F0":
        observation["eef_wrench"] = np.asarray(item["eef_wrench"])
        observation["eef_wrench_is_pad"] = np.asarray(item["eef_wrench_is_pad"])
    if any(key in observation for key in ("actions", "future_wrench", "wrench_window")):
        raise AssertionError("Inference observation contains training targets")
    model = nnx.merge(state.model_def,
        state.params if state.ema_params is None else state.ema_params)
    model.eval()
    metadata = {**config.policy_metadata, "diagnostic_only": True,
                "completed_updates": 3, "checkpoint": None,
                "source_sha256": identity["source_sha256"]}
    # Use the same input/output chain as create_trained_policy. Wrench norms
    # are inserted by ForceBenchDataConfig; inverse output norms need state/actions.
    output_stats = {key: data_config.norm_stats[key] for key in ("state", "actions")}
    policy = Policy(model,
        transforms=[transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(output_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs],
        output_transforms=[*data_config.model_transforms.outputs,
            transforms.Unnormalize(output_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs], metadata=metadata)

    def check_response(response):
        actions = np.asarray(response["actions"])
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise AssertionError(f"Invalid real-model inference output: {actions.shape}")
        return {"shape": list(actions.shape), "finite": True,
                "sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                "min": float(actions.min()), "max": float(actions.max())}

    started = time.monotonic()
    direct = check_response(policy.infer(observation,
        noise=np.zeros((50, 32), dtype=np.float32)))
    direct["elapsed_s"] = time.monotonic() - started

    async def websocket_roundtrip():
        server = WebsocketPolicyServer(policy, host="127.0.0.1", port=0,
                                       metadata=metadata)
        packer = msgpack_numpy.Packer()
        async with ws_server.serve(server._handler, "127.0.0.1", 0,
                                   compression=None, max_size=None) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with ws_client.connect(f"ws://127.0.0.1:{port}",
                                          compression=None, max_size=None) as client:
                received_metadata = msgpack_numpy.unpackb(await client.recv())
                if received_metadata != metadata:
                    raise AssertionError("Websocket metadata changed in transport")
                replies = []
                for _ in range(2):
                    await client.send(packer.pack(observation))
                    result = await client.recv()
                    if isinstance(result, str):
                        raise RuntimeError(result)
                    response = msgpack_numpy.unpackb(result)
                    replies.append({**check_response(response),
                                    "server_timing": response["server_timing"]})
        return {"host": "127.0.0.1", "metadata_matched": True,
                "requests": 2, "replies": replies,
                "server_class": "openpi.serving.websocket_policy_server.WebsocketPolicyServer"}

    socket_result = asyncio.run(asyncio.wait_for(websocket_roundtrip(), timeout=300))
    receipt = {"check": "pi05_preflight_train_infer_websocket", "passed": True,
        "scope": "Three actual GPU training updates plus in-memory EMA-model inference on a recorded observation and loopback websocket transport. No paper scores or 10,000-update checkpoint validation.",
        "identity": identity, "completed_updates": 3, "checkpoint_saved": False,
        "jax_devices": [str(device) for device in jax.devices()],
        "observation_shapes": {key: list(value.shape) for key, value in observation.items()
                               if hasattr(value, "shape")},
        "future_targets_provided_at_inference": False,
        "direct_inference": direct, "websocket": socket_result,
        "elapsed_s": time.monotonic() - started,
        "limitations": ["Inherited GPU environment; not a clean installation.",
            "No checkpoint save/restore or production serving receipt was created.",
            "A three-update model is a software execution check, not a trained paper model."]}
    target = Path(run_root) / "preflight/gpu_inference_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"gpu_validation": str(target), "passed": True}), flush=True)
    return receipt
