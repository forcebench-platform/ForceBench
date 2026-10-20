"""Train the versioned ForceBench recipe; preflight is three updates only."""
import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import time
from runtime_env import ROOT, RUNTIME, configure, require_space
configure()

import jax
import numpy as np
import torch
import train_core as core
from openpi.training import checkpoints, data_loader, sharding
from openpi.training.forcebench_config import make_config, TASKS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def source_identity():
    from portable_common import source_identity as identity
    return identity()


def schema_compatibility():
    # Datasets 3.x calls fixed-length metadata Sequence; old Parquet calls it List.
    from datasets import Features
    original = Features.from_arrow_schema.__func__
    def compatible(cls, schema):
        metadata = dict(schema.metadata or {})
        if b'"_type": "List"' in metadata.get(b"huggingface", b""):
            metadata[b"huggingface"] = metadata[b"huggingface"].replace(b'"_type": "List"', b'"_type": "Sequence"')
            schema = schema.with_metadata(metadata)
        return original(cls, schema)
    Features.from_arrow_schema = classmethod(compatible)


def validate_batch(batch, method, repo_id):
    dataset = Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["total_episodes"] != 200 or info["fps"] != 15:
        raise ValueError("ForceBench training requires the versioned 200-episode, 15 Hz dataset")
    if len(list((dataset / "data").rglob("*.parquet"))) != 200:
        raise ValueError("Expected 200 episode Parquet files")
    obs, actions = batch
    masks = {key: np.asarray(jax.device_get(value)) for key, value in obs.image_masks.items()}
    assert masks["base_0_rgb"].all() and masks["left_wrist_0_rgb"].all()
    assert not masks["right_wrist_0_rgb"].any()
    assert actions.shape[1:] == (50, 32)
    assert (obs.force_history is not None) == (method in {"F1", "F3", "F4"})
    assert (obs.future_wrench is not None) == (method == "F4")
    if obs.force_history is not None:
        assert obs.force_history.shape[1:] == (1, 6)
    if obs.future_wrench is not None:
        assert obs.future_wrench.shape[1:] == (10, 6)
    assert all(np.isfinite(np.asarray(jax.device_get(x))).all() for x in jax.tree.leaves(batch))


def retain_inference_milestone(checkpoint, destination):
    # On the same filesystem, hardlinks retain params after Orbax removes the
    # old recovery checkpoint without duplicating weights or optimizer states.
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("params", "assets"):
        shutil.copytree(checkpoint / name, destination / name, copy_function=os.link)
    for name in ("_CHECKPOINT_METADATA", "forcebench_identity.json"):
        if (checkpoint / name).is_file():
            shutil.copy2(checkpoint / name, destination / name)


def run(args):
    require_space(ROOT, 120)
    config = make_config(args.task, args.method, ROOT, repo_id=args.repo_id,
                         base_params=args.base_params, norm_assets=args.norm_assets, input_identity=args.input_identity, steps=args.steps,
                         new_lr_multiplier=args.new_lr_multiplier, gate_initial=args.gate_initial,
                         batch_size=args.batch_size, resume=args.resume)
    if args.preflight:
        config = dataclasses.replace(config, num_workers=0,
            checkpoint_base_dir=str(ROOT / "preflight/checkpoints"))
    stats_path=Path(config.data.assets.assets_dir)/config.data.repo_id/"norm_stats.json"
    identity = {**config.policy_metadata, "source_sha256": source_identity(),
                "normalization_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest()}
    status = ROOT / ("preflight" if args.preflight else "logs") / f"{args.task}_{args.method.lower()}_status.json"
    protocol_path = config.checkpoint_dir / "training_protocol.json"
    if args.resume:
        previous = json.loads(protocol_path.read_text())
        if previous != {**identity, "batch_size": config.batch_size}:
            raise ValueError("Resume would change recipe, source identity, or batch size")
    elif config.checkpoint_dir.exists() and not args.preflight:
        raise FileExistsError(f"Refusing to overwrite {config.checkpoint_dir}; use --resume with matching identity")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    schema_compatibility()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache/jax"))
    mesh = sharding.make_mesh(config.fsdp_devices)
    layout = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = data_loader.create_data_loader(config, sharding=layout, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)
    validate_batch(batch, args.method, config.data.repo_id)
    manager = None
    if not args.preflight:
        manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir,
            keep_period=None, overwrite=False, resume=args.resume)
        write_json(protocol_path, {**identity, "batch_size": config.batch_size})
    else:
        resuming = False
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    started = time.time()
    write_json(status, {**identity, "stage": "initializing", "pid": os.getpid()})
    state, state_sharding = core.init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = checkpoints.restore_state(manager, state, loader)
        if int(state.step) >= config.num_train_steps:
            manager.close()
            logging.info("Training budget already completed: %s updates", int(state.step))
            return
        for _ in range(int(state.step)):
            batch = next(iterator)
    step_fn = jax.jit(functools.partial(core.train_step, config),
        in_shardings=(replicated, state_sharding, layout),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    limit = 3 if args.preflight else config.num_train_steps
    metrics_path = status.with_suffix(".jsonl")
    for step in range(int(state.step), limit):
        if step % 50 == 0:
            require_space(ROOT, 80)
        with sharding.set_mesh(mesh):
            state, metrics = step_fn(train_rng, state, batch)
        values = {key: float(value) for key, value in jax.device_get(metrics).items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError(values)
        if step % config.log_interval == 0 or args.preflight or step + 1 == limit:
            record = {**identity, "stage": "preflight" if args.preflight else "training",
                "updates": step + 1, "metrics": values, "elapsed_s": time.time() - started}
            write_json(status, record)
            with metrics_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            logging.info("UPDATE %s/%s %s", step + 1, limit, values)
        if manager and ((step + 1) % config.save_interval == 0 or step + 1 == limit):
            require_space(ROOT, 80)
            checkpoints.save_state(manager, state, loader, step)
            manager.wait_until_finished()
            checkpoint = config.checkpoint_dir / str(step)
            from portable_common import preserve_checkpoint_norm
            preserve_checkpoint_norm(checkpoint, config.data.repo_id, stats_path, identity["normalization_sha256"])
            write_json(checkpoint / "forcebench_identity.json", {**identity, "completed_updates": step + 1})
            if (step + 1) % 5_000 == 0 or step + 1 == limit:
                retain_inference_milestone(checkpoint, config.checkpoint_dir / "milestones" / str(step))
        if step + 1 < limit:
            batch = next(iterator)
    if manager:
        manager.close()
    if args.preflight and args.method in {"F1", "F3", "F4"}:
        for module in ("force_spec", "force_cross_attn", "ff_head"):
            if module in state.params:
                assert values[f"{module}/grad_norm"] > 0
                assert values[f"{module}/update_norm"] > 0
    write_json(status, {**identity, "stage": "preflight_passed" if args.preflight else "training_completed",
        "updates": limit, "metrics": values, "elapsed_s": time.time() - started,
        "checkpoint": None if args.preflight else str(config.checkpoint_dir / str(limit - 1)),
        "camera_masks": [True, True, False]})
    if args.preflight and getattr(args, "validate_inference", False):
        from gpu_validation import validate_preflight_policy
        validate_preflight_policy(config, state, identity, ROOT)


if __name__ == "__main__":
    raise SystemExit("Use baselines/pi05/run.py with explicit input paths")
