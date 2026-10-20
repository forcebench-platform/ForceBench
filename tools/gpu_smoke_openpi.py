"""Bounded real-data/model diagnostic for pi0, temporal and ForceVLA packages.

This calls the packaged numerical initialization/update/inference definitions;
it never manufactures a production checkpoint receipt or changes paper values.
Run each family in a separate process with explicit CUDA_VISIBLE_DEVICES and an
external timeout. Three updates are software validation, not paper retraining.
"""
import argparse
import asyncio
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if (path.is_file() and '__pycache__' not in path.parts
                and '.venv' not in path.parts and path.suffix in {'.py', '.json', '.toml', '.lock'}):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('pi0', 'temporal', 'forcevla'), required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--lerobot-home', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--norm-file', type=Path, required=True)
    parser.add_argument('--base-params', type=Path, required=True)
    parser.add_argument('--tokenizer-file', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-snapshot', choices=('base', 'stove_sep14', 'stovepress_sep13'))
    parser.add_argument('--recipe')
    args = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Set CUDA_VISIBLE_DEVICES explicitly')
    if len(args.repo_id.split('/')) != 2 or any(part in ('', '.', '..') for part in args.repo_id.split('/')):
        parser.error('repo-id must be namespace/name')
    home = args.lerobot_home.resolve(strict=True)
    dataset = (home / args.repo_id).resolve(strict=True)
    if not dataset.is_relative_to(home):
        raise ValueError('Dataset must remain inside the explicit root')
    package = REPO_ROOT / 'baselines' / args.family
    run = args.run_dir.resolve()
    base = args.base_params.resolve(strict=True)
    if any(run == p or run.is_relative_to(p) or p.is_relative_to(run) for p in (dataset, base, package)):
        raise ValueError('Run must be separate from source, data and base params')
    info_path = dataset / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if (info['total_episodes'], info['fps']) != (200, 15):
        raise ValueError('Expected 200 real demonstrations at 15 Hz')
    norm = args.norm_file.resolve(strict=True)
    if not {'state', 'actions'}.issubset(json.loads(norm.read_text())['norm_stats']):
        raise ValueError('Missing state/action normalization')
    run.mkdir(parents=True, exist_ok=False)
    for key, rel in {'HF_HOME':'cache/huggingface', 'HF_DATASETS_CACHE':'cache/datasets',
        'HF_HUB_CACHE':'cache/huggingface/hub', 'OPENPI_DATA_HOME':'cache/openpi',
        'JAX_COMPILATION_CACHE_DIR':'cache/jax', 'XDG_CACHE_HOME':'cache/xdg',
        'CUDA_CACHE_PATH':'cache/cuda', 'TMPDIR':'tmp', 'TEMP':'tmp', 'TMP':'tmp',
        'WANDB_DIR':'logs/wandb', 'TORCH_HOME':'cache/torch'}.items():
        path = run / rel
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(HF_LEROBOT_HOME=str(home), HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    sys.dont_write_bytecode = True
    token_target = run / 'cache/openpi/big_vision/paligemma_tokenizer.model'
    token_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.tokenizer_file, token_target)
    selected_source_hash = source_hash(package)
    if args.family == 'temporal':
        if args.method not in ('temporal_fusion', 'dual_path_fusion'):
            raise ValueError('Unknown temporal method')
        sys.path[:0] = [str(package), str(package/'runtime/src'),
            str(package/'runtime/scripts'), str(package/'runtime/packages/openpi-client/src')]
        from recipe import make_config
        config = make_config(args.method, args.repo_id, run, base, num_workers=0)
    elif args.family == 'pi0':
        if args.method != 'pi0':
            raise ValueError('pi0 family requires method pi0')
        sys.path[:0] = [str(package/'pi0_coribench'), str(package/'openpi/src'),
            str(package/'openpi/scripts'), str(package/'openpi/packages/openpi-client/src')]
        from pi0_config import make_config
        config = dataclasses.replace(make_config(args.repo_id, run, base), num_workers=0)
        import compat
        compat.install(dataset)
    else:
        if args.method != 'forcevla' or not args.source_snapshot or not args.recipe:
            raise ValueError('ForceVLA requires method, source snapshot and recipe')
        sys.path.insert(0, str(package))
        from portable_common import activate, inspect_inputs
        paths = inspect_inputs(args)
        activate(args, paths)
        from candidate_recipes import make_config
        config = dataclasses.replace(make_config(args, paths, run/'assets'), num_workers=0)
    # OpenPI's assets_dirs appends the configuration name. An explicit
    # AssetsConfig.assets_dir overrides it (as used by ForceVLA).
    assets_root = config.data.assets.assets_dir or config.assets_dirs
    assets = Path(assets_root) / args.repo_id
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(norm, assets / 'norm_stats.json')
    if sha(norm) != sha(assets / 'norm_stats.json'):
        raise ValueError('Selected normalization bytes changed during import')
    # Match the Arrow List/Sequence compatibility already used by controlled pi05.
    from datasets import Features
    original = Features.from_arrow_schema.__func__
    def compatible(cls, schema):
        metadata = dict(schema.metadata or {})
        if b'"_type": "List"' in metadata.get(b'huggingface', b''):
            metadata[b'huggingface'] = metadata[b'huggingface'].replace(b'"_type": "List"', b'"_type": "Sequence"')
            schema = schema.with_metadata(metadata)
        return original(cls, schema)
    Features.from_arrow_schema = classmethod(compatible)
    import jax
    import numpy as np
    import flax.nnx as nnx
    import train
    from openpi import transforms
    from openpi.training import data_loader, sharding
    from openpi.policies.policy import Policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client as ws_client
    import websockets.asyncio.server as ws_server
    if any(device.platform != 'gpu' for device in jax.devices()):
        raise RuntimeError('JAX GPU unavailable')
    jax.config.update('jax_compilation_cache_dir', str(run/'cache/jax'))
    identity = {'family':args.family, 'method':args.method, 'source_sha256':selected_source_hash,
        'tool_sha256':sha(__file__), 'dataset_info_sha256':sha(info_path),
        'norm_sha256':sha(norm), 'tokenizer_sha256':sha(token_target),
        'base_params':str(base), 'repo_id':args.repo_id, 'config':repr(config),
        'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'jax_devices':[str(device)for device in jax.devices()]}
    (run/'diagnostic_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    mesh = sharding.make_mesh(config.fsdp_devices)
    layout = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    loader = data_loader.create_data_loader(config,sharding=layout,shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)
    train_rng,init_rng = jax.random.split(jax.random.key(config.seed))
    started = time.monotonic()
    state,state_sharding = train.init_train_state(config,init_rng,mesh,resume=False)
    step_fn = jax.jit(functools.partial(train.train_step,config),
        in_shardings=(replicated,state_sharding,layout),
        out_shardings=(state_sharding,replicated),donate_argnums=(1,))
    metrics = []
    for step in range(3):
        with sharding.set_mesh(mesh):
            state,values = step_fn(train_rng,state,batch)
        values = {key:float(value)for key,value in jax.device_get(values).items()}
        if not all(np.isfinite(value)for value in values.values()) or values['grad_norm'] <= 0:
            raise AssertionError('Nonfinite metrics or zero gradient')
        metrics.append(values)
        print(json.dumps({'update':step+1,'metrics':values}),flush=True)
        if step<2:batch=next(iterator)
    dc = config.data.create(config.assets_dirs,config.model)
    raw_dataset = data_loader.create_torch_dataset(dc,50,config.model)
    raw = raw_dataset[min(20,len(raw_dataset)-1)]
    observation = transforms.compose(dc.repack_transforms.inputs)(raw)
    observation.pop('actions',None)
    # Materialize Torch arrays for the actual server serialization protocol.
    observation = {key:np.asarray(value)if hasattr(value,'shape')else value
                   for key,value in observation.items()}
    model = nnx.merge(state.model_def,state.params if state.ema_params is None else state.ema_params)
    model.eval()
    metadata = {'method':args.method,'diagnostic_only':True,'completed_updates':3}
    # Match create_trained_policy: force_history is an observation statistic,
    # while the policy returns state and actions for inverse normalization.
    output_keys = set(dc.action_sequence_keys) | {'state'}
    output_norm_stats = {key:value for key,value in dc.norm_stats.items()
                         if key in output_keys}
    policy = Policy(model,transforms=[transforms.InjectDefaultPrompt(None),
        *dc.data_transforms.inputs,transforms.Normalize(dc.norm_stats,use_quantiles=dc.use_quantile_norm),
        *dc.model_transforms.inputs],output_transforms=[*dc.model_transforms.outputs,
        transforms.Unnormalize(output_norm_stats,use_quantiles=dc.use_quantile_norm),
        *dc.data_transforms.outputs],metadata=metadata)
    def check(response):
        actions = np.asarray(response['actions'])
        if actions.shape!=(50,7) or not np.isfinite(actions).all():
            raise AssertionError(f'Invalid actions {actions.shape}')
        return {'shape':list(actions.shape),'finite':True,'sha256':hashlib.sha256(actions.tobytes()).hexdigest()}
    # Training shards batches over FSDP devices; single-observation serving
    # must not impose the training DATA_AXIS divisibility constraint.
    direct = check(policy.infer(observation))
    async def exchange():
        server = WebsocketPolicyServer(policy,host='127.0.0.1',port=0,metadata=metadata)
        packer = msgpack_numpy.Packer()
        async with ws_server.serve(server._handler,'127.0.0.1',0,compression=None,max_size=None)as listener:
            port = listener.sockets[0].getsockname()[1]
            async with ws_client.connect(f'ws://127.0.0.1:{port}',compression=None,max_size=None)as client:
                if msgpack_numpy.unpackb(await client.recv())!=metadata:
                    raise AssertionError('Metadata transport mismatch')
                await client.send(packer.pack(observation))
                response = await client.recv()
                if isinstance(response,str):raise RuntimeError(response)
                return check(msgpack_numpy.unpackb(response))
    websocket = asyncio.run(asyncio.wait_for(exchange(),timeout=300))
    if selected_source_hash!=source_hash(package):raise RuntimeError('Source changed during validation')
    receipt={'passed':True,'identity':identity,'updates':3,'metrics':metrics,
        'batch_size':config.batch_size,'fsdp_devices':config.fsdp_devices,
        'direct_inference':direct,'websocket':websocket,'elapsed_s':time.monotonic()-started,
        'checkpoint_saved':False,'diagnostic_only':True,
        'limitations':['Three updates do not establish trained model quality or paper reproduction.',
                       'No production checkpoint identity or checkpoint restore was created.',
                       'Selected norms/data are explicit software-check inputs, not a claim of final paper binding.']}
    (run/'gpu_smoke_result.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'receipt':str(run/'gpu_smoke_result.json')}),flush=True)


if __name__=='__main__':
    main()
