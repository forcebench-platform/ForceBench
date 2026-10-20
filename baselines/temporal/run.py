"""Prepare statistics, inspect the recipe, or train a temporal ForceBench policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from recipe import METHODS, description, make_config, validate_repo_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['config', 'check', 'stats', 'train'])
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--lerobot-home', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base-params', type=Path, required=True)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    validate_repo_id(args.repo_id)
    dataset = (args.lerobot_home/args.repo_id).resolve(strict=True)
    run = args.run_dir.resolve()
    if run == dataset or run.is_relative_to(dataset):
        raise ValueError('Run directory must be outside the dataset')
    info_path = dataset/'meta/info.json'
    info = json.loads(info_path.read_text())
    if info['total_episodes'] != 200 or info['fps'] != 15:
        raise ValueError('Expected the selected 200-demonstration, 15 Hz task dataset')
    for key, shape in [('state', [7]), ('actions', [7]), ('eef_wrench', [6])]:
        if info['features'][key]['shape'] != shape:
            raise ValueError(f'Expected {key} shape {shape}')
    for key in ['image', 'wrist_image']:
        if key not in info['features']:
            raise ValueError(f'Missing {key}')
    if args.mode == 'train' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Set CUDA_VISIBLE_DEVICES explicitly')
    if args.mode != 'train':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['HF_LEROBOT_HOME'] = str(args.lerobot_home.resolve())
    os.environ['HF_HOME'] = str(run/'cache/huggingface')
    os.environ['HF_DATASETS_CACHE'] = str(run/'cache/datasets')
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    root = Path(__file__).resolve().parent
    sys.path[:0] = [str(root/'runtime/src'), str(root/'runtime/scripts')]
    config = make_config(args.method, args.repo_id, run, args.base_params.resolve(), args.num_workers)
    identity = {'recipe': description(args.method), 'dataset': str(dataset),
        'info_sha256': hashlib.sha256(info_path.read_bytes()).hexdigest(),
        'config': repr(config)}
    print(json.dumps(identity, indent=2))
    if args.mode == 'config':
        return
    import numpy as np
    from openpi.training import data_loader
    stats_path = config.assets_dirs/args.repo_id/'norm_stats.json'
    marker = run/'stats_identity.json'
    if args.mode in ('check', 'stats'):
        dc = config.data.create(config.assets_dirs, config.model)
        ds = data_loader.create_torch_dataset(dc, 50, config.model)
        sample = data_loader.TransformedDataset(ds, [*dc.repack_transforms.inputs,
            *dc.data_transforms.inputs])[min(20, len(ds)-1)]
        if np.asarray(sample['state']).shape != (32,) or not np.isfinite(sample['state']).all():
            raise ValueError('Invalid augmented state')
        if np.asarray(sample['actions']).shape != (50, 32):
            raise ValueError('Invalid action sequence')
        if args.method == 'dual_path_fusion' and np.asarray(sample['force_history']).shape != (16, 6):
            raise ValueError('Invalid dense force history')
        del ds, sample
        if args.mode == 'check':
            print('Dataset transform check passed; no model loaded or trained.')
            return
        if stats_path.exists() or marker.exists():
            raise FileExistsError('Statistics already exist; choose a fresh run directory')
        import compute_norm_stats
        compute_norm_stats.main(config.name)
        original = stats_path.read_text()
        stats_path.with_name('norm_stats_before_degenerate_guard.json').write_text(original)
        stats = json.loads(original)
        # Preserve the near-constant quantile guard from the evaluated training recipe.
        widened = []
        for key, values in stats['norm_stats'].items():
            if 'q01' not in values:
                continue
            for i, (lo, hi) in enumerate(zip(values['q01'], values['q99'])):
                if hi-lo < 1e-3:
                    values['q01'][i] = values['mean'][i]-1.0
                    values['q99'][i] = values['mean'][i]+1.0
                    widened.append([key, i])
        stats_path.write_text(json.dumps(stats, indent=2))
        identity['norm_sha256'] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        identity['widened_dimensions'] = widened
        marker.write_text(json.dumps(identity, indent=2))
        return
    previous = json.loads(marker.read_text())
    if any(previous[key] != value for key, value in identity.items()):
        raise ValueError('Recipe or dataset changed after statistics; use a fresh run directory')
    if hashlib.sha256(stats_path.read_bytes()).hexdigest() != previous['norm_sha256']:
        raise ValueError('Normalization statistics changed after preparation')
    if config.checkpoint_dir.exists():
        raise FileExistsError('Refusing to overwrite an existing checkpoint directory')
    import jax
    if any(device.platform != 'gpu' for device in jax.devices()):
        raise RuntimeError('JAX GPU unavailable')
    if 32 % jax.device_count():
        raise ValueError('Global batch 32 must be divisible by GPU count')
    import train
    train.main(config)


if __name__ == '__main__':
    main()
