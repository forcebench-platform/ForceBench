"""Portable configuration of the observed 16-frame temporal recipes.

Model and data transforms come from the September 14 training snapshot.
No task data or pretrained weights are selected implicitly.
"""
from pathlib import Path

METHODS = ('temporal_fusion', 'dual_path_fusion')
TASKS = ('weight_sort', 'com_sort', 'faucet_turn', 'faucet_direction',
         'stove_turn', 'stove_press', 'drawer_pull', 'nut_thread')


def validate_repo_id(repo_id):
    if len(repo_id.split('/')) != 2 or any(x in repo_id for x in ('..', '\\')):
        raise ValueError('repo-id must be namespace/name')
    if not all(repo_id.split('/')):
        raise ValueError('repo-id must be namespace/name')


def description(method):
    if method not in METHODS:
        raise ValueError(method)
    return {'method': method, 'history_frames': 16, 'history_rate_hz': 15,
            'wrench_dim': 6, 'action_horizon': 50, 'execution_steps': 50,
            'training_steps': 10000, 'batch_size': 32, 'seed': 42,
            'stats_in_state': True, 'separate_force_tokens': method == 'dual_path_fusion',
            'state_statistics': ['current', 'mean', 'population_std'],
            'force_tokens': 4 if method == 'dual_path_fusion' else 0}


def make_config(method, repo_id, run_dir, base_params, num_workers=4):
    validate_repo_id(repo_id)
    description(method)
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as c
    from openpi.training.weight_loaders import CheckpointWeightLoader
    dual = method == 'dual_path_fusion'
    model = Pi0Config(pi05=True)
    loader = CheckpointWeightLoader(str(base_params))
    if dual:
        model = Pi0Config(pi05=True, force_enabled=True, force_dim=6,
            force_history_len=16, force_num_tokens=4, force_encoder_mode='stats',
            force_fusion_mode='prefix_tokens', force_dropout_prob=0.3, force_noise_std=0.0)
        loader = CheckpointWeightLoader(str(base_params), missing_regex='.*force_spec.*')
    config = c.TrainConfig(name='forcebench_'+method, exp_name='seed42', model=model,
        data=c.LeRobotAutoBioDataConfig(repo_id=repo_id,
            base_config=c.DataConfig(prompt_from_task=True), real_action_dim=7,
            force_enabled=True, force_history_len=16, force_sample_rate=15,
            force_stats_in_state=True, force_history_also=dual, wrap_rpy_action_deltas=True),
        weight_loader=loader, assets_base_dir=str(Path(run_dir)/'assets'),
        checkpoint_base_dir=str(Path(run_dir)/'checkpoints'),
        batch_size=32, num_workers=num_workers, num_train_steps=10000,
        log_interval=50, save_interval=10000, keep_period=None, fsdp_devices=1,
        seed=42, overwrite=False, resume=False, wandb_enabled=False)
    c._CONFIGS_DICT[config.name] = config
    return config
