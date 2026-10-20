"""Prepare/check/train pi0 without changing official data or shared packages."""
import argparse,dataclasses,functools,hashlib,json,os,pathlib,sys
P=pathlib.Path
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['config','check','stats','train'])
    parser.add_argument('--dataset',type=P,required=True)
    parser.add_argument('--repo-id',required=True)
    parser.add_argument('--run-dir',type=P,required=True)
    parser.add_argument('--base-params',type=P,required=True)
    args=parser.parse_args()
    root=P(__file__).resolve().parent.parent
    upstream=root/'openpi'
    sys.path[:0]=[str(upstream/'src'),str(upstream/'scripts')]
    dataset=args.dataset.resolve(strict=True)
    run_dir=args.run_dir.resolve()
    if not args.repo_id or len(P(args.repo_id).parts)!=2 or any(x in args.repo_id for x in ['..','\\']):
        raise ValueError('repo-id must be namespace/name')
    if any('archive' in x.lower() for x in dataset.parts):
        raise ValueError('Archived datasets are not accepted')
    if run_dir == dataset or run_dir.is_relative_to(dataset):
        raise ValueError('Run directory must be outside the formal dataset')
    info_path=dataset/'meta/info.json'
    info=json.loads(info_path.read_text())
    features=info['features']
    for key,shape in [('state',[7]),('actions',[7])]:
        if features[key]['shape'] != shape: raise ValueError(f'{key} must be 7D')
    for key in ['image','wrist_image']:
        if key not in features: raise ValueError(f'Missing {key}')
    if info['fps'] != 15: raise ValueError('This recipe requires 15Hz data; do not resample implicitly')
    os.environ['HF_LEROBOT_HOME']=str(run_dir/'hf_lerobot')
    os.environ['HF_HOME']=str(run_dir/'cache/huggingface')
    os.environ['HF_DATASETS_CACHE']=str(run_dir/'cache/datasets')
    os.environ['WANDB_MODE']='disabled'
    if args.mode != 'train': os.environ['JAX_PLATFORMS']='cpu'
    if args.mode == 'train' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Set CUDA_VISIBLE_DEVICES explicitly before training')
    
    from pi0_config import make_config
    from openpi.training import config as configs,data_loader
    import compat
    compat.install(dataset)
    config=make_config(args.repo_id,run_dir,args.base_params.resolve())
    assert not config.model.pi05 and not config.model.discrete_state_input
    assert config.num_train_steps==10000 and config.batch_size==32 and config.model.action_horizon==50
    identity={'recipe':'pi0_coribench_jax_full_10k_b32_h50_v1','repo_id':args.repo_id,
        'dataset':str(dataset),'info_sha256':hashlib.sha256(info_path.read_bytes()).hexdigest(),
        'base_params':str(args.base_params.resolve()),'config':repr(config),
        'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [P(__file__).resolve(),P(__file__).with_name('pi0_config.py'),P(__file__).with_name('compat.py'),P(__file__).with_name('stats_fast.py'),upstream/'scripts/train.py']}}
    print(json.dumps(identity,indent=2))
    if args.mode=='config': sys.exit()
    
    if args.mode=='check':
        import numpy as np
        ds=data_loader.create_torch_dataset(config.data.create(config.assets_dirs,config.model),50,config.model)
        dc=config.data.create(config.assets_dirs,config.model)
        sample=ds[0]
        for transform in dc.data_transforms.inputs: sample=transform(sample)
        assert sample['actions'].shape==(50,32)
        assert sample['state'].shape==(32,)
        assert sample['image_mask']=={'base_0_rgb':True,'left_wrist_0_rgb':True,'right_wrist_0_rgb':False}
        assert np.isfinite(sample['actions']).all()
        batch_loader=data_loader.create_data_loader(config,num_batches=1,skip_norm_stats=True)
        obs,actions=next(iter(batch_loader))
        assert actions.shape==(32,50,32)
        assert obs.state.shape==(32,32)
        assert obs.tokenized_prompt.shape==(32,48)
        print('FULL BATCH PASSED: 32 x 50 x 32, pi0 token length48, 4 workers.')
        print('CHECK PASSED: real decoded sample, TCP delta actions [50,32], two RGB views; no model updates.')
        sys.exit()
    
    stats_file=config.assets_dirs/args.repo_id/'norm_stats.json'
    if args.mode=='stats':
        if stats_file.exists(): raise FileExistsError(f'Refusing to overwrite {stats_file}')
        from stats_fast import compute
        compute(dataset,stats_file.parent,info['total_frames'],info['total_episodes'])
        marker=run_dir/'stats_identity.json'
        marker.write_text(json.dumps(identity,indent=2))
        sys.exit()
    
    if not args.base_params.is_dir(): raise FileNotFoundError(args.base_params)
    if not stats_file.is_file(): raise FileNotFoundError('Run stats first for this pi0 recipe')
    if json.loads((run_dir/'stats_identity.json').read_text()) != identity:
        raise ValueError('Data/config/source changed after statistics; create a fresh run directory')
    if config.checkpoint_dir.exists(): raise FileExistsError('Checkpoint output already exists; refusing to overwrite')
    import shutil,jax
    if shutil.disk_usage(run_dir).free < 120*1024**3: raise RuntimeError('Need at least 120GiB free before training')
    if not jax.devices() or any(x.platform!='gpu' for x in jax.devices()): raise RuntimeError('JAX GPU unavailable')
    if 32%jax.device_count(): raise RuntimeError('Global batch32 must be divisible by GPU count')
    run_dir.mkdir(parents=True,exist_ok=True)
    (run_dir/'training_identity.json').write_text(json.dumps(identity,indent=2))
    import train
    import logging
    logging.basicConfig(level=logging.INFO)
    train.main(config)

if __name__=='__main__':
    main()
