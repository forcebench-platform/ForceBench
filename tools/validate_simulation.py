#!/usr/bin/env python3
"""Run bounded real simulator checks; these are not benchmark success runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forcebench.registry import task_names


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_outputs(directory, count, steps):
    import numpy as np
    from PIL import Image
    issues, episodes = [], []
    aggregate_path = directory / 'aggregate.json'
    if not aggregate_path.is_file():
        return ['missing aggregate.json'], []
    aggregate = json.loads(aggregate_path.read_text())
    if not aggregate.get('smoke_hold') or aggregate.get('environment_steps') != count * steps:
        issues.append('incorrect hold/step count')
    for seed in range(300, 300 + count):
        folder = directory / f'seed_{seed:03d}'
        summary = json.loads((folder / 'summary.json').read_text())
        arrays = np.load(folder / 'rollout.npz')
        for name, width in (('states', 7), ('actions', 7), ('joint_positions', 7), ('eef_wrenches', 6)):
            value = arrays[name]
            if value.shape != (steps, width) or not np.isfinite(value).all():
                issues.append(f'{seed}/{name}: invalid shape or nonfinite values')
        images = []
        for name in ('main_initial.png', 'wrist_initial.png', 'main_final.png', 'wrist_final.png'):
            path = folder / name
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'))
            if pixels.shape != (224, 224, 3) or float(pixels.std()) < 1:
                issues.append(f'{seed}/{name}: invalid or uniform RGB image')
            images.append({'path': str(path.relative_to(directory)), 'sha256': digest(path),
                           'std': float(pixels.std())})
        if summary.get('physics_rate_hz') != 60 or summary.get('steps') != steps:
            issues.append(f'{seed}: incorrect timing metadata')
        episodes.append({'seed': seed, 'images': images, 'summary_sha256': digest(folder/'summary.json'),
                         'rollout_sha256': digest(folder/'rollout.npz'),
                         'final_success': summary['success'], 'success_is_smoke_metric': True})
    return issues, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--timeout', type=int, default=480)
    parser.add_argument('--tasks', nargs='+', choices=task_names(), default=list(task_names()))
    parser.add_argument('--single-env-check', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output directory; validation never overwrites prior runs')
    if args.steps < 1 or args.timeout < 1:
        parser.error('positive steps and timeout required')
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT)+os.pathsep+env.get('PYTHONPATH',''),
               FORCEBENCH_ASSET_DIR=str(args.assets.resolve()), PYTHONUNBUFFERED='1')
    source = {str(p.relative_to(ROOT)):digest(p) for d in ('forcebench','scripts/evaluation')
              for p in (ROOT/d).rglob('*.py')}
    report = {'schema_version':1, 'scope':'Real GPU simulator reset/hold/observation/outcome checks; no model score reproduction.',
              'python':args.python, 'source_sha256':source, 'cases':[],
              'environment':{k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','FORCEBENCH_ASSET_DIR','LD_PRELOAD','LD_LIBRARY_PATH')}}
    cases = [(task,2) for task in args.tasks]
    if args.single_env_check:
        cases += [(task,1) for task in ('weight_sort','nut_thread') if task in args.tasks]
    for task,count in cases:
        directory=args.output/f'{task}_{count}env'
        log=args.output/f'{task}_{count}env.log'
        command=[args.python,str(ROOT/'scripts/evaluation/inference_forcebench_vectorized.py'),
                 '--task',task,'--seed_start','300','--num_envs',str(count),'--max_steps',str(args.steps),
                 '--smoke_hold','--save_images','--save_rollouts','--no-terminate_on_success',
                 '--final_settle_frames','2','--output_dir',str(directory),
                 '--headless','--enable_cameras','--device','cuda:0']
        start=time.time();timed_out=False
        with log.open('w') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);code=process.wait()
        problems=[];episodes=[]
        if code==0:
            try:problems,episodes=check_outputs(directory,count,args.steps)
            except Exception as exc:problems=[f'{type(exc).__name__}: {exc}']
        else:problems=[f'process exit {code}']
        errors=[line for line in log.read_text(errors='replace').splitlines() if '[Error]' in line]
        if errors:problems += [f'{len(errors)} simulator error log lines']
        row={'task':task,'num_envs':count,'command':command,'seconds':time.time()-start,
             'exit_code':code,'timed_out':timed_out,'passed':not problems,'issues':problems,
             'log_sha256':digest(log),'error_lines':errors[:12],'episodes':episodes}
        report['cases'].append(row)
        report['passed']=all(case['passed'] for case in report['cases'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'task':task,'num_envs':count,'passed':row['passed'],'issues':problems}),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
