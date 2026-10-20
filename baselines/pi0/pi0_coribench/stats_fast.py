"""Full-dataset numeric stats for the exact 50-step, endpoint-padded recipe."""
import json
import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize

def compute(dataset, output, expected_frames, expected_episodes):
    state_stats=normalize.RunningStats()
    action_stats=normalize.RunningStats()
    count=0
    files=sorted((dataset/'data').rglob('*.parquet'))
    if len(files)!=expected_episodes:
        raise ValueError('Expected one Parquet file per episode')
    for file in files:
        table=pq.read_table(file,columns=['state','actions','timestamp'])
        state=np.asarray(table['state'].to_pylist(),dtype=np.float32)
        actions=np.asarray(table['actions'].to_pylist(),dtype=np.float32)
        timestamps=np.asarray(table['timestamp'].to_pylist()).reshape(-1)
        n=len(state)
        if state.shape!=(n,7) or actions.shape!=(n,7) or not n:
            raise ValueError(file)
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError('Nonfinite data')
        if n>1 and not np.allclose(np.diff(timestamps),1/15,atol=1e-3,rtol=0):
            raise ValueError('Nonuniform timestamps; use a timestamp-aware loader')
        for start in range(0,n,256):
            end=min(n,start+256)
            index=np.minimum(np.arange(start,end)[:,None]+np.arange(50)[None,:],n-1)
            a=actions[index].copy()
            a[...,:6]-=state[start:end,None,:6]
            a[...,3:6]=(a[...,3:6]+np.pi)%(2*np.pi)-np.pi
            state_stats.update(np.pad(state[start:end],((0,0),(0,25))))
            action_stats.update(np.pad(a,((0,0),(0,0),(0,25))))
        count+=n
        print(f'STATS {file.name}: {count}/{expected_frames} frames',flush=True)
    if count!=expected_frames: raise ValueError('Frame count mismatch')
    normalize.save(output,{'state':state_stats.get_statistics(),'actions':action_stats.get_statistics()})
    (output/'coverage.json').write_text(json.dumps({'frames':count,'episodes':len(files),'action_horizon':50,
        'action_padding':'repeat episode endpoint','state':'measured TCP xyz/rpy/gripper',
        'normalization':'mean/std','all_frames_included':True},indent=2))
