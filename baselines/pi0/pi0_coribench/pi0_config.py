"""Sidecar CoRiBench configuration; upstream OpenPI remains unmodified."""
import dataclasses
import numpy as np
from openpi import transforms as t
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as c
from openpi.training.optimizer import CosineDecaySchedule
from openpi.training.weight_loaders import CheckpointWeightLoader

@dataclasses.dataclass(frozen=True)
class Inputs(t.DataTransformFn):
    def __call__(self, data):
        state = np.asarray(data['state'], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError('Expected measured TCP xyz/rpy/gripper state[7]')
        images = {}
        for name, key in [('base_0_rgb','image'),('left_wrist_0_rgb','wrist_image')]:
            im = np.asarray(data[key])
            if im.ndim == 3 and im.shape[0] == 3:
                im = np.transpose(im, (1,2,0))
            if im.ndim != 3 or im.shape[-1] != 3 or not np.isfinite(im).all():
                raise ValueError(f'Invalid RGB {key}: {im.shape}')
            if np.issubdtype(im.dtype, np.floating):
                if im.min() < 0 or im.max() > 1:
                    raise ValueError('Float RGB must be in [0,1]')
                im = (im * 255).astype(np.uint8)
            images[name] = im
        images['right_wrist_0_rgb'] = np.zeros_like(images['left_wrist_0_rgb'])
        out = {'image':images,'image_mask':{'base_0_rgb':True,'left_wrist_0_rgb':True,'right_wrist_0_rgb':False},'state':t.pad_to_dim(state,32)}
        if 'actions' in data:
            a = np.asarray(data['actions'], dtype=np.float32)
            if a.shape != (50,7) or not np.isfinite(a).all():
                raise ValueError(f'Expected actions[50,7], got {a.shape}')
            out['actions'] = t.pad_to_dim(a,32)
        if 'prompt' in data: out['prompt'] = data['prompt']
        return out

@dataclasses.dataclass(frozen=True)
class WrapAngles(t.DataTransformFn):
    def __call__(self, data):
        if 'actions' not in data: return data
        a = np.asarray(data['actions']).copy()
        a[...,3:6] = (a[...,3:6] + np.pi) % (2*np.pi) - np.pi
        return {**data,'actions':a}

@dataclasses.dataclass(frozen=True)
class Outputs(t.DataTransformFn):
    def __call__(self, data):
        return {'actions':np.asarray(data['actions'])[...,:7]}

@dataclasses.dataclass(frozen=True)
class Data(c.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(self.create_base_config(assets_dirs,model_config),
            data_transforms=t.Group(
                inputs=[Inputs(),t.DeltaActions(t.make_bool_mask(6,-1)),WrapAngles()],
                outputs=[WrapAngles(),t.AbsoluteActions(t.make_bool_mask(6,-1)),Outputs()]),
            model_transforms=c.ModelTransformFactory()(model_config),
            action_sequence_keys=('actions',),prompt_from_task=True)

def make_config(repo_id, run_dir, base_params):
    return c.TrainConfig(name='coribench_pi0_10k',exp_name='seed42',
        model=Pi0Config(pi05=False,discrete_state_input=False,action_dim=32,action_horizon=50),
        data=Data(repo_id=repo_id),weight_loader=CheckpointWeightLoader(str(base_params)),
        assets_base_dir=str(run_dir/'assets'),checkpoint_base_dir=str(run_dir/'checkpoints'),
        num_train_steps=10000,batch_size=32,seed=42,num_workers=4,fsdp_devices=1,
        lr_schedule=CosineDecaySchedule(warmup_steps=1000,peak_lr=2.5e-5,decay_steps=10000,decay_lr=2.5e-6),
        ema_decay=0.99,save_interval=5000,log_interval=50,overwrite=False,resume=False,wandb_enabled=False)
