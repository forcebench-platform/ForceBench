"""ForceBench configs built directly on the official OpenPI TrainConfig."""
from collections.abc import Mapping, Sequence
import dataclasses
import json
from pathlib import Path
import numpy as np

from openpi import transforms as t
from openpi.models.pi0_config import Pi0Config
from openpi.policies.forcebench_policy import ForceBenchInputs, ForceBenchOutputs, WrapAngles, ClipWrench
from openpi.training import config as upstream
from openpi.training.optimizer import CosineDecaySchedule
from openpi.training.weight_loaders import CheckpointWeightLoader
from openpi.training.forcebench_training import ForceBenchAdamW

TASKS = json.loads(Path(__file__).with_name("forcebench_tasks.json").read_text())
SCHEMA = "forcebench_portable_pi05_candidate_20261014"


@dataclasses.dataclass(frozen=True)
class ForceBenchData(upstream.DataConfig):
    # Temporal queries used by the existing LeRobot loader, separate from policy.
    episodes: Sequence[int] | None = None
    delta_timestamps: Mapping[str, Sequence[float]] = dataclasses.field(default_factory=dict)
    highrate_force_root: str | None = None
    highrate_force_history_len: int = 1
    highrate_force_future_horizon: int = 0


@dataclasses.dataclass(frozen=True)
class ForceBenchDataConfig(upstream.DataConfigFactory):
    method: str = "F0"
    training: bool = True
    future_horizon: int = 10

    def create(self, assets_dirs, model_config):
        base = self.create_base_config(assets_dirs, model_config)
        if base.norm_stats is None:
            raise FileNotFoundError(f"Missing ForceBench normalization assets for {self.repo_id}")
        required = {"state", "actions"}
        if self.method in ("F1", "F3", "F4"):
            required.add("force_history")
        if self.method == "F4" and self.training:
            required.add("future_wrench")
        if not required.issubset(base.norm_stats):
            raise ValueError(f"Missing normalization keys: {required - base.norm_stats.keys()}")
        repack = {"observation/image": "image", "observation/wrist_image": "wrist_image",
                  "observation/state": "state", "actions": "actions", "prompt": "prompt"}
        queries = {}
        if self.method != "F0":
            repack.update(eef_wrench="eef_wrench", eef_wrench_is_pad="eef_wrench_is_pad")
            future = self.future_horizon if self.training and self.method == "F4" else 0
            queries["eef_wrench"] = [i / 15 for i in range(1 + future)]
        data_transforms = t.Group(
            inputs=[ForceBenchInputs(self.method, model_config.action_dim, self.training, self.future_horizon),
                    t.DeltaActions(t.make_bool_mask(6, -1)), WrapAngles()],
            outputs=[WrapAngles(), t.AbsoluteActions(t.make_bool_mask(6, -1)), ForceBenchOutputs()],
        )
        model_transforms = upstream.ModelTransformFactory()(model_config)
        extra = []
        if not self.training and self.method in ("F1", "F3", "F4"):
            # create_trained_policy receives state/action stats only because
            # its inverse transforms operate on outputs, not observations.
            # Normalize wrench explicitly from the same checkpoint's assets.
            extra.append(t.Normalize({"force_history": base.norm_stats["force_history"]}, use_quantiles=True, strict=True))
        extra.append(ClipWrench())
        model_transforms = model_transforms.push(inputs=extra)
        values = {field.name: getattr(base, field.name) for field in dataclasses.fields(upstream.DataConfig)}
        values.update(repack_transforms=t.Group(inputs=[t.RepackTransform(repack)]),
                      data_transforms=data_transforms, model_transforms=model_transforms,
                      action_sequence_keys=("actions",), prompt_from_task=True,
                      delta_timestamps=queries)
        return ForceBenchData(**values)


def make_config(task, method, root, *, repo_id, base_params, norm_assets, input_identity, steps=10_000, new_lr_multiplier=10.0, gate_initial=0.1,
                batch_size=32, resume=False, checkpoint=None):
    if task not in TASKS or method not in {"F0", "F1", "F2", "F3", "F4"}:
        raise ValueError(f"Unknown task/method: {task}/{method}")
    if steps <= 1000 or not 0 < gate_initial < 1:
        raise ValueError("Training must extend beyond warmup; effective gate initial must be in (0,1)")
    root = Path(root).resolve()
    spec = {**TASKS[task], "repo": repo_id}
    force = method in {"F1", "F3", "F4"}
    model = Pi0Config(pi05=True, action_dim=32, action_horizon=50,
                      force_enabled=force, force_dim=6, force_history_len=1,
                      force_num_tokens=6, force_encoder_mode="mlp",
                      force_fusion_mode="gated_cross_attention" if method == "F3" else "prefix_tokens",
                      force_cross_attention_heads=8, force_cross_attention_gate_init=gate_initial,
                      future_force_enabled=method == "F4", future_force_horizon=10,
                      future_force_hidden=256, future_force_loss_weight=0.0003,
                      future_force_stop_action_gradient=False, future_force_input_mode="force_tokens")
    missing = {"F1": ".*force_spec.*", "F3": ".*(?:force_spec|force_cross_attn).*", "F4": ".*(?:force_spec|ff_head).*"}
    loader = CheckpointWeightLoader(str(base_params), **({"missing_regex": missing[method]} if force else {}))
    assets = Path(checkpoint) / "assets" if checkpoint is not None else Path(norm_assets)
    metadata = {"schema": SCHEMA, "task": task, "method": method,
                "action_horizon": 50, "execute_chunk": 50, "control_hz": 15,
                "state_semantics": "measured_tcp_xyz_rpy_gripper", "action_semantics": "absolute_tcp_xyz_rpy_gripper",
                "camera_masks": [True, True, False], "future_labels_at_inference": False,
                "training_steps": steps, "new_module_lr_multiplier": new_lr_multiplier,
                "f3_effective_gate_initial": gate_initial, "portable_inputs": input_identity}
    return upstream.TrainConfig(
        name=f"forcebench_{task}_{method.lower()}", exp_name="seed_42", project_name="ForceBench",
        model=model, data=ForceBenchDataConfig(repo_id=spec["repo"], method=method,
            training=checkpoint is None, assets=upstream.AssetsConfig(assets_dir=str(assets), asset_id=spec["repo"])),
        weight_loader=loader, optimizer=ForceBenchAdamW(new_lr_multiplier),
        lr_schedule=CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-5, decay_steps=steps, decay_lr=2.5e-6),
        assets_base_dir=str(root / "assets"), checkpoint_base_dir=str(root / "checkpoints/pi05" / method / spec["label"]),
        seed=42, batch_size=batch_size, num_workers=4, num_train_steps=steps, fsdp_devices=1,
        ema_decay=0.99, save_interval=5000, keep_period=None, log_interval=50,
        resume=resume, overwrite=False, wandb_enabled=False, policy_metadata=metadata,
    )
