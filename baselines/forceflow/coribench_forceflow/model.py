from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CleanDiffuser"))

from cleandiffuser.diffusion import ContinuousRectifiedFlow
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_condition.resnets import ResNet18
from cleandiffuser.nn_diffusion import DiT1d


class ForceFlowCondition(IdentityCondition):
    """The two-view, state and force-history condition used by ForceFlow."""

    def __init__(self, image_size: int, observation_horizon: int, force_history: int):
        super().__init__(dropout=0.0)
        self.observation_horizon = observation_horizon
        self.force_history = force_history
        self.arm_encoder = ResNet18(image_size, 3, 256)
        self.fixed_encoder = ResNet18(image_size, 3, 256)
        self.state_mlp = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 64))
        self.force_mlp = nn.Sequential(
            nn.Linear(force_history * 6, 128), nn.SiLU(), nn.Linear(128, 128)
        )

    def forward(self, condition: dict[str, torch.Tensor], mask=None) -> dict[str, Any]:
        arm = condition["image_arm"]
        fixed = condition["image_fix"]
        state = condition["lowdim"]
        force = condition["force"]
        batch, horizon = arm.shape[:2]
        if horizon != self.observation_horizon:
            raise ValueError(f"observation horizon {horizon}, expected {self.observation_horizon}")
        if tuple(force.shape[1:]) != (self.force_history, 6):
            raise ValueError(f"force history {tuple(force.shape)}, expected (*,{self.force_history},6)")
        arm_feature = self.arm_encoder(arm.reshape(batch * horizon, *arm.shape[-3:]))
        fixed_feature = self.fixed_encoder(fixed.reshape(batch * horizon, *fixed.shape[-3:]))
        seq_condition = torch.cat(
            [arm_feature.reshape(batch, horizon, -1), fixed_feature.reshape(batch, horizon, -1)],
            dim=-1,
        )
        state_feature = self.state_mlp(state).reshape(batch, -1)
        force_feature = self.force_mlp(force.reshape(batch, -1))
        return {
            "vec_condition": torch.cat([state_feature, force_feature], dim=-1),
            "seq_condition": seq_condition,
            "seq_condition_mask": None,
        }


def build_policy(config: dict[str, Any]) -> ContinuousRectifiedFlow:
    temporal = config["temporal"]
    condition = ForceFlowCondition(
        int(config["image"]["size"]),
        int(temporal["observation_horizon"]),
        int(temporal["force_history"]),
    )
    diffusion = DiT1d(
        x_dim=13,
        x_seq_len=int(temporal["action_horizon"]),
        vec_emb_dim=int(temporal["observation_horizon"]) * 64 + 128,
        seq_emb_dim=512,
        d_model=384,
        n_heads=6,
        depth=12,
        head_type="mlp",
        use_cross_attn=True,
        adaLN_on_cross_attn=True,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
    )
    optimizer = config.get("optimizer") or {}
    # New defaults match pinned upstream ForceFlow/CleanDiffuser. Historical
    # saved CoRiBench configs specify lr/wd but omit type; retain their AdamW
    # interpretation so reading an old run does not change its optimizer.
    legacy_optimizer = bool(optimizer) and "type" not in optimizer
    optimizer_type = optimizer.get("type", "adamw" if legacy_optimizer else "adam")
    diffusion_optim = {
        "optimizer": str(optimizer_type),
        "lr": float(optimizer.get("lr", 1e-4 if legacy_optimizer else 3e-4)),
    }
    weight_decay = optimizer.get("weight_decay", 1e-6 if legacy_optimizer else None)
    if weight_decay is not None:
        diffusion_optim["weight_decay"] = float(weight_decay)
    return ContinuousRectifiedFlow(
        nn_diffusion=diffusion,
        nn_condition=condition,
        optimizer_params={"diffusion": diffusion_optim},
    )
