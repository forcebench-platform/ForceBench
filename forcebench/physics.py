"""Small simulator helpers owned by ForceBench."""

from __future__ import annotations

import torch
import warp as wp


def set_asset_friction(asset, value: float, num_envs: int) -> None:
    """Set static and dynamic friction through the generic PhysX tensor API."""
    if value < 0:
        raise ValueError("friction must be non-negative")
    materials = wp.to_torch(asset.root_view.get_material_properties())
    materials[..., 0] = float(value)
    materials[..., 1] = float(value)
    env_ids = torch.arange(num_envs, dtype=torch.int32, device="cpu")
    asset.root_view.set_material_properties(wp.from_torch(materials.contiguous()), wp.from_torch(env_ids))


def add_diagonal_body_inertia(asset, value: float, num_envs: int) -> None:
    """Add a small diagonal inertia term to reduce high-frequency robot chatter."""
    if value < 0:
        raise ValueError("inertia increment must be non-negative")
    inertias = wp.to_torch(asset.root_view.get_inertias())
    inertias[:, :, [0, 4, 8]] += float(value)
    env_ids = torch.arange(num_envs, dtype=torch.int32, device="cpu")
    asset.root_view.set_inertias(wp.from_torch(inertias.contiguous()), wp.from_torch(env_ids))
