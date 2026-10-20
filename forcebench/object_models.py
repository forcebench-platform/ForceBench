"""Shared physical model for task objects.

Task assets may use different collision representations when their geometry
requires it, but all single-body objects enter the simulator through this
contract.  Only mechanisms with a real degree of freedom should use an
``ArticulationCfg`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


OBJECT_MODEL_SCHEMA_VERSION = "1.0"
FRICTION_COMBINE_MODE = "min"
RESTITUTION_COMBINE_MODE = "min"
FINE_CONTACT_OFFSET_M = 0.00025
STANDARD_CONTACT_OFFSET_M = 0.0005
COARSE_CONTACT_OFFSET_M = 0.001


@dataclass(frozen=True)
class RigidObjectPhysics:
    """Runtime physical parameters for one single-body task object."""

    friction: float
    contact_offset_m: float
    mass_kg: float | None = None
    kinematic: bool = False
    linear_damping: float = 0.5
    angular_damping: float = 1.0

    def validate(self) -> None:
        if self.friction < 0.0:
            raise ValueError("object friction must be non-negative")
        if self.contact_offset_m <= 0.0:
            raise ValueError("object contact offset must be positive")
        if self.mass_kg is not None and self.mass_kg <= 0.0:
            raise ValueError("dynamic object mass must be positive")
        if not self.kinematic and self.mass_kg is None:
            raise ValueError("dynamic rigid objects require mass_kg")
        if self.linear_damping < 0.0 or self.angular_damping < 0.0:
            raise ValueError("object damping must be non-negative")


def make_rigid_object_cfg(
    role: str,
    asset_path: Path,
    physics: RigidObjectPhysics,
    *,
    initial_position_m: tuple[float, float, float] = (0.50, 0.0, 0.0),
    visual_color: tuple[float, float, float] | None = None,
    visual_roughness: float = 0.55,
    visual_metallic: float = 0.10,
    asset_scale: tuple[float, float, float] | None = None,
) -> RigidObjectCfg:
    """Build the benchmark-standard config for a single rigid body."""
    physics.validate()
    if not role or not role.replace("_", "").isalnum():
        raise ValueError(f"invalid object role: {role!r}")
    if not 0.0 <= visual_roughness <= 1.0:
        raise ValueError("visual roughness must be in [0, 1]")
    if not 0.0 <= visual_metallic <= 1.0:
        raise ValueError("visual metallic must be in [0, 1]")
    if asset_scale is not None and (
        len(asset_scale) != 3 or any(float(value) <= 0.0 for value in asset_scale)
    ):
        raise ValueError("asset scale must contain three positive values")
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Task_{role}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_path),
            scale=asset_scale,
            activate_contact_sensors=True,
            visual_material=(
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=visual_color,
                    roughness=visual_roughness,
                    metallic=visual_metallic,
                )
                if visual_color is not None
                else None
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=physics.friction,
                dynamic_friction=physics.friction,
                restitution=0.0,
                friction_combine_mode=FRICTION_COMBINE_MODE,
                restitution_combine_mode=RESTITUTION_COMBINE_MODE,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=physics.kinematic,
                disable_gravity=physics.kinematic,
                max_depenetration_velocity=1.0,
                linear_damping=physics.linear_damping,
                angular_damping=physics.angular_damping,
                max_linear_velocity=10.0,
                max_angular_velocity=100.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=2,
                max_contact_impulse=1.0e4,
            ),
            mass_props=(
                sim_utils.MassPropertiesCfg(mass=physics.mass_kg)
                if physics.mass_kg is not None
                else None
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=physics.contact_offset_m,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=initial_position_m,
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )
