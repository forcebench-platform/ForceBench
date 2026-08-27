"""Isaac Lab runtime plugin for the gear-meshing demo."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import math as math_utils

from forcebench.object_models import (
    FINE_CONTACT_OFFSET_M,
    RigidObjectPhysics,
    make_rigid_object_cfg,
)
from forcebench.task_api import CameraProfile, SceneEntitySpec, TaskRuntime
from . import MANIFEST
from .outcome import evaluate_report
from .spec import (
    SPEC,
    evaluate_final_state,
    sample_initial_state,
)


class GearMeshTask(TaskRuntime):
    manifest = MANIFEST
    # User-selected main view saved by the interactive tuner on 2026-08-24.
    camera_profile = CameraProfile(
        table_position_m=(0.9582311030429754, 0.0, 0.545),
        table_target_m=(0.555, 0.0, -0.01),
        table_rotation_wxyz=(
            0.67249851,
            0.21850801,
            0.21850801,
            0.67249851,
        ),
    )
    def validate_assets(self) -> None:
        for filename in (
            SPEC.gear_base_asset,
            SPEC.medium_gear_asset,
            SPEC.small_gear_asset,
            SPEC.large_gear_asset,
        ):
            path = self.asset_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing gear_mesh asset: {path}")

    def scene_entities(self):
        entities = (
            SceneEntitySpec(
                role="target_base",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "target_base",
                    self.asset_dir / SPEC.gear_base_asset,
                    RigidObjectPhysics(
                        friction=SPEC.fixture_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                        kinematic=True,
                    ),
                    asset_scale=SPEC.asset_scale_xyz,
                ),
                friction=SPEC.fixture_friction,
            ),
            SceneEntitySpec(
                role="small_gear",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "small_gear",
                    self.asset_dir / SPEC.small_gear_asset,
                    RigidObjectPhysics(
                        mass_kg=0.019 * SPEC.mass_scale,
                        friction=SPEC.fixture_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                        linear_damping=0.1,
                        angular_damping=0.1,
                    ),
                    visual_color=(0.35, 0.40, 0.43),
                    visual_roughness=0.24,
                    visual_metallic=0.88,
                    asset_scale=SPEC.asset_scale_xyz,
                ),
                friction=SPEC.fixture_friction,
            ),
            SceneEntitySpec(
                role="large_gear",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "large_gear",
                    self.asset_dir / SPEC.large_gear_asset,
                    RigidObjectPhysics(
                        mass_kg=0.019 * SPEC.mass_scale,
                        friction=SPEC.fixture_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                        linear_damping=0.1,
                        angular_damping=0.1,
                    ),
                    visual_color=(0.29, 0.33, 0.35),
                    visual_roughness=0.28,
                    visual_metallic=0.86,
                    asset_scale=SPEC.asset_scale_xyz,
                ),
                friction=SPEC.fixture_friction,
            ),
            SceneEntitySpec(
                role="medium_gear",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "medium_gear",
                    self.asset_dir / SPEC.medium_gear_asset,
                    RigidObjectPhysics(
                        mass_kg=0.012 * SPEC.mass_scale,
                        friction=SPEC.held_gear_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                    ),
                    # Warm bronze PBR finish echoes the cell's safety accents
                    # while remaining visibly distinct from the steel gears.
                    visual_color=(0.46, 0.25, 0.065),
                    visual_roughness=0.27,
                    visual_metallic=0.82,
                    asset_scale=SPEC.asset_scale_xyz,
                ),
                friction=SPEC.held_gear_friction,
            ),
        )
        for entity in entities:
            entity.validate()
        return entities

    def sample_initial_state(self, seed: int) -> dict[str, object]:
        return sample_initial_state(seed)

    def spawn_scene_prims(self, env) -> None:
        width, depth, height = SPEC.pickup_support_size_m
        support = sim_utils.CuboidCfg(
            size=(width, depth, height),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00025, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=SPEC.fixture_friction,
                dynamic_friction=SPEC.fixture_friction,
                restitution=0.0,
                friction_combine_mode=SPEC.friction_combine_mode,
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.14, 0.17),
                roughness=0.36,
                metallic=0.72,
            ),
        )
        support.func(
            "/World/envs/env_.*/Task_pickup_support",
            support,
            translation=(SPEC.pickup_center_x_m, SPEC.pickup_center_y_m, 0.5 * height),
        )

        # The three texture-free white shaft covers and the center marker are
        # authored directly in normalized/gear_mesh/base.usda.  Keeping them in
        # the referenced rigid-body asset makes PhysX/Fabric move the plate and
        # covers as one object; runtime-spawned child visuals can otherwise
        # remain at stale renderer transforms after the kinematic base moves.

    def reset_scene(self, env, env_ids, layout) -> None:
        if float(layout["initial_separation_m"]) < SPEC.minimum_initial_separation_m:
            raise RuntimeError("target gear center overlaps the pickup fixture")
        count = len(env_ids)
        zero = torch.zeros(count, device=env.device)
        target_yaw = torch.full((count,), float(layout["target_base_yaw_rad"]), device=env.device)
        gear_yaw = torch.full((count,), float(layout["medium_gear_yaw_rad"]), device=env.device)
        target_quaternion = math_utils.quat_from_euler_xyz(zero, zero, target_yaw)
        env.set_entity_root_pose(
            "target_base", env_ids, layout["target_base_position_m"], target_quaternion
        )
        env.set_entity_root_pose(
            "small_gear", env_ids, layout["target_small_gear_position_m"], target_quaternion
        )
        env.set_entity_root_pose(
            "large_gear", env_ids, layout["target_large_gear_position_m"], target_quaternion
        )
        env.set_entity_root_pose(
            "medium_gear",
            env_ids,
            layout["medium_gear_root_position_m"],
            math_utils.quat_from_euler_xyz(zero, zero, gear_yaw),
        )

    @staticmethod
    def _offset_point(env, role: str, offset_m):
        position, quaternion = env.entity_local_pose(role)
        offset = torch.tensor([offset_m], dtype=torch.float32, device=env.device).repeat(len(position), 1)
        return position + math_utils.quat_apply(quaternion, offset)

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        gear_center, gear_quaternion = env.entity_local_pose("medium_gear")
        target_center = self._offset_point(env, "target_base", SPEC.medium_gear_target_offset_m)
        local_z = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
        gear_axis = math_utils.quat_apply(gear_quaternion[env_id : env_id + 1], local_z)[0]
        return evaluate_report(
            evaluate_final_state(
                gear_center_position_m=gear_center[env_id].detach().cpu().numpy(),
                gear_axis_world=gear_axis.detach().cpu().numpy(),
                target_shaft_position_m=target_center[env_id].detach().cpu().numpy(),
                gripper_open_fraction=float(env.gripper_open_fraction_tensor[env_id].item()),
                eef_position_m=env.fingertip_midpoint_pos[env_id].detach().cpu().numpy(),
            ).to_dict()
        )

    def evaluate_final_state(self, env) -> dict[str, object]:
        return self.evaluate_final_state_for(env, 0)

    def privileged_observations(self, env):
        return {
            f"objects/{role}/pose": env.entity_world_pose(role)[0]
            for role in ("target_base", "small_gear", "large_gear", "medium_gear")
        }

    def wrench_reference_position(self, env):
        return self._offset_point(env, "target_base", SPEC.medium_gear_target_offset_m)

    def episode_metadata(self) -> dict[str, object]:
        return {
            "trajectory_scope": "teleoperated_pick_transport_align_mesh_release_retract",
            "source_demo": "Isaac-Forge-GearMesh-Direct-v0",
            "wrist_image_rotation_deg_clockwise": self.camera_profile.wrist_image_rotation_deg,
            "asset_usd_paths": {
                "target_base": str(self.asset_dir / SPEC.gear_base_asset),
                "medium_gear": str(self.asset_dir / SPEC.medium_gear_asset),
                "small_gear": str(self.asset_dir / SPEC.small_gear_asset),
                "large_gear": str(self.asset_dir / SPEC.large_gear_asset),
            },
            "target_medium_gear_offset_m": list(SPEC.medium_gear_target_offset_m),
            "asset_scale_xyz": list(SPEC.asset_scale_xyz),
            "visual_style": (
                "pure_black_base_with_three_opaque_white_shafts_amber_locator_"
                "machined_steel_and_bronze_gears"
            ),
            "contact_profile": {
                "held_gear_friction": SPEC.held_gear_friction,
                "fixture_friction": SPEC.fixture_friction,
                "friction_combine_mode": SPEC.friction_combine_mode,
                "contact_offset_m": 0.00025,
                "grasp_gripper_open_fraction": SPEC.grasp_gripper_open_fraction,
                "flanking_gears_dynamic_on_shafts": True,
            },
            "expert_generation": {
                "pipeline": "privileged_pose_closed_loop_place_release_retract",
                "object_state_overwrite": False,
                "robot_state_overwrite": False,
            },
        }


    def debug_state(self, env):
        output = {}
        for role in ("target_base", "small_gear", "large_gear", "medium_gear"):
            position, quaternion = env.entity_local_pose(role)
            output[role] = {
                "position_m": position[0].detach().cpu().tolist(),
                "quaternion": quaternion[0].detach().cpu().tolist(),
            }
        return output
