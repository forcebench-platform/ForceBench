"""Isaac Lab runtime plugin for the M16 nut-threading demo."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import math as math_utils

from forcebench.object_models import (
    FINE_CONTACT_OFFSET_M,
    RigidObjectPhysics,
    make_rigid_object_cfg,
)
from forcebench.task_api import CameraProfile, SceneEntitySpec, TaskRuntime
from forcebench.nut_assets import validate_nut_assets
from . import MANIFEST
from .outcome import evaluate_report
from .spec import (
    SPEC,
    evaluate_final_state,
    sample_initial_state,
)


# Low-saturation PBR finishes keep the task roles visually distinct while
# matching the surrounding metalworking cell. These values are appearance-only
# and do not participate in layout sampling, collision, or task evaluation.
_MACHINED_STAINLESS = (0.48, 0.51, 0.52)
_BLACK_OXIDE_STEEL = (0.17, 0.19, 0.20)
_BLUE_STEEL_FIXTURE = (0.10, 0.18, 0.20)


class NutThreadTask(TaskRuntime):
    manifest = MANIFEST
    camera_profile = CameraProfile(
        table_position_m=(1.05856784, 0.0, 0.53963208),
        table_target_m=(0.52384004, 0.0, 0.05748461),
        table_focal_length_mm=40.0,
    )

    def validate_assets(self) -> None:
        validate_nut_assets(self.asset_dir)

    def scene_entities(self):
        entities = (
            SceneEntitySpec(
                role="target_bolt",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "target_bolt",
                    self.asset_dir / SPEC.bolt_asset,
                    RigidObjectPhysics(
                        friction=SPEC.bolt_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                        kinematic=True,
                    ),
                    visual_color=_BLACK_OXIDE_STEEL,
                    visual_roughness=0.27,
                    visual_metallic=0.82,
                ),
                friction=SPEC.bolt_friction,
            ),
            SceneEntitySpec(
                role="nut",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "nut",
                    self.asset_dir / SPEC.nut_asset,
                    RigidObjectPhysics(
                        mass_kg=0.03,
                        friction=SPEC.nut_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                    ),
                    visual_color=_MACHINED_STAINLESS,
                    visual_roughness=0.22,
                    visual_metallic=0.88,
                ),
                friction=SPEC.nut_friction,
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
                static_friction=SPEC.bolt_friction,
                dynamic_friction=SPEC.bolt_friction,
                restitution=0.0,
                friction_combine_mode=SPEC.friction_combine_mode,
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=_BLUE_STEEL_FIXTURE,
                roughness=0.34,
                metallic=0.68,
            ),
        )
        support.func(
            "/World/envs/env_.*/Task_pickup_support",
            support,
            translation=(SPEC.pickup_center_x_m, SPEC.pickup_center_y_m, 0.5 * height),
        )

    def reset_scene(self, env, env_ids, layout) -> None:
        if float(layout["initial_separation_m"]) < SPEC.minimum_initial_separation_m:
            raise RuntimeError("target bolt overlaps the pickup fixture")
        count = len(env_ids)
        zero = torch.zeros(count, device=env.device)
        bolt_yaw = torch.full((count,), float(layout["target_bolt_yaw_rad"]), device=env.device)
        nut_yaw = torch.full((count,), float(layout["pickup_nut_yaw_rad"]), device=env.device)
        env.set_entity_root_pose(
            "target_bolt",
            env_ids,
            layout["target_bolt_position_m"],
            math_utils.quat_from_euler_xyz(zero, zero, bolt_yaw),
        )
        env.set_entity_root_pose(
            "nut",
            env_ids,
            layout["pickup_nut_root_position_m"],
            math_utils.quat_from_euler_xyz(zero, zero, nut_yaw),
        )

    @staticmethod
    def _offset_point(env, role: str, z_offset_m: float):
        position, quaternion = env.entity_local_pose(role)
        offset = torch.tensor([[0.0, 0.0, z_offset_m]], dtype=torch.float32, device=env.device).repeat(
            len(position), 1
        )
        return position + math_utils.quat_apply(quaternion, offset)

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        nut_base, nut_quaternion = env.entity_local_pose("nut")
        target_base = self._offset_point(env, "target_bolt", SPEC.target_nut_base_offset_m)
        local_z = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
        nut_axis = math_utils.quat_apply(nut_quaternion[env_id : env_id + 1], local_z)[0]
        return evaluate_report(
            evaluate_final_state(
                nut_base_position_m=nut_base[env_id].detach().cpu().numpy(),
                nut_axis_world=nut_axis.detach().cpu().numpy(),
                target_nut_base_position_m=target_base[env_id].detach().cpu().numpy(),
                gripper_open_fraction=float(env.gripper_open_fraction_tensor[env_id].item()),
                eef_position_m=env.fingertip_midpoint_pos[env_id].detach().cpu().numpy(),
            ).to_dict()
        )

    def evaluate_final_state(self, env) -> dict[str, object]:
        return self.evaluate_final_state_for(env, 0)

    def privileged_observations(self, env):
        return {
            f"objects/{role}/pose": env.entity_world_pose(role)[0]
            for role in ("target_bolt", "nut")
        }

    def wrench_reference_position(self, env):
        return self._offset_point(env, "target_bolt", SPEC.target_nut_base_offset_m)

    def episode_metadata(self) -> dict[str, object]:
        return {
            "trajectory_scope": "teleoperated_pick_transport_engage_thread_release_retract",
            "source_demo": "Isaac-Forge-NutThread-Direct-v0",
            "wrist_image_rotation_deg_clockwise": self.camera_profile.wrist_image_rotation_deg,
            "asset_usd_paths": {
                "nut": str(self.asset_dir / SPEC.nut_asset),
                "target_bolt": str(self.asset_dir / SPEC.bolt_asset),
            },
            "thread_geometry": {
                "pitch_m": SPEC.thread_pitch_m,
                "target_nut_base_offset_m": SPEC.target_nut_base_offset_m,
            },
            "contact_profile": {
                "nut_friction": SPEC.nut_friction,
                "bolt_friction": SPEC.bolt_friction,
                "friction_combine_mode": SPEC.friction_combine_mode,
                "contact_offset_m": 0.00025,
            },
            "expert_generation": {
                "pipeline": "privileged_pose_closed_loop_place_release_retract",
                "object_state_overwrite": False,
                "robot_state_overwrite": False,
            },
        }


    def debug_state(self, env):
        output = {}
        for role in ("target_bolt", "nut"):
            position, quaternion = env.entity_local_pose(role)
            output[role] = {
                "position_m": position[0].detach().cpu().tolist(),
                "quaternion": quaternion[0].detach().cpu().tolist(),
            }
        return output
