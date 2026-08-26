"""Isaac Lab runtime plugin for the peg insertion demo."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import math as math_utils

from coribench.object_models import (
    FINE_CONTACT_OFFSET_M,
    RigidObjectPhysics,
    make_rigid_object_cfg,
)
from coribench.task_api import CameraProfile, SceneEntitySpec, TaskRuntime
from . import MANIFEST
from .outcome import evaluate_report
from .spec import (
    SPEC,
    evaluate_final_state,
    sample_initial_state,
)


# Industrial PBR palette: stainless workpiece, blue anodized pickup fixture,
# and bronze target fixture. The role colors are deliberately distinct enough
# to be named unambiguously in the language prompt. These values affect
# rendering only.
_MACHINED_STAINLESS = (0.50, 0.53, 0.54)
_BRONZE_FIXTURE = (0.52, 0.29, 0.09)
_BLUE_ANODIZED_FIXTURE = (0.05, 0.24, 0.52)


class PegInsertTask(TaskRuntime):
    manifest = MANIFEST
    camera_profile = CameraProfile(
        table_position_m=(1.05856784, 0.0, 0.53963208),
        table_target_m=(0.52384004, 0.0, 0.05748461),
        table_focal_length_mm=40.0,
    )

    def validate_assets(self) -> None:
        for filename in (SPEC.peg_asset, SPEC.hole_asset):
            path = self.asset_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing peg_insert asset: {path}")

    def scene_entities(self):
        hole = self.asset_dir / SPEC.hole_asset
        peg = self.asset_dir / SPEC.peg_asset
        entities = (
            SceneEntitySpec(
                role="target_hole",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "target_hole",
                    hole,
                    RigidObjectPhysics(
                        friction=SPEC.hole_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                        kinematic=True,
                    ),
                    visual_color=_BRONZE_FIXTURE,
                    visual_roughness=0.28,
                    visual_metallic=0.82,
                ),
                friction=SPEC.hole_friction,
            ),
            SceneEntitySpec(
                role="pickup_hole",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "pickup_hole",
                    hole,
                    RigidObjectPhysics(
                        friction=SPEC.hole_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                        kinematic=True,
                    ),
                    visual_color=_BLUE_ANODIZED_FIXTURE,
                    visual_roughness=0.30,
                    visual_metallic=0.74,
                ),
                friction=SPEC.hole_friction,
            ),
            SceneEntitySpec(
                role="peg",
                kind="rigid_object",
                config=make_rigid_object_cfg(
                    "peg",
                    peg,
                    RigidObjectPhysics(
                        mass_kg=0.019,
                        friction=SPEC.peg_friction,
                        contact_offset_m=FINE_CONTACT_OFFSET_M,
                        linear_damping=0.5,
                        angular_damping=5.0,
                    ),
                    visual_color=_MACHINED_STAINLESS,
                    visual_roughness=0.20,
                    visual_metallic=0.90,
                ),
                friction=SPEC.peg_friction,
            ),
        )
        for entity in entities:
            entity.validate()
        return entities

    def sample_initial_state(self, seed: int) -> dict[str, object]:
        return sample_initial_state(seed)

    def spawn_scene_prims(self, env) -> None:
        """Add a shallow fixed floor inside the pickup socket.

        The target-sized socket keeps the peg upright, while this hidden floor
        prevents the peg from settling to the full insertion depth before a
        human has a chance to grasp it.
        """
        support_bottom_z = SPEC.hole_root_z_m
        support_top_z = SPEC.upright_peg_root_z_m
        support_height = support_top_z - support_bottom_z
        support = sim_utils.CuboidCfg(
            size=(SPEC.pickup_support_width_m, SPEC.pickup_support_width_m, support_height),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=FINE_CONTACT_OFFSET_M,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=SPEC.hole_friction,
                dynamic_friction=SPEC.hole_friction,
                restitution=0.0,
                friction_combine_mode=SPEC.friction_combine_mode,
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=_BLUE_ANODIZED_FIXTURE,
                roughness=0.30,
                metallic=0.74,
            ),
        )
        support.func(
            "/World/envs/env_.*/Task_pickup_support",
            support,
            translation=(
                SPEC.pickup_center_x_m,
                SPEC.pickup_center_y_m,
                support_bottom_z + 0.5 * support_height,
            ),
        )

    def reset_scene(self, env, env_ids, layout) -> None:
        separation = float(layout["initial_separation_m"])
        if separation < SPEC.minimum_initial_separation_m:
            raise RuntimeError("target hole overlaps the fixed pickup socket")
        count = len(env_ids)
        zero = torch.zeros(count, device=env.device)
        target_yaw = torch.full((count,), float(layout["target_hole_yaw_rad"]), device=env.device)
        peg_yaw = torch.full((count,), float(layout["peg_upright_yaw_rad"]), device=env.device)
        env.set_entity_root_pose(
            "target_hole",
            env_ids,
            layout["target_hole_position_m"],
            math_utils.quat_from_euler_xyz(zero, zero, target_yaw),
        )
        env.set_entity_root_pose(
            "pickup_hole",
            env_ids,
            layout["pickup_hole_position_m"],
            math_utils.quat_from_euler_xyz(zero, zero, zero),
        )
        env.set_entity_root_pose(
            "peg",
            env_ids,
            layout["peg_position_m"],
            math_utils.quat_from_euler_xyz(zero, zero, peg_yaw),
        )

    @staticmethod
    def _target_mate_position(env):
        position, quaternion = env.entity_local_pose("target_hole")
        offset = torch.tensor(
            [[0.0, 0.0, SPEC.hole_mate_height_m]], dtype=torch.float32, device=env.device
        ).repeat(len(position), 1)
        return position + math_utils.quat_apply(quaternion, offset)

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        peg_position, peg_quaternion = env.entity_local_pose("peg")
        target_position = self._target_mate_position(env)
        local_z = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
        peg_axis = math_utils.quat_apply(
            peg_quaternion[env_id : env_id + 1], local_z
        )[0]
        return evaluate_report(
            evaluate_final_state(
                peg_position_m=peg_position[env_id].detach().cpu().numpy(),
                peg_axis_world=peg_axis.detach().cpu().numpy(),
                target_hole_position_m=target_position[env_id].detach().cpu().numpy(),
                gripper_open_fraction=float(
                    env.gripper_open_fraction_tensor[env_id].item()
                ),
                eef_position_m=env.fingertip_midpoint_pos[env_id]
                .detach()
                .cpu()
                .numpy(),
            ).to_dict()
        )

    def evaluate_final_state(self, env) -> dict[str, object]:
        return self.evaluate_final_state_for(env, 0)

    def privileged_observations(self, env):
        poses = {
            f"objects/{role}/pose": env.entity_world_pose(role)
            for role in ("target_hole", "pickup_hole", "peg")
        }
        # Preserve the legacy single-environment vector contract while exposing
        # one row per cloned workspace to the production 32-environment writer.
        if env.num_envs == 1:
            return {key: value[0] for key, value in poses.items()}
        return poses

    def wrench_reference_position(self, env):
        return self._target_mate_position(env)

    def episode_metadata(self) -> dict[str, object]:
        return {
            "trajectory_scope": "teleoperated_approach_vertical_grasp_lift_transport_align_insert_release_retract",
            "start_tcp_position_m": list(SPEC.start_tcp_position_m),
            "wrist_image_rotation_deg_clockwise": self.camera_profile.wrist_image_rotation_deg,
            "asset_usd_paths": {
                "peg": str(self.asset_dir / SPEC.peg_asset),
                "target_hole": str(self.asset_dir / SPEC.hole_asset),
                "pickup_hole": str(self.asset_dir / SPEC.hole_asset),
            },
            "fixture_color_semantics": {
                "target_hole": "bronze",
                "pickup_hole": "blue_anodized",
            },
            "pickup_support": {
                "width_m": SPEC.pickup_support_width_m,
                "top_z_m": SPEC.upright_peg_root_z_m,
            },
            "contact_profile": {
                "peg_friction": SPEC.peg_friction,
                "gripper_friction": SPEC.gripper_friction,
                "hole_friction": SPEC.hole_friction,
                "effective_gripper_peg_friction": min(SPEC.gripper_friction, SPEC.peg_friction),
                "effective_peg_hole_friction": min(SPEC.peg_friction, SPEC.hole_friction),
                "friction_combine_mode": SPEC.friction_combine_mode,
                "peg_collision_approximation": "convexHull",
                "peg_contact_offset_m": 0.00025,
                "gripper_contact_offset_m": 0.00025,
                "object_solver_velocity_iterations": 2,
            },
            "expert_generation": {
                "pipeline": "privileged_pose_closed_loop_insert_release_retract",
                "object_state_overwrite": False,
                "robot_state_overwrite": False,
            },
        }


    def debug_state(self, env):
        output = {}
        for role in ("target_hole", "pickup_hole", "peg"):
            position, quaternion = env.entity_local_pose(role)
            output[role] = {
                "position_m": position[0].detach().cpu().tolist(),
                "quaternion": quaternion[0].detach().cpu().tolist(),
            }
        return output
