"""Isaac Lab runtime plugin for sorting one visually identical box by weight."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import math as math_utils

from coribench.object_models import (
    COARSE_CONTACT_OFFSET_M,
    RigidObjectPhysics,
    make_rigid_object_cfg,
)
from coribench.task_api import CameraProfile, SceneEntitySpec, TaskRuntime

from . import MANIFEST
from .outcome import evaluate_report
from .spec import SPEC, evaluate_final_state, sample_initial_state


class WeightSortTask(TaskRuntime):
    manifest = MANIFEST
    camera_profile = CameraProfile(
        table_position_m=(1.09205030, 0.0, 0.50249821),
        table_target_m=(0.55, 0.0, 0.03),
    )

    def validate_assets(self) -> None:
        for filename in (SPEC.box_asset, SPEC.source_box_asset, SPEC.container_asset):
            path = self.asset_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing weight_sort asset: {path}")

    def scene_entities(self):
        entity = SceneEntitySpec(
            role="box",
            kind="rigid_object",
            config=make_rigid_object_cfg(
                "box",
                self.asset_dir / SPEC.box_asset,
                RigidObjectPhysics(
                    # Reset overwrites this with the seed-selected fixed mass.
                    mass_kg=SPEC.light_mass_kg,
                    friction=SPEC.box_friction,
                    contact_offset_m=COARSE_CONTACT_OFFSET_M,
                    linear_damping=0.15,
                    angular_damping=0.25,
                ),
                initial_position_m=(SPEC.pickup_x_m, 0.0, SPEC.box_root_z_m),
            ),
            friction=SPEC.box_friction,
        )
        entity.validate()
        return (entity,)

    def sample_initial_state(self, seed: int) -> dict[str, object]:
        return sample_initial_state(seed)

    def _spawn_container(self, name: str, center_xy, material) -> None:
        center_x, center_y = center_xy
        visual = sim_utils.UsdFileCfg(
            usd_path=str(self.asset_dir / SPEC.container_asset),
            scale=(SPEC.container_visual_scale,) * 3,
            visual_material=material,
        )
        visual.func(
            f"/World/envs/env_.*/Task_{name}_container",
            visual,
            translation=(center_x, center_y, 0.0),
        )

        collision = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
        opening_x, opening_y = SPEC.container_opening_size_m
        wall = SPEC.container_wall_thickness_m
        height = SPEC.container_wall_height_m
        parts = {
            "bottom": (
                (opening_x + 2.0 * wall, opening_y + 2.0 * wall, SPEC.container_bottom_thickness_m),
                (center_x, center_y, 0.5 * SPEC.container_bottom_thickness_m),
            ),
            "front": (
                (wall, opening_y + 2.0 * wall, height),
                (center_x - 0.5 * (opening_x + wall), center_y, 0.5 * height),
            ),
            "back": (
                (wall, opening_y + 2.0 * wall, height),
                (center_x + 0.5 * (opening_x + wall), center_y, 0.5 * height),
            ),
            "left": (
                (opening_x, wall, height),
                (center_x, center_y - 0.5 * (opening_y + wall), 0.5 * height),
            ),
            "right": (
                (opening_x, wall, height),
                (center_x, center_y + 0.5 * (opening_y + wall), 0.5 * height),
            ),
        }
        for part_name, (size, translation) in parts.items():
            part = sim_utils.CuboidCfg(size=size, collision_props=collision, visible=False)
            part.func(
                f"/World/envs/env_.*/Task_{name}_container_collision_{part_name}",
                part,
                translation=translation,
            )

    def spawn_scene_prims(self, env) -> None:
        black = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.018, 0.020, 0.022),
            roughness=0.68,
            metallic=0.03,
        )
        blue = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.025, 0.12, 0.55),
            roughness=0.48,
            metallic=0.03,
        )
        self._spawn_container("black", SPEC.black_container_center_xy_m, black)
        self._spawn_container("blue", SPEC.blue_container_center_xy_m, blue)

    def reset_scene(self, env, env_ids, layout) -> None:
        count = len(env_ids)
        box = env.entity("box")
        masses = torch.full(
            (count, 1),
            float(layout["box_mass_kg"]),
            dtype=torch.float32,
            device=env.device,
        )
        box.set_masses_index(masses=masses, env_ids=env_ids)
        zero = torch.zeros(count, device=env.device)
        yaw = torch.full((count,), float(layout["box_yaw_rad"]), device=env.device)
        env.set_entity_root_pose(
            "box",
            env_ids,
            layout["box_position_m"],
            math_utils.quat_from_euler_xyz(zero, zero, yaw),
        )

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        position, _ = env.entity_local_pose("box")
        velocity = env.entity("box").data.root_lin_vel_w.torch[env_id]
        return evaluate_report(
            evaluate_final_state(
                box_kind=str(env.layouts[env_id]["box_kind"]),
                box_position_m=position[env_id].detach().cpu().numpy(),
                box_linear_velocity_m_s=velocity.detach().cpu().numpy(),
                gripper_open_fraction=float(
                    env.gripper_open_fraction_tensor[env_id].item()
                ),
            ).to_dict()
        )

    def evaluate_final_state(self, env) -> dict[str, object]:
        return self.evaluate_final_state_for(env, 0)

    def privileged_observations(self, env):
        pose = env.entity_world_pose("box")
        return {"objects/box/pose": pose[0] if env.num_envs == 1 else pose}

    def wrench_reference_position(self, env):
        center_x = 0.5 * (
            SPEC.black_container_center_xy_m[0] + SPEC.blue_container_center_xy_m[0]
        )
        center_y = 0.5 * (
            SPEC.black_container_center_xy_m[1] + SPEC.blue_container_center_xy_m[1]
        )
        reference = torch.tensor(
            [[center_x, center_y, 0.0]],
            dtype=torch.float32,
            device=env.device,
        )
        return reference.repeat(env.num_envs, 1)

    def episode_metadata(self) -> dict[str, object]:
        return {
            "trajectory_scope": "grasp_one_box_infer_weight_sort_release",
            "asset_usd_paths": {
                "box": str(self.asset_dir / SPEC.box_asset),
                "source_visual": str(self.asset_dir / SPEC.source_box_asset),
                "black_container": str(self.asset_dir / SPEC.container_asset),
                "blue_container": str(self.asset_dir / SPEC.container_asset),
            },
            "box_visual_contract": {
                "same_usd_for_both_masses": True,
                "same_scale_for_both_masses": True,
                "box_visual_scale": SPEC.box_visual_scale,
            },
            "container_visual_contract": {
                "same_usd": True,
                "same_scale": True,
                "container_visual_scale": SPEC.container_visual_scale,
                "colors": {"light_destination": "black", "heavy_destination": "blue"},
            },
            "private_mass_semantics_kg": {
                "light": SPEC.light_mass_kg,
                "heavy": SPEC.heavy_mass_kg,
            },
            "perfect_dataset_balance": {
                "light": "even_seed",
                "heavy": "odd_seed",
            },
            "destinations": {
                "light": "black_container_f27",
                "heavy": "blue_container_f27",
            },
            "scaled_box_size_m": list(SPEC.scaled_box_size_m),
            "scaled_container_size_m": list(SPEC.scaled_container_size_m),
        }
