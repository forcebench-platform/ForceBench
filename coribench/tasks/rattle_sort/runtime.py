"""Isaac Lab runtime for static sorting by front/back center-of-mass offset."""

from __future__ import annotations

import math

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


class RattleSortTask(TaskRuntime):
    manifest = MANIFEST
    camera_profile = CameraProfile(
        table_position_m=(0.95507403, 0.0, 0.595),
        table_target_m=(0.495, 0.0, -0.05),
        table_rotation_wxyz=(
            0.67344552,
            0.21557161,
            0.21557161,
            0.67344552,
        ),
    )

    def __init__(self, asset_dir):
        super().__init__(asset_dir)
        self._inspection_hold_s = 0.0
        self._inspection_max_hold_s = 0.0
        self._inspection_completed = False
        self._inspection_diagnostics: dict[str, float | bool] = {}
        self._inspection_hold_s_batch: list[float] = []
        self._inspection_max_hold_s_batch: list[float] = []
        self._inspection_completed_batch: list[bool] = []
        self._inspection_diagnostics_batch: list[dict[str, float | bool]] = []
        self._default_box_com = None

    def _ensure_batch_state(self, num_envs: int) -> None:
        if len(self._inspection_hold_s_batch) == num_envs:
            return
        self._inspection_hold_s_batch = [0.0 for _ in range(num_envs)]
        self._inspection_max_hold_s_batch = [0.0 for _ in range(num_envs)]
        self._inspection_completed_batch = [False for _ in range(num_envs)]
        self._inspection_diagnostics_batch = [{} for _ in range(num_envs)]

    def _sync_legacy_state(self) -> None:
        if not self._inspection_hold_s_batch:
            return
        self._inspection_hold_s = self._inspection_hold_s_batch[0]
        self._inspection_max_hold_s = self._inspection_max_hold_s_batch[0]
        self._inspection_completed = self._inspection_completed_batch[0]
        self._inspection_diagnostics = dict(self._inspection_diagnostics_batch[0])

    def validate_assets(self) -> None:
        for filename in (
            SPEC.box_asset,
            SPEC.source_box_asset,
            SPEC.container_asset,
            SPEC.conveyor_asset,
            *SPEC.background_parcel_assets,
        ):
            path = self.asset_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing rattle_sort asset: {path}")

    def scene_entities(self):
        box = SceneEntitySpec(
            role="box",
            kind="rigid_object",
            config=make_rigid_object_cfg(
                "box",
                self.asset_dir / SPEC.box_asset,
                RigidObjectPhysics(
                    mass_kg=SPEC.box_total_mass_kg,
                    friction=SPEC.box_friction,
                    contact_offset_m=COARSE_CONTACT_OFFSET_M,
                    linear_damping=0.10,
                    angular_damping=SPEC.box_angular_damping,
                ),
                initial_position_m=(SPEC.pickup_x_m, 0.0, SPEC.box_root_z_m),
            ),
            friction=SPEC.box_friction,
        )
        box.validate()
        return (box,)

    def sample_initial_state(self, seed: int) -> dict[str, object]:
        return sample_initial_state(seed)

    def _spawn_black_basket(self) -> None:
        center_x, center_y = SPEC.black_basket_center_xy_m
        black = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.008, 0.010, 0.012),
            roughness=0.74,
            metallic=0.01,
        )
        visual = sim_utils.UsdFileCfg(
            usd_path=str(self.asset_dir / SPEC.container_asset),
            scale=(SPEC.black_basket_visual_scale,) * 3,
            visual_material=black,
        )
        visual.func(
            "/World/envs/env_.*/Task_black_basket",
            visual,
            translation=(center_x, center_y, 0.0),
        )

        collision = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
        opening_x, opening_y = SPEC.black_basket_opening_size_m
        wall = SPEC.black_basket_wall_thickness_m
        height = SPEC.black_basket_wall_height_m
        bottom = SPEC.black_basket_collision_floor_thickness_m
        parts = {
            "bottom": (
                (opening_x + 2.0 * wall, opening_y + 2.0 * wall, bottom),
                (
                    center_x,
                    center_y,
                    SPEC.black_basket_collision_floor_center_z_m,
                ),
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
                f"/World/envs/env_.*/Task_black_basket_collision_{part_name}",
                part,
                translation=translation,
            )

    def _spawn_conveyor(self, env) -> None:
        yaw = SPEC.conveyor_yaw_rad
        visual = sim_utils.UsdFileCfg(
            usd_path=str(self.asset_dir / SPEC.conveyor_asset),
            scale=SPEC.conveyor_visual_scale_xyz,
        )
        # The A04 root is authored at one end of the two-meter track. Keep that
        # near end beside the table and let the full track continue to the right.
        visual_root_x, visual_root_y = SPEC.conveyor_visual_root_xy_m
        visual_root = (
            "/World/CoRiBenchVisuals"
            if env.num_envs == 1
            else "/World/envs/env_.*/CoRiBenchVisuals"
        )
        visual.func(
            f"{visual_root}/official_conveyor_a04",
            visual,
            translation=(visual_root_x, visual_root_y, SPEC.conveyor_visual_root_z_m),
            orientation=(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)),
        )

        # The official asset supplies appearance only: its source physics were
        # deliberately stripped during normalization. This task-owned cuboid
        # covers the complete belt surface with one deterministic static plane.
        collision = sim_utils.CuboidCfg(
            size=SPEC.conveyor_collision_size_m,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            visible=False,
        )
        collision_center_x, collision_center_y = SPEC.conveyor_collision_center_xy_m
        collision.func(
            "/World/envs/env_.*/Task_conveyor_collision",
            collision,
            translation=(
                collision_center_x,
                collision_center_y,
                SPEC.conveyor_collision_center_z_m,
            ),
        )

    def _spawn_background_parcels(self, env) -> None:
        # Four distinct SimReady Shipping models, placed on the actual top of
        # the official rack shelf (world z=0 after the rack transform).
        placements = (
            (-0.82, -0.82, 0.005, 0.0, 2.00),
            (-0.86, -0.30, 0.005, 0.10, 1.35),
            (-0.76, 0.10, 0.005, 0.0, 1.00),
            (-0.83, 0.80, 0.005, -0.08, 0.68),
        )
        for index, (asset, placement) in enumerate(
            zip(SPEC.background_parcel_assets, placements, strict=True)
        ):
            x, y, z, yaw, scale = placement
            source = sim_utils.UsdFileCfg(
                usd_path=str(self.asset_dir / asset),
                scale=(scale,) * 3,
            )
            visual_root = (
                "/World/CoRiBenchVisuals"
                if env.num_envs == 1
                else "/World/envs/env_.*/CoRiBenchVisuals"
            )
            source.func(
                f"{visual_root}/background_parcel_{index}",
                source,
                translation=(x, y, z),
                orientation=(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)),
            )

    def spawn_scene_prims(self, env) -> None:
        self._spawn_black_basket()
        self._spawn_conveyor(env)
        self._spawn_background_parcels(env)

    def after_scene_setup(self, env) -> None:
        self._ensure_batch_state(env.num_envs)

    def reset_scene(self, env, env_ids, layout) -> None:
        self._ensure_batch_state(env.num_envs)
        count = len(env_ids)
        box = env.entity("box")
        masses = torch.full(
            (count, 1),
            float(layout["box_mass_kg"]),
            dtype=torch.float32,
            device=env.device,
        )
        box.set_masses_index(masses=masses, env_ids=env_ids)

        if self._default_box_com is None:
            self._default_box_com = box.data.body_com_pose_b.torch.clone()
        body_ids = torch.arange(box.num_bodies, dtype=torch.int, device=env.device)
        coms = self._default_box_com[env_ids[:, None], body_ids].clone()
        offset = torch.as_tensor(
            layout["center_of_mass_offset_local_m"],
            dtype=torch.float32,
            device=env.device,
        )
        coms[..., :3] += offset.view(1, 1, 3)
        box.set_coms_index(coms=coms, body_ids=body_ids, env_ids=env_ids)

        zero = torch.zeros(count, device=env.device)
        yaw = torch.full((count,), float(layout["box_yaw_rad"]), device=env.device)
        quaternion = math_utils.quat_from_euler_xyz(zero, zero, yaw)
        env.set_entity_root_pose("box", env_ids, layout["box_position_m"], quaternion)
        for env_index in env_ids.detach().cpu().tolist():
            self._inspection_hold_s_batch[env_index] = 0.0
            self._inspection_max_hold_s_batch[env_index] = 0.0
            self._inspection_completed_batch[env_index] = False
            self._inspection_diagnostics_batch[env_index] = {}
        self._sync_legacy_state()

    def on_control_step(self, env) -> None:
        self._ensure_batch_state(env.num_envs)
        position, _ = env.entity_local_pose("box")
        box = env.entity("box")
        linear_speeds = torch.linalg.vector_norm(box.data.root_lin_vel_w.torch, dim=1)
        angular_speeds = torch.linalg.vector_norm(box.data.root_ang_vel_w.torch, dim=1)
        tcp_box_distances = torch.linalg.vector_norm(
            position - env.fingertip_midpoint_pos, dim=1
        )
        grippers = env.gripper_open_fraction_tensor
        for env_index in range(env.num_envs):
            box_height_m = float(position[env_index, 2].item())
            linear_speed = float(linear_speeds[env_index].item())
            angular_speed = float(angular_speeds[env_index].item())
            tcp_box_distance_m = float(tcp_box_distances[env_index].item())
            gripper_open_fraction = float(grippers[env_index].item())
            held = bool(
                box_height_m >= SPEC.inspection_min_lift_height_m
                and gripper_open_fraction
                <= SPEC.inspection_max_gripper_open_fraction
                and tcp_box_distance_m <= SPEC.inspection_max_tcp_box_distance_m
            )
            stationary = bool(
                linear_speed <= SPEC.inspection_max_box_linear_speed_m_s
                and angular_speed <= SPEC.inspection_max_box_angular_speed_rad_s
            )
            if not self._inspection_completed_batch[env_index]:
                if held and stationary:
                    self._inspection_hold_s_batch[env_index] += float(env.step_dt)
                    self._inspection_max_hold_s_batch[env_index] = max(
                        self._inspection_max_hold_s_batch[env_index],
                        self._inspection_hold_s_batch[env_index],
                    )
                    self._inspection_completed_batch[env_index] = bool(
                        self._inspection_hold_s_batch[env_index] + 1.0e-6
                        >= SPEC.inspection_required_hold_s
                    )
                else:
                    self._inspection_hold_s_batch[env_index] = 0.0
            self._inspection_diagnostics_batch[env_index] = {
                "held": held,
                "stationary": stationary,
                "box_height_m": box_height_m,
                "tcp_box_distance_m": tcp_box_distance_m,
                "box_linear_speed_m_s": linear_speed,
                "box_angular_speed_rad_s": angular_speed,
                "gripper_open_fraction": gripper_open_fraction,
                "required_hold_s": SPEC.inspection_required_hold_s,
            }
        self._sync_legacy_state()

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        self._ensure_batch_state(env.num_envs)
        box_position, _ = env.entity_local_pose("box")
        velocity = env.entity("box").data.root_lin_vel_w.torch[env_id]
        return evaluate_report(
            evaluate_final_state(
                box_kind=str(env.layouts[env_id]["box_kind"]),
                box_position_m=box_position[env_id].detach().cpu().numpy(),
                box_linear_velocity_m_s=velocity.detach().cpu().numpy(),
                gripper_open_fraction=float(
                    env.gripper_open_fraction_tensor[env_id].item()
                ),
            ).to_dict()
        )

    def evaluate_final_state(self, env) -> dict[str, object]:
        return self.evaluate_final_state_for(env, 0)

    def privileged_observations(self, env):
        self._ensure_batch_state(env.num_envs)
        inspection_state = torch.tensor(
            [
                [
                    self._inspection_hold_s_batch[env_index],
                    float(self._inspection_completed_batch[env_index]),
                ]
                for env_index in range(env.num_envs)
            ],
            dtype=torch.float32,
            device=env.device,
        )
        com_offsets = torch.as_tensor(
            [layout["center_of_mass_offset_local_m"] for layout in env.layouts],
            dtype=torch.float32,
            device=env.device,
        )
        observations = {
            "objects/box/pose": env.entity_world_pose("box"),
            "objects/box/center_of_mass_offset_local": com_offsets,
            "inspection/static_hold_state": inspection_state,
        }
        if env.num_envs == 1:
            return {key: value[0] for key, value in observations.items()}
        return observations

    def wrench_reference_position(self, env):
        reference = torch.tensor(
            [[0.5, 0.0, 0.0]], dtype=torch.float32, device=env.device
        )
        return reference.repeat(env.num_envs, 1)

    def debug_state(self, env):
        return {
            "inspection_hold_s": self._inspection_hold_s,
            "inspection_max_hold_s": self._inspection_max_hold_s,
            "inspection_completed": self._inspection_completed,
            "inspection_diagnostics": self._inspection_diagnostics,
            "center_of_mass_offset_local_m": env.layout[
                "center_of_mass_offset_local_m"
            ],
        }

    def episode_metadata(self) -> dict[str, object]:
        return {
            "trajectory_scope": "grasp_equal_mass_box_hold_static_classify_front_back_center_of_mass_release",
            "asset_usd_paths": {
                "box_shell": str(self.asset_dir / SPEC.box_asset),
                "source_visual": str(self.asset_dir / SPEC.source_box_asset),
                "black_basket_f27": str(self.asset_dir / SPEC.container_asset),
                "official_conveyor": SPEC.conveyor_source_url,
                "background_shipping_parcels": [
                    str(self.asset_dir / asset)
                    for asset in SPEC.background_parcel_assets
                ],
            },
            "mass_contract_kg": {
                "uniform_box": SPEC.box_total_mass_kg,
                "imbalanced_box": SPEC.box_total_mass_kg,
                "total_each_case": SPEC.box_total_mass_kg,
            },
            "center_of_mass_contract": {
                "uniform_local_offset_m": [0.0, 0.0, 0.0],
                "axis": "parcel_long_axis_front_back",
                "front_direction": "local +Y = world -X = toward robot",
                "back_direction": "local -Y = world +X = away from robot",
                "imbalanced_local_y_offset_abs_m": SPEC.front_back_com_offset_m,
                "imbalanced_end": "deterministic random front_or_back by seed",
            },
            "perfect_dataset_balance": {
                "uniform": "one per consecutive seed pair",
                "imbalanced": "one per consecutive seed pair",
            },
            "destinations": {
                "imbalanced": "large_black_container_f27_left",
                "uniform": "official_conveyor_belt_a04_right",
            },
            "box_visual_scale_xyz": list(SPEC.box_visual_scale_xyz),
            "scaled_box_size_m": list(SPEC.box_size_m),
            "black_basket_visual_scale": SPEC.black_basket_visual_scale,
            "black_basket_scaled_size_m": list(SPEC.black_basket_scaled_size_m),
            "black_basket_visual_floor_z_m": SPEC.black_basket_visual_floor_z_m,
            "black_basket_collision_floor": {
                "model": "full_depth_static_cuboid_with_unchanged_surface",
                "top_z_m": SPEC.black_basket_visual_floor_z_m,
                "bottom_z_m": SPEC.black_basket_collision_floor_bottom_z_m,
                "center_z_m": SPEC.black_basket_collision_floor_center_z_m,
                "thickness_m": SPEC.black_basket_collision_floor_thickness_m,
            },
            "conveyor_visual_scale_xyz": list(SPEC.conveyor_visual_scale_xyz),
            "conveyor_collision": {
                "model": "single_static_cuboid_covering_full_belt",
                "center_xy_m": list(SPEC.conveyor_collision_center_xy_m),
                "size_m": list(SPEC.conveyor_collision_size_m),
                "surface_z_m": SPEC.conveyor_surface_z_m,
            },
            "main_camera_changed": False,
        }
