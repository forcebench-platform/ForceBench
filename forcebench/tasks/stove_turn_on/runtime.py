"""Isaac Lab runtime for turning on the Lightwheel gas stove."""

from __future__ import annotations

import math

import torch
import warp as wp
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.utils.prims import get_current_stage

from forcebench.task_api import CameraProfile, SceneEntitySpec, TaskRuntime

from . import MANIFEST
from .outcome import evaluate_report
from .spec import (
    SPEC,
    detent_effort_nm,
    detent_resistance_nm,
    evaluate_angle,
    flame_fraction,
    sample_initial_state,
    spec_from_layout,
)


_POT_VISUAL_ASSET = (
    "lightwheel_kitchen/FullKitchen/Collected_KitchenRoom/Pot057/Pot057.usd"
)


def _knob_config(path) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Task_stove_knob",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=2.0,
                max_angular_velocity=12.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
                max_contact_impulse=1.0e4,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=SPEC.knob_friction,
                dynamic_friction=SPEC.knob_friction,
                restitution=0.0,
                friction_combine_mode=SPEC.knob_friction_combine_mode,
                restitution_combine_mode="min",
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=SPEC.target_knob_position_m,
            rot=SPEC.stovetop_orientation_xyzw,
            joint_pos={SPEC.knob_joint_name: 0.0},
            joint_vel={SPEC.knob_joint_name: 0.0},
        ),
        actuators={
            "passive_ignition_detent": ImplicitActuatorCfg(
                joint_names_expr=[SPEC.knob_joint_name],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.003,
                effort_limit_sim=SPEC.max_effort_nm,
                velocity_limit_sim=12.0,
            )
        },
    )


class StoveTurnOnTask(TaskRuntime):
    manifest = MANIFEST
    # Straight-on 62 cm view of the +90-degree stove layout.  The 40-degree
    # pitch keeps the knob centered, retains part of the burner, and crops most
    # of the robot above the frame without introducing a lateral camera yaw.
    camera_profile = CameraProfile(
        table_position_m=(0.918243014, -0.000136329, 0.419075544),
        table_target_m=(0.443295459, -0.000136329, 0.020547226),
        table_rotation_wxyz=(
            0.640856382,
            0.298836239,
            0.298836239,
            0.640856382,
        ),
    )

    _FLAME_PRIM_NAMES = ("outer", "inner")

    def __init__(self, asset_dir):
        super().__init__(asset_dir)
        self._joint_id: int | None = None
        self._episode_spec = SPEC
        self._knob_position_m = SPEC.target_knob_position_m
        self._last_angle_rad = 0.0
        self._last_velocity_rad_s = 0.0
        self._last_effort_nm = 0.0
        self._last_resistance_nm = SPEC.base_friction_torque_nm
        self._peak_abs_effort_nm = 0.0
        self._detent_crossed = False
        self._last_flame_fraction = 0.0
        self._flame_visible = False
        # Isaac Lab clones share one task-runtime object.  Keep the legacy
        # scalar fields for the single-env API, while the physics callback and
        # flame visuals use independent state for every cloned workspace.
        self._episode_specs = []
        self._knob_positions_m = []
        self._last_angle_rad_batch: list[float] = []
        self._last_velocity_rad_s_batch: list[float] = []
        self._last_effort_nm_batch: list[float] = []
        self._last_resistance_nm_batch: list[float] = []
        self._peak_abs_effort_nm_batch: list[float] = []
        self._detent_crossed_batch: list[bool] = []
        self._last_flame_fraction_batch: list[float] = []
        self._flame_visible_batch: list[bool] = []

    def _ensure_batch_state(self, num_envs: int) -> None:
        if len(self._episode_specs) == num_envs:
            return
        self._episode_specs = [SPEC for _ in range(num_envs)]
        self._knob_positions_m = [SPEC.target_knob_position_m for _ in range(num_envs)]
        self._last_angle_rad_batch = [0.0 for _ in range(num_envs)]
        self._last_velocity_rad_s_batch = [0.0 for _ in range(num_envs)]
        self._last_effort_nm_batch = [0.0 for _ in range(num_envs)]
        self._last_resistance_nm_batch = [
            SPEC.base_friction_torque_nm for _ in range(num_envs)
        ]
        self._peak_abs_effort_nm_batch = [0.0 for _ in range(num_envs)]
        self._detent_crossed_batch = [False for _ in range(num_envs)]
        self._last_flame_fraction_batch = [0.0 for _ in range(num_envs)]
        self._flame_visible_batch = [False for _ in range(num_envs)]

    def _sync_legacy_state(self) -> None:
        """Expose env_0 through the established single-environment API."""

        if not self._episode_specs:
            return
        self._episode_spec = self._episode_specs[0]
        self._knob_position_m = self._knob_positions_m[0]
        self._last_angle_rad = self._last_angle_rad_batch[0]
        self._last_velocity_rad_s = self._last_velocity_rad_s_batch[0]
        self._last_effort_nm = self._last_effort_nm_batch[0]
        self._last_resistance_nm = self._last_resistance_nm_batch[0]
        self._peak_abs_effort_nm = self._peak_abs_effort_nm_batch[0]
        self._detent_crossed = self._detent_crossed_batch[0]
        self._last_flame_fraction = self._last_flame_fraction_batch[0]
        self._flame_visible = self._flame_visible_batch[0]

    def validate_assets(self) -> None:
        for relative_path in (
            SPEC.stovetop_asset,
            SPEC.stovetop_environment_asset,
            SPEC.stovetop_texture,
            SPEC.knob_asset,
            _POT_VISUAL_ASSET,
        ):
            path = self.asset_dir / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"missing stove_turn_on asset: {path}")

    def scene_entities(self):
        entity = SceneEntitySpec(
            role="stove_knob",
            kind="articulation",
            config=_knob_config(self.asset_dir / SPEC.knob_asset),
            friction=None,
        )
        entity.validate()
        return (entity,)

    def spawn_scene_prims(self, env) -> None:
        visual = sim_utils.UsdFileCfg(
            usd_path=str(self.asset_dir / SPEC.stovetop_environment_asset)
        )
        visual.func(
            "/World/envs/env_.*/Task_stovetop_visual",
            visual,
            translation=SPEC.stovetop_translation_m,
            orientation=SPEC.stovetop_orientation_xyzw,
        )

        # A scaled kitchen pot provides a natural visual occluder over the
        # active burner.  Authored rigid-body and collision APIs are disabled,
        # so it cannot alter the knob task's contact dynamics.
        pot_visual = sim_utils.UsdFileCfg(
            usd_path=str(self.asset_dir / _POT_VISUAL_ASSET),
            # Pot057 is authored in centimetres.  Convert to metres and retain
            # half of its native physical size for this single-burner stove.
            scale=(0.005, 0.005, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        )
        pot_visual.func(
            "/World/envs/env_.*/Task_stove_burner_pot_visual",
            pot_visual,
            translation=(
                SPEC.burner_center_m[0],
                SPEC.burner_center_m[1],
                0.070,
            ),
        )
        # The matte burner collar shares the measured burner center and fills
        # only the narrow gap below the pot.  It encloses the visual flame disk
        # without touching the surrounding grate in the rendered geometry.
        burner_collar = sim_utils.CylinderCfg(
            radius=0.060,
            height=0.009,
            axis="Z",
            visible=True,
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.012, 0.015, 0.018),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.88,
                metallic=0.04,
                opacity=1.0,
            ),
        )
        burner_collar.func(
            "/World/envs/env_.*/Task_stove_burner_occlusion_collar",
            burner_collar,
            translation=(
                SPEC.burner_center_m[0],
                SPEC.burner_center_m[1],
                0.048,
            ),
        )

        flame_materials = {
            "outer": sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.03, 0.22, 0.95),
                emissive_color=(0.02, 0.30, 2.8),
                roughness=0.15,
                metallic=0.0,
                opacity=0.82,
            ),
            "inner": sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.98, 0.34, 0.025),
                emissive_color=(2.2, 0.18, 0.01),
                roughness=0.18,
                metallic=0.0,
                opacity=0.78,
            ),
        }
        for name, radius, height, z_offset in (
            ("outer", 0.052, 0.0030, 0.0),
            ("inner", 0.037, 0.0035, 0.0015),
        ):
            flame = sim_utils.CylinderCfg(
                radius=radius,
                height=height,
                axis="Z",
                visible=False,
                collision_props=None,
                visual_material=flame_materials[name],
            )
            center = (
                SPEC.burner_center_m[0],
                SPEC.burner_center_m[1],
                SPEC.burner_center_m[2] + z_offset,
            )
            flame.func(
                f"/World/envs/env_.*/Task_stove_flame_{name}",
                flame,
                translation=center,
            )

    def sample_initial_state(self, seed: int) -> dict[str, object]:
        return sample_initial_state(seed)

    def after_scene_setup(self, env) -> None:
        entity = env.entity("stove_knob")
        self._joint_id = entity.joint_names.index(SPEC.knob_joint_name)
        self._ensure_batch_state(env.num_envs)
        stage = get_current_stage()
        for env_index in range(env.num_envs):
            for name in self._FLAME_PRIM_NAMES:
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_index}/Task_stove_flame_{name}"
                )
                if not prim.IsValid():
                    raise RuntimeError(f"stove flame prim {name!r} is missing")

        previous_step = env.sim.step

        def step_with_stove_detent(*args, **kwargs):
            result = previous_step(*args, **kwargs)
            self._physics_tick(env)
            return result

        env.sim.step = step_with_stove_detent

    def _physics_tick(self, env) -> None:
        if self._joint_id is None:
            return
        entity = env.entity("stove_knob")
        positions = wp.to_torch(entity.root_view.get_dof_positions())
        velocities = wp.to_torch(entity.root_view.get_dof_velocities())
        angle = positions[:, self._joint_id]
        velocity = velocities[:, self._joint_id]
        efforts = torch.empty((env.num_envs, 1), dtype=torch.float32, device=env.device)
        for env_index in range(env.num_envs):
            episode_spec = self._episode_specs[env_index]
            efforts[env_index, 0] = detent_effort_nm(
                float(angle[env_index].item()),
                float(velocity[env_index].item()),
                episode_spec,
            )
        entity.set_joint_effort_target_index(
            target=efforts,
            joint_ids=[self._joint_id],
        )
        for env_index in range(env.num_envs):
            episode_spec = self._episode_specs[env_index]
            angle_value = float(angle[env_index].item())
            velocity_value = float(velocity[env_index].item())
            effort_value = float(efforts[env_index, 0].item())
            self._last_angle_rad_batch[env_index] = angle_value
            self._last_velocity_rad_s_batch[env_index] = velocity_value
            self._last_effort_nm_batch[env_index] = effort_value
            self._last_resistance_nm_batch[env_index] = detent_resistance_nm(
                angle_value, episode_spec
            )
            self._peak_abs_effort_nm_batch[env_index] = max(
                self._peak_abs_effort_nm_batch[env_index], abs(effort_value)
            )
            self._detent_crossed_batch[env_index] = bool(
                self._detent_crossed_batch[env_index]
                or evaluate_angle(angle_value, episode_spec)["ignition_detent_crossed"]
            )
            self._update_flame_visual(env_index, angle_value, episode_spec)
        self._sync_legacy_state()

    def _flame_prims(self, env_index: int = 0):
        stage = get_current_stage()
        return tuple(
            stage.GetPrimAtPath(
                f"/World/envs/env_{env_index}/Task_stove_flame_{name}"
            )
            for name in self._FLAME_PRIM_NAMES
        )

    def _update_flame_visual(self, env_index: int, angle_rad: float, episode_spec) -> None:
        intensity = flame_fraction(angle_rad, episode_spec)
        was_visible = self._flame_visible_batch[env_index]
        previous_intensity = self._last_flame_fraction_batch[env_index]
        if intensity <= 0.0:
            if was_visible:
                for prim in self._flame_prims(env_index):
                    UsdGeom.Imageable(prim).MakeInvisible()
            self._flame_visible_batch[env_index] = False
            self._last_flame_fraction_batch[env_index] = 0.0
            return
        if was_visible and abs(intensity - previous_intensity) < 0.01:
            return
        scale = 0.82 + 0.18 * math.sqrt(intensity)
        for prim in self._flame_prims(env_index):
            UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(scale, scale, 1.0))
            if not was_visible:
                UsdGeom.Imageable(prim).MakeVisible()
        self._flame_visible_batch[env_index] = True
        self._last_flame_fraction_batch[env_index] = intensity

    def reset_scene(self, env, env_ids, layout) -> None:
        self._ensure_batch_state(env.num_envs)
        episode_spec = spec_from_layout(layout)
        knob_position_m = tuple(
            float(value) for value in layout["target_knob_position_m"]
        )
        stage = get_current_stage()
        stovetop_translation = tuple(
            float(value) for value in layout["stovetop_translation_m"]
        )
        for env_index in env_ids.detach().cpu().tolist():
            self._episode_specs[env_index] = episode_spec
            self._knob_positions_m[env_index] = knob_position_m
            stove_prim = stage.GetPrimAtPath(
                f"/World/envs/env_{env_index}/Task_stovetop_visual"
            )
            if not stove_prim.IsValid():
                raise RuntimeError("stovetop visual prim is missing")
            UsdGeom.XformCommonAPI(stove_prim).SetTranslate(
                Gf.Vec3d(*stovetop_translation)
            )
            for name in self._FLAME_PRIM_NAMES:
                flame_prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_index}/Task_stove_flame_{name}"
                )
                UsdGeom.XformCommonAPI(flame_prim).SetScale(Gf.Vec3f(1.0, 1.0, 1.0))
                UsdGeom.Imageable(flame_prim).MakeInvisible()

        entity = env.entity("stove_knob")
        root_quaternion = entity.data.default_root_pose.torch[env_ids, 3:7].clone()
        env.set_entity_root_pose(
            "stove_knob",
            env_ids,
            knob_position_m,
            root_quaternion,
        )
        count = len(env_ids)
        joint_position = torch.zeros((count, entity.num_joints), device=env.device)
        joint_velocity = torch.zeros_like(joint_position)
        joint_effort = torch.zeros_like(joint_position)
        entity.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
        entity.set_joint_effort_target_index(target=joint_effort, env_ids=env_ids)
        entity.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        entity.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
        entity.reset(env_ids)
        for env_index in env_ids.detach().cpu().tolist():
            self._last_angle_rad_batch[env_index] = 0.0
            self._last_velocity_rad_s_batch[env_index] = 0.0
            self._last_effort_nm_batch[env_index] = 0.0
            self._last_resistance_nm_batch[env_index] = (
                episode_spec.base_friction_torque_nm
            )
            self._peak_abs_effort_nm_batch[env_index] = 0.0
            self._detent_crossed_batch[env_index] = False
            self._last_flame_fraction_batch[env_index] = 0.0
            self._flame_visible_batch[env_index] = False
        self._sync_legacy_state()

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        self._ensure_batch_state(env.num_envs)
        env_id = int(env_id)
        episode_spec = self._episode_specs[env_id]
        report = evaluate_angle(self._last_angle_rad_batch[env_id], episode_spec)
        report["ignition_detent_crossed"] = self._detent_crossed_batch[env_id]
        report["success"] = bool(
            self._detent_crossed_batch[env_id]
            and self._flame_visible_batch[env_id]
            and report["within_final_angle_tolerance"]
        )
        report["knob_angular_velocity_rad_s"] = self._last_velocity_rad_s_batch[env_id]
        report["load_effort_nm"] = self._last_effort_nm_batch[env_id]
        report["load_resistance_nm"] = self._last_resistance_nm_batch[env_id]
        report["flame_visible"] = self._flame_visible_batch[env_id]
        return evaluate_report(report)

    def evaluate_final_state(self, env) -> dict[str, object]:
        return self.evaluate_final_state_for(env, 0)

    def privileged_observations(self, env):
        entity = env.entity("stove_knob")
        return {
            "objects/stove_knob/pose": env.entity_world_pose("stove_knob")[0],
            "objects/stove_knob/joint_position": entity.data.joint_pos.torch[0],
            "objects/stove_knob/joint_velocity": entity.data.joint_vel.torch[0],
        }

    def wrench_reference_position(self, env):
        self._ensure_batch_state(env.num_envs)
        return torch.tensor(
            self._knob_positions_m,
            dtype=torch.float32,
            device=env.device,
        )

    def debug_state(self, env):
        return {
            "joint_name": SPEC.knob_joint_name,
            "angle_rad": self._last_angle_rad,
            "angular_velocity_rad_s": self._last_velocity_rad_s,
            "load_effort_nm": self._last_effort_nm,
            "load_resistance_nm": self._last_resistance_nm,
            "peak_abs_effort_nm": self._peak_abs_effort_nm,
            "detent_crossed": self._detent_crossed,
            "detent_center_deg": self._episode_spec.detent_center_deg,
            "resistance_peak_angle_deg": self._episode_spec.detent_center_deg,
            "resistance_peak_angle_range_deg": list(
                self._episode_spec.detent_center_range_deg
            ),
            "detent_peak_torque_nm": self._episode_spec.detent_peak_torque_nm,
            "target_final_angle_deg": self._episode_spec.success_angle_deg,
            "final_angle_tolerance_deg": self._episode_spec.final_angle_tolerance_deg,
            "knob_position_m": list(self._knob_position_m),
            "flame_fraction": self._last_flame_fraction,
            "flame_visible": self._flame_visible,
        }

    def episode_metadata(self) -> dict[str, object]:
        return {
            "phase": SPEC.phase,
            "scene_profile": "kitchen_stove",
            "asset_usd_paths": {
                "stovetop": str(self.asset_dir / SPEC.stovetop_asset),
                "stovetop_environment": str(
                    self.asset_dir / SPEC.stovetop_environment_asset
                ),
                "stove_knob": str(self.asset_dir / SPEC.knob_asset),
            },
            "target_knob": "Stovetop012_knob_center",
            "robot_pose_changed": False,
            "main_camera_changed": True,
            "flame_visual": "collision-free emissive burner discs",
            "load_model": {
                "type": "faucet_style_asymmetric_resistance_plus_damping",
                "physics_rate_hz": 60,
                "resistance_peak_angle_deg": self._episode_spec.detent_center_deg,
                "resistance_peak_angle_range_deg": list(
                    self._episode_spec.detent_center_range_deg
                ),
                "detent_rise_width_deg": self._episode_spec.detent_rise_width_deg,
                "detent_drop_width_deg": self._episode_spec.detent_drop_width_deg,
                "post_drop_low_width_deg": self._episode_spec.post_drop_low_width_deg,
                "post_drop_rise_width_deg": self._episode_spec.post_drop_rise_width_deg,
                "post_drop_resistance_nm": self._episode_spec.post_drop_resistance_nm,
                "detent_peak_torque_nm": self._episode_spec.detent_peak_torque_nm,
                "final_angle_tolerance_deg": self._episode_spec.final_angle_tolerance_deg,
                "damping_nm_s_rad": self._episode_spec.damping_nm_s_rad,
                "base_friction_torque_nm": self._episode_spec.base_friction_torque_nm,
                "max_effort_nm": self._episode_spec.max_effort_nm,
                "contact_friction": self._episode_spec.knob_friction,
                "friction_combine_mode": self._episode_spec.knob_friction_combine_mode,
            },
        }
