"""Isaac Lab runtime for the nonlinear-detent faucet valve."""

from __future__ import annotations

import math

import torch
import warp as wp
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.utils.prims import get_current_stage

from forcebench.task_api import SceneEntitySpec, TaskRuntime

from . import MANIFEST
from .outcome import evaluate_report
from .spec import (
    SPEC,
    detent_effort_nm,
    detent_resistance_nm,
    evaluate_angle,
    flow_fraction,
    sample_initial_state,
    spec_from_layout,
)


_CUP09_VISUAL_ASSET = "simready/cup09/cup09.usd"


def _valve_config(path) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Task_faucet_valve",
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
                static_friction=SPEC.valve_friction,
                dynamic_friction=SPEC.valve_friction,
                restitution=0.0,
                friction_combine_mode="min",
                restitution_combine_mode="min",
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=SPEC.valve_position_m,
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={SPEC.valve_joint_name: 0.0},
            joint_vel={SPEC.valve_joint_name: 0.0},
        ),
        actuators={
            "passive_detent": ImplicitActuatorCfg(
                joint_names_expr=[SPEC.valve_joint_name],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.01,
                effort_limit_sim=SPEC.max_effort_nm,
                velocity_limit_sim=12.0,
            )
        },
    )


class FaucetTurnTask(TaskRuntime):
    manifest = MANIFEST

    def __init__(self, asset_dir):
        super().__init__(asset_dir)
        self._joint_id: int | None = None
        self._last_angle_rad = 0.0
        self._last_velocity_rad_s = 0.0
        self._last_effort_nm = 0.0
        self._last_resistance_nm = SPEC.base_friction_torque_nm
        self._peak_abs_effort_nm = 0.0
        self._detent_crossed = False
        self._episode_spec = SPEC
        self._valve_position_m = SPEC.valve_position_m
        self._water_stream_center_m = SPEC.water_stream_center_m
        self._last_flow_fraction = 0.0
        self._water_visible = False
        self._episode_specs = []
        self._valve_positions_m = []
        self._water_stream_centers_m = []
        self._last_angle_rad_batch: list[float] = []
        self._last_velocity_rad_s_batch: list[float] = []
        self._last_effort_nm_batch: list[float] = []
        self._last_resistance_nm_batch: list[float] = []
        self._peak_abs_effort_nm_batch: list[float] = []
        self._detent_crossed_batch: list[bool] = []
        self._last_flow_fraction_batch: list[float] = []
        self._water_visible_batch: list[bool] = []

    def _ensure_batch_state(self, num_envs: int) -> None:
        if len(self._episode_specs) == num_envs:
            return
        self._episode_specs = [SPEC for _ in range(num_envs)]
        self._valve_positions_m = [SPEC.valve_position_m for _ in range(num_envs)]
        self._water_stream_centers_m = [SPEC.water_stream_center_m for _ in range(num_envs)]
        self._last_angle_rad_batch = [0.0 for _ in range(num_envs)]
        self._last_velocity_rad_s_batch = [0.0 for _ in range(num_envs)]
        self._last_effort_nm_batch = [0.0 for _ in range(num_envs)]
        self._last_resistance_nm_batch = [SPEC.base_friction_torque_nm for _ in range(num_envs)]
        self._peak_abs_effort_nm_batch = [0.0 for _ in range(num_envs)]
        self._detent_crossed_batch = [False for _ in range(num_envs)]
        self._last_flow_fraction_batch = [0.0 for _ in range(num_envs)]
        self._water_visible_batch = [False for _ in range(num_envs)]

    def _sync_legacy_state(self) -> None:
        if not self._episode_specs:
            return
        self._episode_spec = self._episode_specs[0]
        self._valve_position_m = self._valve_positions_m[0]
        self._water_stream_center_m = self._water_stream_centers_m[0]
        self._last_angle_rad = self._last_angle_rad_batch[0]
        self._last_velocity_rad_s = self._last_velocity_rad_s_batch[0]
        self._last_effort_nm = self._last_effort_nm_batch[0]
        self._last_resistance_nm = self._last_resistance_nm_batch[0]
        self._peak_abs_effort_nm = self._peak_abs_effort_nm_batch[0]
        self._detent_crossed = self._detent_crossed_batch[0]
        self._last_flow_fraction = self._last_flow_fraction_batch[0]
        self._water_visible = self._water_visible_batch[0]

    def validate_assets(self) -> None:
        for relative_path in (
            SPEC.sink_asset,
            SPEC.sink_environment_asset,
            SPEC.sink_texture,
            SPEC.robot_support_asset,
            SPEC.valve_asset,
            _CUP09_VISUAL_ASSET,
        ):
            path = self.asset_dir / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"missing faucet_turn asset: {path}")

    def scene_entities(self):
        entity = SceneEntitySpec(
            role="faucet_valve",
            kind="articulation",
            config=_valve_config(self.asset_dir / SPEC.valve_asset),
            friction=None,
        )
        entity.validate()
        return (entity,)

    def spawn_scene_prims(self, env) -> None:
        support_visual = sim_utils.UsdFileCfg(
            usd_path=str(self.asset_dir / SPEC.robot_support_asset),
        )
        support_visual.func(
            "/World/envs/env_.*/Task_faucet_robot_support_visual",
            support_visual,
            translation=SPEC.robot_support_position_m,
        )
        support_collision = sim_utils.CuboidCfg(
            size=SPEC.robot_support_size_m,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        support_collision.func(
            "/World/envs/env_.*/Task_faucet_robot_support_collision",
            support_collision,
            translation=SPEC.robot_support_position_m,
        )
        # A tall, wide-mouth Cup09 sits in the basin below the outlet.  Its rim
        # hides the water stream from the policy cameras while keeping the
        # valve and the gripper's rotation envelope unobstructed.
        container_visual = sim_utils.UsdFileCfg(
            usd_path=str(self.asset_dir / _CUP09_VISUAL_ASSET),
            scale=(0.140, 0.140, 0.216),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.34, 0.53, 0.62),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=0.62,
                metallic=0.0,
                opacity=1.0,
            ),
        )
        container_visual.func(
            "/World/envs/env_.*/Task_faucet_stream_container_visual",
            container_visual,
            translation=(
                0.440,
                SPEC.water_outlet_position_m[1] + 0.008,
                0.028,
            ),
        )
        stream = sim_utils.CylinderCfg(
            radius=SPEC.water_stream_radius_m,
            height=SPEC.water_stream_length_m,
            axis="Z",
            visible=False,
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.04, 0.48, 0.82),
                emissive_color=(0.01, 0.06, 0.10),
                roughness=0.08,
                metallic=0.0,
                opacity=0.72,
            ),
        )
        stream.func(
            "/World/envs/env_.*/Task_faucet_water_stream",
            stream,
            translation=SPEC.water_stream_center_m,
        )

    def sample_initial_state(self, seed: int) -> dict[str, object]:
        return sample_initial_state(seed)

    def after_scene_setup(self, env) -> None:
        entity = env.entity("faucet_valve")
        self._joint_id = entity.joint_names.index(SPEC.valve_joint_name)
        self._ensure_batch_state(env.num_envs)
        stage = get_current_stage()
        for env_index in range(env.num_envs):
            lever_prim = stage.GetPrimAtPath(
                f"/World/envs/env_{env_index}/Task_faucet_valve/lever"
            )
            table_path = Sdf.Path(f"/World/envs/env_{env_index}/Table")
            if not lever_prim.IsValid() or not stage.GetPrimAtPath(table_path).IsValid():
                raise RuntimeError("faucet lever or kitchen support table prim is missing")
            # The tuned lever sits partly below the sink deck.  The generic
            # hidden full-table support proxy crosses that space, so filter
            # the lever and only the robot's distal gripper links from that
            # proxy.  Arm-table, robot-lever and robot-connector contact stays
            # active for manipulation and collision avoidance.
            filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(lever_prim)
            filtered_pairs.CreateFilteredPairsRel().AddTarget(table_path)
            for gripper_link_name in (
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
            ):
                gripper_prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_index}/Robot/{gripper_link_name}"
                )
                if not gripper_prim.IsValid():
                    raise RuntimeError(
                        f"Franka gripper link is missing: {gripper_link_name}"
                    )
                gripper_filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(
                    gripper_prim
                )
                gripper_filtered_pairs.CreateFilteredPairsRel().AddTarget(table_path)
            water_prim = stage.GetPrimAtPath(
                f"/World/envs/env_{env_index}/Task_faucet_water_stream"
            )
            if not water_prim.IsValid():
                raise RuntimeError("faucet water-stream prim is missing")
        previous_step = env.sim.step

        def step_with_faucet_detent(*args, **kwargs):
            result = previous_step(*args, **kwargs)
            self._physics_tick(env)
            return result

        env.sim.step = step_with_faucet_detent

    def _physics_tick(self, env) -> None:
        if self._joint_id is None:
            return
        entity = env.entity("faucet_valve")
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
            self._last_resistance_nm_batch[env_index] = detent_resistance_nm(angle_value, episode_spec)
            self._peak_abs_effort_nm_batch[env_index] = max(
                self._peak_abs_effort_nm_batch[env_index], abs(effort_value)
            )
            self._detent_crossed_batch[env_index] = bool(
                self._detent_crossed_batch[env_index]
                or evaluate_angle(angle_value, episode_spec)["detent_crossed"]
            )
            self._update_water_visual(env_index, angle_value, episode_spec)
        self._sync_legacy_state()

    def _update_water_visual(self, env_index: int, angle_rad: float, episode_spec) -> None:
        flow = flow_fraction(angle_rad, episode_spec)
        stage = get_current_stage()
        water_prim = stage.GetPrimAtPath(
            f"/World/envs/env_{env_index}/Task_faucet_water_stream"
        )
        imageable = UsdGeom.Imageable(water_prim)
        if flow <= 0.0:
            if self._water_visible_batch[env_index]:
                imageable.MakeInvisible()
            self._water_visible_batch[env_index] = False
            self._last_flow_fraction_batch[env_index] = 0.0
            return
        radius_scale = 0.45 + 0.55 * math.sqrt(flow)
        UsdGeom.XformCommonAPI(water_prim).SetScale(
            Gf.Vec3f(radius_scale, radius_scale, 1.0)
        )
        if not self._water_visible_batch[env_index]:
            imageable.MakeVisible()
        self._water_visible_batch[env_index] = True
        self._last_flow_fraction_batch[env_index] = flow

    def reset_scene(self, env, env_ids, layout) -> None:
        self._ensure_batch_state(env.num_envs)
        episode_spec = spec_from_layout(layout)
        valve_position_m = episode_spec.valve_position_m
        water_stream_center_m = tuple(
            float(value) for value in layout["water_stream_center_m"]
        )
        stage = get_current_stage()
        sink_translation = tuple(float(value) for value in layout["sink_translation_m"])
        for env_index in env_ids.detach().cpu().tolist():
            self._episode_specs[env_index] = episode_spec
            self._valve_positions_m[env_index] = valve_position_m
            self._water_stream_centers_m[env_index] = water_stream_center_m
            sink_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/TableVisual")
            if not sink_prim.IsValid():
                raise RuntimeError("kitchen sink TableVisual prim is missing")
            UsdGeom.XformCommonAPI(sink_prim).SetTranslate(Gf.Vec3d(*sink_translation))
            water_prim = stage.GetPrimAtPath(
                f"/World/envs/env_{env_index}/Task_faucet_water_stream"
            )
            if not water_prim.IsValid():
                raise RuntimeError("faucet water-stream prim is missing")
            water_xform = UsdGeom.XformCommonAPI(water_prim)
            water_xform.SetTranslate(Gf.Vec3d(*water_stream_center_m))
            water_xform.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))
            UsdGeom.Imageable(water_prim).MakeInvisible()

        entity = env.entity("faucet_valve")
        root_quaternion = entity.data.default_root_pose.torch[env_ids, 3:7].clone()
        env.set_entity_root_pose(
            "faucet_valve",
            env_ids,
            valve_position_m,
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
            self._last_resistance_nm_batch[env_index] = episode_spec.base_friction_torque_nm
            self._peak_abs_effort_nm_batch[env_index] = 0.0
            self._detent_crossed_batch[env_index] = False
            self._last_flow_fraction_batch[env_index] = 0.0
            self._water_visible_batch[env_index] = False
        self._sync_legacy_state()

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        self._ensure_batch_state(env.num_envs)
        episode_spec = self._episode_specs[env_id]
        angle_rad = self._last_angle_rad_batch[env_id]
        report = evaluate_angle(angle_rad, episode_spec)
        report["detent_crossed"] = self._detent_crossed_batch[env_id]
        report["success"] = self._detent_crossed_batch[env_id]
        report["valve_angular_velocity_rad_s"] = self._last_velocity_rad_s_batch[env_id]
        report["load_effort_nm"] = self._last_effort_nm_batch[env_id]
        report["water_visible"] = self._water_visible_batch[env_id]
        backoff_target_deg = (
            episode_spec.detent_center_deg
            + episode_spec.expert_backoff_after_peak_deg
        )
        report["expert_backoff_target_deg"] = backoff_target_deg
        report["expert_backoff_error_deg"] = abs(
            backoff_target_deg - math.degrees(angle_rad)
        )
        report["expert_post_peak_stop"] = bool(
            report["expert_backoff_error_deg"]
            <= episode_spec.expert_backoff_tolerance_deg
            and abs(self._last_velocity_rad_s_batch[env_id]) <= episode_spec.expert_stop_speed_rad_s
        )
        return evaluate_report(report)

    def evaluate_final_state(self, env) -> dict[str, object]:
        return self.evaluate_final_state_for(env, 0)

    def privileged_observations(self, env):
        entity = env.entity("faucet_valve")
        return {
            "objects/faucet_valve/pose": env.entity_world_pose("faucet_valve")[0],
            "objects/faucet_valve/joint_position": entity.data.joint_pos.torch[0],
            "objects/faucet_valve/joint_velocity": entity.data.joint_vel.torch[0],
        }

    def wrench_reference_position(self, env):
        self._ensure_batch_state(env.num_envs)
        reference = torch.tensor(
            [
                [base + offset for base, offset in zip(
                    self._valve_positions_m[env_id],
                    self._episode_specs[env_id].lever_pivot_offset_m,
                )]
                for env_id in range(env.num_envs)
            ],
            dtype=torch.float32,
            device=env.device,
        )
        return reference

    def debug_state(self, env):
        return {
            "joint_name": SPEC.valve_joint_name,
            "angle_rad": self._last_angle_rad,
            "angular_velocity_rad_s": self._last_velocity_rad_s,
            "load_effort_nm": self._last_effort_nm,
            "resistance_torque_nm": self._last_resistance_nm,
            "peak_abs_effort_nm": self._peak_abs_effort_nm,
            "detent_crossed": self._detent_crossed,
            "detent_center_deg": self._episode_spec.detent_center_deg,
            "detent_peak_torque_nm": self._episode_spec.detent_peak_torque_nm,
            "valve_position_m": list(self._valve_position_m),
            "water_stream_center_m": list(self._water_stream_center_m),
            "flow_fraction": self._last_flow_fraction,
            "water_visible": self._water_visible,
        }

    def episode_metadata(self) -> dict[str, object]:
        return {
            "phase": SPEC.phase,
            "scene_profile": "kitchen_sink",
            "asset_usd_paths": {
                "faucet_valve": str(self.asset_dir / SPEC.valve_asset),
            },
            "sink_asset": str(self.asset_dir / SPEC.sink_asset),
            "robot_pose_changed": False,
            "main_camera_changed": False,
            "water_visual": {
                "type": "collision_free_translucent_stream",
                "onset": "episode_detent_center_deg",
                "radius_tracks": "flow_fraction",
            },
            "load_model": {
                "type": "continuous_rise_drop_capture_rise_resistance_plus_damping",
                "physics_rate_hz": 60,
                "detent_center_deg": self._episode_spec.detent_center_deg,
                "detent_rise_width_deg": self._episode_spec.detent_rise_width_deg,
                "detent_drop_width_deg": self._episode_spec.detent_drop_width_deg,
                "post_drop_low_width_deg": self._episode_spec.post_drop_low_width_deg,
                "post_drop_rise_width_deg": self._episode_spec.post_drop_rise_width_deg,
                "post_drop_resistance_nm": self._episode_spec.post_drop_resistance_nm,
                "detent_peak_torque_nm": self._episode_spec.detent_peak_torque_nm,
                "damping_nm_s_rad": self._episode_spec.damping_nm_s_rad,
                "base_friction_torque_nm": self._episode_spec.base_friction_torque_nm,
                "max_effort_nm": self._episode_spec.max_effort_nm,
            },
        }
