"""Isaac Lab runtime for the pure-rotation force-disambiguated faucet."""

from __future__ import annotations

import math

import torch
import warp as wp
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.utils.prims import get_current_stage

from forcebench.task_api import SceneEntitySpec
from forcebench.tasks.faucet_turn.runtime import FaucetTurnTask

from . import MANIFEST
from .outcome import evaluate_report
from .spec import (
    SPEC,
    directional_resistance_nm,
    evaluate_turn_angle,
    flow_fraction,
    sample_initial_state,
    spec_from_layout,
    turn_effort_nm,
)


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
                friction_combine_mode=SPEC.valve_friction_combine_mode,
                restitution_combine_mode="min",
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=SPEC.valve_position_m,
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={SPEC.turn_joint_name: 0.0},
            joint_vel={SPEC.turn_joint_name: 0.0},
        ),
        actuators={
            "directional_turn_load": ImplicitActuatorCfg(
                joint_names_expr=[SPEC.turn_joint_name],
                stiffness=0.0,
                damping=0.015,
                # A small physical joint friction holds the selected valve
                # state after release.  The much larger direction-dependent
                # load below still defines the probe/switch signal and drop.
                friction=0.040,
                # The Coulomb-like load is updated at 60 Hz.  A modest
                # reflected inertia prevents low-speed +/- one-step chatter
                # after the gripper releases the lever, without changing the
                # requested resistance-versus-angle curve.
                armature=0.05,
                effort_limit_sim=SPEC.turn_max_effort_nm,
                velocity_limit_sim=12.0,
            ),
        },
    )


class FaucetForceTurnTask(FaucetTurnTask):
    """Reuse the accepted faucet scene while replacing only its valve model."""

    manifest = MANIFEST

    def __init__(self, asset_dir):
        super().__init__(asset_dir)
        self._turn_joint_id: int | None = None
        self._correct_direction_detent_crossed = False
        self._wrong_direction_detent_crossed = False
        self._positive_direction_attempted = False
        self._negative_direction_attempted = False
        self._correct_direction_detent_crossed_batch: list[bool] = []
        self._wrong_direction_detent_crossed_batch: list[bool] = []
        self._positive_direction_attempted_batch: list[bool] = []
        self._negative_direction_attempted_batch: list[bool] = []

    def _ensure_batch_state(self, num_envs: int) -> None:
        super()._ensure_batch_state(num_envs)
        if len(self._correct_direction_detent_crossed_batch) == num_envs:
            return
        self._episode_specs = [SPEC for _ in range(num_envs)]
        self._correct_direction_detent_crossed_batch = [False for _ in range(num_envs)]
        self._wrong_direction_detent_crossed_batch = [False for _ in range(num_envs)]
        self._positive_direction_attempted_batch = [False for _ in range(num_envs)]
        self._negative_direction_attempted_batch = [False for _ in range(num_envs)]

    def _sync_legacy_state(self) -> None:
        super()._sync_legacy_state()
        if not self._correct_direction_detent_crossed_batch:
            return
        self._correct_direction_detent_crossed = (
            self._correct_direction_detent_crossed_batch[0]
        )
        self._wrong_direction_detent_crossed = (
            self._wrong_direction_detent_crossed_batch[0]
        )
        self._positive_direction_attempted = self._positive_direction_attempted_batch[0]
        self._negative_direction_attempted = self._negative_direction_attempted_batch[0]

    def validate_assets(self) -> None:
        for relative_path in (
            SPEC.sink_asset,
            SPEC.sink_environment_asset,
            SPEC.sink_texture,
            SPEC.robot_support_asset,
            SPEC.valve_asset,
            "simready/cup09/cup09.usd",
        ):
            path = self.asset_dir / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"missing faucet_force_turn asset: {path}")

    def scene_entities(self):
        entity = SceneEntitySpec(
            role="faucet_valve",
            kind="articulation",
            config=_valve_config(self.asset_dir / SPEC.valve_asset),
            friction=None,
        )
        entity.validate()
        return (entity,)

    def sample_initial_state(self, seed: int) -> dict[str, object]:
        return sample_initial_state(seed)

    def after_scene_setup(self, env) -> None:
        # The parent applies the already accepted table/gripper collision
        # filters and installs a simulation-step callback. Dynamic dispatch
        # makes that callback invoke this class's _physics_tick.
        super().after_scene_setup(env)
        entity = env.entity("faucet_valve")
        self._turn_joint_id = entity.joint_names.index(SPEC.turn_joint_name)
        self._joint_id = self._turn_joint_id

    def _physics_tick(self, env) -> None:
        if self._turn_joint_id is None:
            return
        entity = env.entity("faucet_valve")
        positions = wp.to_torch(entity.root_view.get_dof_positions())
        velocities = wp.to_torch(entity.root_view.get_dof_velocities())
        angle = positions[:, self._turn_joint_id]
        angular_velocity = velocities[:, self._turn_joint_id]
        efforts = torch.empty((env.num_envs, 1), dtype=torch.float32, device=env.device)
        for env_index in range(env.num_envs):
            efforts[env_index, 0] = turn_effort_nm(
                float(angle[env_index].item()),
                float(angular_velocity[env_index].item()),
                self._episode_specs[env_index],
            )
        entity.set_joint_effort_target_index(
            target=efforts,
            joint_ids=[self._turn_joint_id],
        )

        for env_index in range(env.num_envs):
            episode_spec = self._episode_specs[env_index]
            angle_value = float(angle[env_index].item())
            velocity_value = float(angular_velocity[env_index].item())
            effort_value = float(efforts[env_index, 0].item())
            angle_report = evaluate_turn_angle(angle_value, episode_spec)

            self._last_angle_rad_batch[env_index] = angle_value
            self._last_velocity_rad_s_batch[env_index] = velocity_value
            self._last_effort_nm_batch[env_index] = effort_value
            self._last_resistance_nm_batch[env_index] = directional_resistance_nm(
                angle_value,
                episode_spec,
                direction_hint=velocity_value,
            )
            self._peak_abs_effort_nm_batch[env_index] = max(
                self._peak_abs_effort_nm_batch[env_index], abs(effort_value)
            )
            self._correct_direction_detent_crossed_batch[env_index] = bool(
                self._correct_direction_detent_crossed_batch[env_index]
                or angle_report["correct_direction_detent_crossed"]
            )
            self._wrong_direction_detent_crossed_batch[env_index] = bool(
                self._wrong_direction_detent_crossed_batch[env_index]
                or angle_report["wrong_direction_detent_crossed"]
            )
            threshold = math.radians(episode_spec.direction_attempt_threshold_deg)
            if angle_value >= threshold:
                self._positive_direction_attempted_batch[env_index] = True
            elif angle_value <= -threshold:
                self._negative_direction_attempted_batch[env_index] = True
            if self._correct_direction_detent_crossed_batch[env_index]:
                self._detent_crossed_batch[env_index] = True
            self._update_water_visual(env_index, angle_value, episode_spec)
        self._sync_legacy_state()

    def _update_water_visual(self, env_index: int, angle_rad: float, episode_spec) -> None:
        flow = (
            max(
                episode_spec.minimum_flow_fraction,
                flow_fraction(angle_rad, episode_spec),
            )
            if self._correct_direction_detent_crossed_batch[env_index]
            else 0.0
        )
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
        env_index_list = env_ids.detach().cpu().tolist()
        for env_index in env_index_list:
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
        for env_index in env_index_list:
            self._last_angle_rad_batch[env_index] = 0.0
            self._last_velocity_rad_s_batch[env_index] = 0.0
            self._last_effort_nm_batch[env_index] = 0.0
            self._last_resistance_nm_batch[env_index] = episode_spec.base_friction_torque_nm
            self._peak_abs_effort_nm_batch[env_index] = 0.0
            self._correct_direction_detent_crossed_batch[env_index] = False
            self._wrong_direction_detent_crossed_batch[env_index] = False
            self._positive_direction_attempted_batch[env_index] = False
            self._negative_direction_attempted_batch[env_index] = False
            self._detent_crossed_batch[env_index] = False
            self._last_flow_fraction_batch[env_index] = 0.0
            self._water_visible_batch[env_index] = False
        self._sync_legacy_state()

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        self._ensure_batch_state(env.num_envs)
        env_id = int(env_id)
        episode_spec = self._episode_specs[env_id]
        report = evaluate_turn_angle(self._last_angle_rad_batch[env_id], episode_spec)
        report["correct_direction_detent_crossed"] = (
            self._correct_direction_detent_crossed_batch[env_id]
        )
        report["wrong_direction_detent_crossed"] = (
            self._wrong_direction_detent_crossed_batch[env_id]
        )
        report["water_visible"] = self._water_visible_batch[env_id]
        report["valve_angular_velocity_rad_s"] = self._last_velocity_rad_s_batch[env_id]
        report["load_effort_nm"] = self._last_effort_nm_batch[env_id]
        report["load_resistance_nm"] = self._last_resistance_nm_batch[env_id]
        gripper_open_fraction = float(
            env.gripper_open_fraction_tensor[env_id].item()
        )
        tcp_height_m = float(env.fingertip_midpoint_pos[env_id, 2].item())
        report["gripper_open_fraction"] = gripper_open_fraction
        report["min_gripper_open_fraction"] = (
            episode_spec.min_gripper_open_fraction
        )
        report["gripper_released"] = bool(
            gripper_open_fraction >= episode_spec.min_gripper_open_fraction
        )
        report["tcp_height_m"] = tcp_height_m
        report["minimum_final_tcp_height_m"] = (
            episode_spec.minimum_final_tcp_height_m
        )
        report["tcp_lifted_clear"] = bool(
            tcp_height_m >= episode_spec.minimum_final_tcp_height_m
        )
        report["max_final_valve_speed_rad_s"] = (
            episode_spec.max_final_valve_speed_rad_s
        )
        report["valve_stationary"] = bool(
            abs(self._last_velocity_rad_s_batch[env_id])
            <= episode_spec.max_final_valve_speed_rad_s
        )
        report["positive_direction_attempted"] = (
            self._positive_direction_attempted_batch[env_id]
        )
        report["negative_direction_attempted"] = (
            self._negative_direction_attempted_batch[env_id]
        )
        report["direction_switch_observed"] = bool(
            report["positive_direction_attempted"]
            and report["negative_direction_attempted"]
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
            "objects/faucet_valve/turn_position": entity.data.joint_pos.torch[
                0, self._turn_joint_id : self._turn_joint_id + 1
            ],
        }

    def wrench_reference_position(self, env):
        self._ensure_batch_state(env.num_envs)
        return torch.tensor(
            [
                (
                    position[0] + episode_spec.lever_pivot_offset_m[0],
                    position[1] + episode_spec.lever_pivot_offset_m[1],
                    position[2] + episode_spec.lever_pivot_offset_m[2],
                )
                for env_index, (position, episode_spec) in enumerate(
                    zip(self._valve_positions_m, self._episode_specs)
                )
            ],
            dtype=torch.float32,
            device=env.device,
        )

    def debug_state(self, env):
        return {
            "turn_joint_name": SPEC.turn_joint_name,
            "angle_rad": self._last_angle_rad,
            "angular_velocity_rad_s": self._last_velocity_rad_s,
            "load_effort_nm": self._last_effort_nm,
            "load_resistance_nm": self._last_resistance_nm,
            "peak_abs_effort_nm": self._peak_abs_effort_nm,
            "correct_direction_detent_crossed": self._correct_direction_detent_crossed,
            "wrong_direction_detent_crossed": self._wrong_direction_detent_crossed,
            "direction_switch_observed": bool(
                self._positive_direction_attempted
                and self._negative_direction_attempted
            ),
            "easy_direction": self._episode_spec.easy_direction_name,
            "easy_direction_sign": self._episode_spec.easy_direction_sign,
            "hard_direction": self._episode_spec.hard_direction_name,
            "hard_direction_sign": self._episode_spec.hard_direction_sign,
            "detent_center_magnitude_deg": (
                self._episode_spec.detent_center_magnitude_deg
            ),
            "target_final_angle_deg": self._episode_spec.target_final_angle_deg,
            "water_visible": self._water_visible,
        }

    def episode_metadata(self) -> dict[str, object]:
        spec = self._episode_spec
        return {
            "phase": SPEC.phase,
            "scene_profile": "kitchen_sink",
            "asset_usd_paths": {
                "sink": str(self.asset_dir / SPEC.sink_asset),
                "sink_environment": str(self.asset_dir / SPEC.sink_environment_asset),
                "faucet_valve": str(self.asset_dir / SPEC.valve_asset),
            },
            "source_task": "faucet_turn",
            "source_task_modified": True,
            "mechanism_dofs": [spec.turn_joint_name],
            "expert_grasp_radius_m": spec.expert_grasp_radius_m,
            "success_contract": {
                "signed_final_angle_band_after_detent_deg": [0.6, 2.1],
                "min_gripper_open_fraction": spec.min_gripper_open_fraction,
                "min_final_tcp_lift_m": spec.min_final_tcp_lift_m,
                "minimum_final_tcp_height_m": spec.minimum_final_tcp_height_m,
                "max_final_valve_speed_rad_s": spec.max_final_valve_speed_rad_s,
            },
            "load_model": {
                "type": (
                    "center_continuous_early_split_gradual_rise_"
                    "sudden_drop_plus_damping"
                ),
                "physics_rate_hz": 60,
                "easy_direction": spec.easy_direction_name,
                "easy_direction_sign": spec.easy_direction_sign,
                "hard_direction": spec.hard_direction_name,
                "hard_direction_sign": spec.hard_direction_sign,
                "resistance_peak_magnitude_deg": spec.detent_center_magnitude_deg,
                "resistance_peak_angle_range_deg": list(
                    spec.detent_center_range_deg
                ),
                "direction_split_width_deg": spec.direction_split_width_deg,
                "detent_drop_width_deg": spec.detent_drop_width_deg,
                "easy_shoulder_torque_nm": spec.easy_shoulder_torque_nm,
                "hard_shoulder_torque_nm": spec.hard_shoulder_torque_nm,
                "easy_peak_torque_nm": spec.easy_peak_torque_nm,
                "hard_peak_torque_nm": spec.hard_peak_torque_nm,
                "hard_to_easy_peak_ratio": (
                    spec.hard_peak_torque_nm / spec.easy_peak_torque_nm
                ),
                "base_friction_torque_nm": spec.base_friction_torque_nm,
                "post_drop_resistance_nm": spec.post_drop_resistance_nm,
                "final_angle_tolerance_deg": spec.final_angle_tolerance_deg,
                "max_effort_nm": spec.turn_max_effort_nm,
                "contact_friction": spec.valve_friction,
                "friction_combine_mode": spec.valve_friction_combine_mode,
            },
        }
