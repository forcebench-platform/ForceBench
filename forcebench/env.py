"""Task-agnostic ForceBench environment built directly on Isaac Lab."""

from __future__ import annotations

import os

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import combine_frame_transforms

from .control import (
    ARM_CONTROL_MODE,
    JointServoLimits,
    SmoothPositionConfig,
    SynchronizedQuinticTrajectory,
    shortest_angular_delta,
    unwrap_joint_positions,
)
from .physics import add_diagonal_body_inertia, set_asset_friction
from .scene_profiles import get_scene_profile
from .task_api import SceneEntitySpec, TaskRuntime


def _change_wrench_frame(source_force, source_torque, source_position, target_position):
    displacement = target_position - source_position
    return source_force, source_torque + torch.cross(displacement, source_force, dim=-1)


class ForceBenchEnv(DirectRLEnv):
    """Shared robot, camera, teleoperation and recording runtime for all demos."""

    _CAMERA_DATA_TYPES = {
        "rgb",
        "simple_shading_constant_diffuse",
        "simple_shading_diffuse_mdl",
        "simple_shading_full_mdl",
    }

    def __init__(
        self,
        cfg,
        render_mode=None,
        *,
        task_runtime: TaskRuntime,
        image_size=None,
        main_image_size=224,
        wrist_image_size=224,
        camera_data_type="rgb",
        enable_cameras=True,
        **kwargs,
    ):
        self._enable_cameras = bool(enable_cameras)
        configured_num_envs = int(cfg.scene.num_envs)
        if configured_num_envs <= 0:
            raise ValueError("ForceBench requires at least one environment")
        if camera_data_type not in self._CAMERA_DATA_TYPES:
            raise ValueError(f"unsupported camera data type: {camera_data_type}")
        self.task = task_runtime
        # The manifest is the one authoritative visual contract for a demo.
        # Launchers and environment variables cannot silently change it.
        cfg.scene_profile = self.task.manifest.scene_profile
        self._arm_control_mode = ARM_CONTROL_MODE
        self._arm_trajectory: SynchronizedQuinticTrajectory | None = None
        self._joint_target = None
        self._joint_command = None
        self._raw_tcp_rpy: torch.Tensor | None = None
        self._continuous_tcp_rpy: torch.Tensor | None = None
        wrist_rotation_deg = int(self.task.camera_profile.wrist_image_rotation_deg)
        if wrist_rotation_deg % 90:
            raise ValueError("wrist image rotation must be a multiple of 90 degrees")
        self._wrist_image_rotation_quadrants = (wrist_rotation_deg // 90) % 4
        if image_size is not None:
            main_image_size = wrist_image_size = int(image_size)
        self._main_image_size = int(main_image_size)
        self._wrist_image_size = int(wrist_image_size)
        if self._main_image_size < 16 or self._wrist_image_size < 16:
            raise ValueError("camera image sizes must be at least 16 pixels")
        self._camera_data_type = camera_data_type
        self._gripper_target = torch.ones(configured_num_envs, device=cfg.sim.device)
        self._gripper_start = torch.ones_like(self._gripper_target)
        self._gripper_command = torch.ones_like(self._gripper_target)
        self._teleop_substep = 0
        self._capture_wrench = False
        self._physics_timestamps = []
        self._physics_wrench_joint = []
        self._sim_step_counter = 0
        self._current_position_error_m = 0.0
        self._current_rotation_error_rad = 0.0
        self._max_position_error_m = 0.0
        self._max_rotation_error_rad = 0.0
        self._control_steps = 0
        self._physics_start_step = 0
        self._last_physics_sample_step = -1
        self._active_seed = int(cfg.seed or 0)
        self._pending_seed = self._active_seed
        self._layout = self.task.sample_initial_state(self._active_seed)
        self._active_seeds = [self._active_seed] * configured_num_envs
        self._pending_seeds = list(self._active_seeds)
        self._layouts = [dict(self._layout) for _ in range(configured_num_envs)]
        self._entities: dict[str, Articulation | RigidObject] = {}
        self._entity_specs: dict[str, SceneEntitySpec] = {}
        zero = torch.zeros(1, device=cfg.sim.device)
        identity = math_utils.quat_from_euler_xyz(zero, zero, zero)[0]
        self._runtime_xyzw = bool(identity[-1].abs() > 0.5)
        if not self._runtime_xyzw:
            raise RuntimeError("ForceBench requires Isaac Lab XYZW quaternions")

        super().__init__(cfg, render_mode, **kwargs)

        set_asset_friction(self._robot, float(cfg.robot_friction), self.scene.num_envs)
        add_diagonal_body_inertia(self._robot, float(cfg.robot_inertia_increment), self.scene.num_envs)
        for role, spec in self._entity_specs.items():
            if spec.friction is not None:
                set_asset_friction(self._entities[role], spec.friction, self.scene.num_envs)

        self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")
        self._hand_index = self._robot.body_names.index("panda_hand")
        # PhysX reports the force transmitted from the parent to this link by
        # its incoming joint.  For the force_sensor link this is the wrist
        # panda_hand_joint fixed joint, and the six-vector is expressed in that
        # joint frame.  This is the canonical ForceBench wrench; no pose-derived
        # coordinate transform or gravity compensation is applied to it.
        self.force_sensor_joint_name = "panda_hand_joint"
        self.force_sensor_joint_raw = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_joint_smooth = torch.zeros_like(self.force_sensor_joint_raw)
        self._latest_solved_joint_wrench = torch.zeros(
            (self.num_envs, len(self._robot.body_names), 6), device=self.device
        )
        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_joint_raw)

        # Derived values are internal diagnostics for scripted experts.  They
        # are not observation fields and are not persisted in official HDF5.
        self.force_sensor_world = torch.zeros_like(self.force_sensor_joint_raw)
        self.force_sensor_world_raw = torch.zeros_like(self.force_sensor_joint_raw)
        self.force_sensor_world_smooth = torch.zeros_like(self.force_sensor_joint_raw)
        self.force_sensor_tool_smooth = torch.zeros_like(self.force_sensor_joint_raw)
        self._arm_joint_ids = list(range(7))
        self._gripper_joint_ids = [7, 8]
        self.ctrl_target_joint_pos = self._robot.data.joint_pos.torch.clone()
        self._raw_arm_joint_pos = self.ctrl_target_joint_pos[:, :7].clone()
        self._continuous_arm_joint_pos = self._raw_arm_joint_pos.clone()
        self._smooth_position_config = SmoothPositionConfig(
            actuator_stiffness=float(cfg.servo_arm_stiffness),
            actuator_damping=float(cfg.servo_arm_damping),
            dls_damping=float(cfg.smooth_ik_dls_damping),
            max_position_error_m=float(cfg.smooth_ik_max_position_error_m),
            max_rotation_error_rad=float(cfg.smooth_ik_max_rotation_error_rad),
            max_joint_step_rad=float(cfg.smooth_ik_max_joint_step_rad),
            servo_limits=JointServoLimits(
                max_velocity_rad_s=float(cfg.servo_max_velocity_rad_s),
                max_acceleration_rad_s2=float(cfg.servo_max_acceleration_rad_s2),
                max_jerk_rad_s3=float(cfg.servo_max_jerk_rad_s3),
                min_duration_s=float(cfg.servo_min_duration_s),
                constraint_samples=int(cfg.servo_constraint_samples),
            ),
        )
        self._smooth_position_config.validate()
        self._joint_target = self.ctrl_target_joint_pos[:, :7].clone()
        self._joint_command = self._joint_target.clone()
        self._arm_trajectory = SynchronizedQuinticTrajectory(
            self._smooth_position_config.servo_limits
        )
        self._arm_trajectory.reset(self._joint_target)
        self.joint_pos = self._robot.data.joint_pos.torch.clone()
        self.joint_pos[:, :7] = self._continuous_arm_joint_pos
        self.joint_vel = self._robot.data.joint_vel.torch.clone()
        self.prev_joint_pos = self.joint_pos[:, :7].clone()
        self.fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.fingertip_midpoint_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_pos = torch.zeros_like(self.fingertip_midpoint_pos)
        self.prev_fingertip_quat = torch.zeros_like(self.fingertip_midpoint_quat)
        self.ee_linvel_fd = torch.zeros_like(self.fingertip_midpoint_pos)
        self.ee_angvel_fd = torch.zeros_like(self.fingertip_midpoint_pos)
        self.joint_vel_fd = torch.zeros_like(self.prev_joint_pos)
        self._wrist_pos_in_hand = torch.tensor(
            (self.task.camera_profile.wrist_position_in_hand_m,), device=self.device
        ).repeat(self.num_envs, 1)
        self._wrist_quat_in_hand = torch.tensor(
            (self._runtime_quaternion(self.task.camera_profile.wrist_rotation_wxyz),), device=self.device
        ).repeat(self.num_envs, 1)

        self._original_sim_step = self.sim.step

        def sim_step_with_wrench(*args, **kwargs):
            result = self._original_sim_step(*args, **kwargs)
            self._sim_step_counter += 1
            # Read the reaction produced by this completed solve. The arm state
            # is never overwritten; both supported controllers consume the
            # dynamically updated articulation state.
            self._latest_solved_joint_wrench = wp.to_torch(
                self._robot.root_view.get_link_incoming_joint_force()
            ).clone()
            self._capture_physics_wrench_sample()
            return result

        self.sim.step = sim_step_with_wrench
        self._update_intermediate_values(self.physics_dt)
        self._reset_arm_controller_from_current_state()
        self.prev_fingertip_pos.copy_(self.fingertip_midpoint_pos)
        self.prev_fingertip_quat.copy_(self.fingertip_midpoint_quat)
        self._update_wrench()
        self.task.after_scene_setup(self)

    def _runtime_quaternion(self, quaternion_wxyz):
        w, x, y, z = quaternion_wxyz
        return (x, y, z, w) if self._runtime_xyzw else (w, x, y, z)

    def _setup_scene(self):
        self._scene_profile = get_scene_profile(self.cfg.scene_profile)
        scene_profile = self._scene_profile
        self._official_asset_root = os.environ.get(
            "FORCEBENCH_ISAAC_ASSET_ROOT", ISAAC_NUCLEUS_DIR
        )

        def scene_asset(relative_path: str) -> str:
            local_prefix = "forcebench://assets/"
            if relative_path.startswith(local_prefix):
                local_path = self.task.asset_dir / relative_path[len(local_prefix) :]
                if not local_path.is_file():
                    raise FileNotFoundError(f"missing ForceBench scene asset: {local_path}")
                return str(local_path)
            if not self._official_asset_root:
                raise RuntimeError(
                    "Isaac Sim official asset root is unavailable; set FORCEBENCH_ISAAC_ASSET_ROOT "
                    "or repair the task manifest/asset mapping"
                )
            return f"{self._official_asset_root.rstrip('/')}/{relative_path.lstrip('/')}"

        if scene_profile.room_usd_path is not None:
            room_usd_path = scene_asset(scene_profile.room_usd_path)
            room_prim_path = (
                "/World/ForceBenchLab"
                if self.cfg.scene.num_envs == 1
                else "/World/envs/env_.*/ForceBenchLab"
            )
            room_env0_path = (
                "/World/ForceBenchLab"
                if self.cfg.scene.num_envs == 1
                else "/World/envs/env_0/ForceBenchLab"
            )
            room_cfg = sim_utils.UsdFileCfg(
                usd_path=room_usd_path,
                scale=scene_profile.room_scale,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
            room_cfg.func(
                room_prim_path,
                room_cfg,
                translation=scene_profile.room_translation_m,
                orientation=scene_profile.room_orientation_xyzw,
            )
            if scene_profile.hidden_room_prims:
                from isaaclab.sim.utils.prims import get_current_stage

                stage = get_current_stage()
                for child_name in scene_profile.hidden_room_prims:
                    prim = stage.GetPrimAtPath(f"{room_env0_path}/{child_name}")
                    if not prim.IsValid():
                        raise RuntimeError(
                            f"official room asset is missing expected prim {child_name!r}; "
                            "the selected Isaac asset pack may be incompatible"
                        )
                    # Deactivation removes both the visual and any authored
                    # collision under the official room's built-in table.
                    prim.SetActive(False)

        visual_root = (
            "/World/ForceBenchVisuals"
            if self.cfg.scene.num_envs == 1
            else "/World/envs/env_.*/ForceBenchVisuals"
        )
        if self.cfg.scene.num_envs > 1:
            sim_utils.create_prim("/World/envs/env_0/ForceBenchVisuals", prim_type="Xform")
        for box in scene_profile.visual_boxes:
            box_cfg = sim_utils.CuboidCfg(
                size=box.size_m,
                collision_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=box.color,
                    emissive_color=box.emissive_color,
                    roughness=box.roughness,
                    metallic=box.metallic,
                ),
            )
            box_cfg.func(
                f"{visual_root}/{box.name}",
                box_cfg,
                translation=box.translation_m,
            )

        ground_cfg = sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_profile.ground_color),
            visible=scene_profile.ground_visible,
        )
        ground_cfg.func(
            (
                "/World/ground"
                if self.cfg.scene.num_envs == 1
                else "/World/envs/env_.*/Ground"
            ),
            ground_cfg,
            translation=(0.0, 0.0, scene_profile.ground_center_z_m),
        )
        table_cfg = sim_utils.CuboidCfg(
            size=scene_profile.table_size_m,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=scene_profile.table_color,
                roughness=scene_profile.table_roughness,
                metallic=scene_profile.table_metallic,
            ),
            visible=scene_profile.table_visible,
        )
        table_cfg.func(
            "/World/envs/env_.*/Table",
            table_cfg,
            translation=scene_profile.table_translation_m,
        )
        if scene_profile.table_visual_usd_path is not None:
            table_visual_usd_path = scene_asset(scene_profile.table_visual_usd_path)
            table_visual_cfg = sim_utils.UsdFileCfg(
                usd_path=table_visual_usd_path,
                scale=scene_profile.table_visual_scale,
            )
            table_visual_cfg.func(
                "/World/envs/env_.*/TableVisual",
                table_visual_cfg,
                translation=scene_profile.table_visual_translation_m,
                orientation=scene_profile.table_visual_orientation_xyzw,
            )
            if scene_profile.table_visual_collision_prim is not None:
                # Some instanceable visuals keep authored colliders in a
                # dedicated branch. Deactivate that branch and retain the one
                # invariant ForceBench support collider at exactly z=0.
                from isaaclab.sim.utils.prims import get_current_stage

                table_collisions = get_current_stage().GetPrimAtPath(
                    "/World/envs/env_0/TableVisual/"
                    f"{scene_profile.table_visual_collision_prim}"
                )
                if not table_collisions.IsValid():
                    raise RuntimeError(
                        "external table visual is missing collision branch "
                        f"{scene_profile.table_visual_collision_prim!r}"
                    )
                table_collisions.SetActive(False)

        self._robot = Articulation(self.cfg.robot)
        solver_iterations = int(self.cfg.sim.physics.max_position_iteration_count)
        for spec in self.task.scene_entities():
            spec.validate()
            if spec.role in self._entities:
                raise ValueError(f"duplicate scene role: {spec.role}")
            rigid_props = getattr(getattr(spec.config, "spawn", None), "rigid_props", None)
            if rigid_props is not None:
                rigid_props.solver_position_iteration_count = solver_iterations
            entity = Articulation(spec.config) if spec.kind == "articulation" else RigidObject(spec.config)
            self._entities[spec.role] = entity
            self._entity_specs[spec.role] = spec

        self.task.spawn_scene_prims(self)
        if self._enable_cameras:
            profile = self.task.camera_profile

            def pinhole(focal_length, clipping_range):
                return sim_utils.PinholeCameraCfg(
                    focal_length=focal_length,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=clipping_range,
                )

            multi_camera = self.cfg.scene.num_envs > 1
            camera_class = TiledCamera if multi_camera else Camera
            camera_cfg_class = TiledCameraCfg if multi_camera else CameraCfg
            camera_prefix = "/World/envs/env_.*/" if multi_camera else "/World/envs/env_0/"
            self._wrist_camera = camera_class(
                camera_cfg_class(
                    prim_path=f"{camera_prefix}wrist_cam",
                    update_period=0.0,
                    height=self._wrist_image_size,
                    width=self._wrist_image_size,
                    data_types=[self._camera_data_type],
                    spawn=pinhole(profile.wrist_focal_length_mm, (0.05, 2.0)),
                    offset=CameraCfg.OffsetCfg(
                        pos=profile.wrist_position_in_hand_m,
                        rot=self._runtime_quaternion(profile.wrist_rotation_wxyz),
                        convention="opengl",
                    ),
                )
            )
            self._table_camera = camera_class(
                camera_cfg_class(
                    prim_path=f"{camera_prefix}table_cam",
                    update_period=0.0,
                    height=self._main_image_size,
                    width=self._main_image_size,
                    data_types=[self._camera_data_type],
                    spawn=pinhole(profile.table_focal_length_mm, (0.05, 6.0)),
                    offset=CameraCfg.OffsetCfg(
                        pos=profile.table_position_m,
                        rot=self._runtime_quaternion(profile.table_rotation_wxyz),
                        convention="opengl",
                    ),
                )
            )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()
        self.scene.articulations["robot"] = self._robot
        for role, entity in self._entities.items():
            if self._entity_specs[role].kind == "articulation":
                self.scene.articulations[role] = entity
            else:
                self.scene.rigid_objects[role] = entity
        light_cfg = sim_utils.DomeLightCfg(
            intensity=scene_profile.dome_light_intensity,
            color=scene_profile.dome_light_color,
        )
        light_cfg.func("/World/Light", light_cfg)

    @property
    def robot(self):
        return self._robot

    @property
    def arm_control_mode(self) -> str:
        return self._arm_control_mode

    @property
    def scene_profile_name(self) -> str:
        return self._scene_profile.name

    @property
    def scene_asset_paths(self) -> dict[str, str]:
        def resolved(path: str) -> str:
            local_prefix = "forcebench://assets/"
            if path.startswith(local_prefix):
                return str(self.task.asset_dir / path[len(local_prefix) :])
            return f"{self._official_asset_root.rstrip('/')}/{path.lstrip('/')}"

        paths: dict[str, str] = {}
        if self._scene_profile.room_usd_path is not None:
            paths["scene_room"] = resolved(self._scene_profile.room_usd_path)
        if self._scene_profile.table_visual_usd_path is not None:
            paths["scene_table_visual"] = resolved(self._scene_profile.table_visual_usd_path)
        return paths

    def entity(self, role: str):
        try:
            return self._entities[role]
        except KeyError as exc:
            raise KeyError(f"task {self.task.manifest.name!r} has no scene role {role!r}") from exc

    @property
    def entity_roles(self) -> tuple[str, ...]:
        return tuple(self._entities)

    def entity_world_pose(self, role: str):
        entity = self.entity(role)
        return torch.cat((entity.data.root_pos_w.torch, entity.data.root_quat_w.torch), dim=-1)

    def entity_local_pose(self, role: str):
        entity = self.entity(role)
        return entity.data.root_pos_w.torch - self.scene.env_origins, entity.data.root_quat_w.torch

    def set_entity_root_pose(self, role: str, env_ids, position_m, quaternion) -> None:
        entity = self.entity(role)
        pose = entity.data.default_root_pose.torch[env_ids].clone()
        velocity = torch.zeros_like(entity.data.default_root_vel.torch[env_ids])
        position = torch.as_tensor(position_m, dtype=torch.float32, device=self.device)
        if position.shape != (3,):
            raise ValueError(f"{role} position must have shape (3,), got {tuple(position.shape)}")
        quat = torch.as_tensor(quaternion, dtype=torch.float32, device=self.device)
        if quat.shape == (4,):
            quat = quat.unsqueeze(0).repeat(len(env_ids), 1)
        if quat.shape != (len(env_ids), 4):
            raise ValueError(f"{role} quaternion has invalid shape {tuple(quat.shape)}")
        pose[:, :3] = position.unsqueeze(0) + self.scene.env_origins[env_ids]
        pose[:, 3:7] = quat
        entity.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        entity.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        entity.reset(env_ids)

    def set_episode_seed(self, seed: int) -> None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._pending_seed = seed
        self._pending_seeds = [seed] * self.num_envs

    def set_episode_seeds(self, seeds) -> None:
        """Set one deterministic pending seed per cloned environment."""

        values = [int(seed) for seed in seeds]
        if len(values) != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} episode seeds, received {len(values)}"
            )
        if any(seed < 0 for seed in values) or len(set(values)) != len(values):
            raise ValueError("parallel episode seeds must be distinct non-negative integers")
        self._pending_seeds = values
        self._pending_seed = values[0]

    @property
    def active_seed(self) -> int:
        return self._active_seed

    @property
    def active_seeds(self) -> tuple[int, ...]:
        return tuple(self._active_seeds)

    @property
    def layout(self) -> dict[str, object]:
        return dict(self._layout)

    @property
    def layouts(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(layout) for layout in self._layouts)

    @staticmethod
    def _limit_vector_norm(value: torch.Tensor, maximum: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(value, dim=1, keepdim=True)
        return value * torch.clamp(float(maximum) / torch.clamp(norm, min=1.0e-9), max=1.0)

    def _reset_arm_controller_from_current_state(self) -> None:
        if self._arm_trajectory is None:
            raise RuntimeError("joint trajectory is not initialized")
        current = self._continuous_arm_joint_pos.clone()
        self._joint_target = current.clone()
        self._joint_command = current.clone()
        self._arm_trajectory.reset(current)
        self.ctrl_target_joint_pos[:, :7] = current
        self._teleop_substep = int(self.cfg.decimation)

    def _reset_tcp_continuity(self) -> None:
        self._raw_tcp_rpy = None
        self._continuous_tcp_rpy = None

    def _batch_value(self, value, width: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.shape == (width,):
            tensor = tensor.unsqueeze(0).repeat(self.num_envs, 1)
        if tensor.shape != (self.num_envs, width) or not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} must be finite [{width}] or [{self.num_envs}, {width}]"
            )
        return tensor

    def _batch_gripper(self, value) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.repeat(self.num_envs)
        if tensor.shape != (self.num_envs,) or not torch.isfinite(tensor).all():
            raise ValueError(
                f"gripper target must be finite scalar or [{self.num_envs}]"
            )
        return torch.clamp(tensor, 0.0, 1.0)

    def set_cartesian_teleop_target(
        self,
        target_position_m,
        target_quaternion_xyzw,
        gripper_open_fraction: float,
    ) -> None:
        """Solve one bounded absolute pose with the canonical legacy DLS IK."""
        position = self._batch_value(target_position_m, 3, "Cartesian target position")
        quaternion = self._batch_value(
            target_quaternion_xyzw, 4, "Cartesian target quaternion"
        )
        quaternion_norm = torch.linalg.vector_norm(quaternion, dim=1, keepdim=True)
        if bool(torch.any(quaternion_norm < 1.0e-8)):
            raise ValueError("Cartesian target quaternion cannot be zero")
        quaternion = quaternion / quaternion_norm
        requested_position = position
        requested_quaternion = quaternion
        current_position = self.fingertip_midpoint_pos
        current_quaternion = self.fingertip_midpoint_quat
        position_error, rotation_error = math_utils.compute_pose_error(
            current_position,
            current_quaternion,
            requested_position,
            requested_quaternion,
            rot_error_type="axis_angle",
        )
        position_norm = float(torch.linalg.vector_norm(position_error, dim=1).max().item())
        rotation_norm = float(torch.linalg.vector_norm(rotation_error, dim=1).max().item())
        self._current_position_error_m = position_norm
        self._current_rotation_error_rad = rotation_norm
        self._max_position_error_m = max(self._max_position_error_m, position_norm)
        self._max_rotation_error_rad = max(self._max_rotation_error_rad, rotation_norm)
        self._control_steps += 1

        limited_error = torch.cat(
            (
                self._limit_vector_norm(
                    position_error, self._smooth_position_config.max_position_error_m
                ),
                self._limit_vector_norm(
                    rotation_error, self._smooth_position_config.max_rotation_error_rad
                ),
            ),
            dim=1,
        )
        jacobian = self._robot.data.body_link_jacobian_w.torch[
            :, self.fingertip_body_idx - 1, :6, :7
        ]
        regularizer = (self._smooth_position_config.dls_damping**2) * torch.eye(
            6, dtype=jacobian.dtype, device=self.device
        ).unsqueeze(0)
        delta_joint_position = jacobian.transpose(1, 2) @ torch.linalg.solve(
            jacobian @ jacobian.transpose(1, 2) + regularizer,
            limited_error.unsqueeze(-1),
        )
        delta_joint_position = torch.clamp(
            delta_joint_position.squeeze(-1),
            -self._smooth_position_config.max_joint_step_rad,
            self._smooth_position_config.max_joint_step_rad,
        )
        self.set_joint_teleop_target(
            self._robot.data.joint_pos.torch[:, :7] + delta_joint_position,
            gripper_open_fraction,
        )

    def set_joint_teleop_target(self, joint_target, gripper_open_fraction: float) -> None:
        """Set a bounded joint target for the implicit position drive."""
        if self._arm_trajectory is None:
            raise RuntimeError("joint trajectory is not initialized")
        target = self._batch_value(joint_target, 7, "joint target")
        limits = self._robot.data.soft_joint_pos_limits.torch[:, :7]
        target = torch.clamp(target, limits[..., 0], limits[..., 1])
        if self._joint_command is None:
            self._joint_command = self._continuous_arm_joint_pos.clone()
        self._joint_target = target
        self._arm_trajectory.plan(target)
        self._joint_command = self._arm_trajectory.position.clone()
        self._gripper_start = self._gripper_command.clone()
        self._gripper_target = self._batch_gripper(gripper_open_fraction)
        self._teleop_substep = 0

    def set_cartesian_rpy_target(self, target_xyz_rpy, gripper_open_fraction: float) -> None:
        target = self._batch_value(target_xyz_rpy, 6, "Cartesian XYZ/RPY target")
        quaternion = math_utils.quat_from_euler_xyz(
            target[:, 3], target[:, 4], target[:, 5]
        )
        self.set_cartesian_teleop_target(
            target[:, :3], quaternion, gripper_open_fraction
        )

    def reset(self, *args, **kwargs):
        # A persistent expert worker reuses this object across episodes.  Clear
        # every process-local sensor accumulator before DirectRLEnv advances
        # the freshly restored scene, otherwise the next episode starts with
        # the previous episode's EMA and high-rate capture state.
        if hasattr(self, "_capture_wrench"):
            self.discard_wrench_capture()
        for name in (
            "force_sensor_joint_raw",
            "force_sensor_joint_smooth",
            "force_sensor_world",
            "force_sensor_world_raw",
            "force_sensor_world_smooth",
            "force_sensor_tool_smooth",
            "force_sensor_smooth",
            "force_sensor_task_uncompensated_smooth",
            "force_sensor_task_compensated_smooth",
            "_latest_solved_joint_wrench",
        ):
            value = getattr(self, name, None)
            if value is not None:
                value.zero_()
        self._current_position_error_m = 0.0
        self._current_rotation_error_rad = 0.0
        self._max_position_error_m = 0.0
        self._max_rotation_error_rad = 0.0
        self._control_steps = 0
        result = super().reset(*args, **kwargs)
        self._update_intermediate_values(self.physics_dt)
        self._reset_arm_controller_from_current_state()
        self._gripper_start = self.gripper_open_fraction_tensor.clone()
        self._gripper_target = self._gripper_start.clone()
        self._gripper_command = self._gripper_start.clone()
        self._teleop_substep = int(self.cfg.decimation)
        return result

    def _reset_idx(self, env_ids):
        DirectRLEnv._reset_idx(self, env_ids)
        reset_ids = [int(value) for value in env_ids.detach().cpu().tolist()]
        for env_id in reset_ids:
            seed = int(self._pending_seeds[env_id])
            self._active_seeds[env_id] = seed
            self._layouts[env_id] = self.task.sample_initial_state(seed)
        self._active_seed = int(self._active_seeds[0])
        self._pending_seed = int(self._pending_seeds[0])
        self._layout = dict(self._layouts[0])

        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, :7] = torch.tensor(self.cfg.start_joint_positions, device=self.device)
        joint_pos[:, 7:9] = 0.04
        joint_vel = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids] = joint_pos
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self._robot.set_joint_effort_target_index(
            target=torch.zeros((len(env_ids), 7), device=self.device),
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._raw_arm_joint_pos[env_ids] = joint_pos[:, :7]
        self._continuous_arm_joint_pos[env_ids] = joint_pos[:, :7]
        self._robot.reset(env_ids)
        for env_id in reset_ids:
            one_id = torch.tensor([env_id], dtype=torch.int64, device=self.device)
            self.task.reset_scene(self, one_id, self._layouts[env_id])
        self.actions.zero_()
        self._reset_tcp_continuity()
        self.step_sim_no_action()
        self.prev_joint_pos = self.joint_pos[:, :7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
        self.ee_angvel_fd.zero_()
        self.ee_linvel_fd.zero_()
        self._reset_arm_controller_from_current_state()

    @property
    def gripper_open_fraction_tensor(self):
        return torch.clamp(torch.mean(self._robot.data.joint_pos.torch[:, 7:9], dim=1) / 0.04, 0.0, 1.0)

    @property
    def tcp_rpy(self) -> torch.Tensor:
        """Continuous XYZ Euler representation of the measured TCP orientation."""
        if self._continuous_tcp_rpy is None:
            raise RuntimeError("TCP orientation has not been initialized")
        return self._continuous_tcp_rpy

    @property
    def policy_state(self) -> torch.Tensor:
        """Canonical measured TCP/gripper state consumed by data and policies."""
        return torch.cat(
            (
                self.fingertip_midpoint_pos,
                self.tcp_rpy,
                (2.0 * self.gripper_open_fraction_tensor - 1.0).unsqueeze(-1),
            ),
            dim=-1,
        )

    @property
    def gripper_open_fraction(self) -> float:
        return float(self.gripper_open_fraction_tensor[0].item())

    def restore_fixed_arm_pose(self, arm_joint_position) -> None:
        arm = torch.as_tensor(arm_joint_position, dtype=torch.float32, device=self.device)
        if arm.shape != (7,):
            raise ValueError("fixed arm pose must have shape (7,)")
        env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        joint_pos = self._robot.data.joint_pos.torch.clone()
        joint_pos[:, :7] = arm.unsqueeze(0)
        joint_pos[:, 7:9] = 0.04
        self.ctrl_target_joint_pos[:] = joint_pos
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self._robot.set_joint_effort_target_index(
            target=torch.zeros((len(env_ids), 7), device=self.device),
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
        self._raw_arm_joint_pos[env_ids] = joint_pos[:, :7]
        self._continuous_arm_joint_pos[env_ids] = joint_pos[:, :7]
        self._robot.reset()
        self._reset_tcp_continuity()
        self.step_sim_no_action()
        self._update_intermediate_values(self.physics_dt)
        self._reset_arm_controller_from_current_state()
        gripper = torch.ones(self.num_envs, device=self.device)
        self._gripper_target = gripper.clone()
        self._gripper_start = gripper.clone()
        self._gripper_command = gripper.clone()
        self._teleop_substep = int(self.cfg.decimation)

    def restore_joint_state(self, joint_position, joint_velocity=None) -> None:
        """Restore one settled nine-DOF pose before an isolated expert episode."""
        position = torch.as_tensor(joint_position, dtype=torch.float32, device=self.device)
        if position.shape != (9,) or not torch.isfinite(position).all():
            raise ValueError("restored joint position must be a finite [9] vector")
        if joint_velocity is None:
            velocity = torch.zeros_like(position)
        else:
            velocity = torch.as_tensor(joint_velocity, dtype=torch.float32, device=self.device)
            if velocity.shape != (9,) or not torch.isfinite(velocity).all():
                raise ValueError("restored joint velocity must be a finite [9] vector")
        env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        if len(env_ids) != 1:
            raise ValueError("fixed expert-pose restore currently supports one environment")
        position_batch = position.unsqueeze(0)
        velocity_batch = velocity.unsqueeze(0)
        self.ctrl_target_joint_pos[:] = position_batch
        self._robot.set_joint_position_target_index(target=position_batch, env_ids=env_ids)
        self._robot.set_joint_effort_target_index(
            target=torch.zeros((len(env_ids), 7), device=self.device),
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids,
        )
        self._robot.write_joint_position_to_sim_index(position=position_batch, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=velocity_batch, env_ids=env_ids)
        self._raw_arm_joint_pos[env_ids] = position_batch[:, :7]
        self._continuous_arm_joint_pos[env_ids] = position_batch[:, :7]
        self._robot.reset()
        self.force_sensor_joint_smooth.zero_()
        self.force_sensor_world_smooth.zero_()
        self.force_sensor_tool_smooth.zero_()
        self.force_sensor_smooth.zero_()
        self._reset_tcp_continuity()
        self.step_sim_no_action()
        self._update_intermediate_values(self.physics_dt)
        self._reset_arm_controller_from_current_state()
        gripper = float(torch.clamp(torch.mean(position[7:9]) / 0.04, 0.0, 1.0).item())
        gripper_batch = torch.full((self.num_envs,), gripper, device=self.device)
        self._gripper_target = gripper_batch.clone()
        self._gripper_start = gripper_batch.clone()
        self._gripper_command = gripper_batch.clone()
        self._teleop_substep = int(self.cfg.decimation)

    def _pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

    def _apply_action(self):
        if self._arm_trajectory is None or self._joint_target is None:
            raise RuntimeError("smooth-position controller is not initialized")
        steps = max(int(self.cfg.decimation), 1)
        self._teleop_substep = min(self._teleop_substep + 1, steps)
        alpha = self._teleop_substep / steps
        self._joint_command = self._arm_trajectory.step(self.physics_dt)
        self._gripper_command = self._gripper_start + alpha * (
            self._gripper_target - self._gripper_start
        )
        self.ctrl_target_joint_pos[:, :7] = self._joint_command
        self.ctrl_target_joint_pos[:, 7:9] = 0.04 * self._gripper_command.unsqueeze(-1)
        self._robot.set_joint_position_target_index(target=self.ctrl_target_joint_pos)

    def _get_rewards(self):
        # No task predicate is evaluated during a trajectory.
        return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

    def _get_dones(self):
        self._update_intermediate_values(self.physics_dt)
        self.task.on_control_step(self)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(timeout), timeout

    def _get_observations(self):
        return {"policy": self.policy_state}

    def _update_intermediate_values(self, dt):
        self.fingertip_midpoint_pos = (
            self._robot.data.body_pos_w.torch[:, self.fingertip_body_idx] - self.scene.env_origins
        )
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w.torch[:, self.fingertip_body_idx]
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(self.fingertip_midpoint_quat)
        raw_tcp_rpy = torch.stack((roll, pitch, yaw), dim=-1)
        if self._raw_tcp_rpy is None or self._continuous_tcp_rpy is None:
            self._continuous_tcp_rpy = raw_tcp_rpy.clone()
        else:
            self._continuous_tcp_rpy = self._continuous_tcp_rpy + shortest_angular_delta(
                raw_tcp_rpy, self._raw_tcp_rpy
            )
        self._raw_tcp_rpy = raw_tcp_rpy.clone()
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w.torch[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w.torch[:, self.fingertip_body_idx]
        raw_joint_pos = self._robot.data.joint_pos.torch.clone()
        self._continuous_arm_joint_pos = unwrap_joint_positions(
            raw_joint_pos[:, :7],
            self._raw_arm_joint_pos,
            self._continuous_arm_joint_pos,
        )
        self._raw_arm_joint_pos = raw_joint_pos[:, :7].clone()
        self.joint_pos = raw_joint_pos
        self.joint_pos[:, :7] = self._continuous_arm_joint_pos
        self.joint_vel = self._robot.data.joint_vel.torch.clone()
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        rot_diff = math_utils.quat_mul(
            self.fingertip_midpoint_quat, math_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff *= torch.sign(rot_diff[:, -1 if self._runtime_xyzw else 0]).unsqueeze(-1)
        self.ee_angvel_fd = math_utils.axis_angle_from_quat(rot_diff) / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()
        self.joint_vel_fd = (self.joint_pos[:, :7] - self.prev_joint_pos) / dt
        self.prev_joint_pos = self.joint_pos[:, :7].clone()
        if hasattr(self, "force_sensor_joint_raw"):
            self._update_wrench()

    def step_sim_no_action(self):
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._update_intermediate_values(self.physics_dt)

    def _joint_wrench_to_world(self, joint_wrench):
        joint_quaternion_world = self._robot.data.body_quat_w.torch[:, self.force_sensor_body_idx]
        world_force = math_utils.quat_apply(joint_quaternion_world, joint_wrench[:, :3])
        world_torque = math_utils.quat_apply(joint_quaternion_world, joint_wrench[:, 3:6])
        return torch.cat((world_force, world_torque), dim=-1)

    def _update_wrench(self):
        self.force_sensor_joint_raw = self._latest_solved_joint_wrench[
            :, self.force_sensor_body_idx
        ]
        self.force_sensor_joint_smooth = (
            0.25 * self.force_sensor_joint_raw + 0.75 * self.force_sensor_joint_smooth
        )

        # The policy-facing wrench is the incoming fixed-joint reaction in the
        # joint frame.  Keep raw and EMA-filtered variants separately.
        self.force_sensor_smooth = self.force_sensor_joint_smooth
        self.force_sensor_tool_smooth = self.force_sensor_joint_smooth

        self.force_sensor_world_raw = self._joint_wrench_to_world(self.force_sensor_joint_raw)
        self.force_sensor_world = self.force_sensor_world_raw
        self.force_sensor_world_smooth = self._joint_wrench_to_world(
            self.force_sensor_joint_smooth
        )
        reference = self.task.wrench_reference_position(self)
        force, torque = _change_wrench_frame(
            self.force_sensor_world_smooth[:, :3],
            self.force_sensor_world_smooth[:, 3:6],
            torch.zeros((self.num_envs, 3), device=self.device),
            reference,
        )
        self.force_sensor_task_uncompensated_smooth = torch.cat((force, torque), dim=-1)
        # Task/world channels are internal expert diagnostics.  They are not
        # part of the model observation or official storage contract.
        self.force_sensor_task_compensated_smooth = self.force_sensor_task_uncompensated_smooth

    def evaluate_final_success(self) -> dict[str, object]:
        return self.task.evaluate_final_state(self)

    @property
    def arm_controller_profile(self) -> dict[str, object]:
        return self._smooth_position_config.profile(
            physics_rate_hz=int(round(1.0 / self.physics_dt)),
            control_rate_hz=15,
        )

    @property
    def arm_servo_profile(self) -> dict[str, object]:
        """Backward-compatible name for the canonical controller profile."""
        return self.arm_controller_profile

    @property
    def wrench_profile(self) -> dict[str, object]:
        return {
            "source": "link_incoming_joint_force",
            "link": "force_sensor",
            "joint": self.force_sensor_joint_name,
            "frame": "incoming_joint",
            "direction": "parent_on_child",
            "component_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "policy_filter": {"type": "none"},
            "stored_control_channels": {
                "raw": {"filter": "none"},
                "ema": {"filter": "ema", "alpha": 0.25},
            },
            "coordinate_transform": "none",
            "gravity_compensation": False,
        }

    @property
    def arm_controller_diagnostics(self) -> dict[str, float | int]:
        return {
            "current_position_error_m": self._current_position_error_m,
            "current_rotation_error_rad": self._current_rotation_error_rad,
            "episode_max_position_error_m": self._max_position_error_m,
            "episode_max_rotation_error_rad": self._max_rotation_error_rad,
            "control_steps": self._control_steps,
        }

    def reset_cameras(self):
        if not self._enable_cameras:
            raise RuntimeError("cameras are disabled for this ForceBench environment")
        self._apply_wrist_mount()
        self._table_camera.reset()
        self._wrist_camera.reset()

    def update_cameras(self):
        if not self._enable_cameras:
            raise RuntimeError("cameras are disabled for this ForceBench environment")
        self._apply_wrist_mount()
        self.sim.render()
        self._table_camera.update(self.physics_dt, force_recompute=True)
        self._wrist_camera.update(self.physics_dt, force_recompute=True)

    def _apply_wrist_mount(self):
        if not self._enable_cameras:
            return
        pos, quat = combine_frame_transforms(
            self._robot.data.body_pos_w.torch[:, self._hand_index],
            self._robot.data.body_quat_w.torch[:, self._hand_index],
            self._wrist_pos_in_hand,
            self._wrist_quat_in_hand,
        )
        self._wrist_camera.set_world_poses(pos, quat, convention="opengl")

    @property
    def table_rgb(self):
        if not self._enable_cameras:
            raise RuntimeError("table camera is disabled")
        return self._table_camera.data.output[self._camera_data_type]

    @property
    def wrist_rgb(self):
        if not self._enable_cameras:
            raise RuntimeError("wrist camera is disabled")
        image = self._wrist_camera.data.output[self._camera_data_type]
        if self._wrist_image_rotation_quadrants:
            # torch.rot90 uses counter-clockwise-positive k; the profile uses
            # the conventional display direction where positive is clockwise.
            image = torch.rot90(
                image,
                k=-self._wrist_image_rotation_quadrants,
                dims=(1, 2),
            )
        return image

    def start_wrench_capture(self):
        self._physics_timestamps.clear()
        self._physics_wrench_joint.clear()
        self._physics_start_step = self._sim_step_counter
        self._last_physics_sample_step = self._sim_step_counter
        self._capture_wrench = True

    def _capture_physics_wrench_sample(self):
        if not self._capture_wrench or self._sim_step_counter <= self._last_physics_sample_step:
            return
        joint = self._latest_solved_joint_wrench[:, self.force_sensor_body_idx]
        # Derive time from the number of captured solves.  The simulation
        # context can wrap ``sim.step`` while environments are recreated, so a
        # global step counter is not a reliable clock for a per-episode stream.
        self._physics_timestamps.append((len(self._physics_timestamps) + 1) * self.physics_dt)
        self._physics_wrench_joint.append(joint.detach().clone())
        self._last_physics_sample_step = self._sim_step_counter

    def stop_wrench_capture(self):
        self._capture_wrench = False
        if not self._physics_timestamps:
            raise ValueError("physics-rate wrench capture is empty")
        wrench = torch.stack(self._physics_wrench_joint).cpu().numpy().astype(np.float32)
        if self.num_envs == 1:
            wrench = wrench[:, 0]
        return {
            "timestamp": np.asarray(self._physics_timestamps, dtype=np.float64),
            "eef_wrench_joint_raw": wrench,
        }

    def discard_wrench_capture(self):
        self._capture_wrench = False
        self._physics_timestamps.clear()
        self._physics_wrench_joint.clear()
