"""Simulator configuration owned entirely by ForceBench."""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from .control import (
    DEFAULT_SMOOTH_POSITION_CONFIG,
    DEFAULT_SERVO_LIMITS,
    JointServoLimits,
)
from .scene_profiles import DEFAULT_SCENE_PROFILE, get_scene_profile


ASSET_DIR = Path(os.environ.get("FORCEBENCH_ASSET_DIR", "/home/pc/ForceBench/assets")).resolve()
# One benchmark-wide reset pose.  Individual tasks may not override it: object
# layouts must be designed around a common robot start state.
START_JOINT_POSITIONS = (
    0.0086831,
    0.0687144,
    -0.0087775,
    -2.3834085,
    0.0004833,
    2.4512002,
    0.7849010,
)
ROBOT_CONTACT_FRICTION = 1.5
FRICTION_COMBINE_MODE = "min"
GRIPPER_VELOCITY_LIMIT_M_S = 0.08
ARM_DRIVE_STIFFNESS = DEFAULT_SMOOTH_POSITION_CONFIG.actuator_stiffness
ARM_DRIVE_DAMPING = DEFAULT_SMOOTH_POSITION_CONFIG.actuator_damping


def _robot_config() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_DIR / "franka_mimic.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=2.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=2,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00025, rest_offset=0.0),
            physics_material=RigidBodyMaterialCfg(
                static_friction=ROBOT_CONTACT_FRICTION,
                dynamic_friction=ROBOT_CONTACT_FRICTION,
                restitution=0.0,
                friction_combine_mode=FRICTION_COMBINE_MODE,
                restitution_combine_mode="min",
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": START_JOINT_POSITIONS[0],
                "panda_joint2": START_JOINT_POSITIONS[1],
                "panda_joint3": START_JOINT_POSITIONS[2],
                "panda_joint4": START_JOINT_POSITIONS[3],
                "panda_joint5": START_JOINT_POSITIONS[4],
                "panda_joint6": START_JOINT_POSITIONS[5],
                "panda_joint7": START_JOINT_POSITIONS[6],
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=ARM_DRIVE_STIFFNESS,
                damping=ARM_DRIVE_DAMPING,
                friction=0.0,
                armature=1.0e-3,
                effort_limit_sim=87.0,
                velocity_limit_sim=124.6,
            ),
            "arm_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=ARM_DRIVE_STIFFNESS,
                damping=ARM_DRIVE_DAMPING,
                friction=0.0,
                armature=1.0e-3,
                effort_limit_sim=12.0,
                velocity_limit_sim=149.5,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=30.0,
                velocity_limit_sim=GRIPPER_VELOCITY_LIMIT_M_S,
                stiffness=4000.0,
                damping=120.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )


@configclass
class ForceBenchEnvCfg(DirectRLEnvCfg):
    """One-env arm-control configuration shared by every task plugin."""

    decimation = 4
    action_space = 7
    observation_space = 7
    state_space = 0
    episode_length_s = 24.0 * 60.0 * 60.0
    start_joint_positions = START_JOINT_POSITIONS
    robot_friction = ROBOT_CONTACT_FRICTION
    robot_inertia_increment = 0.01
    servo_max_velocity_rad_s = DEFAULT_SERVO_LIMITS.max_velocity_rad_s
    servo_max_acceleration_rad_s2 = DEFAULT_SERVO_LIMITS.max_acceleration_rad_s2
    servo_max_jerk_rad_s3 = DEFAULT_SERVO_LIMITS.max_jerk_rad_s3
    servo_min_duration_s = DEFAULT_SERVO_LIMITS.min_duration_s
    servo_constraint_samples = DEFAULT_SERVO_LIMITS.constraint_samples
    servo_arm_stiffness = DEFAULT_SMOOTH_POSITION_CONFIG.actuator_stiffness
    servo_arm_damping = DEFAULT_SMOOTH_POSITION_CONFIG.actuator_damping
    smooth_ik_dls_damping = DEFAULT_SMOOTH_POSITION_CONFIG.dls_damping
    smooth_ik_max_position_error_m = (
        DEFAULT_SMOOTH_POSITION_CONFIG.max_position_error_m
    )
    smooth_ik_max_rotation_error_rad = (
        DEFAULT_SMOOTH_POSITION_CONFIG.max_rotation_error_rad
    )
    smooth_ik_max_joint_step_rad = DEFAULT_SMOOTH_POSITION_CONFIG.max_joint_step_rad
    scene_profile = DEFAULT_SCENE_PROFILE

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, -9.81),
        render_interval=decimation,
        physics=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=2,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode=FRICTION_COMBINE_MODE,
            restitution_combine_mode="min",
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    robot = _robot_config()


def configure_runtime(
    cfg: ForceBenchEnvCfg,
    *,
    device: str,
    physics_rate_hz: int,
    solver_iterations: int,
    num_envs: int = 1,
    env_spacing: float = 2.0,
) -> None:
    """Apply collection-rate settings without any task-specific assumptions."""
    robot_path = ASSET_DIR / "franka_mimic.usd"
    if not robot_path.is_file():
        raise FileNotFoundError(f"missing ForceBench robot asset: {robot_path}")
    physics_rate_hz = int(physics_rate_hz)
    if physics_rate_hz != 60:
        raise ValueError("ForceBench collection physics_rate_hz is fixed to 60")
    if int(solver_iterations) != 64:
        raise ValueError("ForceBench solver_position_iterations is fixed to 64")
    num_envs = int(num_envs)
    env_spacing = float(env_spacing)
    if num_envs <= 0:
        raise ValueError("ForceBench num_envs must be positive")
    if env_spacing <= 0.0:
        raise ValueError("ForceBench env_spacing must be positive")
    get_scene_profile(cfg.scene_profile)
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = env_spacing
    cfg.scene.replicate_physics = True
    cfg.scene.filter_collisions = True
    cfg.scene.clone_in_fabric = False
    cfg.sim.dt = 1.0 / physics_rate_hz
    cfg.decimation = physics_rate_hz // 15
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics.max_position_iteration_count = int(solver_iterations)
    cfg.robot.spawn.usd_path = str(robot_path)
    cfg.robot.spawn.rigid_props.solver_position_iteration_count = int(solver_iterations)
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = int(solver_iterations)
    for name in ("arm_shoulder", "arm_forearm"):
        cfg.robot.actuators[name].stiffness = float(cfg.servo_arm_stiffness)
        cfg.robot.actuators[name].damping = float(cfg.servo_arm_damping)

    JointServoLimits(
        max_velocity_rad_s=float(cfg.servo_max_velocity_rad_s),
        max_acceleration_rad_s2=float(cfg.servo_max_acceleration_rad_s2),
        max_jerk_rad_s3=float(cfg.servo_max_jerk_rad_s3),
        min_duration_s=float(cfg.servo_min_duration_s),
        constraint_samples=int(cfg.servo_constraint_samples),
    ).validate()
    for name in (
        "servo_arm_stiffness",
        "servo_arm_damping",
        "smooth_ik_dls_damping",
        "smooth_ik_max_position_error_m",
        "smooth_ik_max_rotation_error_rad",
        "smooth_ik_max_joint_step_rad",
    ):
        if float(getattr(cfg, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")
