"""Isaac Lab runtime for the RADMANSO cabinet on the box-sort table."""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import math as math_utils

from forcebench.task_api import CameraProfile, SceneEntitySpec, TaskRuntime

from . import MANIFEST
from .outcome import evaluate_report
from .spec import (
    BOTTOM_JOINT,
    DRAWER_JOINTS,
    SPEC,
    evaluate_final_state,
    sample_initial_state,
)


def _orientation(roll: float, pitch: float, yaw: float, device: str = "cpu"):
    return math_utils.quat_from_euler_xyz(
        torch.tensor([float(roll)], device=device),
        torch.tensor([float(pitch)], device=device),
        torch.tensor([float(yaw)], device=device),
    )


def _cabinet_config(usd_path) -> ArticulationCfg:
    quaternion = _orientation(
        SPEC.cabinet_roll_rad,
        SPEC.cabinet_pitch_rad,
        SPEC.cabinet_yaw_rad,
    )[0]
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Task_cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            scale=(SPEC.cabinet_scale,) * 3,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=SPEC.cabinet_position_m,
            rot=tuple(float(value) for value in quaternion),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Start with every drawer held shut.  reset_scene releases only the
            # production middle drawer by changing its live gains.
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=[*DRAWER_JOINTS.values(), BOTTOM_JOINT],
                effort_limit_sim=120.0,
                velocity_limit_sim=1.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )


class RadmansoDrawerSortTableTask(TaskRuntime):
    manifest = MANIFEST
    camera_profile = CameraProfile(
        table_position_m=(0.95894094, 0.0, 0.635),
        table_target_m=(0.55, 0.0, 0.13),
        table_rotation_wxyz=(
            0.66654819,
            0.23603709,
            0.23603709,
            0.66654819,
        ),
    )

    def __init__(self, asset_dir):
        super().__init__(asset_dir)
        self._drawer_ids: dict[str, int] = {}
        self._selected_drawer = "middle"
        self._layout: dict[str, object] = {}
        self._wrench_bias = None
        self._latch_broken = False
        self._breakaway_frame: int | None = None
        self._steps_past_breakaway = 0
        self._resistance_force_n = SPEC.free_resistance_n
        self._peak_resistance_n = 0.0
        self._peak_axial_pull_n = 0.0
        self._frames_armed = 0
        self._stable_target_frames = 0
        self._max_opening_world_m = 0.0
        self._resistance_start_open_m = SPEC.resistance_start_open_m
        self._success_open_m = SPEC.success_open_m
        self._breakaway_open_m = SPEC.breakaway_open_m
        self._breakaway_resistance_n = SPEC.breakaway_resistance_n
        # One TaskRuntime instance is shared by all Isaac Lab clones.  The
        # scalar fields above remain the established env_0 API; these arrays
        # keep the mechanism state isolated for every cloned workspace.
        self._selected_drawers: list[str] = []
        self._layouts_batch: list[dict[str, object]] = []
        self._wrench_biases: list[torch.Tensor | None] = []
        self._latch_broken_batch: list[bool] = []
        self._breakaway_frames: list[int | None] = []
        self._steps_past_breakaway_batch: list[int] = []
        self._resistance_force_n_batch: list[float] = []
        self._peak_resistance_n_batch: list[float] = []
        self._peak_axial_pull_n_batch: list[float] = []
        self._frames_armed_batch: list[int] = []
        self._stable_target_frames_batch: list[int] = []
        self._max_opening_world_m_batch: list[float] = []
        self._resistance_start_open_m_batch: list[float] = []
        self._success_open_m_batch: list[float] = []
        self._breakaway_open_m_batch: list[float] = []
        self._breakaway_resistance_n_batch: list[float] = []

    def _ensure_batch_state(self, num_envs: int) -> None:
        if len(self._selected_drawers) == num_envs:
            return
        self._selected_drawers = ["middle" for _ in range(num_envs)]
        self._layouts_batch = [{} for _ in range(num_envs)]
        self._wrench_biases = [None for _ in range(num_envs)]
        self._latch_broken_batch = [False for _ in range(num_envs)]
        self._breakaway_frames = [None for _ in range(num_envs)]
        self._steps_past_breakaway_batch = [0 for _ in range(num_envs)]
        self._resistance_force_n_batch = [SPEC.free_resistance_n for _ in range(num_envs)]
        self._peak_resistance_n_batch = [0.0 for _ in range(num_envs)]
        self._peak_axial_pull_n_batch = [0.0 for _ in range(num_envs)]
        self._frames_armed_batch = [0 for _ in range(num_envs)]
        self._stable_target_frames_batch = [0 for _ in range(num_envs)]
        self._max_opening_world_m_batch = [0.0 for _ in range(num_envs)]
        self._resistance_start_open_m_batch = [SPEC.resistance_start_open_m for _ in range(num_envs)]
        self._success_open_m_batch = [SPEC.success_open_m for _ in range(num_envs)]
        self._breakaway_open_m_batch = [SPEC.breakaway_open_m for _ in range(num_envs)]
        self._breakaway_resistance_n_batch = [SPEC.breakaway_resistance_n for _ in range(num_envs)]

    def _sync_legacy_state(self) -> None:
        if not self._selected_drawers:
            return
        self._selected_drawer = self._selected_drawers[0]
        self._layout = dict(self._layouts_batch[0])
        self._wrench_bias = self._wrench_biases[0]
        self._latch_broken = self._latch_broken_batch[0]
        self._breakaway_frame = self._breakaway_frames[0]
        self._steps_past_breakaway = self._steps_past_breakaway_batch[0]
        self._resistance_force_n = self._resistance_force_n_batch[0]
        self._peak_resistance_n = self._peak_resistance_n_batch[0]
        self._peak_axial_pull_n = self._peak_axial_pull_n_batch[0]
        self._frames_armed = self._frames_armed_batch[0]
        self._stable_target_frames = self._stable_target_frames_batch[0]
        self._max_opening_world_m = self._max_opening_world_m_batch[0]
        self._resistance_start_open_m = self._resistance_start_open_m_batch[0]
        self._success_open_m = self._success_open_m_batch[0]
        self._breakaway_open_m = self._breakaway_open_m_batch[0]
        self._breakaway_resistance_n = self._breakaway_resistance_n_batch[0]

    @property
    def _usd_path(self):
        return self.asset_dir / SPEC.asset_relative_path

    def validate_assets(self) -> None:
        if not self._usd_path.is_file():
            raise FileNotFoundError(f"missing ArtVIP RADMANSO cabinet: {self._usd_path}")

    def scene_entities(self):
        entity = SceneEntitySpec(
            role="cabinet",
            kind="articulation",
            config=_cabinet_config(self._usd_path),
            friction=None,
        )
        entity.validate()
        return (entity,)

    def sample_initial_state(self, seed: int) -> dict[str, object]:
        return sample_initial_state(seed)

    def after_scene_setup(self, env) -> None:
        names = list(env.entity("cabinet").joint_names)
        required = (*DRAWER_JOINTS.values(), BOTTOM_JOINT)
        missing = [name for name in required if name not in names]
        if missing:
            raise RuntimeError(f"RADMANSO cabinet is missing joints {missing}; found {names}")
        self._drawer_ids = {
            drawer: names.index(joint_name)
            for drawer, joint_name in DRAWER_JOINTS.items()
        }
        self._ensure_batch_state(env.num_envs)

        # Refresh resistance at the configured physics rate (60 Hz for the
        # formal RADMANSO collection). A 15 Hz control-rate update would make
        # update would make the 18 mm high-load band a coarse staircase.
        previous_step = env.sim.step

        def step_with_resistance(*step_args, **step_kwargs):
            result = previous_step(*step_args, **step_kwargs)
            self._apply_resistance(env)
            return result

        env.sim.step = step_with_resistance

    def _fresh_drawer_positions(self, env) -> torch.Tensor:
        entity = env.entity("cabinet")
        return wp.to_torch(entity.root_view.get_dof_positions())

    def _apply_resistance(self, env) -> None:
        if not self._drawer_ids:
            return
        self._ensure_batch_state(env.num_envs)
        entity = env.entity("cabinet")
        positions = self._fresh_drawer_positions(env)
        efforts = torch.zeros(
            (env.num_envs, entity.num_joints), dtype=torch.float32, device=env.device
        )
        for env_index in range(env.num_envs):
            drawer_id = self._drawer_ids[self._selected_drawers[env_index]]
            opening = max(0.0, -float(positions[env_index, drawer_id].item()))
            if (
                not self._latch_broken_batch[env_index]
                and opening >= self._breakaway_open_m_batch[env_index]
            ):
                self._steps_past_breakaway_batch[env_index] += 1
                if self._steps_past_breakaway_batch[env_index] >= SPEC.breakaway_hold_steps:
                    self._latch_broken_batch[env_index] = True
                    self._breakaway_frames[env_index] = self._frames_armed_batch[env_index]
            else:
                self._steps_past_breakaway_batch[env_index] = 0
            resistance = SPEC.resistance_force_n(
                opening,
                resistance_start_open_m=self._resistance_start_open_m_batch[env_index],
                breakaway_open_m=self._breakaway_open_m_batch[env_index],
                breakaway_resistance_n=self._breakaway_resistance_n_batch[env_index],
                broken=self._latch_broken_batch[env_index],
            )
            self._resistance_force_n_batch[env_index] = resistance
            self._peak_resistance_n_batch[env_index] = max(
                self._peak_resistance_n_batch[env_index], resistance
            )
            efforts[env_index, drawer_id] = resistance
        entity.set_joint_effort_target_index(target=efforts)
        self._sync_legacy_state()

    def _write_selected_drawer_safety_limit(self, env, env_ids, drawer_id: int) -> None:
        limits = torch.zeros(
            (len(env_ids), 1, 2), dtype=torch.float32, device=env.device
        )
        limits[:, 0, 0] = -float(SPEC.max_safe_open_m)
        entity = env.entity("cabinet")
        entity.write_joint_position_limit_to_sim_index(
            limits=limits,
            joint_ids=[drawer_id],
            env_ids=env_ids,
        )

    def reset_scene(self, env, env_ids, layout) -> None:
        self._ensure_batch_state(env.num_envs)
        selected_drawer = str(layout["selected_drawer"])
        if selected_drawer not in self._drawer_ids:
            raise ValueError(f"unsupported selected drawer: {selected_drawer!r}")
        entity = env.entity("cabinet")
        count = len(env_ids)
        quaternion = _orientation(
            float(layout["cabinet_roll_rad"]),
            float(layout["cabinet_pitch_rad"]),
            float(layout["cabinet_yaw_rad"]),
            device=env.device,
        ).repeat(count, 1)
        env.set_entity_root_pose("cabinet", env_ids, layout["cabinet_position_m"], quaternion)
        joint_position = torch.zeros((count, entity.num_joints), device=env.device)
        entity.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
        entity.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
        entity.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(joint_position), env_ids=env_ids
        )
        stiffness = torch.full_like(joint_position, 2000.0)
        damping = torch.full_like(joint_position, 100.0)
        selected_id = self._drawer_ids[selected_drawer]
        resistance_start = float(layout["resistance_start_open_m"])
        success_open = float(layout["success_open_m"])
        breakaway_open = float(layout["breakaway_open_m"])
        breakaway_resistance = float(layout["breakaway_resistance_n"])
        if not (
            0.0
            < resistance_start
            < success_open
            < breakaway_open
            < SPEC.max_safe_open_m
        ):
            raise ValueError("sampled resistance layout is not safely ordered")
        self._write_selected_drawer_safety_limit(env, env_ids, selected_id)
        stiffness[:, selected_id] = 0.0
        damping[:, selected_id] = 2.0
        entity.write_joint_stiffness_to_sim_index(
            stiffness=stiffness, env_ids=env_ids
        )
        entity.write_joint_damping_to_sim_index(damping=damping, env_ids=env_ids)
        entity.set_joint_effort_target_index(
            target=torch.zeros((count, entity.num_joints), device=env.device),
            env_ids=env_ids,
        )
        entity.reset(env_ids)
        for env_index in env_ids.detach().cpu().tolist():
            self._selected_drawers[env_index] = selected_drawer
            self._layouts_batch[env_index] = dict(layout)
            self._wrench_biases[env_index] = None
            self._latch_broken_batch[env_index] = False
            self._breakaway_frames[env_index] = None
            self._steps_past_breakaway_batch[env_index] = 0
            self._resistance_force_n_batch[env_index] = SPEC.free_resistance_n
            self._peak_resistance_n_batch[env_index] = 0.0
            self._peak_axial_pull_n_batch[env_index] = 0.0
            self._frames_armed_batch[env_index] = 0
            self._stable_target_frames_batch[env_index] = 0
            self._max_opening_world_m_batch[env_index] = 0.0
            self._resistance_start_open_m_batch[env_index] = resistance_start
            self._success_open_m_batch[env_index] = success_open
            self._breakaway_open_m_batch[env_index] = breakaway_open
            self._breakaway_resistance_n_batch[env_index] = breakaway_resistance
        self._sync_legacy_state()

    def arm_after_franka_pregrasp(
        self, env, wrench_bias_world, env_id: int = 0
    ) -> None:
        self._ensure_batch_state(env.num_envs)
        env_id = int(env_id)
        bias = torch.as_tensor(wrench_bias_world, dtype=torch.float32, device=env.device)
        if bias.shape != (6,) or not torch.isfinite(bias).all():
            raise ValueError("pregrasp wrench bias must be a finite 6-vector")
        self._wrench_biases[env_id] = bias.detach().clone()
        self._frames_armed_batch[env_id] = 0
        self._peak_axial_pull_n_batch[env_id] = 0.0
        self._stable_target_frames_batch[env_id] = 0
        self._sync_legacy_state()

    def axial_pull_n(self, env, env_id: int = 0) -> float:
        self._ensure_batch_state(env.num_envs)
        env_id = int(env_id)
        bias = self._wrench_biases[env_id]
        if bias is None:
            return 0.0
        wrench = env.force_sensor_world[env_id] - bias
        axis = torch.as_tensor(
            self._layouts_batch[env_id]["pull_direction_world"],
            dtype=wrench.dtype,
            device=wrench.device,
        )
        return float(torch.dot(wrench[:3], axis).item())

    def drawer_position_m(self, env, env_id: int = 0) -> float:
        if not self._drawer_ids:
            raise RuntimeError("drawer joint has not been initialised")
        self._ensure_batch_state(env.num_envs)
        env_id = int(env_id)
        drawer_id = self._drawer_ids[self._selected_drawers[env_id]]
        return float(env.entity("cabinet").data.joint_pos.torch[env_id, drawer_id].item())

    def drawer_opening_world_m(self, env, env_id: int = 0) -> float:
        return max(0.0, -self.drawer_position_m(env, env_id))

    def drawer_velocity_world_mps(self, env, env_id: int = 0) -> float:
        self._ensure_batch_state(env.num_envs)
        env_id = int(env_id)
        drawer_id = self._drawer_ids[self._selected_drawers[env_id]]
        velocity = float(
            env.entity("cabinet").data.joint_vel.torch[env_id, drawer_id].item()
        )
        return -velocity

    def designed_resistance_n_for(self, env_id: int) -> float:
        return float(self._resistance_force_n_batch[int(env_id)])

    def latch_broken_for(self, env_id: int) -> bool:
        return bool(self._latch_broken_batch[int(env_id)])

    @property
    def designed_resistance_n(self) -> float:
        return float(self._resistance_force_n)

    @property
    def latch_broken(self) -> bool:
        return bool(self._latch_broken)

    def on_control_step(self, env) -> None:
        self._ensure_batch_state(env.num_envs)
        for env_id in range(env.num_envs):
            if self._wrench_biases[env_id] is None:
                continue
            self._frames_armed_batch[env_id] += 1
            opening = self.drawer_opening_world_m(env, env_id)
            speed = abs(self.drawer_velocity_world_mps(env, env_id))
            self._max_opening_world_m_batch[env_id] = max(
                self._max_opening_world_m_batch[env_id], opening
            )
            self._peak_axial_pull_n_batch[env_id] = max(
                self._peak_axial_pull_n_batch[env_id], self.axial_pull_n(env, env_id)
            )
            at_target = (
                abs(opening - self._success_open_m_batch[env_id])
                <= SPEC.success_tolerance_m
            )
            self._stable_target_frames_batch[env_id] = (
                self._stable_target_frames_batch[env_id] + 1
                if at_target and speed <= 0.010
                else 0
            )
        self._sync_legacy_state()

    def evaluate_final_state_for(self, env, env_id: int) -> dict[str, object]:
        self._ensure_batch_state(env.num_envs)
        env_id = int(env_id)
        report = evaluate_final_state(
            self.drawer_position_m(env, env_id),
            self._selected_drawers[env_id],
            latch_broken=self._latch_broken_batch[env_id],
            stable_frames=self._stable_target_frames_batch[env_id],
            peak_resistance_n=self._peak_resistance_n_batch[env_id],
            success_open_m=self._success_open_m_batch[env_id],
            resistance_start_open_m=self._resistance_start_open_m_batch[env_id],
            breakaway_open_m=self._breakaway_open_m_batch[env_id],
            breakaway_resistance_n=self._breakaway_resistance_n_batch[env_id],
        )
        report.update(
            {
                "drawer_velocity_world_mps": self.drawer_velocity_world_mps(env, env_id),
                "axial_pull_n": self.axial_pull_n(env, env_id),
                "peak_axial_pull_n": self._peak_axial_pull_n_batch[env_id],
                "max_opening_world_m": self._max_opening_world_m_batch[env_id],
                "breakaway_frame": self._breakaway_frames[env_id],
            }
        )
        return evaluate_report(report)

    def evaluate_final_state(self, env) -> dict[str, object]:
        return self.evaluate_final_state_for(env, 0)

    def instruction(self, layout) -> str:
        drawer = str(layout["selected_drawer"])
        return (
            f"Grip the protruding round handle on the {drawer} drawer of the "
            "walnut cabinet, pull toward the robot until resistance clearly rises, "
            "then stop and hold the drawer open without pulling it out."
        )

    def privileged_observations(self, env):
        entity = env.entity("cabinet")
        self._ensure_batch_state(env.num_envs)
        resistance_state = torch.tensor(
            [
                [
                    self.drawer_opening_world_m(env, env_id),
                    self._resistance_force_n_batch[env_id],
                    self.axial_pull_n(env, env_id),
                    1.0 if self._latch_broken_batch[env_id] else 0.0,
                    self._success_open_m_batch[env_id],
                ]
                for env_id in range(env.num_envs)
            ], dtype=torch.float32, device=env.device,
        )
        return {
            "objects/cabinet/pose": torch.cat(env.entity_local_pose("cabinet"), dim=-1),
            "mechanism/joint_position": entity.data.joint_pos.torch,
            "mechanism/resistance_state": resistance_state,
        }

    def wrench_reference_position(self, env):
        self._ensure_batch_state(env.num_envs)
        return torch.tensor(
            [list(SPEC.handle_center_m(drawer)) for drawer in self._selected_drawers],
            dtype=torch.float32, device=env.device,
        )

    def debug_state(self, env):
        return {
            "selected_drawer": self._selected_drawer,
            "drawer_joint": DRAWER_JOINTS[self._selected_drawer],
            "drawer_position_m": self.drawer_position_m(env),
            "drawer_opening_world_m": self.drawer_opening_world_m(env),
            "drawer_velocity_world_mps": self.drawer_velocity_world_mps(env),
            "designed_resistance_n": self._resistance_force_n,
            "peak_resistance_n": self._peak_resistance_n,
            "axial_pull_n": self.axial_pull_n(env),
            "peak_axial_pull_n": self._peak_axial_pull_n,
            "stable_target_frames": self._stable_target_frames,
            "latch_broken": self._latch_broken,
            "breakaway_frame": self._breakaway_frame,
            "max_opening_world_m": self._max_opening_world_m,
            "resistance_start_open_m": self._resistance_start_open_m,
            "success_open_m": self._success_open_m,
            "breakaway_open_m": self._breakaway_open_m,
            "breakaway_resistance_n": self._breakaway_resistance_n,
            "max_safe_open_m": SPEC.max_safe_open_m,
            "cabinet_position_m": list(SPEC.cabinet_position_m),
            "cabinet_orientation_rpy_rad": [
                SPEC.cabinet_roll_rad,
                SPEC.cabinet_pitch_rad,
                SPEC.cabinet_yaw_rad,
            ],
            "pull_direction_world": list(SPEC.pull_direction_world),
        }

    def episode_metadata(self) -> dict[str, object]:
        return {
            "mechanism": "artvip_radmanso_middle_drawer_on_box_sort_table",
            "asset_usd_path": str(self._usd_path),
            "selectable_drawers": {"middle": DRAWER_JOINTS["middle"]},
            "selected_drawer": self._selected_drawer,
            "drawer_joint": DRAWER_JOINTS[self._selected_drawer],
            "always_locked_joint": BOTTOM_JOINT,
            "scene_profile": "simple_room_sort_table",
            "cabinet_scale": SPEC.cabinet_scale,
            "resistance_profile": {
                "type": "free_then_cubic_rise_then_irreversible_release",
                "free_resistance_n": SPEC.free_resistance_n,
                "resistance_start_open_m": self._resistance_start_open_m,
                "resistance_band_length_m": (
                    self._breakaway_open_m - self._resistance_start_open_m
                ),
                "success_open_m": self._success_open_m,
                "success_tolerance_m": SPEC.success_tolerance_m,
                "breakaway_open_m": self._breakaway_open_m,
                "breakaway_resistance_n": self._breakaway_resistance_n,
                "stop_resistance_n": SPEC.stop_resistance_n,
                "post_breakaway_resistance_n": SPEC.post_breakaway_resistance_n,
                "max_safe_open_m": SPEC.max_safe_open_m,
            },
            "success_criterion": (
                "drawer stable in target window after entering high resistance, "
                "with breakaway latch intact"
            ),
            "cabinet_orientation_rpy_rad": [
                SPEC.cabinet_roll_rad,
                SPEC.cabinet_pitch_rad,
                SPEC.cabinet_yaw_rad,
            ],
        }
