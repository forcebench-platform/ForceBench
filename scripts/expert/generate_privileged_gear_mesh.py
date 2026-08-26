#!/usr/bin/env python3
"""Generate gear_mesh experts from exact object/shaft pose feedback."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--artifact_dir", type=Path)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--start_seed", type=int, default=0)
parser.add_argument("--max_attempts_per_episode", type=int, default=1)
parser.add_argument("--main_image_size", type=int, default=224)
parser.add_argument("--wrist_image_size", type=int, default=224)
parser.add_argument("--physics_rate_hz", type=int, choices=(60,), default=60)
parser.add_argument("--solver_position_iterations", type=int, choices=(64,), default=64)
parser.add_argument("--translation_speed_m_s", type=float, default=0.09)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or args.max_attempts_per_episode < 1 or args.start_seed < 0:
    parser.error("episode counts must be positive and seed must be non-negative")

launcher = AppLauncher(args)
simulation_app = launcher.app

from coribench.privileged_expert_runner import collect_privileged_placements
from coribench.privileged_placement import PrivilegedPlacementPlan
from coribench.tasks.gear_mesh.spec import SPEC


def plan(env) -> PrivilegedPlacementPlan:
    return PrivilegedPlacementPlan(
        object_role="medium_gear",
        pickup_tcp_z_m=SPEC.pickup_grasp_tcp_z_m,
        lift_tcp_z_m=0.220,
        grasp_open_fraction=SPEC.grasp_gripper_open_fraction,
        desired_object_yaw_rad=float(
            env.layout["target_base_yaw_rad"]
            + math.radians(SPEC.expert_twist_alignment_offset_deg)
        ),
        placement_clearances_m=(0.100, 0.050, 0.025, 0.012, 0.006, 0.0),
        translation_speed_m_s=args.translation_speed_m_s,
        object_position_tolerance_m=0.0015,
        retract_height_m=0.12,
    )


try:
    collect_privileged_placements(args=args, task_name="gear_mesh", plan_factory=plan)
finally:
    simulation_app.close()
