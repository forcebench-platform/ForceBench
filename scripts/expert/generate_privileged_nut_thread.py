#!/usr/bin/env python3
"""Generate nut_thread experts from exact nut/bolt pose feedback."""

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
parser.add_argument("--translation_speed_m_s", type=float, default=0.075)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or args.max_attempts_per_episode < 1 or args.start_seed < 0:
    parser.error("episode counts must be positive and seed must be non-negative")

launcher = AppLauncher(args)
simulation_app = launcher.app

from forcebench.privileged_expert_runner import collect_privileged_placements
from forcebench.privileged_placement import PrivilegedPlacementPlan
from forcebench.tasks.nut_thread.spec import SPEC


def plan(env) -> PrivilegedPlacementPlan:
    return PrivilegedPlacementPlan(
        object_role="nut",
        pickup_tcp_z_m=0.018,
        lift_tcp_z_m=0.170,
        grasp_open_fraction=0.0,
        desired_object_yaw_rad=float(env.layout["target_bolt_yaw_rad"]),
        placement_clearances_m=(0.050, 0.025, 0.012, 0.006, 0.003, 0.0),
        translation_speed_m_s=args.translation_speed_m_s,
        # The unchanged final outcome contract is tighter and remains the
        # authority after release.  This gate only lets the expert proceed
        # through the loaded IK controller's ~1.1 mm steady-state residual so
        # the nut can physically settle on the thread before final checking.
        object_position_tolerance_m=0.00175,
        # A right-hand thread descends under clockwise rotation when viewed
        # from above.  Half a turn matches the remaining ~1 mm engagement for
        # the 2 mm pitch while the controller maintains gentle downward load.
        thread_rotation_rad=-1.0 * math.pi,
        thread_yaw_step_rad=0.025,
        thread_downward_bias_m=0.0015,
        retract_height_m=0.08,
    )


try:
    collect_privileged_placements(args=args, task_name="nut_thread", plan_factory=plan)
finally:
    simulation_app.close()
