"""Simulator-free layout and final-state definition for nut threading."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class NutThreadSpec:
    nut_asset: str = "normalized/nut_thread/nut.usda"
    bolt_asset: str = "normalized/nut_thread/bolt.usda"
    target_bolt_x_range_m: tuple[float, float] = (0.58, 0.68)
    target_bolt_y_range_m: tuple[float, float] = (-0.11, 0.11)
    target_bolt_root_z_m: float = 0.0
    initial_yaw_rad: float = 0.0
    pickup_center_x_m: float = 0.49
    pickup_center_y_m: float = 0.0
    pickup_nut_root_z_m: float = 0.010
    pickup_support_size_m: tuple[float, float, float] = (0.018, 0.018, 0.010)
    minimum_initial_separation_m: float = 0.14
    bolt_base_height_m: float = 0.010
    bolt_shank_height_m: float = 0.025
    thread_pitch_m: float = 0.002
    target_thread_pitch_offset: float = 1.5
    max_center_error_m: float = 0.0025
    min_thread_height_error_m: float = -0.002
    # Half a thread pitch tolerates the sub-millimetre contact-settling
    # variation of a physically engaged nut without admitting an unthreaded
    # nut resting on top of the 2 mm-pitch bolt.
    max_thread_height_error_m: float = 0.001
    max_tilt_deg: float = 10.0
    min_gripper_open_fraction: float = 0.65
    min_retract_distance_m: float = 0.07
    min_retract_height_m: float = 0.055
    gripper_friction: float = 1.5
    nut_friction: float = 0.75
    bolt_friction: float = 0.25
    friction_combine_mode: str = "min"
    # The nut root remains about 8 mm below the fingertip midpoint in the
    # grasp.  Tracking the object root at this height makes the scripted lift
    # prove that the nut actually came off the pickup support.
    scripted_lift_nut_root_z_m: float = 0.160

    @property
    def target_nut_base_offset_m(self) -> float:
        return self.bolt_base_height_m + self.bolt_shank_height_m - self.thread_pitch_m * self.target_thread_pitch_offset


SPEC = NutThreadSpec()


def sample_initial_state(seed: int, spec: NutThreadSpec = SPEC) -> dict[str, object]:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    pickup_xy = np.asarray([spec.pickup_center_x_m, spec.pickup_center_y_m], dtype=np.float64)
    target_xy = None
    for _ in range(128):
        candidate = np.asarray(
            [rng.uniform(*spec.target_bolt_x_range_m), rng.uniform(*spec.target_bolt_y_range_m)],
            dtype=np.float64,
        )
        if np.linalg.norm(candidate - pickup_xy) >= spec.minimum_initial_separation_m:
            target_xy = candidate
            break
    if target_xy is None:
        raise RuntimeError("configured nut-thread layout cannot satisfy minimum separation")
    separation = float(np.linalg.norm(target_xy - pickup_xy))
    return {
        "seed": seed,
        "target_bolt_position_m": [float(target_xy[0]), float(target_xy[1]), spec.target_bolt_root_z_m],
        "target_bolt_yaw_rad": spec.initial_yaw_rad,
        "pickup_nut_root_position_m": [
            spec.pickup_center_x_m,
            spec.pickup_center_y_m,
            spec.pickup_nut_root_z_m,
        ],
        "pickup_nut_yaw_rad": spec.initial_yaw_rad,
        "target_nut_base_position_m": [
            float(target_xy[0]),
            float(target_xy[1]),
            spec.target_bolt_root_z_m + spec.target_nut_base_offset_m,
        ],
        "initial_separation_m": separation,
    }


@dataclass(frozen=True)
class SuccessReport:
    success: bool
    centered: bool
    threaded: bool
    upright: bool
    released: bool
    retracted: bool
    center_error_m: float
    thread_height_error_m: float
    tilt_deg: float
    gripper_open_fraction: float
    retract_distance_m: float
    retract_height_m: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_final_state(
    *,
    nut_base_position_m,
    nut_axis_world,
    target_nut_base_position_m,
    gripper_open_fraction: float,
    eef_position_m,
    spec: NutThreadSpec = SPEC,
) -> SuccessReport:
    nut = np.asarray(nut_base_position_m, dtype=np.float64)
    axis = np.asarray(nut_axis_world, dtype=np.float64)
    target = np.asarray(target_nut_base_position_m, dtype=np.float64)
    eef = np.asarray(eef_position_m, dtype=np.float64)
    if any(value.shape != (3,) for value in (nut, axis, target, eef)):
        raise ValueError("nut, axis, target and end-effector positions must be 3-vectors")
    if not all(np.isfinite(value).all() for value in (nut, axis, target, eef)):
        raise ValueError("final-state inputs must be finite")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-9:
        raise ValueError("nut axis cannot be zero")
    axis /= axis_norm

    center_error = float(np.linalg.norm(nut[:2] - target[:2]))
    height_error = float(nut[2] - target[2])
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
    retract_distance = float(np.linalg.norm(eef[:2] - nut[:2]))
    retract_height = float(eef[2] - nut[2])
    centered = center_error <= spec.max_center_error_m
    threaded = spec.min_thread_height_error_m <= height_error <= spec.max_thread_height_error_m
    upright = tilt_deg <= spec.max_tilt_deg
    released = float(gripper_open_fraction) >= spec.min_gripper_open_fraction
    retracted = retract_distance >= spec.min_retract_distance_m or retract_height >= spec.min_retract_height_m
    return SuccessReport(
        success=bool(centered and threaded and upright and released and retracted),
        centered=bool(centered),
        threaded=bool(threaded),
        upright=bool(upright),
        released=bool(released),
        retracted=bool(retracted),
        center_error_m=center_error,
        thread_height_error_m=height_error,
        tilt_deg=tilt_deg,
        gripper_open_fraction=float(gripper_open_fraction),
        retract_distance_m=retract_distance,
        retract_height_m=retract_height,
    )
