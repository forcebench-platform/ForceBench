"""Simulator-free peg insertion layout and final-state definition."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PegInsertSpec:
    peg_asset: str = "normalized/peg_insert/peg.usda"
    hole_asset: str = "normalized/peg_insert/hole.usda"
    hole_x_range_m: tuple[float, float] = (0.58, 0.68)
    hole_y_range_m: tuple[float, float] = (-0.12, 0.12)
    pickup_center_x_m: float = 0.50
    pickup_center_y_m: float = 0.0
    minimum_initial_separation_m: float = 0.14
    hole_root_z_m: float = 0.0
    hole_mate_height_m: float = 0.003
    upright_peg_root_z_m: float = 0.015
    initial_yaw_rad: float = 0.0
    pickup_support_width_m: float = 0.012
    start_tcp_position_m: tuple[float, float, float] = (0.50, 0.0, 0.18)
    max_center_error_m: float = 0.003
    max_base_height_error_m: float = 0.006
    max_tilt_deg: float = 12.0
    min_gripper_open_fraction: float = 0.65
    min_retract_distance_m: float = 0.07
    min_retract_height_m: float = 0.055
    gripper_friction: float = 1.5
    peg_friction: float = 3.0
    hole_friction: float = 0.01
    friction_combine_mode: str = "min"


SPEC = PegInsertSpec()

# A release gate is deliberately tighter than the final benchmark predicate.
# It is used while the fingers still retain the peg, leaving time to re-close
# and re-center before an incipient slip becomes an unrecoverable ejection.
RELEASE_GATE_MAX_CENTER_ERROR_M = 0.0015
RELEASE_GATE_MAX_HEIGHT_ERROR_M = 0.0055
RELEASE_GATE_MAX_TILT_DEG = 8.0


def evaluate_release_stability_batch(
    center_error_m,
    height_error_m,
    tilt_deg,
) -> np.ndarray:
    """Return the per-environment pre-release stability gate."""

    center = np.asarray(center_error_m, dtype=np.float64)
    height = np.asarray(height_error_m, dtype=np.float64)
    tilt = np.asarray(tilt_deg, dtype=np.float64)
    if center.ndim != 1 or height.shape != center.shape or tilt.shape != center.shape:
        raise ValueError("release-gate inputs must be same-shaped vectors")
    if not (np.isfinite(center).all() and np.isfinite(height).all() and np.isfinite(tilt).all()):
        raise ValueError("release-gate inputs must be finite")
    return (
        (center <= RELEASE_GATE_MAX_CENTER_ERROR_M)
        & (np.abs(height) <= RELEASE_GATE_MAX_HEIGHT_ERROR_M)
        & (tilt <= RELEASE_GATE_MAX_TILT_DEG)
    )


def sample_initial_state(seed: int, spec: PegInsertSpec = SPEC) -> dict[str, object]:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    pickup_xy = np.asarray([spec.pickup_center_x_m, spec.pickup_center_y_m], dtype=np.float64)
    target_xy = None
    for _ in range(128):
        candidate = np.asarray(
            [rng.uniform(*spec.hole_x_range_m), rng.uniform(*spec.hole_y_range_m)], dtype=np.float64
        )
        if np.linalg.norm(candidate - pickup_xy) >= spec.minimum_initial_separation_m:
            target_xy = candidate
            break
    if target_xy is None:
        raise RuntimeError("configured layout ranges cannot satisfy minimum separation")
    separation = float(np.linalg.norm(pickup_xy - target_xy))
    if separation < spec.minimum_initial_separation_m:
        raise RuntimeError("sampled target hole overlaps the protected pickup region")
    return {
        "seed": seed,
        "target_hole_position_m": [float(target_xy[0]), float(target_xy[1]), spec.hole_root_z_m],
        "target_hole_yaw_rad": spec.initial_yaw_rad,
        "pickup_hole_position_m": [spec.pickup_center_x_m, spec.pickup_center_y_m, spec.hole_root_z_m],
        "peg_position_m": [float(pickup_xy[0]), float(pickup_xy[1]), spec.upright_peg_root_z_m],
        "peg_upright_yaw_rad": spec.initial_yaw_rad,
        "initial_separation_m": separation,
    }


@dataclass(frozen=True)
class SuccessReport:
    success: bool
    centered: bool
    inserted: bool
    upright: bool
    released: bool
    retracted: bool
    center_error_m: float
    base_height_error_m: float
    tilt_deg: float
    gripper_open_fraction: float
    retract_distance_m: float
    retract_height_m: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_final_state(
    *,
    peg_position_m,
    peg_axis_world,
    target_hole_position_m,
    gripper_open_fraction: float,
    eef_position_m,
    spec: PegInsertSpec = SPEC,
) -> SuccessReport:
    peg = np.asarray(peg_position_m, dtype=np.float64)
    target = np.asarray(target_hole_position_m, dtype=np.float64)
    axis = np.asarray(peg_axis_world, dtype=np.float64)
    eef = np.asarray(eef_position_m, dtype=np.float64)
    if any(value.shape != (3,) for value in (peg, target, axis, eef)):
        raise ValueError("peg, target hole, axis and end-effector positions must be 3-vectors")
    if not all(np.isfinite(value).all() for value in (peg, target, axis, eef)):
        raise ValueError("final-state inputs must be finite")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        raise ValueError("peg axis cannot be zero")
    axis = axis / axis_norm
    center_error = float(np.linalg.norm(peg[:2] - target[:2]))
    height_error = float(peg[2] - target[2])
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
    retract_distance = float(np.linalg.norm(eef[:2] - peg[:2]))
    retract_height = float(eef[2] - peg[2])
    centered = center_error <= spec.max_center_error_m
    inserted = abs(height_error) <= spec.max_base_height_error_m
    upright = tilt_deg <= spec.max_tilt_deg
    released = float(gripper_open_fraction) >= spec.min_gripper_open_fraction
    retracted = retract_distance >= spec.min_retract_distance_m or retract_height >= spec.min_retract_height_m
    return SuccessReport(
        success=bool(centered and inserted and upright and released and retracted),
        centered=bool(centered),
        inserted=bool(inserted),
        upright=bool(upright),
        released=bool(released),
        retracted=bool(retracted),
        center_error_m=center_error,
        base_height_error_m=height_error,
        tilt_deg=tilt_deg,
        gripper_open_fraction=float(gripper_open_fraction),
        retract_distance_m=retract_distance,
        retract_height_m=retract_height,
    )
