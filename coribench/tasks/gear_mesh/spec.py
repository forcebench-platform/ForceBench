"""Simulator-free layout and final-state definition for gear meshing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GearMeshSpec:
    gear_base_asset: str = "normalized/gear_mesh/base.usda"
    medium_gear_asset: str = "normalized/gear_mesh/medium.usda"
    small_gear_asset: str = "normalized/gear_mesh/small.usda"
    large_gear_asset: str = "normalized/gear_mesh/large.usda"
    # The complete Factory Gear Mesh assembly is enlarged uniformly.  Keeping
    # the scale here makes the visual mesh, SDF collision and layout contract
    # agree instead of applying a render-only transform.
    asset_scale_xyz: tuple[float, float, float] = (2.0, 2.0, 2.0)
    mass_scale: float = 8.0
    # Move the doubled 300 mm base plate to the right side of the workbench so
    # it remains separated from the pickup gear and fully inside the tabletop.
    target_base_x_range_m: tuple[float, float] = (0.64, 0.70)
    target_base_y_range_m: tuple[float, float] = (-0.08, 0.08)
    target_base_root_z_m: float = 0.0
    target_initial_yaw_rad: float = 0.0
    # Center the pickup phase on the same mesh phase used by the privileged O
    # key, then add a small seeded perturbation below.
    # The medium gear has 40 teeth (9 degree pitch); its tooth tips are centered
    # at 4.5 + 9k degrees, while both flanking gears present a valley along the
    # line of centers.  A -4.5 degree medium-gear yaw therefore puts a tooth tip
    # into each valley.  The flanking gears also remain free to rotate on their
    # shafts, as in the source task.
    initial_medium_gear_yaw_deg: float = -4.5
    # Prevent an uncorrected drop from starting in the exactly compatible mesh
    # phase while keeping the pickup orientation close to that phase.
    initial_medium_gear_yaw_jitter_abs_range_deg: tuple[float, float] = (2.0, 3.5)
    expert_twist_alignment_offset_deg: float = -4.5
    medium_gear_target_offset_m: tuple[float, float, float] = (0.04050, 0.0, 0.010)
    small_gear_target_offset_m: tuple[float, float, float] = (0.10150, 0.0, 0.010)
    large_gear_target_offset_m: tuple[float, float, float] = (-0.06050, 0.0, 0.010)
    pickup_center_x_m: float = 0.40
    pickup_center_y_m: float = 0.0
    pickup_gear_root_z_m: float = 0.010
    # The doubled medium gear has an approximately 60 mm hub inside an 84 mm
    # toothed flange.  The grasp plane and support height scale with it.
    pickup_grasp_tcp_z_m: float = 0.040
    pickup_support_size_m: tuple[float, float, float] = (0.040, 0.040, 0.010)
    # A 70% opening gives 28 mm per finger and a light preload on the enlarged
    # hub while remaining inside the Franka hand's 80 mm travel.
    grasp_gripper_open_fraction: float = 0.70
    minimum_initial_separation_m: float = 0.28
    max_center_error_m: float = 0.005
    max_height_error_m: float = 0.006
    max_tilt_deg: float = 12.0
    min_gripper_open_fraction: float = 0.65
    min_retract_distance_m: float = 0.10
    min_retract_height_m: float = 0.09
    gripper_friction: float = 1.5
    held_gear_friction: float = 0.75
    fixture_friction: float = 0.25
    friction_combine_mode: str = "min"


SPEC = GearMeshSpec()


def _rotated_xy(offset_xyz, yaw_rad: float) -> np.ndarray:
    offset = np.asarray(offset_xyz, dtype=np.float64)
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    return np.asarray(
        [cosine * offset[0] - sine * offset[1], sine * offset[0] + cosine * offset[1]],
        dtype=np.float64,
    )


def sample_initial_medium_gear_yaw_jitter_deg(
    seed: int, spec: GearMeshSpec = SPEC
) -> float:
    """Return a small reproducible, non-zero pickup tooth-phase offset."""

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    minimum, maximum = spec.initial_medium_gear_yaw_jitter_abs_range_deg
    if not 0.0 < minimum <= maximum:
        raise ValueError("Gear yaw jitter magnitude range must be positive and ordered")
    # Keep this stream independent so adding yaw jitter does not alter the
    # established seed-to-layout XY mapping.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x47454152]))
    magnitude = float(rng.uniform(minimum, maximum))
    sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
    return sign * magnitude


def sample_initial_state(seed: int, spec: GearMeshSpec = SPEC) -> dict[str, object]:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    pickup_center = np.asarray([spec.pickup_center_x_m, spec.pickup_center_y_m], dtype=np.float64)
    target_base_xy = None
    target_center_xy = None
    target_yaw = None
    for _ in range(128):
        candidate_base = np.asarray(
            [rng.uniform(*spec.target_base_x_range_m), rng.uniform(*spec.target_base_y_range_m)],
            dtype=np.float64,
        )
        candidate_yaw = spec.target_initial_yaw_rad
        candidate_center = candidate_base + _rotated_xy(spec.medium_gear_target_offset_m, candidate_yaw)
        if np.linalg.norm(candidate_center - pickup_center) >= spec.minimum_initial_separation_m:
            target_base_xy = candidate_base
            target_center_xy = candidate_center
            target_yaw = candidate_yaw
            break
    if target_base_xy is None or target_center_xy is None or target_yaw is None:
        raise RuntimeError("configured gear layout cannot satisfy minimum separation")

    separation = float(np.linalg.norm(target_center_xy - pickup_center))
    small_xy = target_base_xy + _rotated_xy(spec.small_gear_target_offset_m, target_yaw)
    large_xy = target_base_xy + _rotated_xy(spec.large_gear_target_offset_m, target_yaw)
    yaw_jitter_deg = sample_initial_medium_gear_yaw_jitter_deg(seed, spec)
    return {
        "seed": seed,
        "target_base_position_m": [
            float(target_base_xy[0]),
            float(target_base_xy[1]),
            spec.target_base_root_z_m,
        ],
        "target_base_yaw_rad": target_yaw,
        "target_medium_shaft_position_m": [
            float(target_center_xy[0]),
            float(target_center_xy[1]),
            spec.target_base_root_z_m + spec.medium_gear_target_offset_m[2],
        ],
        "target_small_gear_position_m": [
            float(small_xy[0]),
            float(small_xy[1]),
            spec.target_base_root_z_m + spec.small_gear_target_offset_m[2],
        ],
        "target_large_gear_position_m": [
            float(large_xy[0]),
            float(large_xy[1]),
            spec.target_base_root_z_m + spec.large_gear_target_offset_m[2],
        ],
        "pickup_gear_center_position_m": [
            spec.pickup_center_x_m,
            spec.pickup_center_y_m,
            spec.pickup_gear_root_z_m,
        ],
        "medium_gear_root_position_m": [
            spec.pickup_center_x_m,
            spec.pickup_center_y_m,
            spec.pickup_gear_root_z_m,
        ],
        "medium_gear_yaw_rad": float(
            spec.target_initial_yaw_rad
            + np.deg2rad(spec.initial_medium_gear_yaw_deg + yaw_jitter_deg)
        ),
        "medium_gear_yaw_jitter_deg": yaw_jitter_deg,
        "initial_separation_m": separation,
    }


@dataclass(frozen=True)
class SuccessReport:
    success: bool
    centered: bool
    seated: bool
    upright: bool
    released: bool
    retracted: bool
    center_error_m: float
    height_error_m: float
    tilt_deg: float
    gripper_open_fraction: float
    retract_distance_m: float
    retract_height_m: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_final_state(
    *,
    gear_center_position_m,
    gear_axis_world,
    target_shaft_position_m,
    gripper_open_fraction: float,
    eef_position_m,
    spec: GearMeshSpec = SPEC,
) -> SuccessReport:
    gear = np.asarray(gear_center_position_m, dtype=np.float64)
    axis = np.asarray(gear_axis_world, dtype=np.float64)
    target = np.asarray(target_shaft_position_m, dtype=np.float64)
    eef = np.asarray(eef_position_m, dtype=np.float64)
    if any(value.shape != (3,) for value in (gear, axis, target, eef)):
        raise ValueError("gear, axis, target shaft and end-effector positions must be 3-vectors")
    if not all(np.isfinite(value).all() for value in (gear, axis, target, eef)):
        raise ValueError("final-state inputs must be finite")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-9:
        raise ValueError("gear axis cannot be zero")
    axis /= axis_norm

    center_error = float(np.linalg.norm(gear[:2] - target[:2]))
    height_error = float(gear[2] - target[2])
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))
    retract_distance = float(np.linalg.norm(eef[:2] - gear[:2]))
    retract_height = float(eef[2] - gear[2])
    centered = center_error <= spec.max_center_error_m
    seated = abs(height_error) <= spec.max_height_error_m
    upright = tilt_deg <= spec.max_tilt_deg
    released = float(gripper_open_fraction) >= spec.min_gripper_open_fraction
    retracted = retract_distance >= spec.min_retract_distance_m or retract_height >= spec.min_retract_height_m
    return SuccessReport(
        success=bool(centered and seated and upright and released and retracted),
        centered=bool(centered),
        seated=bool(seated),
        upright=bool(upright),
        released=bool(released),
        retracted=bool(retracted),
        center_error_m=center_error,
        height_error_m=height_error,
        tilt_deg=tilt_deg,
        gripper_open_fraction=float(gripper_open_fraction),
        retract_distance_m=retract_distance,
        retract_height_m=retract_height,
    )
