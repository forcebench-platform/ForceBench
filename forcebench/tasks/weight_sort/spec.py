"""Simulator-free layout and success definition for weight-based box sorting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


BOX_KINDS = ("light", "heavy")


@dataclass(frozen=True)
class WeightSortSpec:
    box_asset: str = "normalized/weight_sort/box.usda"
    source_box_asset: str = "simready/flatbox_a02/flatbox_a02.usd"
    container_asset: str = "simready/container_f27/container_f27.usd"
    source_box_native_size_m: tuple[float, float, float] = (
        0.1505928114,
        0.2117998227,
        0.0835589990,
    )
    box_visual_scale: float = 0.36
    light_mass_kg: float = 0.06
    heavy_mass_kg: float = 0.18
    box_friction: float = 1.5
    friction_combine_mode: str = "min"
    box_root_z_m: float = 0.003
    pickup_x_m: float = 0.67
    # The pickup and both destinations are shifted toward the camera-facing
    # part of the reachable table so the task objects fill the lower view.
    # Keep the low grasp outside the central near-base singular region observed
    # at x=0.62--0.65 m under the canonical 60 Hz smooth-position controller.
    pickup_x_range_m: tuple[float, float] = (0.67, 0.72)
    pickup_y_range_m: tuple[float, float] = (-0.20, 0.08)
    pickup_yaw_center_rad: float = np.pi / 2.0
    black_container_center_xy_m: tuple[float, float] = (0.52, -0.16)
    blue_container_center_xy_m: tuple[float, float] = (0.52, 0.16)
    container_native_size_m: tuple[float, float, float] = (
        1.1583822370,
        0.7957980931,
        0.5223035157,
    )
    container_visual_scale: float = 0.16
    container_opening_size_m: tuple[float, float] = (0.16, 0.10)
    container_wall_thickness_m: float = 0.012
    container_wall_height_m: float = 0.075
    container_bottom_thickness_m: float = 0.006
    placement_center_margin_m: float = 0.018
    max_box_root_z_in_container_m: float = 0.060
    min_gripper_open_fraction: float = 0.65
    max_settled_linear_speed_m_s: float = 0.15

    @property
    def scaled_box_size_m(self) -> tuple[float, float, float]:
        return tuple(value * self.box_visual_scale for value in self.source_box_native_size_m)

    @property
    def scaled_container_size_m(self) -> tuple[float, float, float]:
        return tuple(
            value * self.container_visual_scale for value in self.container_native_size_m
        )


SPEC = WeightSortSpec()


def sample_initial_state(seed: int, spec: WeightSortSpec = SPEC) -> dict[str, object]:
    """Sample one box position and assign an exactly balanced seed-parity mass."""
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    position_seed = np.random.SeedSequence(seed).spawn(1)[0]
    position_rng = np.random.default_rng(position_seed)
    position = [
        float(position_rng.uniform(*spec.pickup_x_range_m)),
        float(position_rng.uniform(*spec.pickup_y_range_m)),
        spec.box_root_z_m,
    ]
    box_kind = BOX_KINDS[seed % len(BOX_KINDS)]
    box_mass_kg = spec.light_mass_kg if box_kind == "light" else spec.heavy_mass_kg
    destination = "black_container" if box_kind == "light" else "blue_container"
    return {
        "seed": seed,
        "box_position_m": position,
        "box_yaw_rad": float(spec.pickup_yaw_center_rad),
        "box_kind": box_kind,
        "box_mass_kg": box_mass_kg,
        "destination": destination,
        "black_container_center_xy_m": list(spec.black_container_center_xy_m),
        "blue_container_center_xy_m": list(spec.blue_container_center_xy_m),
    }


def _inside_centered_rectangle(
    point_xy: np.ndarray,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    margin: float,
) -> bool:
    half = 0.5 * np.asarray(size_xy, dtype=np.float64) - float(margin)
    if np.any(half <= 0.0):
        raise ValueError("rectangle margin leaves no valid interior")
    return bool(np.all(np.abs(point_xy - np.asarray(center_xy, dtype=np.float64)) <= half))


@dataclass(frozen=True)
class SuccessReport:
    success: bool
    box_kind: str
    destination: str
    box_in_correct_container: bool
    box_in_black_container: bool
    box_in_blue_container: bool
    box_location: str
    placement_failure_reason: str | None
    box_position_m: tuple[float, float, float]
    box_released: bool
    box_settled: bool
    box_root_z_m: float
    box_speed_m_s: float
    gripper_open_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_final_state(
    *,
    box_kind: str,
    box_position_m,
    box_linear_velocity_m_s,
    gripper_open_fraction: float,
    spec: WeightSortSpec = SPEC,
) -> SuccessReport:
    if box_kind not in BOX_KINDS:
        raise ValueError(f"box_kind must be one of {BOX_KINDS}, got {box_kind!r}")
    position = np.asarray(box_position_m, dtype=np.float64)
    velocity = np.asarray(box_linear_velocity_m_s, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("box position and linear velocity must be 3-vectors")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("final-state inputs must be finite")
    if not np.isfinite(gripper_open_fraction):
        raise ValueError("gripper_open_fraction must be finite")

    destination = "black_container" if box_kind == "light" else "blue_container"
    valid_container_height = 0.0 <= float(position[2]) <= spec.max_box_root_z_in_container_m
    box_in_black_container = _inside_centered_rectangle(
        position[:2],
        spec.black_container_center_xy_m,
        spec.container_opening_size_m,
        spec.placement_center_margin_m,
    ) and valid_container_height
    box_in_blue_container = _inside_centered_rectangle(
        position[:2],
        spec.blue_container_center_xy_m,
        spec.container_opening_size_m,
        spec.placement_center_margin_m,
    ) and valid_container_height
    box_in_correct_container = (
        box_in_black_container if box_kind == "light" else box_in_blue_container
    )
    if box_in_black_container:
        box_location = "black_container"
    elif box_in_blue_container:
        box_location = "blue_container"
    else:
        box_location = "outside_containers"
    placement_failure_reason = None
    if not box_in_correct_container:
        placement_failure_reason = (
            "box_in_wrong_container"
            if box_in_black_container or box_in_blue_container
            else "box_not_in_any_container"
        )
    box_speed = float(np.linalg.norm(velocity))
    box_settled = box_speed <= spec.max_settled_linear_speed_m_s
    box_released = float(gripper_open_fraction) >= spec.min_gripper_open_fraction
    return SuccessReport(
        success=bool(box_in_correct_container and box_released and box_settled),
        box_kind=box_kind,
        destination=destination,
        box_in_correct_container=bool(box_in_correct_container),
        box_in_black_container=bool(box_in_black_container),
        box_in_blue_container=bool(box_in_blue_container),
        box_location=box_location,
        placement_failure_reason=placement_failure_reason,
        box_position_m=tuple(float(value) for value in position),
        box_released=bool(box_released),
        box_settled=bool(box_settled),
        box_root_z_m=float(position[2]),
        box_speed_m_s=box_speed,
        gripper_open_fraction=float(gripper_open_fraction),
    )
