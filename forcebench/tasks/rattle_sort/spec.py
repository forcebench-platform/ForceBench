"""Simulator-free specification for front/back equal-mass balance sorting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


BOX_KINDS = ("uniform", "imbalanced")


@dataclass(frozen=True)
class RattleSortSpec:
    box_asset: str = "normalized/rattle_sort/box_shell.usda"
    source_box_asset: str = "simready/flatbox_a07/flatbox_a07.usd"
    container_asset: str = "simready/container_f27/container_f27.usd"
    conveyor_asset: str = "simready/conveyor_a04/conveyor_a04.usd"
    conveyor_source_url: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/6.0/Isaac/Props/Conveyors/ConveyorBelt_A04.usd"
    )
    background_parcel_assets: tuple[str, ...] = (
        "simready/shipping_background/Assets/simready_content/common_assets/props/"
        "multidepthbox_a04/multidepthbox_a04_base.usd",
        "simready/shipping_background/Assets/simready_content/common_assets/props/"
        "cubebox_a03/cubebox_a03_base.usd",
        "simready/shipping_background/Assets/simready_content/common_assets/props/"
        "longbox_a04/longbox_a04_base.usd",
        "simready/shipping_background/Assets/simready_content/common_assets/props/"
        "flatbox_a12/flatbox_a12_base.usd",
    )
    # Audited on L20 from the downloaded NVIDIA source using UsdGeom.BBoxCache.
    source_box_native_size_m: tuple[float, float, float] = (
        0.2560426890850067,
        0.3172025680541992,
        0.11038243770599365,
    )
    box_visual_scale_xyz: tuple[float, float, float] = (
        0.21090233114241308,
        0.37830715159751177,
        0.21090311542137577,
    )
    box_size_m: tuple[float, float, float] = (
        0.054,
        0.120,
        0.02328,
    )
    box_total_mass_kg: float = 0.18
    # The parcel's long local-Y axis is rotated by +90 degrees around world Z.
    # Local +Y therefore points toward the robot (world -X, the parcel front),
    # while local -Y points away from it (world +X, the parcel back).
    # A 20 mm offset yields about 0.035 N*m of gravity moment at 0.18 kg:
    # clearly measurable, while still permitting a stable center grasp.
    front_back_com_offset_m: float = 0.020
    box_friction: float = 1.5
    box_angular_damping: float = 1.0
    box_root_z_m: float = 0.002
    pickup_x_m: float = 0.46
    # Keep the complete open-gripper approach envelope away from the near
    # corner of the black basket collision walls.  The former far-X/negative-Y
    # corner allowed (0.52, -0.06) and made seed 40 contact the basket edge.
    # This 80 x 85 mm rectangle preserves substantial pose diversity while
    # adding at least 30 mm in X and 35 mm in Y at the hazardous corner.
    pickup_region_profile: str = "basket_edge_safe_rectangle_v2"
    pickup_x_range_m: tuple[float, float] = (0.40, 0.48)
    pickup_y_range_m: tuple[float, float] = (-0.025, 0.06)
    pickup_yaw_rad: float = np.pi / 2.0

    # Camera-horizontal world Y: negative is left, positive is right. Both
    # destinations sit forward in the task-focused view to avoid dead space.
    black_basket_center_xy_m: tuple[float, float] = (0.64, -0.22)
    black_basket_visual_scale: float = 0.23
    black_basket_opening_size_m: tuple[float, float] = (0.235, 0.145)
    black_basket_wall_thickness_m: float = 0.014
    black_basket_wall_height_m: float = 0.105
    black_basket_bottom_thickness_m: float = 0.008
    # Measured from the official F27 mesh: native interior floor top
    # 0.1422365233 m * visual scale 0.23 = 0.0327144004 m.
    black_basket_visual_floor_z_m: float = 0.0327144
    # Preserve the authored floor surface while extending the invisible floor
    # down through the table collision volume.  An 8 mm slab can be crossed in
    # one 60 Hz step by a released parcel; this full-depth slab cannot.
    black_basket_collision_floor_bottom_z_m: float = -0.050
    black_basket_max_box_root_z_m: float = 0.095

    # The reachable drop zone sits at the near end of a long conveyor that
    # continues away from the table toward camera-right (positive world Y).
    # Its robot-facing outer frame edge remains flush with the basket edge, and
    # the two destinations form a compact row under the top-down main camera.
    conveyor_center_xy_m: tuple[float, float] = (0.69093371, 0.22)
    conveyor_visual_root_xy_m: tuple[float, float] = (0.69093371, 0.12)
    # Native bounds are 2.001 x 1.151 x 2.311 m; the belt top is z=1.781 m.
    # Non-uniform scaling yields a realistic 1.30 x 0.37 m straight conveyor
    # while preserving floor-to-belt height. Its long axis is rotated onto Y.
    conveyor_visual_scale_xyz: tuple[float, float, float] = (0.65, 0.32, 0.47)
    conveyor_visual_root_z_m: float = -0.81885
    conveyor_yaw_rad: float = np.pi / 2.0
    # The visual belt mesh spans local X=[-0.000036, 1.999399] m and
    # local Y=[-0.450031, 0.449968] m. After the authored scale and +90 deg
    # yaw, a 0.28 x 1.30 m cuboid covers the complete belt surface while
    # remaining just inside its side edges.  Extend the hidden collider from
    # the belt surface down to the worktable collider's bottom at z=-0.050 m;
    # this keeps the visible/contact surface unchanged while preventing a
    # parcel from crossing the full support in one 60 Hz physics step.
    conveyor_collision_center_xy_m: tuple[float, float] = (0.69093371, 0.77)
    conveyor_collision_size_m: tuple[float, float, float] = (0.28, 1.30, 0.068)
    conveyor_surface_z_m: float = 0.018
    conveyor_collision_center_z_m: float = -0.016
    conveyor_usable_size_m: tuple[float, float] = (0.24, 0.22)
    conveyor_max_box_root_z_m: float = 0.070

    placement_center_margin_m: float = 0.014
    min_gripper_open_fraction: float = 0.65
    max_settled_box_speed_m_s: float = 0.15
    # The long local-Y edge is rotated onto world X, leaving a 54 mm grasp
    # width in the 80 mm Franka gripper. Classification uses a static hold.
    inspection_min_lift_height_m: float = 0.070
    inspection_max_gripper_open_fraction: float = 0.85
    inspection_max_tcp_box_distance_m: float = 0.11
    # The prompt requires a 2-to-3 second static inspection.  At 15 Hz, 35
    # frames leave enough time for one settling sample followed by a full two
    # seconds of consecutive control-observable stability.
    inspection_expert_hold_frames: int = 35
    inspection_required_hold_s: float = 2.0
    inspection_max_control_position_step_m: float = 0.003
    inspection_max_control_rotation_step_rad: float = 0.020
    inspection_max_box_linear_speed_m_s: float = 0.030
    inspection_max_box_angular_speed_rad_s: float = 0.30

    @property
    def black_basket_scaled_size_m(self) -> tuple[float, float, float]:
        native = (1.1583822370, 0.7957980931, 0.5223035157)
        return tuple(value * self.black_basket_visual_scale for value in native)

    @property
    def black_basket_collision_floor_center_z_m(self) -> float:
        return 0.5 * (
            self.black_basket_visual_floor_z_m
            + self.black_basket_collision_floor_bottom_z_m
        )

    @property
    def black_basket_collision_floor_thickness_m(self) -> float:
        return (
            self.black_basket_visual_floor_z_m
            - self.black_basket_collision_floor_bottom_z_m
        )


SPEC = RattleSortSpec()


def box_kind_for_seed(seed: int) -> str:
    """Choose a deterministic random order inside each exactly balanced seed pair."""
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    pair = seed // len(BOX_KINDS)
    rng = np.random.default_rng(np.random.SeedSequence([pair, 0x434F4D]))
    return str(rng.permutation(BOX_KINDS)[seed % len(BOX_KINDS)])


def sample_initial_state(seed: int, spec: RattleSortSpec = SPEC) -> dict[str, object]:
    """Sample one visually identical uniform or front/back-imbalanced parcel."""
    seed = int(seed)
    box_kind = box_kind_for_seed(seed)
    pose_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x52415454]))
    position = [
        float(pose_rng.uniform(*spec.pickup_x_range_m)),
        float(pose_rng.uniform(*spec.pickup_y_range_m)),
        spec.box_root_z_m,
    ]
    heavy_end = "center"
    com_offset_y = 0.0
    if box_kind == "imbalanced":
        heavy_end = "front" if int(pose_rng.integers(0, 2)) == 0 else "back"
        com_offset_y = (
            spec.front_back_com_offset_m
            if heavy_end == "front"
            else -spec.front_back_com_offset_m
        )
    sine = float(np.sin(spec.pickup_yaw_rad))
    cosine = float(np.cos(spec.pickup_yaw_rad))
    com_offset_world = [-sine * com_offset_y, cosine * com_offset_y, 0.0]
    destination = "black_basket" if box_kind == "imbalanced" else "conveyor"
    return {
        "seed": seed,
        "box_kind": box_kind,
        "box_position_m": position,
        "box_yaw_rad": float(spec.pickup_yaw_rad),
        "box_mass_kg": spec.box_total_mass_kg,
        "total_mass_kg": spec.box_total_mass_kg,
        "center_of_mass_offset_local_m": [0.0, com_offset_y, 0.0],
        "center_of_mass_offset_world_m": com_offset_world,
        "imbalance_axis": "front_back",
        "heavy_end": heavy_end,
        "destination": destination,
        "pickup_region_profile": spec.pickup_region_profile,
        "pickup_x_range_m": list(spec.pickup_x_range_m),
        "pickup_y_range_m": list(spec.pickup_y_range_m),
        "black_basket_center_xy_m": list(spec.black_basket_center_xy_m),
        "conveyor_center_xy_m": list(spec.conveyor_center_xy_m),
    }


def _inside_centered_rectangle(point_xy, center_xy, size_xy, margin: float) -> bool:
    point = np.asarray(point_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    half = 0.5 * np.asarray(size_xy, dtype=np.float64) - float(margin)
    if point.shape != (2,) or center.shape != (2,) or np.any(half <= 0.0):
        raise ValueError("invalid centered-rectangle arguments")
    return bool(np.all(np.abs(point - center) <= half))


@dataclass(frozen=True)
class SuccessReport:
    success: bool
    box_kind: str
    destination: str
    box_in_correct_destination: bool
    box_in_black_basket: bool
    box_on_conveyor: bool
    box_location: str
    placement_failure_reason: str | None
    box_position_m: tuple[float, float, float]
    inspection_completed: bool
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
    inspection_completed: bool,
    gripper_open_fraction: float,
    spec: RattleSortSpec = SPEC,
) -> SuccessReport:
    if box_kind not in BOX_KINDS:
        raise ValueError(f"box_kind must be one of {BOX_KINDS}, got {box_kind!r}")
    position = np.asarray(box_position_m, dtype=np.float64)
    velocity = np.asarray(box_linear_velocity_m_s, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("box state values must be 3-vectors")
    if not all(np.isfinite(value).all() for value in (position, velocity)):
        raise ValueError("box state values must be finite")
    if not np.isfinite(gripper_open_fraction):
        raise ValueError("gripper_open_fraction must be finite")

    destination = "black_basket" if box_kind == "imbalanced" else "conveyor"
    box_in_black_basket = _inside_centered_rectangle(
        position[:2],
        spec.black_basket_center_xy_m,
        spec.black_basket_opening_size_m,
        spec.placement_center_margin_m,
    ) and 0.0 <= float(position[2]) <= spec.black_basket_max_box_root_z_m
    box_on_conveyor = _inside_centered_rectangle(
        position[:2],
        spec.conveyor_center_xy_m,
        spec.conveyor_usable_size_m,
        spec.placement_center_margin_m,
    ) and 0.0 <= float(position[2]) <= spec.conveyor_max_box_root_z_m
    box_in_correct_destination = (
        box_in_black_basket if destination == "black_basket" else box_on_conveyor
    )
    if box_in_black_basket:
        box_location = "black_basket"
    elif box_on_conveyor:
        box_location = "conveyor"
    else:
        box_location = "outside_destinations"
    if box_in_correct_destination:
        placement_failure_reason = None
    elif box_location != "outside_destinations":
        placement_failure_reason = "parcel_in_wrong_destination"
    else:
        placement_failure_reason = "parcel_not_in_any_destination"
    speed = float(np.linalg.norm(velocity))
    box_settled = speed <= spec.max_settled_box_speed_m_s
    box_released = float(gripper_open_fraction) >= spec.min_gripper_open_fraction
    return SuccessReport(
        success=bool(
            box_in_correct_destination
            and inspection_completed
            and box_released
            and box_settled
        ),
        box_kind=box_kind,
        destination=destination,
        box_in_correct_destination=bool(box_in_correct_destination),
        box_in_black_basket=bool(box_in_black_basket),
        box_on_conveyor=bool(box_on_conveyor),
        box_location=box_location,
        placement_failure_reason=placement_failure_reason,
        box_position_m=tuple(float(value) for value in position),
        inspection_completed=bool(inspection_completed),
        box_released=bool(box_released),
        box_settled=bool(box_settled),
        box_root_z_m=float(position[2]),
        box_speed_m_s=speed,
        gripper_open_fraction=float(gripper_open_fraction),
    )
