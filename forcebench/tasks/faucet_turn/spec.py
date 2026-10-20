"""Simulator-free faucet detent parameters, load curve, and success state."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from dataclasses import replace
from typing import Mapping


@dataclass(frozen=True)
class FaucetTurnSpec:
    sink_asset: str = "lightwheel_kitchen/Sink054/Sink054.usd"
    sink_environment_asset: str = "lightwheel_kitchen/Sink054/Sink054_environment.usda"
    sink_texture: str = "lightwheel_kitchen/Sink054/texutre/T_Sink054_BC001.png"
    robot_support_asset: str = (
        "lightwheel_kitchen/Sink054/robot_base_counter_extension.usda"
    )
    valve_asset: str = "lightwheel_kitchen/Sink054/faucet_detent_valve.usda"
    sink_translation_m: tuple[float, float, float] = (0.45, 0.33, -0.049304)
    sink_yaw_deg: float = -90.0
    countertop_z_m: float = 0.0
    robot_support_position_m: tuple[float, float, float] = (-0.04, 0.0, -0.025)
    robot_support_size_m: tuple[float, float, float] = (0.52, 0.50, 0.05)
    # Keep the user-requested articulation root separate from the visible
    # handle pivot.  This local pivot is the position selected in the L20 live
    # tuning window; the joint axis follows the center of the movable head.
    valve_position_m: tuple[float, float, float] = (0.653, 0.015, -0.037)
    lever_pivot_offset_m: tuple[float, float, float] = (
        0.01,
        0.0,
        -0.01,
    )
    # Positive task angle is reported as 0..110 degrees, while the authored
    # faucet now opens about the opposite physical Y direction.
    lever_rotation_sign: float = -1.0
    # The visible source handle's narrow rod is not centered on its parent
    # Xform.  At the 90 mm grasp radius its centerline is 13.866 mm toward -Y;
    # this follows the visible rod's authored 4.1-degree tilt.  The user-tuned
    # head position needs this near-tip grasp to cross the detent reliably.
    lever_rod_centerline_y_offset_m: float = -0.013866420774313262
    lever_rod_centerline_x_offset_m: float = -0.00009
    expert_grasp_radius_m: float = 0.090
    # The accepted complete tuning snapshot places the control 127.796 mm
    # from the spout centerline.
    minimum_valve_pipe_clearance_m: float = 0.12
    water_outlet_position_m: tuple[float, float, float] = (
        0.491172,
        0.142796,
        0.141474,
    )
    water_stream_center_m: tuple[float, float, float] = (
        0.491172,
        0.142796,
        0.071474,
    )
    # The visible stream terminates inside the receiving container rather than
    # continuing through its floor to the bottom of the sink.
    water_stream_length_m: float = 0.140
    water_stream_radius_m: float = 0.0045
    valve_joint_name: str = "valve_joint"
    closed_angle_deg: float = 0.0
    detent_center_deg: float = 25.0
    # The critical angle is the resistance peak.  The approach is deliberately
    # broad while the post-peak unloading interval is much narrower.
    detent_rise_width_deg: float = 10.0
    detent_drop_width_deg: float = 0.6
    # Hold a short low-load capture zone immediately after snap-through, then
    # raise the resistance again so the desired stop is bounded on both sides.
    post_drop_low_width_deg: float = 1.5
    post_drop_rise_width_deg: float = 4.0
    post_drop_resistance_nm: float = 0.14
    # The faucet lever rotates in the vertical X-Z plane about the Y shaft.
    # Stop before the lever can continue down into the sink deck.
    open_limit_deg: float = 110.0
    success_angle_deg: float = 26.75
    detent_peak_torque_nm: float = 0.20
    damping_nm_s_rad: float = 0.055
    base_friction_torque_nm: float = 0.045
    friction_transition_speed_rad_s: float = 0.08
    max_effort_nm: float = 0.55
    valve_friction: float = 1.4
    minimum_flow_fraction: float = 0.10
    detent_center_range_deg: tuple[float, float] = (20.0, 30.0)
    expert_load_drop_fraction: float = 0.45
    expert_overshoot_after_peak_deg: float = 0.95
    expert_backoff_after_peak_deg: float = 0.65
    expert_backoff_tolerance_deg: float = 0.8
    expert_stop_speed_rad_s: float = 0.08
    phase: str = "interactive_detent"

    def validate(self) -> None:
        if not self.closed_angle_deg < self.detent_center_deg < self.open_limit_deg:
            raise ValueError("detent center must lie strictly inside the valve range")
        if not self.detent_center_deg < self.success_angle_deg <= self.open_limit_deg:
            raise ValueError("success angle must be beyond the detent and inside the valve range")
        if (
            self.detent_rise_width_deg <= 0.0
            or self.detent_drop_width_deg <= 0.0
            or self.detent_peak_torque_nm <= 0.0
        ):
            raise ValueError("detent rise/drop widths and peak torque must be positive")
        if self.detent_drop_width_deg >= self.detent_rise_width_deg:
            raise ValueError("detent unloading must be narrower than its rising interval")
        if self.post_drop_low_width_deg <= 0.0 or self.post_drop_rise_width_deg <= 0.0:
            raise ValueError("post-drop low and rising intervals must be positive")
        if not (
            self.base_friction_torque_nm
            < self.post_drop_resistance_nm
            < self.detent_peak_torque_nm
        ):
            raise ValueError("post-drop resistance must lie between base and peak")
        if (
            self.detent_center_range_deg[1]
            + self.detent_drop_width_deg
            + self.post_drop_low_width_deg
            + self.post_drop_rise_width_deg
            > self.open_limit_deg
        ):
            raise ValueError("post-drop resistance rise must fit inside the valve range")
        if self.damping_nm_s_rad < 0.0 or self.base_friction_torque_nm < 0.0:
            raise ValueError("damping and base friction must be non-negative")
        if self.friction_transition_speed_rad_s <= 0.0:
            raise ValueError("friction transition speed must be positive")
        if self.max_effort_nm < self.detent_peak_torque_nm:
            raise ValueError("maximum effort must not clip the detent peak")
        if not 0.0 <= self.minimum_flow_fraction <= 1.0:
            raise ValueError("minimum flow fraction must be in [0, 1]")
        if self.water_stream_length_m <= 0.0 or self.water_stream_radius_m <= 0.0:
            raise ValueError("water stream dimensions must be positive")
        if any(value <= 0.0 for value in self.robot_support_size_m):
            raise ValueError("robot support dimensions must be positive")
        support_top_z = (
            self.robot_support_position_m[2] + 0.5 * self.robot_support_size_m[2]
        )
        if not math.isclose(support_top_z, self.countertop_z_m, abs_tol=1.0e-9):
            raise ValueError("robot support top must match the countertop height")
        if self.minimum_valve_pipe_clearance_m <= 0.0:
            raise ValueError("minimum valve-pipe clearance must be positive")
        if self.expert_grasp_radius_m <= 0.0:
            raise ValueError("expert grasp radius must be positive")
        if self.lever_rotation_sign not in (-1.0, 1.0):
            raise ValueError("lever rotation sign must be -1 or +1")
        valve_pivot_y_m = self.valve_position_m[1] + self.lever_pivot_offset_m[1]
        valve_pipe_clearance = abs(self.water_outlet_position_m[1] - valve_pivot_y_m)
        if valve_pipe_clearance < self.minimum_valve_pipe_clearance_m:
            raise ValueError("valve must remain clear of the faucet pipe")
        for name, bounds in (("detent center", self.detent_center_range_deg),):
            if len(bounds) != 2 or not math.isfinite(bounds[0]) or not math.isfinite(bounds[1]):
                raise ValueError(f"{name} range must contain two finite values")
            if bounds[0] > bounds[1]:
                raise ValueError(f"{name} range must be ascending")
        if not (
            self.closed_angle_deg
            < self.detent_center_range_deg[0]
            <= self.detent_center_range_deg[1]
            < self.open_limit_deg
        ):
            raise ValueError("randomized detent centers must remain inside the valve range")
        if not 0.0 < self.expert_load_drop_fraction < 1.0:
            raise ValueError("expert load-drop fraction must lie in (0, 1)")
        if self.expert_overshoot_after_peak_deg <= self.detent_drop_width_deg:
            raise ValueError("expert overshoot must pass the detent unloading interval")
        if not (
            0.0
            < self.expert_backoff_after_peak_deg
            < self.expert_overshoot_after_peak_deg
        ):
            raise ValueError("expert backoff must remain after and closer to the peak")
        if self.expert_backoff_tolerance_deg <= 0.0 or self.expert_stop_speed_rad_s <= 0.0:
            raise ValueError("expert backoff tolerance and stop speed must be positive")


SPEC = FaucetTurnSpec()
SPEC.validate()


def _smoothstep01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _soft_linear01(value: float, edge_fraction: float = 0.05) -> float:
    """Near-linear C1 ramp with short eased ends and a bounded peak slope."""

    value = max(0.0, min(1.0, float(value)))
    edge = max(1.0e-6, min(0.49, float(edge_fraction)))
    slope = 1.0 / (1.0 - edge)
    if value < edge:
        return slope * value * value / (2.0 * edge)
    if value <= 1.0 - edge:
        return slope * (value - 0.5 * edge)
    remainder = 1.0 - value
    return 1.0 - slope * remainder * remainder / (2.0 * edge)


def detent_resistance_nm(
    angle_rad: float,
    spec: FaucetTurnSpec = SPEC,
) -> float:
    """Return the positive resistance envelope at ``angle_rad``.

    Before the detent, resistance rises gently from the base level and then
    climbs rapidly over the wider approach interval.  The critical angle is
    the unique maximum; immediately afterward the load falls back to its base
    level over the much narrower unloading interval.  A short low-load capture
    zone follows, then resistance rises again to bound the intended stop.
    Smoothstep joins keep both torque and slope continuous at every boundary.
    """

    angle_deg = math.degrees(float(angle_rad))
    if not math.isfinite(angle_deg):
        raise ValueError("valve angle must be finite")
    peak_deg = spec.detent_center_deg
    rise_start_deg = peak_deg - spec.detent_rise_width_deg
    drop_end_deg = peak_deg + spec.detent_drop_width_deg
    low_end_deg = drop_end_deg + spec.post_drop_low_width_deg
    post_rise_end_deg = low_end_deg + spec.post_drop_rise_width_deg
    base = spec.base_friction_torque_nm
    peak = spec.detent_peak_torque_nm
    shoulder = base + 0.10 * (peak - base)

    if angle_deg <= rise_start_deg:
        denominator = max(1.0e-9, rise_start_deg - spec.closed_angle_deg)
        progress = _smoothstep01(
            (angle_deg - spec.closed_angle_deg) / denominator
        )
        return base + progress * (shoulder - base)
    if angle_deg <= peak_deg:
        progress = _smoothstep01(
            (angle_deg - rise_start_deg) / spec.detent_rise_width_deg
        )
        return shoulder + progress * (peak - shoulder)
    if angle_deg < drop_end_deg:
        progress = _soft_linear01(
            (angle_deg - peak_deg) / spec.detent_drop_width_deg
        )
        return peak + progress * (base - peak)
    if angle_deg <= low_end_deg:
        return base
    if angle_deg < post_rise_end_deg:
        progress = _smoothstep01(
            (angle_deg - low_end_deg) / spec.post_drop_rise_width_deg
        )
        return base + progress * (spec.post_drop_resistance_nm - base)
    return spec.post_drop_resistance_nm


def detent_effort_nm(
    angle_rad: float,
    angular_velocity_rad_s: float,
    spec: FaucetTurnSpec = SPEC,
) -> float:
    """Return continuous passive effort opposing the current rotation."""

    angle = float(angle_rad)
    velocity = float(angular_velocity_rad_s)
    if not math.isfinite(angle) or not math.isfinite(velocity):
        raise ValueError("valve angle and angular velocity must be finite")
    resistance = detent_resistance_nm(angle, spec)
    direction = math.tanh(
        velocity / spec.friction_transition_speed_rad_s
    )
    damping = -spec.damping_nm_s_rad * velocity
    effort = -resistance * direction + damping
    return max(-spec.max_effort_nm, min(spec.max_effort_nm, effort))


def flow_fraction(angle_rad: float, spec: FaucetTurnSpec = SPEC) -> float:
    """Map valve angle to the requested post-detent water-flow convention."""

    angle_deg = math.degrees(float(angle_rad))
    if angle_deg < spec.detent_center_deg:
        return 0.0
    progress = min(
        1.0,
        max(
            0.0,
            (angle_deg - spec.detent_center_deg)
            / (spec.open_limit_deg - spec.detent_center_deg),
        ),
    )
    return 1.0 - progress * (1.0 - spec.minimum_flow_fraction)


def evaluate_angle(angle_rad: float, spec: FaucetTurnSpec = SPEC) -> dict[str, object]:
    angle_deg = math.degrees(float(angle_rad))
    crossed = angle_deg >= spec.success_angle_deg
    return {
        "success": crossed,
        "detent_crossed": crossed,
        "valve_angle_deg": angle_deg,
        "flow_fraction": flow_fraction(angle_rad, spec),
    }


def spec_from_layout(
    layout: Mapping[str, object], spec: FaucetTurnSpec = SPEC
) -> FaucetTurnSpec:
    """Build the load-curve parameters sampled for one episode."""

    profile = layout["valve_load_profile"]
    if not isinstance(profile, Mapping):
        raise ValueError("valve_load_profile must be a mapping")
    episode_spec = replace(
        spec,
        valve_position_m=tuple(float(value) for value in layout["valve_position_m"]),
        water_outlet_position_m=tuple(
            float(value) for value in layout["water_outlet_position_m"]
        ),
        water_stream_center_m=tuple(
            float(value) for value in layout["water_stream_center_m"]
        ),
        detent_center_deg=float(profile["detent_center_deg"]),
        detent_peak_torque_nm=float(profile["detent_peak_torque_nm"]),
        success_angle_deg=float(profile["success_angle_deg"]),
    )
    episode_spec.validate()
    return episode_spec


def sample_initial_state(seed: int, spec: FaucetTurnSpec = SPEC) -> dict[str, object]:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = random.Random(seed)
    offsets = (0.0, 0.0, 0.0)
    sink_translation = tuple(
        base + offset for base, offset in zip(spec.sink_translation_m, offsets)
    )
    valve_position = tuple(
        base + offset for base, offset in zip(spec.valve_position_m, offsets)
    )
    water_outlet_position = tuple(
        base + offset for base, offset in zip(spec.water_outlet_position_m, offsets)
    )
    water_stream_center = tuple(
        base + offset for base, offset in zip(spec.water_stream_center_m, offsets)
    )
    detent_center_deg = rng.uniform(*spec.detent_center_range_deg)
    # Like stove_turn_on, the seed changes the detent location but never its
    # strength.  Otherwise seed becomes a shortcut for the physical property.
    detent_peak_torque_nm = spec.detent_peak_torque_nm
    success_angle_deg = min(
        spec.open_limit_deg,
        detent_center_deg + spec.expert_overshoot_after_peak_deg,
    )
    rise_start_deg = detent_center_deg - spec.detent_rise_width_deg
    drop_end_deg = detent_center_deg + spec.detent_drop_width_deg
    low_end_deg = drop_end_deg + spec.post_drop_low_width_deg
    post_rise_end_deg = low_end_deg + spec.post_drop_rise_width_deg
    overshoot_target_deg = detent_center_deg + spec.expert_overshoot_after_peak_deg
    backoff_target_deg = detent_center_deg + spec.expert_backoff_after_peak_deg
    load_profile = {
        key: value
        for key, value in asdict(spec).items()
        if key
        in {
            "closed_angle_deg",
            "detent_rise_width_deg",
            "detent_drop_width_deg",
            "post_drop_low_width_deg",
            "post_drop_rise_width_deg",
            "post_drop_resistance_nm",
            "open_limit_deg",
            "damping_nm_s_rad",
            "base_friction_torque_nm",
            "friction_transition_speed_rad_s",
            "max_effort_nm",
        }
    }
    load_profile.update(
        {
            "detent_center_deg": detent_center_deg,
            "success_angle_deg": success_angle_deg,
            "detent_peak_torque_nm": detent_peak_torque_nm,
        }
    )
    return {
        "seed": seed,
        "phase": spec.phase,
        "sink_translation_m": list(sink_translation),
        "sink_yaw_deg": spec.sink_yaw_deg,
        "countertop_z_m": spec.countertop_z_m + offsets[2],
        "valve_position_m": list(valve_position),
        "water_outlet_position_m": list(water_outlet_position),
        "water_stream_center_m": list(water_stream_center),
        "valve_load_profile": load_profile,
        "expert_detent_sequence": {
            "rise_start_angle_deg": rise_start_deg,
            "critical_peak_angle_deg": detent_center_deg,
            "drop_end_angle_deg": drop_end_deg,
            "low_load_end_angle_deg": low_end_deg,
            "post_drop_rise_end_angle_deg": post_rise_end_deg,
            "overshoot_target_deg": overshoot_target_deg,
            "backoff_target_deg": backoff_target_deg,
            "backoff_tolerance_deg": spec.expert_backoff_tolerance_deg,
            "stop_speed_rad_s": spec.expert_stop_speed_rad_s,
            "load_drop_threshold_nm": spec.expert_load_drop_fraction
            * (detent_peak_torque_nm - spec.base_friction_torque_nm),
            "load_drop_fraction": spec.expert_load_drop_fraction,
        },
        "randomization": {
            "sink_offset_m": list(offsets),
            "geometry_randomized": False,
            "detent_center_range_deg": list(spec.detent_center_range_deg),
            "detent_peak_torque_nm": detent_peak_torque_nm,
            "detent_peak_torque_randomized": False,
        },
    }
