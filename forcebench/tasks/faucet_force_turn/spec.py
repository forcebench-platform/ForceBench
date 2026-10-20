"""Simulator-free force-disambiguated faucet layout and load model."""

from __future__ import annotations

import math
import random
from forcebench.seed_balance import balanced_sign, VERSION as DIRECTION_VERSION
from dataclasses import asdict, dataclass, replace
from typing import Mapping


_DIRECTION_NAMES = {-1: "clockwise", 1: "counterclockwise"}


@dataclass(frozen=True)
class FaucetForceTurnSpec:
    # Scene, camera, sink placement and lever geometry are copied from the
    # accepted faucet_turn task. Only the bidirectional load model is replaced.
    sink_asset: str = "lightwheel_kitchen/Sink054/Sink054.usd"
    sink_environment_asset: str = "lightwheel_kitchen/Sink054/Sink054_environment.usda"
    sink_texture: str = "lightwheel_kitchen/Sink054/texutre/T_Sink054_BC001.png"
    robot_support_asset: str = (
        "lightwheel_kitchen/Sink054/robot_base_counter_extension.usda"
    )
    valve_asset: str = "lightwheel_kitchen/Sink054/faucet_force_turn_valve.usda"
    sink_translation_m: tuple[float, float, float] = (0.45, 0.33, -0.049304)
    sink_yaw_deg: float = -90.0
    countertop_z_m: float = 0.0
    robot_support_position_m: tuple[float, float, float] = (-0.04, 0.0, -0.025)
    robot_support_size_m: tuple[float, float, float] = (0.52, 0.50, 0.05)
    valve_position_m: tuple[float, float, float] = (0.653, 0.015, -0.037)
    lever_pivot_offset_m: tuple[float, float, float] = (0.01, 0.0, -0.01)
    lever_rotation_sign: float = -1.0
    lever_rod_centerline_y_offset_m: float = -0.013866420774313262
    lever_rod_centerline_x_offset_m: float = -0.00009
    # Grasp the middle-upper portion of the narrow rod instead of its top tip.
    expert_grasp_radius_m: float = 0.065
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
    water_stream_length_m: float = 0.140
    water_stream_radius_m: float = 0.0045

    turn_joint_name: str = "valve_joint"

    closed_angle_deg: float = 0.0
    turn_limit_deg: float = 45.0
    detent_center_magnitude_deg: float = 25.0
    detent_center_range_deg: tuple[float, float] = (20.0, 30.0)
    detent_rise_width_deg: float = 25.0
    detent_drop_width_deg: float = 0.6
    post_drop_low_width_deg: float = 1.5
    post_drop_rise_width_deg: float = 4.0
    post_drop_resistance_nm: float = 0.14
    # Match the current faucet_turn contract: the accepted final angle is the
    # complete post-peak low-load zone [detent + 0.60, detent + 2.10] degrees.
    final_target_after_peak_deg: float = 1.35
    final_angle_tolerance_deg: float = 0.75
    # Success is checked after the gripper releases and lifts clear, so elastic
    # preload or a still-moving valve cannot produce a false positive.
    min_gripper_open_fraction: float = 0.65
    min_final_tcp_lift_m: float = 0.09
    max_final_valve_speed_rad_s: float = 0.01

    # Both directions are identical at zero. A C1 smoothstep separates them
    # over the first five degrees, followed by a gradual rise to the sampled
    # peak and a much narrower 0.6-degree unloading interval.
    easy_direction_sign: int = 1
    direction_split_width_deg: float = 5.0
    easy_shoulder_torque_nm: float = 0.070
    hard_shoulder_torque_nm: float = 0.160
    easy_peak_torque_nm: float = 0.20
    hard_peak_torque_nm: float = 0.46
    damping_nm_s_rad: float = 0.055
    base_friction_torque_nm: float = 0.045
    friction_transition_speed_rad_s: float = 0.08
    turn_max_effort_nm: float = 0.55
    valve_friction: float = 1.8
    valve_friction_combine_mode: str = "max"
    direction_attempt_threshold_deg: float = 4.0
    minimum_flow_fraction: float = 0.10
    phase: str = "force_disambiguated_turn"

    @property
    def hard_direction_sign(self) -> int:
        return -self.easy_direction_sign

    @property
    def easy_direction_name(self) -> str:
        return _DIRECTION_NAMES[self.easy_direction_sign]

    @property
    def hard_direction_name(self) -> str:
        return _DIRECTION_NAMES[self.hard_direction_sign]

    @property
    def target_final_angle_deg(self) -> float:
        return self.easy_direction_sign * (
            self.detent_center_magnitude_deg + self.final_target_after_peak_deg
        )

    @property
    def closed_grasp_reference_height_m(self) -> float:
        return (
            self.valve_position_m[2]
            + self.lever_pivot_offset_m[2]
            + self.expert_grasp_radius_m
        )

    @property
    def minimum_final_tcp_height_m(self) -> float:
        return self.closed_grasp_reference_height_m + self.min_final_tcp_lift_m

    # Compatibility aliases keep inherited faucet scene utilities small.
    @property
    def valve_joint_name(self) -> str:
        return self.turn_joint_name

    @property
    def open_limit_deg(self) -> float:
        return self.turn_limit_deg

    @property
    def detent_center_deg(self) -> float:
        return self.detent_center_magnitude_deg

    @property
    def success_angle_deg(self) -> float:
        return self.detent_center_magnitude_deg + self.detent_drop_width_deg

    @property
    def detent_peak_torque_nm(self) -> float:
        return self.easy_peak_torque_nm

    @property
    def max_effort_nm(self) -> float:
        return self.turn_max_effort_nm

    def validate(self) -> None:
        if self.easy_direction_sign not in _DIRECTION_NAMES:
            raise ValueError("easy direction sign must be -1 or +1")
        if not 0.0 < self.detent_center_magnitude_deg < self.turn_limit_deg:
            raise ValueError("detent center must lie inside both turn limits")
        if not (
            0.0
            < self.detent_center_range_deg[0]
            <= self.detent_center_range_deg[1]
            < self.turn_limit_deg
        ):
            raise ValueError("random detent range must fit inside both turn limits")
        if not math.isclose(
            self.detent_rise_width_deg,
            self.detent_center_magnitude_deg,
            abs_tol=1.0e-9,
        ):
            raise ValueError("the nominal load must rise continuously from center")
        if not 0.0 < self.direction_split_width_deg < self.detent_rise_width_deg:
            raise ValueError("direction split must fit inside the gradual rise")
        if not 0.0 < self.detent_drop_width_deg < self.detent_rise_width_deg:
            raise ValueError("the unloading interval must be narrower than the rise")
        if self.post_drop_low_width_deg <= 0.0 or self.post_drop_rise_width_deg <= 0.0:
            raise ValueError("post-drop capture intervals must be positive")
        if not (
            self.base_friction_torque_nm
            < self.easy_shoulder_torque_nm
            < self.easy_peak_torque_nm
            < self.hard_peak_torque_nm
            <= self.turn_max_effort_nm
        ):
            raise ValueError("easy-direction torques are not correctly ordered")
        if not (
            self.base_friction_torque_nm
            < self.hard_shoulder_torque_nm
            < self.hard_peak_torque_nm
            and self.easy_shoulder_torque_nm < self.hard_shoulder_torque_nm
        ):
            raise ValueError("hard-direction torques are not correctly ordered")
        if not (
            self.base_friction_torque_nm
            < self.post_drop_resistance_nm
            < self.easy_peak_torque_nm
        ):
            raise ValueError("post-drop resistance must be between base and easy peak")
        if self.final_target_after_peak_deg < self.detent_drop_width_deg:
            raise ValueError("final target must reach the unloaded side of the peak")
        if self.final_angle_tolerance_deg <= 0.0:
            raise ValueError("final angle tolerance must be positive")
        if not 0.0 <= self.min_gripper_open_fraction <= 1.0:
            raise ValueError("minimum gripper opening fraction must be in [0, 1]")
        if self.min_final_tcp_lift_m <= 0.0:
            raise ValueError("minimum final TCP lift must be positive")
        if self.max_final_valve_speed_rad_s <= 0.0:
            raise ValueError("maximum final valve speed must be positive")
        if (
            self.detent_center_range_deg[1]
            + self.detent_drop_width_deg
            + self.post_drop_low_width_deg
            + self.post_drop_rise_width_deg
            > self.turn_limit_deg
        ):
            raise ValueError("complete load curve must fit inside both turn limits")
        if (
            self.detent_center_range_deg[1]
            + self.final_target_after_peak_deg
            + self.final_angle_tolerance_deg
            > self.turn_limit_deg
        ):
            raise ValueError("final target band must fit inside both turn limits")
        if min(
            self.damping_nm_s_rad,
            self.base_friction_torque_nm,
            self.friction_transition_speed_rad_s,
            self.valve_friction,
            self.direction_attempt_threshold_deg,
        ) <= 0.0:
            raise ValueError("turn and contact parameters must be positive")
        if self.valve_friction_combine_mode not in {"min", "multiply", "average", "max"}:
            raise ValueError("unsupported friction combine mode")
        support_top = self.robot_support_position_m[2] + 0.5 * self.robot_support_size_m[2]
        if not math.isclose(support_top, self.countertop_z_m, abs_tol=1.0e-9):
            raise ValueError("robot support top must match countertop")
        pivot_y = self.valve_position_m[1] + self.lever_pivot_offset_m[1]
        if abs(self.water_outlet_position_m[1] - pivot_y) < self.minimum_valve_pipe_clearance_m:
            raise ValueError("valve must remain clear of the faucet pipe")
        if self.lever_rotation_sign not in (-1.0, 1.0):
            raise ValueError("lever rotation sign must be -1 or +1")


SPEC = FaucetForceTurnSpec()
SPEC.validate()


def _smoothstep01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _load_envelope_nm(
    magnitude_deg: float,
    shoulder_torque_nm: float,
    peak_torque_nm: float,
    spec: FaucetForceTurnSpec,
) -> float:
    peak = spec.detent_center_magnitude_deg
    rise_start = peak - spec.detent_rise_width_deg
    split_end = rise_start + spec.direction_split_width_deg
    drop_end = peak + spec.detent_drop_width_deg
    low_end = drop_end + spec.post_drop_low_width_deg
    post_end = low_end + spec.post_drop_rise_width_deg
    base = spec.base_friction_torque_nm

    if magnitude_deg <= rise_start:
        return base
    if magnitude_deg <= split_end:
        progress = _smoothstep01(
            (magnitude_deg - rise_start) / spec.direction_split_width_deg
        )
        return base + progress * (shoulder_torque_nm - base)
    if magnitude_deg <= peak:
        progress = _smoothstep01((magnitude_deg - split_end) / (peak - split_end))
        return shoulder_torque_nm + progress * (peak_torque_nm - shoulder_torque_nm)
    if magnitude_deg < drop_end:
        progress = _smoothstep01((magnitude_deg - peak) / spec.detent_drop_width_deg)
        return peak_torque_nm + progress * (base - peak_torque_nm)
    if magnitude_deg <= low_end:
        return base
    if magnitude_deg < post_end:
        progress = _smoothstep01((magnitude_deg - low_end) / spec.post_drop_rise_width_deg)
        return base + progress * (spec.post_drop_resistance_nm - base)
    return spec.post_drop_resistance_nm


def directional_resistance_nm(
    angle_rad: float,
    spec: FaucetForceTurnSpec = SPEC,
    *,
    direction_hint: float = 0.0,
) -> float:
    """Return the hidden easy/hard, center-continuous resistance envelope."""

    angle = float(angle_rad)
    hint = float(direction_hint)
    if not math.isfinite(angle) or not math.isfinite(hint):
        raise ValueError("turn angle and direction hint must be finite")
    signed_source = angle if abs(angle) > 1.0e-9 else hint
    direction_sign = 0 if signed_source == 0.0 else (1 if signed_source > 0.0 else -1)
    is_easy = direction_sign in {0, spec.easy_direction_sign}
    shoulder = spec.easy_shoulder_torque_nm if is_easy else spec.hard_shoulder_torque_nm
    peak = spec.easy_peak_torque_nm if is_easy else spec.hard_peak_torque_nm
    return _load_envelope_nm(abs(math.degrees(angle)), shoulder, peak, spec)


def turn_effort_nm(
    angle_rad: float,
    angular_velocity_rad_s: float,
    spec: FaucetForceTurnSpec = SPEC,
) -> float:
    """Return the always-active bidirectional passive turn effort."""

    angle = float(angle_rad)
    velocity = float(angular_velocity_rad_s)
    if not all(math.isfinite(value) for value in (angle, velocity)):
        raise ValueError("faucet state must be finite")
    resistance = directional_resistance_nm(angle, spec, direction_hint=velocity)
    direction = math.tanh(velocity / spec.friction_transition_speed_rad_s)
    effort = -resistance * direction - spec.damping_nm_s_rad * velocity
    return max(-spec.turn_max_effort_nm, min(spec.turn_max_effort_nm, effort))


def evaluate_turn_angle(
    angle_rad: float,
    spec: FaucetForceTurnSpec = SPEC,
) -> dict[str, object]:
    angle_deg = math.degrees(float(angle_rad))
    if not math.isfinite(angle_deg):
        raise ValueError("turn angle must be finite")
    easy_progress = spec.easy_direction_sign * angle_deg
    hard_progress = spec.hard_direction_sign * angle_deg
    # Crossing is defined at the end of the sharp unloading interval, not at
    # the later center of the post-drop capture plateau.
    detent_crossing = spec.detent_center_magnitude_deg + spec.detent_drop_width_deg
    correct_crossed = bool(
        easy_progress >= detent_crossing
        or math.isclose(
            easy_progress, detent_crossing, rel_tol=0.0, abs_tol=1.0e-9
        )
    )
    wrong_crossed = bool(
        hard_progress >= detent_crossing
        or math.isclose(
            hard_progress, detent_crossing, rel_tol=0.0, abs_tol=1.0e-9
        )
    )
    final_error = abs(angle_deg - spec.target_final_angle_deg)
    within = bool(
        final_error <= spec.final_angle_tolerance_deg
        or math.isclose(
            final_error,
            spec.final_angle_tolerance_deg,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    )
    return {
        "success": correct_crossed and within,
        "correct_direction_detent_crossed": correct_crossed,
        "wrong_direction_detent_crossed": wrong_crossed,
        "valve_angle_deg": angle_deg,
        "easy_direction_sign": spec.easy_direction_sign,
        "easy_direction": spec.easy_direction_name,
        "hard_direction_sign": spec.hard_direction_sign,
        "hard_direction": spec.hard_direction_name,
        "target_final_angle_deg": spec.target_final_angle_deg,
        "final_angle_error_deg": final_error,
        "final_angle_tolerance_deg": spec.final_angle_tolerance_deg,
        "within_final_angle_tolerance": within,
    }


def evaluate_paper_turn_angle(
    angle_rad: float,
    spec: FaucetForceTurnSpec = SPEC,
) -> dict[str, object]:
    """Paper-v1 final angle: signed progress reaches C, with no upper bound.

    Keep the historical capture-band function above unchanged for old records
    and load/diagnostic bookkeeping. This version is the public task judge.
    """
    report = evaluate_turn_angle(angle_rad, spec)
    progress = spec.easy_direction_sign * report["valve_angle_deg"]
    reached = bool(progress >= spec.detent_center_magnitude_deg or math.isclose(
        progress, spec.detent_center_magnitude_deg, rel_tol=0.0, abs_tol=1.0e-9
    ))
    report.update({
        "success": reached,
        "final_angle_reached": reached,
        "signed_final_angle_deg": progress,
        "minimum_signed_final_angle_deg": spec.detent_center_magnitude_deg,
        "scoring_protocol": "forcebench-paper-v1-faucet-direction-lower-bound",
    })
    return report


def flow_fraction(angle_rad: float, spec: FaucetForceTurnSpec = SPEC) -> float:
    """Map progress in the sampled easy direction to visible water flow."""

    progress = spec.easy_direction_sign * math.degrees(float(angle_rad))
    if progress < spec.detent_center_magnitude_deg:
        return 0.0
    normalized = min(
        1.0,
        max(
            0.0,
            (progress - spec.detent_center_magnitude_deg)
            / (spec.turn_limit_deg - spec.detent_center_magnitude_deg),
        ),
    )
    return 1.0 - normalized * (1.0 - spec.minimum_flow_fraction)


def spec_from_layout(
    layout: Mapping[str, object],
    spec: FaucetForceTurnSpec = SPEC,
) -> FaucetForceTurnSpec:
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
        detent_center_magnitude_deg=float(profile["detent_center_magnitude_deg"]),
        detent_rise_width_deg=float(profile["detent_rise_width_deg"]),
        final_target_after_peak_deg=float(profile["final_target_after_peak_deg"]),
        easy_direction_sign=int(profile["easy_direction_sign"]),
        direction_split_width_deg=float(profile["direction_split_width_deg"]),
        easy_shoulder_torque_nm=float(profile["easy_shoulder_torque_nm"]),
        hard_shoulder_torque_nm=float(profile["hard_shoulder_torque_nm"]),
        easy_peak_torque_nm=float(profile["easy_peak_torque_nm"]),
        hard_peak_torque_nm=float(profile["hard_peak_torque_nm"]),
        final_angle_tolerance_deg=float(profile["final_angle_tolerance_deg"]),
    )
    episode_spec.validate()
    return episode_spec


def sample_initial_state(
    seed: int,
    spec: FaucetForceTurnSpec = SPEC,
) -> dict[str, object]:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = random.Random(seed)
    detent = rng.uniform(*spec.detent_center_range_deg)
    easy_sign = rng.choice((-1, 1))
    easy_sign = balanced_sign(seed, easy_sign)
    target_magnitude = detent + spec.final_target_after_peak_deg
    load_profile = {
        key: value
        for key, value in asdict(spec).items()
        if key
        in {
            "closed_angle_deg",
            "turn_limit_deg",
            "detent_drop_width_deg",
            "post_drop_low_width_deg",
            "post_drop_rise_width_deg",
            "post_drop_resistance_nm",
            "final_target_after_peak_deg",
            "final_angle_tolerance_deg",
            "damping_nm_s_rad",
            "base_friction_torque_nm",
            "friction_transition_speed_rad_s",
            "turn_max_effort_nm",
        }
    }
    load_profile.update(
        {
            "detent_center_magnitude_deg": detent,
            "detent_rise_width_deg": detent,
            "easy_direction_sign": easy_sign,
            "easy_direction": _DIRECTION_NAMES[easy_sign],
            "hard_direction_sign": -easy_sign,
            "hard_direction": _DIRECTION_NAMES[-easy_sign],
            "direction_split_width_deg": spec.direction_split_width_deg,
            "easy_shoulder_torque_nm": spec.easy_shoulder_torque_nm,
            "hard_shoulder_torque_nm": spec.hard_shoulder_torque_nm,
            "easy_peak_torque_nm": spec.easy_peak_torque_nm,
            "hard_peak_torque_nm": spec.hard_peak_torque_nm,
        }
    )
    return {
        "seed": seed,
        "phase": spec.phase,
        "sink_translation_m": list(spec.sink_translation_m),
        "sink_yaw_deg": spec.sink_yaw_deg,
        "countertop_z_m": spec.countertop_z_m,
        "valve_position_m": list(spec.valve_position_m),
        "water_outlet_position_m": list(spec.water_outlet_position_m),
        "water_stream_center_m": list(spec.water_stream_center_m),
        "valve_load_profile": load_profile,
        "valve_detent_sequence": {
            "rise_start_magnitude_deg": 0.0,
            "critical_peak_magnitude_deg": detent,
            "drop_end_magnitude_deg": detent + spec.detent_drop_width_deg,
            "target_final_magnitude_deg": target_magnitude,
            "target_final_angle_deg": easy_sign * target_magnitude,
        },
        "randomization": {
            "easy_direction_choices": ["clockwise", "counterclockwise"],
            "easy_direction_uniform": True,
            "direction_assignment_version": DIRECTION_VERSION,
            "detent_center_range_deg": list(spec.detent_center_range_deg),
            "resistance_peak_angle_range_deg": list(spec.detent_center_range_deg),
            "direction_split_width_deg": spec.direction_split_width_deg,
            "easy_peak_torque_nm": spec.easy_peak_torque_nm,
            "hard_peak_torque_nm": spec.hard_peak_torque_nm,
            "peak_torques_randomized": False,
        },
    }
