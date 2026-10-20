"""Simulator-free press-and-turn stove layout and directional load model."""

from __future__ import annotations

import math
import random
from forcebench.seed_balance import balanced_sign, VERSION as DIRECTION_VERSION
from dataclasses import asdict, dataclass, replace
from typing import Mapping


_DIRECTION_NAMES = {-1: "clockwise", 1: "counterclockwise"}


@dataclass(frozen=True)
class StovePressTurnSpec:
    # Reuse the accepted stove scene exactly; only the independently simulated
    # center knob gains an axial press joint and bidirectional rotation.
    stovetop_asset: str = "lightwheel_kitchen/Stovetop012/Stovetop012.usd"
    stovetop_environment_asset: str = (
        "lightwheel_kitchen/Stovetop012/Stovetop012_environment.usda"
    )
    stovetop_texture: str = (
        "lightwheel_kitchen/Stovetop012/texture/T_Stovetop012_BC001.png"
    )
    knob_asset: str = "lightwheel_kitchen/Stovetop012/stove_press_turn_knob.usda"
    stovetop_translation_m: tuple[float, float, float] = (
        0.435837259,
        0.242254671,
        0.022642156,
    )
    stovetop_yaw_deg: float = 0.0
    stovetop_orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    target_knob_local_position_m: tuple[float, float, float] = (
        0.0074582,
        -0.242391,
        -0.00209493,
    )
    target_knob_position_m: tuple[float, float, float] = (
        0.443295459,
        -0.000136329,
        0.020547226,
    )
    burner_center_m: tuple[float, float, float] = (
        0.435843146821,
        0.268566305108,
        0.047,
    )

    press_joint_name: str = "stove_press_joint"
    turn_joint_name: str = "stove_turn_joint"
    press_lower_limit_m: float = -0.008
    press_upper_limit_m: float = 0.0
    press_engage_position_m: float = -0.007
    press_return_stiffness_n_m: float = 1000.0
    press_return_damping_n_s_m: float = 15.0
    press_max_effort_n: float = 60.0
    press_velocity_limit_m_s: float = 0.20

    closed_angle_deg: float = 0.0
    turn_limit_deg: float = 120.0
    detent_center_magnitude_deg: float = 75.0
    detent_center_range_deg: tuple[float, float] = (60.0, 90.0)
    detent_rise_width_deg: float = 75.0
    detent_drop_width_deg: float = 0.5
    post_drop_low_width_deg: float = 0.0
    post_drop_rise_width_deg: float = 4.0
    post_drop_resistance_nm: float = 0.08
    success_after_peak_deg: float = 0.5
    success_angle_magnitude_deg: float = 75.5
    final_angle_tolerance_deg: float = 3.0

    # The two directions are identical at the center, then separate smoothly
    # over a short shoulder interval before continuing to the detent peak.
    # This provides an early force cue without a discontinuous torque step.
    easy_direction_sign: int = 1
    direction_split_width_deg: float = 8.0
    easy_shoulder_torque_nm: float = 0.045
    hard_shoulder_torque_nm: float = 0.12
    easy_peak_torque_nm: float = 0.12
    hard_peak_torque_nm: float = 0.34
    damping_nm_s_rad: float = 0.055
    base_friction_torque_nm: float = 0.025
    friction_transition_speed_rad_s: float = 0.08
    turn_max_effort_nm: float = 0.42
    unpressed_lock_stiffness_nm_rad: float = 18.0
    unpressed_lock_damping_nm_s_rad: float = 0.45
    unpressed_lock_max_effort_nm: float = 1.20
    # After the detent has latched, releasing the axial button captures the
    # settled angle with a compliant, near-critically-damped hold.  These
    # gains are deliberately much softer than the pre-engagement lock.
    latched_hold_stiffness_nm_rad: float = 0.60
    latched_hold_damping_nm_s_rad: float = 0.085
    latched_hold_max_effort_nm: float = 0.12
    knob_friction: float = 2.2
    knob_friction_combine_mode: str = "max"
    direction_attempt_threshold_deg: float = 5.0
    phase: str = "press_then_force_disambiguated_turn"

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
    def target_angle_magnitude_deg(self) -> float:
        # Retain the expert's stopping target 3 deg beyond the crossing.
        # This target/tolerance is diagnostic, not an evaluation success gate.
        return self.success_angle_magnitude_deg + self.final_angle_tolerance_deg

    @property
    def target_final_angle_deg(self) -> float:
        return self.easy_direction_sign * self.target_angle_magnitude_deg

    def validate(self) -> None:
        if self.easy_direction_sign not in _DIRECTION_NAMES:
            raise ValueError("easy direction sign must be -1 or +1")
        if not self.press_lower_limit_m < self.press_engage_position_m < 0.0:
            raise ValueError("press engagement must lie inside the negative press range")
        if self.press_upper_limit_m != 0.0:
            raise ValueError("the released press position must be zero")
        if min(
            self.press_return_stiffness_n_m,
            self.press_return_damping_n_s_m,
            self.press_max_effort_n,
            self.press_velocity_limit_m_s,
        ) <= 0.0:
            raise ValueError("press actuator parameters must be positive")
        if not 0.0 < self.detent_center_magnitude_deg < self.turn_limit_deg:
            raise ValueError("detent center must lie inside the bidirectional turn range")
        if not (
            0.0
            < self.detent_center_range_deg[0]
            <= self.detent_center_range_deg[1]
            < self.turn_limit_deg
        ):
            raise ValueError("randomized detent magnitudes must fit inside the turn range")
        if (
            self.detent_rise_width_deg <= 0.0
            or self.detent_drop_width_deg <= 0.0
            or self.detent_drop_width_deg >= self.detent_rise_width_deg
        ):
            raise ValueError("the detent must rise gradually and unload over a narrower interval")
        if not 0.0 < self.direction_split_width_deg < self.detent_rise_width_deg:
            raise ValueError("direction split must fit inside the gradual rise interval")
        if self.post_drop_low_width_deg < 0.0 or self.post_drop_rise_width_deg <= 0.0:
            raise ValueError("post-drop intervals must be non-negative/positive")
        if not (
            self.base_friction_torque_nm
            < self.post_drop_resistance_nm
            < self.easy_peak_torque_nm
            < self.hard_peak_torque_nm
            <= self.turn_max_effort_nm
        ):
            raise ValueError("directional resistance torques are not strictly ordered")
        if not (
            self.base_friction_torque_nm
            < self.easy_shoulder_torque_nm
            < self.easy_peak_torque_nm
            and self.base_friction_torque_nm
            < self.hard_shoulder_torque_nm
            < self.hard_peak_torque_nm
            and self.easy_shoulder_torque_nm < self.hard_shoulder_torque_nm
        ):
            raise ValueError("directional shoulder torques are not correctly ordered")
        if self.success_after_peak_deg < self.detent_drop_width_deg:
            raise ValueError("success angle must reach the post-peak minimum")
        if self.final_angle_tolerance_deg <= 0.0:
            raise ValueError("final angle tolerance must be positive")
        if (
            self.detent_center_range_deg[1]
            + self.success_after_peak_deg
            + 2.0 * self.final_angle_tolerance_deg
            > self.turn_limit_deg
        ):
            raise ValueError("the final target band must fit inside both turn limits")
        if (
            self.detent_center_range_deg[1]
            + self.detent_drop_width_deg
            + self.post_drop_low_width_deg
            + self.post_drop_rise_width_deg
            > self.turn_limit_deg
        ):
            raise ValueError("the complete load curve must fit inside both turn limits")
        if min(
            self.damping_nm_s_rad,
            self.base_friction_torque_nm,
            self.friction_transition_speed_rad_s,
            self.unpressed_lock_stiffness_nm_rad,
            self.unpressed_lock_damping_nm_s_rad,
            self.unpressed_lock_max_effort_nm,
            self.latched_hold_stiffness_nm_rad,
            self.latched_hold_damping_nm_s_rad,
            self.latched_hold_max_effort_nm,
            self.knob_friction,
            self.direction_attempt_threshold_deg,
        ) <= 0.0:
            raise ValueError("turn resistance and lock parameters must be positive")
        if self.unpressed_lock_max_effort_nm <= self.turn_max_effort_nm:
            raise ValueError("the unpressed lock must be stronger than either pressed direction")
        if self.knob_friction_combine_mode not in {"min", "multiply", "average", "max"}:
            raise ValueError("unsupported knob friction combine mode")

        yaw = math.radians(self.stovetop_yaw_deg)
        local_x, local_y, local_z = self.target_knob_local_position_m
        rotated_local = (
            math.cos(yaw) * local_x - math.sin(yaw) * local_y,
            math.sin(yaw) * local_x + math.cos(yaw) * local_y,
            local_z,
        )
        expected_knob = tuple(
            base + local for base, local in zip(self.stovetop_translation_m, rotated_local)
        )
        if any(
            abs(expected - actual) > 1.0e-8
            for expected, actual in zip(expected_knob, self.target_knob_position_m)
        ):
            raise ValueError("target knob position must follow the source asset transform")
        quaternion_norm = sum(value * value for value in self.stovetop_orientation_xyzw)
        if abs(quaternion_norm - 1.0) > 1.0e-6:
            raise ValueError("stovetop orientation must be a unit quaternion")


SPEC = StovePressTurnSpec()
SPEC.validate()


def _smoothstep01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def is_press_engaged(press_position_m: float, spec: StovePressTurnSpec = SPEC) -> bool:
    position = float(press_position_m)
    if not math.isfinite(position):
        raise ValueError("press position must be finite")
    return position <= spec.press_engage_position_m


def press_fraction(press_position_m: float, spec: StovePressTurnSpec = SPEC) -> float:
    position = float(press_position_m)
    if not math.isfinite(position):
        raise ValueError("press position must be finite")
    return max(0.0, min(1.0, position / spec.press_lower_limit_m))


def _load_envelope_nm(
    magnitude_deg: float,
    shoulder_torque_nm: float,
    peak_torque_nm: float,
    spec: StovePressTurnSpec,
) -> float:
    peak_deg = spec.detent_center_magnitude_deg
    rise_start_deg = peak_deg - spec.detent_rise_width_deg
    drop_end_deg = peak_deg + spec.detent_drop_width_deg
    low_end_deg = drop_end_deg + spec.post_drop_low_width_deg
    post_rise_end_deg = low_end_deg + spec.post_drop_rise_width_deg
    base = spec.base_friction_torque_nm

    split_end_deg = rise_start_deg + spec.direction_split_width_deg

    if magnitude_deg <= rise_start_deg:
        return base
    if magnitude_deg <= split_end_deg:
        progress = _smoothstep01(
            (magnitude_deg - rise_start_deg) / spec.direction_split_width_deg
        )
        return base + progress * (shoulder_torque_nm - base)
    if magnitude_deg <= peak_deg:
        progress = _smoothstep01(
            (magnitude_deg - split_end_deg) / (peak_deg - split_end_deg)
        )
        return shoulder_torque_nm + progress * (peak_torque_nm - shoulder_torque_nm)
    if magnitude_deg < drop_end_deg:
        progress = _smoothstep01(
            (magnitude_deg - peak_deg) / spec.detent_drop_width_deg
        )
        return peak_torque_nm + progress * (base - peak_torque_nm)
    if magnitude_deg <= low_end_deg:
        return base
    if magnitude_deg < post_rise_end_deg:
        progress = _smoothstep01(
            (magnitude_deg - low_end_deg) / spec.post_drop_rise_width_deg
        )
        return base + progress * (spec.post_drop_resistance_nm - base)
    return spec.post_drop_resistance_nm


def directional_resistance_nm(
    angle_rad: float,
    spec: StovePressTurnSpec = SPEC,
    *,
    direction_hint: float = 0.0,
) -> float:
    """Return the hidden easy/hard resistance envelope for a signed angle."""

    angle = float(angle_rad)
    hint = float(direction_hint)
    if not math.isfinite(angle) or not math.isfinite(hint):
        raise ValueError("turn angle and direction hint must be finite")
    signed_source = angle if abs(angle) > 1.0e-9 else hint
    direction_sign = 0 if signed_source == 0.0 else (1 if signed_source > 0.0 else -1)
    is_easy = direction_sign in {0, spec.easy_direction_sign}
    shoulder = (
        spec.easy_shoulder_torque_nm if is_easy else spec.hard_shoulder_torque_nm
    )
    peak = spec.easy_peak_torque_nm if is_easy else spec.hard_peak_torque_nm
    return _load_envelope_nm(abs(math.degrees(angle)), shoulder, peak, spec)


def turn_effort_nm(
    angle_rad: float,
    angular_velocity_rad_s: float,
    press_position_m: float,
    spec: StovePressTurnSpec = SPEC,
    *,
    mechanism_latched: bool = False,
    unpressed_hold_angle_rad: float = 0.0,
) -> float:
    """Return passive turn effort, including the lock before axial engagement.

    Releasing the axial press locks the knob at the angle where it was
    released.  Before the detent this is the strong press-to-unlock gate; after
    the detent it becomes a compliant hold that absorbs release transients.
    The runtime updates ``unpressed_hold_angle_rad`` on every
    pressed-to-released transition, so neither mode pulls the knob back to
    zero.
    """

    angle = float(angle_rad)
    velocity = float(angular_velocity_rad_s)
    press_position = float(press_position_m)
    hold_angle = float(unpressed_hold_angle_rad)
    if not all(
        math.isfinite(value)
        for value in (angle, velocity, press_position, hold_angle)
    ):
        raise ValueError("turn state must be finite")
    if not is_press_engaged(press_position, spec):
        if mechanism_latched:
            effort = (
                -spec.latched_hold_stiffness_nm_rad * (angle - hold_angle)
                - spec.latched_hold_damping_nm_s_rad * velocity
            )
            limit = spec.latched_hold_max_effort_nm
        else:
            effort = (
                -spec.unpressed_lock_stiffness_nm_rad * (angle - hold_angle)
                - spec.unpressed_lock_damping_nm_s_rad * velocity
            )
            limit = spec.unpressed_lock_max_effort_nm
        return max(-limit, min(limit, effort))

    resistance = directional_resistance_nm(angle, spec, direction_hint=velocity)
    direction = math.tanh(velocity / spec.friction_transition_speed_rad_s)
    effort = -resistance * direction - spec.damping_nm_s_rad * velocity
    limit = spec.turn_max_effort_nm
    return max(-limit, min(limit, effort))


def evaluate_turn_angle(
    angle_rad: float,
    spec: StovePressTurnSpec = SPEC,
) -> dict[str, object]:
    angle_deg = math.degrees(float(angle_rad))
    if not math.isfinite(angle_deg):
        raise ValueError("turn angle must be finite")
    signed_easy_progress_deg = spec.easy_direction_sign * angle_deg
    signed_hard_progress_deg = spec.hard_direction_sign * angle_deg
    correct_direction_detent_crossed = (
        signed_easy_progress_deg >= spec.success_angle_magnitude_deg
    )
    wrong_direction_detent_crossed = (
        signed_hard_progress_deg >= spec.success_angle_magnitude_deg
    )
    final_angle_error_deg = abs(angle_deg - spec.target_final_angle_deg)
    within_final_angle_tolerance = (
        final_angle_error_deg <= spec.final_angle_tolerance_deg
    )
    return {
        "success": correct_direction_detent_crossed,
        "detent_crossing_magnitude_deg": spec.success_angle_magnitude_deg,
        "success_requires_final_angle_tolerance": False,
        "correct_direction_detent_crossed": correct_direction_detent_crossed,
        "wrong_direction_detent_crossed": wrong_direction_detent_crossed,
        "knob_angle_deg": angle_deg,
        "easy_direction_sign": spec.easy_direction_sign,
        "easy_direction": spec.easy_direction_name,
        "hard_direction_sign": spec.hard_direction_sign,
        "hard_direction": spec.hard_direction_name,
        "target_final_angle_deg": spec.target_final_angle_deg,
        "final_angle_error_deg": final_angle_error_deg,
        "final_angle_tolerance_deg": spec.final_angle_tolerance_deg,
        "within_final_angle_tolerance": within_final_angle_tolerance,
    }


def flame_fraction(angle_rad: float, spec: StovePressTurnSpec = SPEC) -> float:
    """Map progress in the sampled easy direction to burner intensity."""

    progress_deg = spec.easy_direction_sign * math.degrees(float(angle_rad))
    if progress_deg < spec.detent_center_magnitude_deg:
        return 0.0
    fraction = min(
        1.0,
        max(
            0.0,
            (progress_deg - spec.detent_center_magnitude_deg)
            / (spec.turn_limit_deg - spec.detent_center_magnitude_deg),
        ),
    )
    return 0.55 + 0.45 * fraction


def spec_from_layout(
    layout: Mapping[str, object],
    spec: StovePressTurnSpec = SPEC,
) -> StovePressTurnSpec:
    profile = layout["knob_load_profile"]
    if not isinstance(profile, Mapping):
        raise ValueError("knob_load_profile must be a mapping")
    episode_spec = replace(
        spec,
        detent_center_magnitude_deg=float(profile["detent_center_magnitude_deg"]),
        detent_rise_width_deg=float(profile["detent_rise_width_deg"]),
        success_angle_magnitude_deg=float(profile["success_angle_magnitude_deg"]),
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
    spec: StovePressTurnSpec = SPEC,
) -> dict[str, object]:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = random.Random(seed)
    detent_center_deg = rng.uniform(*spec.detent_center_range_deg)
    easy_direction_sign = rng.choice((-1, 1))
    easy_direction_sign = balanced_sign(seed, easy_direction_sign)
    detent_rise_width_deg = detent_center_deg - spec.closed_angle_deg
    success_angle_deg = detent_center_deg + spec.success_after_peak_deg
    target_angle_magnitude_deg = success_angle_deg + spec.final_angle_tolerance_deg
    signed_success_edges = sorted(
        (
            easy_direction_sign * success_angle_deg,
            easy_direction_sign * spec.turn_limit_deg,
        )
    )

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
            "success_after_peak_deg",
            "final_angle_tolerance_deg",
            "damping_nm_s_rad",
            "base_friction_torque_nm",
            "friction_transition_speed_rad_s",
            "turn_max_effort_nm",
            "unpressed_lock_stiffness_nm_rad",
            "unpressed_lock_damping_nm_s_rad",
            "unpressed_lock_max_effort_nm",
            "latched_hold_stiffness_nm_rad",
            "latched_hold_damping_nm_s_rad",
            "latched_hold_max_effort_nm",
        }
    }
    load_profile.update(
        {
            "detent_center_magnitude_deg": detent_center_deg,
            "detent_rise_width_deg": detent_rise_width_deg,
            "success_angle_magnitude_deg": success_angle_deg,
            "target_angle_magnitude_deg": target_angle_magnitude_deg,
            "easy_direction_sign": easy_direction_sign,
            "easy_direction": _DIRECTION_NAMES[easy_direction_sign],
            "hard_direction_sign": -easy_direction_sign,
            "hard_direction": _DIRECTION_NAMES[-easy_direction_sign],
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
        "stovetop_translation_m": list(spec.stovetop_translation_m),
        "stovetop_yaw_deg": spec.stovetop_yaw_deg,
        "stovetop_orientation_xyzw": list(spec.stovetop_orientation_xyzw),
        "target_knob_position_m": list(spec.target_knob_position_m),
        "burner_center_m": list(spec.burner_center_m),
        "press_gate": {
            "lower_limit_m": spec.press_lower_limit_m,
            "engage_position_m": spec.press_engage_position_m,
            "return_stiffness_n_m": spec.press_return_stiffness_n_m,
        },
        "knob_load_profile": load_profile,
        "knob_detent_sequence": {
            "rise_start_magnitude_deg": 0.0,
            "critical_peak_magnitude_deg": detent_center_deg,
            "drop_end_magnitude_deg": detent_center_deg + spec.detent_drop_width_deg,
            "success_magnitude_deg": success_angle_deg,
            "target_final_angle_deg": easy_direction_sign * target_angle_magnitude_deg,
            "final_success_angle_range_deg": signed_success_edges,
            "success_criterion": "pressed_correct_direction_detent_crossing_latched",
            "success_requires_final_angle_tolerance": False,
            "success_is_latched_event": True,
        },
        "randomization": {
            "easy_direction_choices": ["clockwise", "counterclockwise"],
            "easy_direction_uniform": True,
            "direction_assignment_version": DIRECTION_VERSION,
            "detent_center_range_deg": list(spec.detent_center_range_deg),
            "resistance_peak_angle_range_deg": list(spec.detent_center_range_deg),
            "direction_split_width_deg": spec.direction_split_width_deg,
            "easy_shoulder_torque_nm": spec.easy_shoulder_torque_nm,
            "hard_shoulder_torque_nm": spec.hard_shoulder_torque_nm,
            "easy_peak_torque_nm": spec.easy_peak_torque_nm,
            "hard_peak_torque_nm": spec.hard_peak_torque_nm,
            "peak_torques_randomized": False,
        },
    }
