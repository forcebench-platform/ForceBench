"""Simulator-free stove layout, detent load curve, and success state."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class StoveTurnOnSpec:
    stovetop_asset: str = "lightwheel_kitchen/Stovetop012/Stovetop012.usd"
    stovetop_environment_asset: str = (
        "lightwheel_kitchen/Stovetop012/Stovetop012_environment.usda"
    )
    stovetop_texture: str = (
        "lightwheel_kitchen/Stovetop012/texture/T_Stovetop012_BC001.png"
    )
    knob_asset: str = "lightwheel_kitchen/Stovetop012/stove_detent_knob.usda"
    # Rotate the accepted single-burner cooktop +90 degrees from its previous
    # world yaw, then translate it so the independently simulated center knob
    # stays at the accepted world position below the initial gripper.
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
        # The retained burner's concentric mesh components are authored at
        # local (0.000005888, 0.026311634), rather than at the USD root.
        0.435843146821,
        0.268566305108,
        0.047,
    )
    knob_joint_name: str = "stove_knob_joint"
    closed_angle_deg: float = 0.0
    # The seed-selectable detent center is the end of a long, gradual rise.
    detent_center_deg: float = 75.0
    detent_rise_width_deg: float = 75.0
    # Unload over only half a degree so the resistance drop is conspicuous.
    detent_drop_width_deg: float = 0.5
    # The minimum is a single turning point rather than a flat low plateau.
    post_drop_low_width_deg: float = 0.0
    post_drop_rise_width_deg: float = 4.0
    post_drop_resistance_nm: float = 0.08
    open_limit_deg: float = 120.0
    success_after_peak_deg: float = 0.5
    success_angle_deg: float = 75.5
    # Native task evaluation must finish near the post-drop minimum; merely
    # crossing it and stopping much farther forward is not sufficient.
    final_angle_tolerance_deg: float = 3.0
    detent_peak_torque_nm: float = 0.12
    # Match faucet_turn: a fixed passive viscous torque -c * angular_velocity.
    damping_nm_s_rad: float = 0.055
    base_friction_torque_nm: float = 0.025
    friction_transition_speed_rad_s: float = 0.08
    max_effort_nm: float = 0.35
    knob_friction: float = 2.2
    knob_friction_combine_mode: str = "max"
    # The seed randomizes the actual resistance-peak angle, not an offset proxy.
    detent_center_range_deg: tuple[float, float] = (60.0, 90.0)
    phase: str = "ignition_detent"

    def validate(self) -> None:
        if not self.closed_angle_deg < self.detent_center_deg < self.open_limit_deg:
            raise ValueError("detent center must lie strictly inside the knob range")
        if not self.detent_center_deg < self.success_angle_deg <= self.open_limit_deg:
            raise ValueError("success angle must be beyond the detent")
        if (
            self.detent_rise_width_deg <= 0.0
            or self.detent_drop_width_deg <= 0.0
            or self.detent_peak_torque_nm <= 0.0
        ):
            raise ValueError("detent rise/drop widths and peak torque must be positive")
        if self.detent_drop_width_deg >= self.detent_rise_width_deg:
            raise ValueError("detent unloading must be narrower than its rising interval")
        if self.post_drop_low_width_deg < 0.0 or self.post_drop_rise_width_deg <= 0.0:
            raise ValueError("post-drop low interval must be non-negative and rise positive")
        if not (
            self.base_friction_torque_nm
            < self.post_drop_resistance_nm
            < self.detent_peak_torque_nm
        ):
            raise ValueError("post-drop resistance must lie between base and peak")
        if self.success_after_peak_deg < self.detent_drop_width_deg:
            raise ValueError("success angle must reach the detent minimum")
        if self.final_angle_tolerance_deg <= 0.0:
            raise ValueError("final angle tolerance must be positive")
        if self.damping_nm_s_rad < 0.0 or self.base_friction_torque_nm < 0.0:
            raise ValueError("damping and friction must be non-negative")
        if self.friction_transition_speed_rad_s <= 0.0:
            raise ValueError("friction transition speed must be positive")
        if self.max_effort_nm < self.detent_peak_torque_nm:
            raise ValueError("maximum effort must not clip the detent peak")
        if self.knob_friction <= 0.0:
            raise ValueError("knob friction must be positive")
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
        if not (
            self.closed_angle_deg
            < self.detent_center_range_deg[0]
            <= self.detent_center_range_deg[1]
            < self.open_limit_deg
        ):
            raise ValueError("randomized detent centers must remain inside the knob range")
        if (
            self.detent_center_range_deg[1]
            + self.detent_drop_width_deg
            + self.post_drop_low_width_deg
            + self.post_drop_rise_width_deg
            > self.open_limit_deg
        ):
            raise ValueError("post-drop resistance rise must fit inside the knob range")
        if (
            self.detent_center_range_deg[1]
            + self.success_after_peak_deg
            + self.final_angle_tolerance_deg
            > self.open_limit_deg
        ):
            raise ValueError("randomized final-angle bands must fit inside the knob range")


SPEC = StoveTurnOnSpec()
SPEC.validate()


def _smoothstep01(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def detent_resistance_nm(
    angle_rad: float,
    spec: StoveTurnOnSpec = SPEC,
) -> float:
    """Return the faucet-style positive resistance envelope at ``angle_rad``."""

    angle_deg = math.degrees(float(angle_rad))
    if not math.isfinite(angle_deg):
        raise ValueError("knob angle must be finite")
    peak_deg = spec.detent_center_deg
    rise_start_deg = peak_deg - spec.detent_rise_width_deg
    drop_end_deg = peak_deg + spec.detent_drop_width_deg
    low_end_deg = drop_end_deg + spec.post_drop_low_width_deg
    post_rise_end_deg = low_end_deg + spec.post_drop_rise_width_deg
    base = spec.base_friction_torque_nm
    peak = spec.detent_peak_torque_nm

    if angle_deg <= rise_start_deg:
        return base
    if angle_deg <= peak_deg:
        progress = _smoothstep01(
            (angle_deg - rise_start_deg) / spec.detent_rise_width_deg
        )
        return base + progress * (peak - base)
    if angle_deg < drop_end_deg:
        progress = _smoothstep01(
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
    spec: StoveTurnOnSpec = SPEC,
) -> float:
    """Return continuous passive effort opposing the current rotation."""

    angle = float(angle_rad)
    velocity = float(angular_velocity_rad_s)
    if not math.isfinite(angle) or not math.isfinite(velocity):
        raise ValueError("knob angle and angular velocity must be finite")
    resistance = detent_resistance_nm(angle, spec)
    direction = math.tanh(
        velocity / spec.friction_transition_speed_rad_s
    )
    damping = -spec.damping_nm_s_rad * velocity
    effort = -resistance * direction + damping
    return max(-spec.max_effort_nm, min(spec.max_effort_nm, effort))


def flame_fraction(angle_rad: float, spec: StoveTurnOnSpec = SPEC) -> float:
    """Map the post-detent range to a visible burner intensity."""

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
    return 0.55 + 0.45 * progress


def evaluate_angle(angle_rad: float, spec: StoveTurnOnSpec = SPEC) -> dict[str, object]:
    angle_deg = math.degrees(float(angle_rad))
    ignition_detent_crossed = angle_deg >= spec.success_angle_deg
    final_angle_error_deg = abs(angle_deg - spec.success_angle_deg)
    within_final_angle_tolerance = (
        final_angle_error_deg <= spec.final_angle_tolerance_deg
    )
    return {
        "success": ignition_detent_crossed and within_final_angle_tolerance,
        "ignition_detent_crossed": ignition_detent_crossed,
        "knob_angle_deg": angle_deg,
        "target_final_angle_deg": spec.success_angle_deg,
        "final_angle_error_deg": final_angle_error_deg,
        "final_angle_tolerance_deg": spec.final_angle_tolerance_deg,
        "within_final_angle_tolerance": within_final_angle_tolerance,
        "burner_lit": flame_fraction(angle_rad, spec) > 0.0,
        "flame_fraction": flame_fraction(angle_rad, spec),
    }


def spec_from_layout(
    layout: Mapping[str, object], spec: StoveTurnOnSpec = SPEC
) -> StoveTurnOnSpec:
    profile = layout["knob_load_profile"]
    if not isinstance(profile, Mapping):
        raise ValueError("knob_load_profile must be a mapping")
    episode_spec = replace(
        spec,
        detent_center_deg=float(profile["detent_center_deg"]),
        detent_rise_width_deg=float(profile["detent_rise_width_deg"]),
        detent_peak_torque_nm=float(profile["detent_peak_torque_nm"]),
        success_angle_deg=float(profile["success_angle_deg"]),
        final_angle_tolerance_deg=float(profile["final_angle_tolerance_deg"]),
    )
    episode_spec.validate()
    return episode_spec


def sample_initial_state(seed: int, spec: StoveTurnOnSpec = SPEC) -> dict[str, object]:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = random.Random(seed)
    detent_center_deg = rng.uniform(*spec.detent_center_range_deg)
    # Match faucet_turn: the hidden peak angle changes with seed, while the
    # resistance magnitude stays fixed so seeds do not change task strength.
    detent_peak_torque_nm = spec.detent_peak_torque_nm
    detent_rise_width_deg = detent_center_deg - spec.closed_angle_deg
    success_angle_deg = min(
        spec.open_limit_deg,
        detent_center_deg + spec.success_after_peak_deg,
    )
    rise_start_deg = detent_center_deg - detent_rise_width_deg
    drop_end_deg = detent_center_deg + spec.detent_drop_width_deg
    low_end_deg = drop_end_deg + spec.post_drop_low_width_deg
    post_rise_end_deg = low_end_deg + spec.post_drop_rise_width_deg
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
            "success_after_peak_deg",
            "final_angle_tolerance_deg",
            "damping_nm_s_rad",
            "base_friction_torque_nm",
            "friction_transition_speed_rad_s",
            "max_effort_nm",
        }
    }
    load_profile.update(
        {
            "detent_center_deg": detent_center_deg,
            "detent_rise_width_deg": detent_rise_width_deg,
            "detent_peak_torque_nm": detent_peak_torque_nm,
            "success_angle_deg": success_angle_deg,
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
        "knob_load_profile": load_profile,
        "knob_detent_sequence": {
            "rise_start_angle_deg": rise_start_deg,
            "critical_peak_angle_deg": detent_center_deg,
            "drop_end_angle_deg": drop_end_deg,
            "low_load_end_angle_deg": low_end_deg,
            "post_drop_rise_end_angle_deg": post_rise_end_deg,
            "success_angle_deg": success_angle_deg,
        },
        "randomization": {
            "detent_center_range_deg": list(spec.detent_center_range_deg),
            "resistance_peak_angle_range_deg": list(spec.detent_center_range_deg),
            "detent_peak_torque_nm": spec.detent_peak_torque_nm,
            "detent_peak_torque_randomized": False,
            "damping_nm_s_rad": spec.damping_nm_s_rad,
            "damping_randomized": False,
        },
    }
