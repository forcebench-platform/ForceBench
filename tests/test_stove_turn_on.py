import math
from pathlib import Path

import pytest

from forcebench.registry import get_task_manifest, task_names
from forcebench.task_api import CameraProfile
from forcebench.tasks.stove_turn_on.spec import (
    SPEC,
    detent_effort_nm,
    detent_resistance_nm,
    evaluate_angle,
    flame_fraction,
    sample_initial_state,
    spec_from_layout,
)


def test_stove_task_is_discoverable_and_uses_task_focused_camera():
    assert "stove_turn_on" in task_names()
    manifest = get_task_manifest("stove_turn_on")
    assert manifest.runtime_class.endswith(":StoveTurnOnTask")
    runtime_source = (
        Path(__file__).resolve().parents[1]
        / "forcebench"
        / "tasks"
        / "stove_turn_on"
        / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "camera_profile = CameraProfile(" in runtime_source
    assert "table_position_m=(0.918243014, -0.000136329, 0.419075544)" in runtime_source
    assert "table_target_m=(0.443295459, -0.000136329, 0.020547226)" in runtime_source
    assert "0.640856382" in runtime_source
    assert "0.298836239" in runtime_source
    assert "standby_joint_positions" not in runtime_source
    assert CameraProfile().table_position_m == (1.30, 0.00, 0.69)


def test_stove_knob_uses_high_friction_contact_without_min_clamping():
    assert SPEC.knob_friction == pytest.approx(2.2)
    assert SPEC.knob_friction_combine_mode == "max"
    runtime_source = (
        Path(__file__).resolve().parents[1]
        / "forcebench"
        / "tasks"
        / "stove_turn_on"
        / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "friction_combine_mode=SPEC.knob_friction_combine_mode" in runtime_source


def test_stove_layout_keeps_geometry_fixed_and_randomizes_only_the_load():
    layout = sample_initial_state(3)
    assert layout == sample_initial_state(3)
    assert layout != sample_initial_state(4)
    assert layout["stovetop_translation_m"] == list(SPEC.stovetop_translation_m)
    assert layout["stovetop_yaw_deg"] == SPEC.stovetop_yaw_deg
    assert layout["stovetop_orientation_xyzw"] == list(SPEC.stovetop_orientation_xyzw)
    assert layout["target_knob_position_m"] == list(SPEC.target_knob_position_m)
    assert layout["burner_center_m"] == list(SPEC.burner_center_m)
    assert SPEC.stovetop_translation_m == pytest.approx(
        (0.435837259, 0.242254671, 0.022642156)
    )
    assert SPEC.target_knob_position_m[0] == pytest.approx(0.443295459)
    assert SPEC.burner_center_m == pytest.approx(
        (0.435837259, 0.242254671, 0.055)
    )
    assert SPEC.stovetop_yaw_deg == 0.0
    assert SPEC.stovetop_orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert SPEC.target_knob_position_m[1] < SPEC.burner_center_m[1]
    assert SPEC.detent_center_range_deg == (60.0, 90.0)
    sampled_peak_angle_deg = layout["knob_load_profile"]["detent_center_deg"]
    assert 60.0 <= sampled_peak_angle_deg <= 90.0
    assert layout["randomization"]["resistance_peak_angle_range_deg"] == [60.0, 90.0]
    assert layout["knob_load_profile"]["detent_peak_torque_nm"] == pytest.approx(
        SPEC.detent_peak_torque_nm
    )
    assert layout["randomization"]["detent_peak_torque_randomized"] is False
    assert SPEC.damping_nm_s_rad == pytest.approx(0.055)
    assert layout["knob_load_profile"]["damping_nm_s_rad"] == pytest.approx(
        SPEC.damping_nm_s_rad
    )
    assert layout["randomization"]["damping_nm_s_rad"] == pytest.approx(
        SPEC.damping_nm_s_rad
    )
    assert layout["randomization"]["damping_randomized"] is False
    assert {
        sample_initial_state(seed)["knob_load_profile"]["damping_nm_s_rad"]
        for seed in range(32)
    } == {SPEC.damping_nm_s_rad}
    sampled_angles = [
        sample_initial_state(seed)["knob_load_profile"]["detent_center_deg"]
        for seed in range(256)
    ]
    assert min(sampled_angles) >= 60.0
    assert max(sampled_angles) <= 90.0
    assert len({round(value, 9) for value in sampled_angles}) == len(sampled_angles)
    assert {
        sample_initial_state(seed)["knob_load_profile"]["detent_peak_torque_nm"]
        for seed in range(32)
    } == {SPEC.detent_peak_torque_nm}
    episode_spec = spec_from_layout(layout)
    assert episode_spec.success_angle_deg == pytest.approx(
        episode_spec.detent_center_deg + episode_spec.success_after_peak_deg
    )
    assert episode_spec.final_angle_tolerance_deg == pytest.approx(3.0)
    sequence = layout["knob_detent_sequence"]
    assert sequence["critical_peak_angle_deg"] == pytest.approx(
        episode_spec.detent_center_deg
    )
    assert sequence["rise_start_angle_deg"] == pytest.approx(SPEC.closed_angle_deg)
    assert sequence["critical_peak_angle_deg"] < sequence["drop_end_angle_deg"]
    assert sequence["drop_end_angle_deg"] == pytest.approx(
        sequence["success_angle_deg"]
    )
    assert sequence["success_angle_deg"] == pytest.approx(
        sequence["low_load_end_angle_deg"]
    )
    assert sequence["low_load_end_angle_deg"] < sequence["post_drop_rise_end_angle_deg"]


def test_stove_resistance_rises_slowly_then_drops_sharply_to_a_unique_minimum():
    rise_start_deg = SPEC.detent_center_deg - SPEC.detent_rise_width_deg
    drop_end_deg = SPEC.detent_center_deg + SPEC.detent_drop_width_deg
    low_end_deg = drop_end_deg + SPEC.post_drop_low_width_deg
    post_rise_end_deg = low_end_deg + SPEC.post_drop_rise_width_deg
    angles_deg = [
        index / 100.0 for index in range(round(100.0 * SPEC.open_limit_deg) + 1)
    ]
    samples = [
        detent_resistance_nm(math.radians(angle_deg)) for angle_deg in angles_deg
    ]
    peak_index = max(range(len(samples)), key=samples.__getitem__)
    assert angles_deg[peak_index] == pytest.approx(SPEC.detent_center_deg)
    assert samples[peak_index] == pytest.approx(SPEC.detent_peak_torque_nm)
    assert detent_resistance_nm(math.radians(0.0)) == pytest.approx(
        SPEC.base_friction_torque_nm
    )
    assert detent_resistance_nm(math.radians(drop_end_deg)) == pytest.approx(
        SPEC.base_friction_torque_nm
    )
    assert detent_resistance_nm(math.radians(low_end_deg)) == pytest.approx(
        SPEC.base_friction_torque_nm
    )
    assert detent_resistance_nm(math.radians(post_rise_end_deg)) == pytest.approx(
        SPEC.post_drop_resistance_nm
    )
    rising = [
        detent_resistance_nm(math.radians(value))
        for value in (
            rise_start_deg,
            rise_start_deg + 0.25 * SPEC.detent_rise_width_deg,
            rise_start_deg + 0.50 * SPEC.detent_rise_width_deg,
            rise_start_deg + 0.75 * SPEC.detent_rise_width_deg,
            SPEC.detent_center_deg,
        )
    ]
    falling = [
        detent_resistance_nm(math.radians(value))
        for value in (
            SPEC.detent_center_deg,
            SPEC.detent_center_deg + 0.125,
            SPEC.detent_center_deg + 0.250,
            SPEC.detent_center_deg + 0.375,
            drop_end_deg,
        )
    ]
    assert rising == sorted(rising)
    assert falling == sorted(falling, reverse=True)
    assert SPEC.detent_drop_width_deg < SPEC.detent_rise_width_deg
    pre_drop_samples = samples[: peak_index + 1]
    drop_end_index = round(100.0 * drop_end_deg)
    drop_samples = samples[peak_index : drop_end_index + 1]
    maximum_rise_step = max(
        abs(right - left) for left, right in zip(pre_drop_samples, pre_drop_samples[1:])
    )
    maximum_drop_step = max(
        abs(right - left) for left, right in zip(drop_samples, drop_samples[1:])
    )
    assert maximum_drop_step > 50.0 * maximum_rise_step


def test_stove_effort_opposes_velocity_without_a_static_torque_jump():
    center = math.radians(SPEC.detent_center_deg)
    assert detent_effort_nm(center, 0.0) == pytest.approx(0.0)
    assert detent_effort_nm(center, 1.0) < 0.0
    assert detent_effort_nm(center, -1.0) > 0.0
    speed = 1.0
    expected_passive_effort = (
        -SPEC.detent_peak_torque_nm
        * math.tanh(speed / SPEC.friction_transition_speed_rad_s)
        - SPEC.damping_nm_s_rad * speed
    )
    assert detent_effort_nm(center, speed) == pytest.approx(expected_passive_effort)
    assert detent_effort_nm(center, -speed) == pytest.approx(-expected_passive_effort)
    assert abs(detent_effort_nm(center, 100.0)) <= SPEC.max_effort_nm


def test_burner_ignites_only_after_the_detent():
    assert SPEC.detent_center_range_deg == (60.0, 90.0)
    assert SPEC.detent_center_deg == pytest.approx(75.0)
    assert flame_fraction(math.radians(SPEC.detent_center_deg - 0.1)) == 0.0
    assert flame_fraction(math.radians(SPEC.detent_center_deg)) > 0.0
    assert not evaluate_angle(math.radians(SPEC.success_angle_deg - 0.1))["success"]
    report = evaluate_angle(math.radians(SPEC.success_angle_deg))
    assert report["success"]
    assert report["burner_lit"]


def test_native_success_requires_finishing_near_the_post_drop_minimum():
    target_deg = SPEC.success_angle_deg
    tolerance_deg = SPEC.final_angle_tolerance_deg
    within = evaluate_angle(math.radians(target_deg + tolerance_deg))
    assert within["success"]
    assert within["within_final_angle_tolerance"]
    assert within["final_angle_error_deg"] == pytest.approx(tolerance_deg)

    too_far = evaluate_angle(math.radians(target_deg + tolerance_deg + 0.01))
    assert too_far["ignition_detent_crossed"]
    assert too_far["burner_lit"]
    assert not too_far["within_final_angle_tolerance"]
    assert not too_far["success"]
