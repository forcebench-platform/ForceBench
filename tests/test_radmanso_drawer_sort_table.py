import math

import pytest

from coribench.tasks.radmanso_drawer_sort_table.spec import (
    SPEC,
    evaluate_final_state,
    sample_initial_state,
)


def test_cabinet_stands_upright_on_the_table():
    assert SPEC.cabinet_position_m[2] == pytest.approx(0.0)
    assert SPEC.cabinet_roll_rad == pytest.approx(0.0)
    assert SPEC.cabinet_pitch_rad == pytest.approx(0.0)
    assert SPEC.cabinet_yaw_rad == pytest.approx(math.pi)
    assert SPEC.cabinet_scale == pytest.approx(0.45)
    assert SPEC.cabinet_position_m[0] > 0.3


def test_drawer_front_is_perpendicular_to_table_edge_and_points_to_robot():
    direction = SPEC.pull_direction_world
    assert math.sqrt(sum(value * value for value in direction)) == pytest.approx(1.0)
    assert direction[0] == pytest.approx(0.0, abs=1.0e-9)
    assert direction[1] == pytest.approx(1.0)
    assert SPEC.top_handle_center_m[1] > SPEC.cabinet_position_m[1]
    assert SPEC.middle_handle_center_m[1] > SPEC.cabinet_position_m[1]
    assert SPEC.top_handle_center_m[2] > SPEC.middle_handle_center_m[2]


def test_layout_and_negative_opening_joint_contract():
    first = sample_initial_state(4)
    second = sample_initial_state(5)
    assert first == sample_initial_state(4)
    for layout in (first, second):
        assert layout["selected_drawer"] == "middle"
        assert layout["drawer_joint"] == "PrismaticJoint_middle"
        assert layout["handle_center_m"] == list(SPEC.middle_handle_center_m)
        assert SPEC.resistance_start_range_m[0] <= layout["resistance_start_open_m"] <= SPEC.resistance_start_range_m[1]
        assert SPEC.resistance_band_length_range_m[0] <= layout["resistance_band_length_m"] <= SPEC.resistance_band_length_range_m[1]
        assert layout["resistance_start_open_m"] < layout["success_open_m"] < layout["breakaway_open_m"]
        assert layout["breakaway_open_m"] < layout["max_safe_open_m"] < 0.30
    outside = SPEC.success_joint_position_m + SPEC.success_tolerance_m + 0.001
    assert not evaluate_final_state(outside)["success"]
    assert evaluate_final_state(SPEC.success_joint_position_m, "middle")["success"]


def test_two_hundred_seeds_randomize_band_and_dynamic_success_target_safely():
    layouts = [sample_initial_state(seed) for seed in range(200)]
    starts = {round(float(layout["resistance_start_open_m"]), 6) for layout in layouts}
    lengths = {round(float(layout["resistance_band_length_m"]), 6) for layout in layouts}
    targets = {round(float(layout["success_open_m"]), 6) for layout in layouts}
    peaks = {round(float(layout["breakaway_resistance_n"]), 6) for layout in layouts}
    assert len(starts) > 190
    assert len(lengths) > 190
    assert len(targets) > 190
    assert len(peaks) > 190

    for layout in layouts:
        assert layout["selected_drawer"] == "middle"
        assert layout["drawer_joint"] == "PrismaticJoint_middle"
        start = float(layout["resistance_start_open_m"])
        length = float(layout["resistance_band_length_m"])
        target = float(layout["success_open_m"])
        breakaway = float(layout["breakaway_open_m"])
        peak = float(layout["breakaway_resistance_n"])
        assert 0.045 <= start <= 0.090
        assert 0.012 <= length <= 0.030
        assert 5.5 <= peak <= 9.0
        assert breakaway == pytest.approx(start + length)
        assert start < target < breakaway < SPEC.max_safe_open_m
        assert SPEC.resistance_force_n(
            target,
            resistance_start_open_m=start,
            breakaway_open_m=breakaway,
            breakaway_resistance_n=peak,
        ) == pytest.approx(SPEC.stop_resistance_n, abs=1.0e-10)
        report = evaluate_final_state(
            -target,
            str(layout["selected_drawer"]),
            stable_frames=SPEC.success_stable_frames,
            peak_resistance_n=SPEC.stop_resistance_n,
            success_open_m=target,
            resistance_start_open_m=start,
            breakaway_open_m=breakaway,
            breakaway_resistance_n=peak,
        )
        assert report["success"]
        assert not report["pulled_out"]


def test_resistance_curve_has_free_rise_and_irreversible_release_regions():
    assert SPEC.resistance_force_n(0.0) == pytest.approx(SPEC.free_resistance_n)
    assert SPEC.resistance_force_n(SPEC.resistance_start_open_m) == pytest.approx(
        SPEC.free_resistance_n
    )
    middle = 0.5 * (SPEC.resistance_start_open_m + SPEC.breakaway_open_m)
    assert SPEC.free_resistance_n < SPEC.resistance_force_n(middle) < SPEC.breakaway_resistance_n
    assert SPEC.resistance_force_n(SPEC.breakaway_open_m) == pytest.approx(
        SPEC.breakaway_resistance_n
    )
    assert SPEC.resistance_force_n(middle, broken=True) == pytest.approx(
        SPEC.post_breakaway_resistance_n
    )


def test_breakaway_and_unstable_final_states_fail():
    q = SPEC.success_joint_position_m
    assert not evaluate_final_state(
        q, latch_broken=True, stable_frames=SPEC.success_stable_frames
    )["success"]
    assert not evaluate_final_state(q, stable_frames=SPEC.success_stable_frames - 1)[
        "success"
    ]
    assert evaluate_final_state(q, stable_frames=SPEC.success_stable_frames)["success"]
