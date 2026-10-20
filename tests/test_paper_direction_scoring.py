"""Published final-angle rule is independent of the historical capture band."""
from dataclasses import replace
import math
import pytest

from forcebench.tasks.faucet_force_turn.spec import SPEC, evaluate_paper_turn_angle, evaluate_turn_angle
from forcebench.tasks.faucet_force_turn.outcome import CRITERIA, evaluate_report


@pytest.mark.parametrize("direction", (-1, 1))
@pytest.mark.parametrize("offset,success", ((-0.01, False), (0.0, True), (0.1, True), (1.35, True), (10.0, True)))
def test_paper_lower_bound_has_no_capture_band(direction, offset, success):
    spec = replace(SPEC, easy_direction_sign=direction)
    report = evaluate_paper_turn_angle(math.radians(direction * (spec.detent_center_magnitude_deg + offset)), spec)
    assert report["final_angle_reached"] is success
    assert report["success"] is success
    report.update(water_visible=True, gripper_released=True, tcp_lifted_clear=True, valve_stationary=True)
    assert evaluate_report(report)["success"] is success
    assert not evaluate_paper_turn_angle(math.radians(-direction * (spec.detent_center_magnitude_deg + 10)), spec)["success"]
    if offset == 10:
        assert not evaluate_turn_angle(math.radians(direction * (spec.detent_center_magnitude_deg + offset)), spec)["success"]


@pytest.mark.parametrize("failed", [item.field for item in CRITERIA])
def test_each_complete_task_condition_is_required(failed):
    report = {item.field: True for item in CRITERIA}
    report["success"] = True
    report[failed] = False
    assert not evaluate_report(report)["success"]


def test_returning_below_peak_does_not_pass_using_latched_crossing():
    report = evaluate_paper_turn_angle(math.radians(SPEC.easy_direction_sign * (SPEC.detent_center_magnitude_deg - 1)))
    report.update(correct_direction_detent_crossed=True, water_visible=True,
                  gripper_released=True, tcp_lifted_clear=True, valve_stationary=True)
    assert not evaluate_report(report)["success"]
