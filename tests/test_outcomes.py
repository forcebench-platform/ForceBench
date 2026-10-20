import importlib
import inspect

import pytest

from forcebench.outcome import OUTCOME_SCHEMA_VERSION
from forcebench.registry import task_names


@pytest.mark.parametrize("task_name", task_names())
def test_every_task_has_an_explicit_outcome_judge(task_name):
    module = importlib.import_module(f"forcebench.tasks.{task_name}.outcome")
    assert module.CRITERIA
    passing = {criterion.field: criterion.expected for criterion in module.CRITERIA}
    passing["success"] = True
    report = module.evaluate_report(passing)
    assert report["success"]
    assert report["failure_reasons"] == []
    assert report["outcome"]["schema_version"] == OUTCOME_SCHEMA_VERSION
    assert report["outcome"]["source_success_consistent"]


@pytest.mark.parametrize("task_name", task_names())
def test_each_outcome_criterion_has_a_stable_failure_code(task_name):
    module = importlib.import_module(f"forcebench.tasks.{task_name}.outcome")
    baseline = {criterion.field: criterion.expected for criterion in module.CRITERIA}
    for criterion in module.CRITERIA:
        failing = {**baseline, criterion.field: not criterion.expected, "success": True}
        report = module.evaluate_report(failing)
        assert not report["success"]
        assert report["failure_reasons"] == [criterion.failure_code]
        assert not report["outcome"]["source_success_consistent"]


def test_rattle_sort_has_no_stationary_inspection_success_gate():
    outcome = importlib.import_module("forcebench.tasks.rattle_sort.outcome")
    spec = importlib.import_module("forcebench.tasks.rattle_sort.spec")

    assert tuple(criterion.field for criterion in outcome.CRITERIA) == (
        "box_in_correct_destination",
        "box_released",
        "box_settled",
    )
    assert "inspection_completed" not in inspect.signature(
        spec.evaluate_final_state
    ).parameters

    report = outcome.evaluate_report(
        {
            "success": True,
            "box_in_correct_destination": True,
            "box_released": True,
            "box_settled": True,
        }
    )
    assert report["success"]
    assert report["failure_reasons"] == []


@pytest.mark.parametrize("box_kind", ("uniform", "imbalanced"))
@pytest.mark.parametrize("inspection_completed", (False, True))
def test_rattle_sort_geometry_and_outcome_agree_without_inspection_gate(
    box_kind, inspection_completed
):
    """Both scoring layers must accept a complete placement without a hold."""
    outcome = importlib.import_module("forcebench.tasks.rattle_sort.outcome")
    spec = importlib.import_module("forcebench.tasks.rattle_sort.spec")
    center = (
        spec.SPEC.conveyor_center_xy_m
        if box_kind == "uniform"
        else spec.SPEC.black_basket_center_xy_m
    )
    source = spec.evaluate_final_state(
        box_kind=box_kind,
        box_position_m=(*center, 0.04),
        box_linear_velocity_m_s=(0.0, 0.0, 0.0),
        gripper_open_fraction=1.0,
    ).to_dict()
    report = outcome.evaluate_report(
        {**source, "inspection_completed": inspection_completed}
    )
    assert source["success"]
    assert report["success"]
    assert report["inspection_completed"] is inspection_completed
    assert report["outcome"]["source_success_consistent"]


@pytest.mark.parametrize(
    ("changes", "failure_code"),
    (
        ({"box_position_m": (0.0, 0.0, 0.04)}, "parcel_misclassified"),
        ({"gripper_open_fraction": 0.64}, "parcel_not_released"),
        ({"box_linear_velocity_m_s": (0.151, 0.0, 0.0)}, "parcel_not_settled"),
    ),
)
def test_rattle_sort_inspection_cannot_override_incomplete_placement(
    changes, failure_code
):
    outcome = importlib.import_module("forcebench.tasks.rattle_sort.outcome")
    spec = importlib.import_module("forcebench.tasks.rattle_sort.spec")
    state = {
        "box_kind": "uniform",
        "box_position_m": (*spec.SPEC.conveyor_center_xy_m, 0.04),
        "box_linear_velocity_m_s": (0.0, 0.0, 0.0),
        "gripper_open_fraction": 1.0,
    }
    source = spec.evaluate_final_state(**{**state, **changes}).to_dict()
    report = outcome.evaluate_report({**source, "inspection_completed": True})
    assert not source["success"]
    assert not report["success"]
    assert report["failure_reasons"] == [failure_code]
    assert report["outcome"]["source_success_consistent"]
