import importlib
import inspect

import pytest

from coribench.outcome import OUTCOME_SCHEMA_VERSION
from coribench.registry import task_names


@pytest.mark.parametrize("task_name", task_names())
def test_every_task_has_an_explicit_outcome_judge(task_name):
    module = importlib.import_module(f"coribench.tasks.{task_name}.outcome")
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
    module = importlib.import_module(f"coribench.tasks.{task_name}.outcome")
    baseline = {criterion.field: criterion.expected for criterion in module.CRITERIA}
    for criterion in module.CRITERIA:
        failing = {**baseline, criterion.field: not criterion.expected, "success": True}
        report = module.evaluate_report(failing)
        assert not report["success"]
        assert report["failure_reasons"] == [criterion.failure_code]
        assert not report["outcome"]["source_success_consistent"]


def test_rattle_sort_has_no_stationary_inspection_success_gate():
    outcome = importlib.import_module("coribench.tasks.rattle_sort.outcome")
    spec = importlib.import_module("coribench.tasks.rattle_sort.spec")

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
