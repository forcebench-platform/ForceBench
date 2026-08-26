"""Shared, simulator-free primitives for task-specific outcome judges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


OUTCOME_SCHEMA_VERSION = "coribench_task_outcome_v1"


@dataclass(frozen=True)
class OutcomeCriterion:
    """One boolean condition in a task's final success contract."""

    field: str
    failure_code: str
    description: str
    expected: bool = True

    def __post_init__(self) -> None:
        for name, value in (("field", self.field), ("failure_code", self.failure_code)):
            if not value or not value.replace("_", "").isalnum():
                raise ValueError(f"invalid outcome criterion {name}: {value!r}")
        if not self.description.strip():
            raise ValueError("outcome criterion description cannot be empty")


def evaluate_criteria(
    report: Mapping[str, object],
    criteria: tuple[OutcomeCriterion, ...],
) -> dict[str, object]:
    """Return a report whose verdict is derived only from declared criteria.

    Task ``spec.py`` modules compute geometric and mechanism predicates.  The
    colocated ``outcome.py`` declares which predicates constitute success.
    Keeping this last step separate makes failure labels stable for collection,
    evaluation and offline analysis without duplicating geometry thresholds.
    """

    if not criteria:
        raise ValueError("an outcome judge must declare at least one criterion")
    fields = [criterion.field for criterion in criteria]
    if len(fields) != len(set(fields)):
        raise ValueError(f"duplicate outcome criterion fields: {fields}")

    result = dict(report)
    criterion_results: dict[str, bool] = {}
    descriptions: dict[str, str] = {}
    failure_reasons: list[str] = []
    for criterion in criteria:
        if criterion.field not in report:
            raise KeyError(f"outcome report is missing {criterion.field!r}")
        value = report[criterion.field]
        if not isinstance(value, bool):
            raise TypeError(
                f"outcome field {criterion.field!r} must be bool, got {type(value).__name__}"
            )
        passed = value is criterion.expected
        criterion_results[criterion.field] = passed
        descriptions[criterion.field] = criterion.description
        if not passed:
            failure_reasons.append(criterion.failure_code)

    success = all(criterion_results.values())
    source_success = report.get("success")
    source_success_bool = source_success if isinstance(source_success, bool) else None
    result["success"] = success
    result["failure_reasons"] = failure_reasons
    result["outcome"] = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "success": success,
        "criteria": criterion_results,
        "criterion_descriptions": descriptions,
        "failure_reasons": failure_reasons,
        "source_success": source_success_bool,
        "source_success_consistent": (
            source_success_bool is None or source_success_bool == success
        ),
    }
    return result
