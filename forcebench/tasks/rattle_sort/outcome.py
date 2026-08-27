"""Final outcome contract for force-based parcel sorting."""

from forcebench.outcome import OutcomeCriterion, evaluate_criteria


CRITERIA = (
    OutcomeCriterion(
        "box_in_correct_destination",
        "parcel_misclassified",
        "The parcel is in the destination matching its mass distribution.",
    ),
    OutcomeCriterion(
        "box_released",
        "parcel_not_released",
        "The gripper released the parcel.",
    ),
    OutcomeCriterion(
        "box_settled",
        "parcel_not_settled",
        "The parcel is settled at submission.",
    ),
)


def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
