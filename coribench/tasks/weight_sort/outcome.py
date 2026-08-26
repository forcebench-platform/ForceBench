"""Final outcome contract for force-based weight sorting."""

from coribench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion(
        "box_in_correct_container",
        "box_misclassified",
        "The box is in the container matching its hidden mass.",
    ),
    OutcomeCriterion("box_released", "box_not_released", "The gripper released the box."),
    OutcomeCriterion("box_settled", "box_not_settled", "The box is settled inside the container."),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
