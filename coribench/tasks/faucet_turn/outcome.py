"""Final outcome contract for faucet detent traversal and water flow."""

from coribench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion("detent_crossed", "detent_not_crossed", "The valve crossed the hidden detent."),
    OutcomeCriterion("water_visible", "water_not_flowing", "Water remains visibly on at submission."),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
