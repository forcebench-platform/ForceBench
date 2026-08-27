"""Final outcome contract for stove ignition."""

from forcebench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion(
        "ignition_detent_crossed",
        "ignition_detent_not_crossed",
        "The knob crossed the hidden ignition detent.",
    ),
    OutcomeCriterion(
        "within_final_angle_tolerance",
        "knob_not_at_ignition_angle",
        "The knob stopped near the sampled ignition angle.",
    ),
    OutcomeCriterion("flame_visible", "burner_not_lit", "The burner flame remains visible at submission."),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
