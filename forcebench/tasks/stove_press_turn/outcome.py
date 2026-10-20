"""Success is the latched, press-engaged crossing of the correct detent."""

from forcebench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion(
        "correct_direction_detent_crossed",
        "easy_direction_detent_not_crossed",
        "While axially engaged, the knob crossed the complete detent in the "
        "hidden lower-resistance direction; stopping angle and release are not gates.",
    ),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
