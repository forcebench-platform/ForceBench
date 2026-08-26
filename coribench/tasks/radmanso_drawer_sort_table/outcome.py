"""Final outcome contract for stopping the selected drawer in resistance."""

from coribench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion(
        "drawer_open_enough",
        "drawer_not_in_resistance",
        "The selected drawer entered the resistance region.",
    ),
    OutcomeCriterion("within_target_window", "drawer_missed_target", "The drawer stopped inside the target window."),
    OutcomeCriterion("stable_at_target", "drawer_not_stable", "The drawer remained stable at the target."),
    OutcomeCriterion(
        "entered_high_resistance",
        "resistance_not_detected",
        "The trajectory reached the designed high-resistance band.",
    ),
    OutcomeCriterion("latch_broken", "drawer_latch_broken", "The breakaway latch remained intact.", expected=False),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
