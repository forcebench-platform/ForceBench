"""Final outcome contract for peg insertion."""

from forcebench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion("centered", "peg_not_centered", "The peg axis is centered on the target hole."),
    OutcomeCriterion("inserted", "peg_not_inserted", "The peg reached the target insertion depth."),
    OutcomeCriterion("upright", "peg_tilted", "The inserted peg remains upright."),
    OutcomeCriterion("released", "peg_not_released", "The gripper released the peg."),
    OutcomeCriterion("retracted", "gripper_not_retracted", "The gripper retracted after release."),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
