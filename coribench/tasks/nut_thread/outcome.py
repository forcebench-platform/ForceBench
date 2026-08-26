"""Final outcome contract for nut engagement and threading."""

from coribench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion("centered", "nut_not_centered", "The nut axis is centered on the bolt."),
    OutcomeCriterion("threaded", "nut_not_threaded", "The nut reached the threaded target height."),
    OutcomeCriterion("upright", "nut_tilted", "The nut axis is upright."),
    OutcomeCriterion("released", "nut_not_released", "The gripper released the nut."),
    OutcomeCriterion("retracted", "gripper_not_retracted", "The gripper retracted after release."),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
