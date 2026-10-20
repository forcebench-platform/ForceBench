"""Final outcome contract for gear placement and meshing."""

from forcebench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion("centered", "gear_not_centered", "The medium gear is centered on its shaft."),
    OutcomeCriterion("seated", "gear_not_seated", "The medium gear is seated at the mesh height."),
    OutcomeCriterion("upright", "gear_tilted", "The medium gear axis is upright."),
    OutcomeCriterion("released", "gear_not_released", "The gripper released the gear."),
    OutcomeCriterion("retracted", "gripper_not_retracted", "The gripper retracted after release."),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
