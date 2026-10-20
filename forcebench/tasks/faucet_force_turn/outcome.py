"""Final contract for force-disambiguated faucet turning."""

from forcebench.outcome import OutcomeCriterion, evaluate_criteria

CRITERIA = (
    OutcomeCriterion(
        "correct_direction_detent_crossed",
        "easy_direction_detent_not_crossed",
        "The valve crossed the detent in the hidden lower-resistance direction.",
    ),
    OutcomeCriterion(
        "within_final_angle_tolerance",
        "valve_not_at_sampled_target",
        "The valve stopped in the signed low-load band from detent + 0.6 to detent + 2.1 degrees.",
    ),
    OutcomeCriterion(
        "water_visible",
        "water_not_flowing",
        "Water remains visibly on at submission.",
    ),
    OutcomeCriterion(
        "gripper_released",
        "gripper_not_released",
        "The gripper is at least 65 percent open.",
    ),
    OutcomeCriterion(
        "tcp_lifted_clear",
        "tcp_not_lifted_clear",
        "The gripper lifted at least 9 cm above the closed-angle grasp height.",
    ),
    OutcomeCriterion(
        "valve_stationary",
        "valve_still_moving",
        "The released valve is stationary within 0.01 rad/s.",
    ),
)

def evaluate_report(report):
    return evaluate_criteria(report, CRITERIA)
