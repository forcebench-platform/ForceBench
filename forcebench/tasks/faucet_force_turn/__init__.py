"""Force-disambiguated faucet manifest."""

from forcebench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="faucet_force_turn",
    prompt=(
        "Use force feedback to identify the lower-resistance direction, rotate "
        "the faucet lever past the randomized detent, and leave the water on."
    ),
    runtime_class=(
        "forcebench.tasks.faucet_force_turn.runtime:FaucetForceTurnTask"
    ),
    dataset_subdir="faucet_force_turn",
    evaluation_max_steps=1500,
    arm_actuator_stiffness=80.0,
    arm_actuator_damping=12.0,
    gripper_actuator_stiffness=20000.0,
    gripper_actuator_damping=600.0,
    scene_profile="kitchen_sink",
    description=(
        "Pure-rotation kitchen faucet with a hidden easy/hard direction and "
        "randomized snap-through detent."
    ),
)
