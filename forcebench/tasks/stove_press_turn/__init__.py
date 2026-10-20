"""Force-disambiguated stove press-and-turn manifest."""

from forcebench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="stove_press_turn",
    prompt=(
        "Press and hold the center stove knob, then turn it in the "
        "lower-resistance direction past the detent."
    ),
    runtime_class="forcebench.tasks.stove_press_turn.runtime:StovePressTurnTask",
    dataset_subdir="stove_press_turn",
    arm_actuator_stiffness=80.0,
    arm_actuator_damping=12.0,
    gripper_actuator_stiffness=20000.0,
    gripper_actuator_damping=600.0,
    scene_profile="kitchen_stove",
    description=(
        "Axially gated stove knob with a hidden per-seed easy direction and "
        "gradual-rise/sudden-drop resistance; success latches when the "
        "press-engaged knob crosses the lower-resistance detent."
    ),
    evaluation_max_steps=1500,
)
