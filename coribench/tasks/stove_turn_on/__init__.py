"""Gas-stove turn-on task manifest, safe to import without Isaac Sim."""

from coribench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="stove_turn_on",
    prompt="Turn the center stove knob past the detent to ignite the burner.",
    runtime_class="coribench.tasks.stove_turn_on.runtime:StoveTurnOnTask",
    dataset_subdir="stove_turn_on",
    scene_profile="kitchen_stove",
    description="Lightwheel gas-stove knob manipulation with a nonlinear ignition detent.",
)
