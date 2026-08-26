"""Peg insertion task manifest (safe to import before Isaac Sim starts)."""

from coribench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="peg_insert",
    prompt=(
        "Pick up the silver rod from the blue fixture, insert it into the bronze "
        "target hole, release it, and retract the gripper."
    ),
    runtime_class="coribench.tasks.peg_insert.runtime:PegInsertTask",
    dataset_subdir="peg_insert",
    scene_profile="precision_ash_veneer",
    description="Teleoperated vertical grasp, transport, insertion, release, and retraction.",
    evaluation_max_steps=1500,
)
