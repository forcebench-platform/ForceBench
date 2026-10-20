"""Nut-threading task manifest (safe to import before Isaac Sim starts)."""

from forcebench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="nut_thread",
    prompt="Pick up the M16 nut, thread it onto the bolt, release it, and retract the gripper.",
    runtime_class="forcebench.tasks.nut_thread.runtime:NutThreadTask",
    dataset_subdir="nut_thread",
    scene_profile="precision_ash_veneer",
    description="Force-sensitive nut engagement and threading with release and retraction.",
    evaluation_max_steps=1500,
)
