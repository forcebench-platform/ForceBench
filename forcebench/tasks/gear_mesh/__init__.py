"""Gear-meshing task manifest (safe to import before Isaac Sim starts)."""

from forcebench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="gear_mesh",
    prompt=(
        "Pick up the medium gear, place it on the center shaft so that it meshes "
        "with both flanking gears, release it, and retract the gripper."
    ),
    runtime_class="forcebench.tasks.gear_mesh.runtime:GearMeshTask",
    dataset_subdir="gear_mesh",
    scene_profile="gear_assembly_cell",
    description="Force-sensitive gear alignment, meshing, release, and retraction.",
)
