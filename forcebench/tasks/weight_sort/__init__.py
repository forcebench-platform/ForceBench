"""Weight-based box sorting task manifest (safe before Isaac Sim starts)."""

from forcebench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="weight_sort",
    prompt=(
        "Pick up the box to determine whether it is light or heavy. "
        "Put a light box into the black container or a heavy box into the blue "
        "container at the side of the table, then release it."
    ),
    runtime_class="forcebench.tasks.weight_sort.runtime:WeightSortTask",
    dataset_subdir="weight_sort",
    evaluation_max_steps=1500,
    scene_profile="weight_sort_residential",
    description="Sort one box into the black or blue container using grasp feedback.",
)
