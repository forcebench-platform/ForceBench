"""Static balance parcel sorting manifest, safe before Isaac Sim starts."""

from coribench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="rattle_sort",
    prompt=(
        "Lift and inspect the parcel without shaking it. Put an imbalanced parcel in the left "
        "black basket or a uniform parcel on the right conveyor, then release it."
    ),
    runtime_class="coribench.tasks.rattle_sort.runtime:RattleSortTask",
    dataset_subdir="rattle_sort",
    scene_profile="parcel_sort_station",
    description=(
        "Sort visually identical equal-total-mass parcels by sensing centered versus "
        "front/back-offset mass."
    ),
)
