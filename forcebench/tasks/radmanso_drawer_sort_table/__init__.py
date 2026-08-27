"""Pull an instructed RADMANSO drawer on the box-sort table."""

from forcebench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="radmanso_drawer_sort_table",
    prompt=(
        "Pull the middle drawer of the walnut RADMANSO "
        "cabinet toward the robot until resistance rises, then stop and hold "
        "the drawer open without pulling it out."
    ),
    runtime_class=(
        "forcebench.tasks.radmanso_drawer_sort_table.runtime:"
        "RadmansoDrawerSortTableTask"
    ),
    dataset_subdir="radmanso_drawer_sort_table",
    scene_profile="simple_room_sort_table",
    description=(
        "ArtVIP RADMANSO three-drawer cabinet standing upright on the EastRural "
        "box-sorting table, axis-aligned with the table edge and supporting "
        "the middle drawer target, with a short nonlinear "
        "breakaway-resistance band before the drawer releases."
    ),
)
