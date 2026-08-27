# Expert backend plugins

`scripts/expert/collect_expert.py` is the only public expert-collection
supervisor. Every backend module in this directory belongs to exactly one task
and exports:

- one `ExpertBackendManifest` named `MANIFEST`;
- one `ExpertBackend` class referenced by `MANIFEST.backend_class`;
- a `build_command(ExpertRequest)` implementation that attempts exactly one
  seed and writes successful episodes with `HDF5EpisodeWriter`.

To connect a new task, add its task package and generator, then add a
non-underscore module such as `new_task.py`. The registry discovers it
automatically; do not add a task switch to the collection supervisor.

The supervisor owns seed parsing, retries, resume behavior,
logs, optional videos, final HDF5 validation, and formal/scratch storage
separation. Backends only translate the canonical request into the task
generator's argv.

`weight_sort`, `rattle_sort`, `stove_turn_on`, `radmanso_drawer_sort_table`,
and `peg_insert` are deliberate exceptions to the per-seed
backend interface. Their public `run_native.sh expert` route always uses one
Isaac process with exactly 32 cloned, collision-filtered workspaces and batched
TiledCamera rendering. Existing successful seeds are skipped and every batch
records one 32-tile mosaic video. The former one-process-per-seed and
persistent stage-rebuild generators are retired. Stove environments retain
independent seed-specific detent curves and flame state. RADMANSO and peg also
keep per-environment mechanism/fixture state. Other task adapters keep
their existing task-specific process lifecycle.

Set `enabled=False` plus a concrete `disabled_reason` when task/expert code
must remain in-tree but formal collection is not ready. Disabled plugins are
shown by `./run_native.sh expert --list`, but are excluded from backend lookup
and cannot be launched.

Assembly experts may use declared privileged object/fixture poses as teacher
state. They must still command the shared environment controller and let PhysX
move the scene; robot/object state overwrite is forbidden. `gear_mesh`,
`nut_thread`, and `peg_insert` follow this contract. A learned checkpoint is not
part of their collection interface.

Every formal expert episode records both incoming-joint wrench channels:

- `forces/control/eef_wrench_joint_raw` (canonical model input);
- `forces/control/eef_wrench_joint_ema` (`alpha=0.25`, synchronized auxiliary
  channel).

The 60 Hz physics stream remains raw.
