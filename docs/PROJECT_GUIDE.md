# ForceBench project guide

## 1. Authoritative locations

| Purpose | Path |
| --- | --- |
| Source and bundled runtime | `/home/pc/ForceBench` |
| Formal datasets | `/home/pc/ForceBench_data/datasets` |
| Formal checkpoints | `/home/pc/ForceBench_data/checkpoints` |
| Formal evaluations | `/home/pc/ForceBench_data/evaluations` |
| Disposable smoke/previews/logs | `/home/pc/ForceBench_scratch` |
| Historical archives | `/home/pc/ForceBench_archive` |

`forcebench/paths.py` is the Python authority for these roots. Override them only
with `FORCEBENCH_DATA_ROOT`, `FORCEBENCH_SCRATCH_ROOT` and
`FORCEBENCH_ARCHIVE_ROOT`. Generated data does not belong in the source tree.

## 2. Fixed simulator contract

- Physics rate: 60 Hz.
- Position solver iterations: 64.
- Policy/control rate: 15 Hz (`decimation=4`).
- Manual collection and evaluation use one environment. Production cloned
  experts use exactly 32 isolated environments in one Isaac process.
- Robot asset: `assets/franka_mimic.usd`.
- Arm drive: implicit joint-position PD, stiffness 80, damping 12.
- Gripper: implicit position drive using the two finger joints.

`configure_runtime()` rejects other physics or solver values. Tasks may define
objects, randomization, success predicates and camera profiles; they may not
silently change the benchmark-wide robot/controller contract.

## 3. Canonical arm controller

There is one supported arm controller, identified as `smooth_position`:

1. Accept an absolute world-frame TCP pose.
2. Normalize its quaternion and bound each command to 25 mm translation and
   0.12 rad rotation from the measured TCP pose.
3. Express the current and target TCP pose plus geometric Jacobian in the robot
   base frame.
4. Solve with Isaac Lab's official `SmoothPositionController` configured as:

   ```text
   command_type="pose"
   use_relative_mode=False
   ik_method="dls"
   lambda_val=0.05
   ```

5. Limit the solved joint change to 0.08 rad per 15 Hz command, clamp it to the
   robot soft joint limits, and plan a synchronized quintic joint trajectory.
6. Send the trajectory to the implicit joint-position drives at 60 Hz.

The environment does not overwrite articulation state and does not expose a
TSI, PI-hold, damping-ablation or controller-selection interface.

### Angle continuity

The environment maintains continuous measured joint angles and continuous
measured TCP RPY. A raw `+pi/-pi` representation change is converted to the
shortest angular delta before it reaches IK, recorded state, policy input or
replay. Policy RPY targets are likewise represented on the nearest branch to
the current measured TCP RPY.

## 4. Observation, action and wrench

The seven-dimensional policy state is measured after physics:

```text
[tcp_x, tcp_y, tcp_z, tcp_roll, tcp_pitch, tcp_yaw, gripper]
```

TCP RPY is continuous. `gripper` is `-1` closed and `+1` open. It is not a
target-state or state-overwrite value.

The seven-dimensional action is:

```text
[target_tcp_x, target_tcp_y, target_tcp_z,
 target_tcp_roll, target_tcp_pitch, target_tcp_yaw, target_gripper]
```

The canonical six-dimensional wrench is the incoming fixed-joint reaction on
the `force_sensor` link through `panda_hand_joint`, expressed in the incoming
joint frame as `[Fx, Fy, Fz, Tx, Ty, Tz]` (parent on child). No coordinate
transform or gravity compensation is applied. Both raw and EMA (`alpha=0.25`)
streams are recorded; training conversion and policy inference use the raw
stream. EMA remains a synchronized auxiliary channel for analysis and expert
control diagnostics.

## 5. Formal HDF5 data

New episodes use the stable schema name `forcebench_hdf5`. Formal writers accept successful episodes
only and write atomically. Each episode contains:

- measured 7-D state and absolute 7-D action;
- measured joint position, joint velocity and TCP pose;
- canonical main/wrist RGB frames;
- raw and EMA control-rate wrench;
- raw 60 Hz physics-rate wrench;
- timestamps, task outcome verdict (`success`, stable failure codes and
  per-criterion results), seed, realized layout, controller/wrench profile,
  task metadata and declared privileged teacher fields.

Device-specific teleoperation readings are deliberately excluded. The file
stores the measured robot state and the canonical target action, independent of
whether that action came from teleoperation, a scripted expert or another source.

Old schemas/controllers are historical artifacts. They are kept in the archive,
not mixed into current datasets.

## 6. Standard commands

All Isaac Lab commands go through `run_native.sh`:

```bash
cd /home/pc/ForceBench

# GELLO collection; defaults to /home/pc/ForceBench_data/datasets/<task>
FORCEBENCH_TASK=weight_sort ./run_native.sh collect

# Disposable smoke output; defaults below /home/pc/ForceBench_scratch
FORCEBENCH_TASK=weight_sort ./run_native.sh smoke

# List task-bound automated expert plugins
./run_native.sh expert --list

# Cloned expert collection. This is the supported production topology: one Isaac
# process, 32 cloned/collision-filtered workspaces and batched TiledCamera RGB.
# Existing validated seeds are skipped; every batch writes a 32-tile video.
./run_native.sh expert --task weight_sort --seeds 0-199
./run_native.sh expert --task rattle_sort --seeds 0-199
./run_native.sh expert --task stove_turn_on --seeds 0-199
./run_native.sh expert --task radmanso_drawer_sort_table --seeds 0-199
./run_native.sh expert --task peg_insert --seeds 0-199

# Assembly experts are task-specific privileged-pose teachers. They read the
# exact object/fixture pose, but command the normal SmoothPosition controller;
# they never overwrite robot or object state and require no checkpoint.
./run_native.sh expert --task gear_mesh --seeds 0-19 --workers 2
./run_native.sh expert --task nut_thread --seeds 0-19 --workers 2

# Policy evaluation; execute one 50-frame action chunk per inference by default.
# Select a registered adapter with FORCEBENCH_POLICY_ADAPTER.
FORCEBENCH_POLICY_ADAPTER=openpi_ws \
FORCEBENCH_TASK=weight_sort ./run_native.sh policy 200 600 \
  /home/pc/ForceBench_data/evaluations/weight_sort/model_name/seed_200

# Replay a current canonical episode
./run_native.sh replay /home/pc/ForceBench_data/datasets/<task>/<episode>.hdf5
```

Evaluation checks task success after every control frame and terminates once the
predicate is stable. `FORCEBENCH_EXECUTE_CHUNK` may override the default 50 for a
declared experiment; there are no hidden orientation or controller overrides.

## 7. Adding a model adapter

Policy evaluation consumes one `PolicyObservation` and accepts only a
`PolicyActionChunk` in the canonical absolute TCP action space. Transport and
model-specific key layouts live below `forcebench/policy_adapters/`, not in the
evaluation loop.

Built-in adapters are `openpi_ws`, `forcevla_ws` and the native
`forcebench_ws`. Add a non-underscore Python module exporting `MANIFEST` and a
`PolicyAdapter` implementation; it is discovered automatically. An adapter may
translate relative or joint actions internally, but it must return 15 Hz
absolute world-frame TCP `xyz/rpy/gripper` targets. Adapter-specific JSON
options may be supplied through `FORCEBENCH_POLICY_ADAPTER_OPTIONS`.

## 8. Adding a task/demo and its expert collector

Do not add task switches to the shared environment. Add a package below
`forcebench/tasks/<task_name>/` containing:

- `__init__.py`: exports one validated `TaskManifest` named `MANIFEST`;
- `spec.py`: immutable geometry/randomization constants and final-state predicates;
- `outcome.py`: combines those predicates into the task's authoritative verdict
  and stable failure codes without importing Isaac Sim;
- `runtime.py`: a `TaskRuntime` implementation for scene entities, reset,
  instruction, privileged observations and the outcome-enriched final report.

The task registry discovers packages automatically. Add task-specific assets below
`assets/normalized/<task_name>/`, record provenance/hashes, and add focused
task-contract tests. Scripted demo generators belong in `scripts/expert/` and
must call the shared `ForceBenchEnv` controller API. They must not implement a
second robot controller or write formal data outside the canonical writer.

Automated expert collection for ordinary tasks is a task-bound plugin contract. Add a
non-underscore module below `forcebench/expert_adapters/` that exports one
`ExpertBackendManifest` and one `ExpertBackend` for exactly that task. Its
`build_command()` translates the canonical request into one isolated generator
invocation for one seed. `weight_sort`, `rattle_sort`, `stove_turn_on`,
`radmanso_drawer_sort_table`, and `peg_insert` intentionally bypass
that per-seed adapter contract: `collect_expert.py` routes all five to
`parallel_sort_collection.py`, which launches exactly 32 cloned workspaces in
one Isaac process, skips validated seeds, fills a final partial batch without
writing duplicate HDF5, and records a 32-tile video. Their retired single-seed
and persistent-rebuild generators are not retained. Stove clones retain an
independent seed-specific detent load model and flame state per environment.
Other backends keep their
task-specific process lifecycle. See
`forcebench/expert_adapters/README.md`.

Current collection status:

- `weight_sort`, `rattle_sort`, `stove_turn_on`, and
  `radmanso_drawer_sort_table`: enabled through the default 32-clone route;
- `peg_insert`: routed only through the 32-clone 60 Hz collector; formal
  collection stays gated until its 32/32 release-stability smoke passes;
- `faucet_turn`, `gear_mesh`, and `nut_thread`:
  enabled through task-bound adapters;

`gear_mesh`, `nut_thread` and `peg_insert` use task-owned privileged-state
teachers to obtain exact assembly trajectories. The retired learned
force-skill expert path is historical and exists only in the external archive.
For `faucet_turn`, scene geometry and peak resistance (0.20 N·m) are fixed;
seed randomizes only the detent angle. Its expert detects the transition from
measured wrist torque, following the same load-then-recovery pattern as
`stove_turn_on`.

## 9. Repository hygiene

- Keep reusable collection, evaluation, expert, conversion and task interfaces.
- Put disposable diagnostics under the scratch root; remove them when the
  investigation ends.
- Do not keep timestamped source backups, PID files, generated logs, previews,
  `__pycache__` or `.pytest_cache` in the repository.
- Archive superseded formal data/checkpoints with a manifest instead of deleting
  or mixing them with current artifacts.
- Run static compilation, unit tests and a no-render simulator precision smoke
  after changing the shared controller.
