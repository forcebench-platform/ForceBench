# Benchmark standardization record

## Current canonical contract

- One arm controller only: Isaac Lab `DifferentialIKController` with absolute
  pose commands and DLS (`lambda=0.05`).
- The controller consumes measured TCP pose, the base-frame geometric Jacobian
  and continuous measured joint positions. Its result is joint-step limited,
  joint-limit clamped and sent through the shared quintic trajectory to implicit
  position drives.
- No state overwrite, operational-space impedance branch, PI hold, controller
  selection environment variable or evaluation-only orientation/damping guard.
- Joint angles and measured TCP RPY are continuous across the `-pi/+pi` branch.
- Physics is fixed at 60 Hz, policy control at 15 Hz and solver position
  iterations at 64.
- Formal data uses the stable schema name `forcebench_hdf5` and excludes
  device-specific teleoperation readings. Previous datasets and checkpoints are
  retained under the external archive root and are not accepted as new formal
  inputs.
- Expert collection stores synchronized raw and EMA incoming-joint wrench
  channels. Raw is the model input; EMA is auxiliary. Task-specific generators
  connect through auto-discovered expert plugins and the single
  `run_native.sh expert` supervisor.
- Gear mesh, nut thread and peg insert use direct privileged-pose teachers;
  the previous learned force-skill collection path is archived and unavailable.
- Faucet geometry and detent strength (0.20 N·m) are fixed across seeds; only
  its detent angle is randomized. Its transition detector consumes measured
  wrist torque, not the analytic resistance curve.
- Policy inference uses one canonical observation/action contract. Model and
  transport differences are isolated in auto-discovered policy adapters.

## Storage separation

- code/runtime: `/home/pc/ForceBench`
- formal datasets/checkpoints/evaluations: `/home/pc/ForceBench_data`
- smoke tests, previews and disposable outputs: `/home/pc/ForceBench_scratch`
- immutable historical artifacts: `/home/pc/ForceBench_archive`

The archive manifest records original paths, sizes and move time. New smoke or
debug output must never be written below the formal `datasets/` directory.
