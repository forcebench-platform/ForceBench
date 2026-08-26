# CoRiBench

CoRiBench is a task-plugin benchmark built directly on Isaac Lab. Every task
shares one robot, one rendering contract, one observation/action contract and
one arm controller.

Canonical runtime:

- physics: 60 Hz, PhysX position solver iterations: 64;
- policy/control: 15 Hz;
- arm: Isaac Lab `DifferentialIKController`, absolute pose, DLS (`lambda=0.05`),
  followed by a bounded quintic joint trajectory and implicit joint-position PD;
- state: measured TCP `xyz`, continuous `rpy`, measured gripper opening;
- action: absolute target TCP `xyz/rpy` and target gripper opening;
- wrench: raw and EMA `panda_hand_joint` incoming constraint-reaction wrench in
  the joint frame, with raw used by training and inference;
- model interface: auto-discovered policy adapters with one canonical observation/action contract;
- expert interface: task-bound, auto-discovered collectors behind
  `./run_native.sh expert`;
- formal episode format: `coribench_hdf5` without device-specific teleoperation fields.

Use `./run_native.sh` for Isaac Lab entry points. See `PROJECT_GUIDE.md` for the
storage layout, data contract, evaluation behavior and task-plugin interface.
