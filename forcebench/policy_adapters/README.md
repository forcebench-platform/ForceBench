# Adding a policy adapter

Add one non-underscore Python module in this directory. It must export a
validated `PolicyAdapterManifest` named `MANIFEST` and a `PolicyAdapter`
subclass referenced by `MANIFEST.adapter_class`.

The evaluator always constructs `PolicyObservation` with:

- measured world TCP `xyz/rpy/gripper` state;
- canonical main and wrist RGB;
- the raw joint-frame six-dimensional wrench (the synchronized EMA channel is
  stored in HDF5 but is not the policy input);
- prompt and control-step index.

The adapter owns all model-specific repacking, transport and conversion. It
must return `PolicyActionChunk` containing one or more 15 Hz absolute
world-frame TCP `xyz/rpy/gripper` targets. The shared evaluator validates that
contract before executing any frame.

Use `openpi_ws.py` and `forcevla_ws.py` for compatibility examples. New servers
should prefer the explicit native request/response contract in
`forcebench_ws.py`.
