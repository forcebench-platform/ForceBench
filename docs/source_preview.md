# Source preview

This release track makes the available implementation inspectable. A missing
historical handoff archive is **not** a prerequisite for publishing source.
It remains relevant to claims that depend on the evidence in that archive.

## Scope

- Eight simulation task specifications, environment code, policy adapters,
  vectorized evaluation code and packaged baseline runtime groups.
- Reported paper tables, explicitly labeled as reference values.
- No datasets, trained checkpoints, private evaluation logs, credentials,
  or simulator/third-party asset payloads.
- Gear/Peg implementations are retained for historical source dependencies;
  they are excluded from the default eight-task registry.
- No complete real-robot deployment stack is provided.

The implementation is a **source preview**, not a verified reproduction.
The complete historical audit and a fully runnable benchmark release are
separate milestones.

## Install the lightweight package

Use Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m forcebench list-tasks
python -m forcebench list-tasks --json
```

This installs the simulator-free registry and inspection CLI. It does **not**
install Isaac Sim, CUDA, PyTorch, or all model runtimes. `baselines/` contains
incompatible OpenPI variants; install each in a separate environment using its
own README.

For CPU development:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python tools/verify_paper_reference.py
```

Read `VALIDATION.md` in the exported preview for the observed test result and
remaining failures. A reference-table check validates transcription, not policy
performance. No test is silently omitted to make a release appear successful.

## Assets and environment checks

Before starting a simulator, inspect its direct file requirements:

```bash
python -m forcebench doctor --asset-dir /path/to/authorized/assets
python -m forcebench doctor --asset-dir /path/to/authorized/assets --task nut_thread
```

`doctor` returns a nonzero code if direct files or simulator modules are missing.
Even a zero result only checks these prerequisites: it does not validate
transitive USD references, textures, asset licenses, physical behavior or GPU
execution. See the machine-readable list in `forcebench/asset_requirements.json`.
Use assets acquired under their respective terms; this preview does not
distribute an asset bundle or promise that upstream files alone match the
historical processed scenes.

Storage defaults to `$HOME/ForceBench/{data,scratch,archive}`. Override with
`FORCEBENCH_HOME` or individually with `FORCEBENCH_DATA_ROOT`,
`FORCEBENCH_SCRATCH_ROOT`, and `FORCEBENCH_ARCHIVE_ROOT`. Paths are resolved at
import time, without creating directories. Keep these roots distinct and outside
the checkout.

Linux simulation entry points require an independently configured Isaac
environment and complete assets:

```bash
export FORCEBENCH_ISAACLAB_ROOT=/path/to/IsaacLab
export FORCEBENCH_CONDA_SH=/path/to/miniforge/etc/profile.d/conda.sh
export FORCEBENCH_CONDA_ENV=your_isaac_environment
export FORCEBENCH_ASSET_DIR=/path/to/authorized/assets
bash run_native.sh --help
python -m scripts.expert.collect_expert --list
```

The packaged batch collectors cover Weight Sort, CoM Sort, Faucet Turn,
Stove Turn, Drawer Pull and Nut Thread. Faucet Direction and Stove Press
collection entries are not packaged; the CLI reports this explicitly.
Do not infer collection readiness from a task appearing in the registry.

The launcher supports `expert`, `policy-vector`, `replay` and `policy-replay`.
Retired collection/smoke modes are rejected before simulator initialization.
Clean GPU execution has not been validated for this assembled source.

## Evaluation semantics to preserve

- New evaluations use seeds 300–499 and a maximum of 1,500 control steps
  as the paper reference, with explicit model, norm and asset selection.
- Keep first full success, a fixed post-success stability interval, and
  batch-end state separate. Historical batch-end settling is not a uniform
  per-episode stability test.
- Faucet Direction's online termination and historical offline scoring differ;
  they must not be silently interchanged.
- CoM Sort has an unresolved mismatch between its outcome contract and a
  stationary-inspection gate in its specification.
- Nut data revisions, scene-seed overlap and model-to-dataset binding are not
  fully established for every historical run.
- Preserve per-task actuator settings and candidate model defaults until
  their provenance is resolved. Do not change physics or scoring just to match
  the paper table.

Save new run records separately from historical reference values. A successful
new rollout does not retroactively establish which historical checkpoint or
dataset produced the paper.

## Release milestones

| Milestone | Required | Not required |
| --- | --- | --- |
| Source preview | Accurate scope, no secrets/private handoff material, consistent distribution permissions, working documented CPU entry points, visible validation limits | The missing 180-item archive, all raw logs, all weights, or reproduction of every reported number |
| Runnable simulation release | Distributable/acquirable asset recipe, supported clean environment, eight-task GPU smoke, working representative train/serve/evaluate path, resolved default scoring contracts | A claim of bitwise reproduction |
| Verified paper reproduction | Per-result model/data/norm/scorer identity and executed validation for the claimed scope | Replacing historical results with new runs |

The original strict release checker concerns the third milestone. It must not
be used as a blanket prohibition on a clearly labeled source preview.
