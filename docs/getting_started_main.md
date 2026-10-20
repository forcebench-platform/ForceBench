# Getting started

Use three separate environment types: a lightweight Python 3.11+ environment
for repository tools, Python 3.12 for Isaac Sim/Isaac Lab, and an independent
Python 3.11 environment for each model runtime family. Installing all baseline
variants into one environment would mix incompatible `openpi` packages.

## 1. Inspect the benchmark without a GPU

From the repository root:

```bash
python3.11 -m venv .venv-tools
source .venv-tools/bin/activate
python -m pip install -e .
python -m forcebench list-tasks --json
python tools/verify_paper_reference.py
```

On Windows, activate with `.venv-tools\Scripts\Activate.ps1`. For development
checks, install `python -m pip install -e ".[dev]"` and run
`python -m pytest tests -q`. Observed counts and skip reasons are recorded in
[the public report](../validation/public_report.json).

## 2. Prepare the simulator environment

Simulation is exercised on Linux with the PhysX backend and RTX renderer.
The observed reference environment has these exact package versions:

| Component | Observed version |
|---|---|
| Python | 3.12.13 |
| Isaac Sim packages | 6.0.1.0 |
| Isaac Lab pip distribution | 3.0.0b2.post1 |
| `isaaclab` extension metadata | 6.1.14 |
| `isaaclab_physx` | 1.1.3 |
| `isaaclab_assets` | 0.3.4 |
| `isaaclab_tasks` | 1.10.9 |
| `isaaclab_contrib` | 0.4.4 |
| `isaaclab_rl` | 0.5.5 |
| PyTorch | 2.11.0+cu128 |
| NumPy | 2.3.1 |
| USD Python (`usd-core`) | 25.11 |
| Warp | 1.13.0 |
| OpenCV headless | 4.13.0.90 |
| h5py / msgpack / websockets / Pillow | 3.16.0 / 1.2.1 / 12.0 / 12.2.0 |

Use the official **`isaaclab==3.0.0b2.post1`** wheel for this release.
Its SHA256 is
`dd32886588479ffd70f7348019aeac1582eb9ad16c40244f41d9f458f96f73c3`
(6,070,415 bytes). The `isaacsim` extra pins Isaac Sim 6.0.1.0.
The extension metadata values in the table are internal extension versions,
not independently installable pip releases: `isaaclab==6.1.14` is not a public
pip package version.

All 429 core Python files, 20 RL Python files, six extension configuration
files and nine Kit application files match the inherited reference source.
Several extension initializers differ only to locate the wheel's directory
layout. Two Factory-task camera changes in the inherited tree are outside
ForceBench's execution path; their environment implementations and camera
patch modules are not loaded. Kit may load Factory/FORGE agent metadata. The exact inherited Git commit remains unknown. Source comparison
and the scoped wheel execution checks in the public report establish this
installation target; they are separate from a complete fresh simulator install.

Use NVIDIA's [Isaac Lab installation documentation](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/setup/installation/index.html)
for platform, driver and license prerequisites, and its [wheel installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/setup/installation/isaaclab_pip_installation.html)
for running custom scripts. In a new Python 3.12 environment:

```bash
conda create -n forcebench-sim python=3.12.13
conda activate forcebench-sim
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaaclab[isaacsim]==3.0.0b2.post1' \
  torch==2.11.0 numpy==2.3.1 --extra-index-url https://pypi.nvidia.com
python -m pip install opencv-python-headless==4.13.0.90 h5py==3.16.0 \
  msgpack==1.2.1 websockets==12.0
cd /path/to/ForceBench
python -m pip install -e .
export FORCEBENCH_SIM_PYTHON="$(command -v python)"
```

This uses the published wheel and does not require an unidentified source
checkout. Keep the simulator environment separate from each model environment.
`run_native.sh` and the expert collector use this selected interpreter for wheel
installations. Existing source launchers remain supported when selected; no
local Conda script or Isaac Lab source directory is required for wheel mode.
To verify the downloaded wheel before installing it, use
`pip download --no-deps isaaclab==3.0.0b2.post1 --extra-index-url https://pypi.nvidia.com`
and compare its SHA256 with the value above.

Check the installed metadata against the report before continuing. Do not
install the model runtimes into this simulator environment. A kit-less/Newton
installation does not supply the PhysX/RTX execution path used here.

On minimal Linux images, Isaac Sim also needs the platform's X11 native
libraries, including `libXt`, `libSM` and `libICE`, even for headless rendering.
A `CXXABI_1.3.15` error indicates an incompatible `libstdc++` selected by the
process. The observed startup was repaired by using the environment's matching
`libstdc++` and supplying the missing native libraries; use the platform's
supported libraries and record any process overrides in validation output.
Review NVIDIA's terms when installing and running its software.

## 3. Set storage and import separately acquired assets

Keep run data outside the checkout:

```bash
export FORCEBENCH_DATA_ROOT=/path/to/forcebench-data
export FORCEBENCH_SCRATCH_ROOT=/path/to/forcebench-scratch
export FORCEBENCH_ARCHIVE_ROOT=/path/to/forcebench-archive
```

These storage roots must be distinct. Acquire the models/textures described in
[the asset guide](assets.md) under their upstream terms, preserving the expected
relative paths. Then, in the simulator environment:

```bash
python tools/prepare_assets.py \
  --source-dir /path/to/legally-acquired-assets \
  --asset-dir /path/to/forcebench-assets \
  --check-only
python tools/prepare_assets.py \
  --source-dir /path/to/legally-acquired-assets \
  --asset-dir /path/to/forcebench-assets
export FORCEBENCH_ASSET_DIR=/path/to/forcebench-assets

python -m forcebench doctor --asset-dir "$FORCEBENCH_ASSET_DIR"
python tools/prepare_nut_assets.py --asset-dir "$FORCEBENCH_ASSET_DIR" --check-only
python tools/verify_nut_asset_composition.py --asset-dir "$FORCEBENCH_ASSET_DIR"
```

The importer checks 300 pinned files for all default tasks, uses bundled
authored wrappers where appropriate, and refuses different existing files.
Its renderer checks use the installed `kit/mdl/core/Base` directory; pass
`--runtime-material-dir` explicitly if automatic discovery is unavailable.
`doctor` checks prerequisites; the Nut composition check additionally verifies
source SDF geometry and rigid-body structure. Neither replaces a GPU rollout.

## 4. Run a bounded simulator check

Run the checker with a fresh output directory. It initializes real simulator
environments, steps hold actions, and checks finite observations and RGB images:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validate_simulation.py \
  --python "$(command -v python)" \
  --assets "$FORCEBENCH_ASSET_DIR" \
  --output "$FORCEBENCH_SCRATCH_ROOT/validation/simulator-first-check" \
  --tasks weight_sort --steps 10
```

Omit `--tasks` to request all eight default tasks. This requires sufficient GPU
memory and startup time. A successful hold check verifies environment execution
and observation plumbing; it does not test learned manipulation success.

## 5. Install a model runtime separately

Create one Linux/Python 3.11 environment per runtime family:

| Family | Source directory | Entry-point guide |
|---|---|---|
| pi0.5 F0–F4 | `baselines/pi05/runtime` | [pi0.5](../baselines/pi05/README.md) |
| Temporal Prompt / Dual-Path | `baselines/temporal/runtime` | [Temporal](../baselines/temporal/README.md) |
| pi0 | `baselines/pi0/openpi` | [pi0](../baselines/pi0/README.md) |
| ForceVLA | The explicitly selected `baselines/forcevla/runtime_*` | [ForceVLA](../baselines/forcevla/README.md) |
| ForceFlow | `baselines/forceflow` | [ForceFlow](../baselines/forceflow/README.md) |

For an OpenPI family, use its packaged dependency recipe and the public
report's environment reconciliation. For example, the pi0.5 source workspace
uses `uv sync --frozen --no-install-workspace --no-dev --python 3.11` from
`baselines/pi05/runtime`; its guide also records the pinned LeRobot fallback.
ForceFlow uses its own `requirements.txt` after selecting the appropriate CUDA
PyTorch build.
ForceVLA requires an explicit source snapshot and recipe; do not infer either
from a task name. A dependency reference alone is not a clean-install receipt.

All ten systems passed bounded GPU model execution checks. Nine OpenPI systems
used three real updates and recorded-observation inference with loopback
websocket exchange. ForceFlow used two-update checks plus strict checkpoint
loading, new-checkpoint reload and native websocket inference. A fresh isolated
Python environment was checked for pi0.5/F4 only, using the pinned LeRobot
commit through its official source archive when Git transport failed. The
other model checks used existing environments; see the public report for the
exact dependency and input scope.

Supply the selected dataset, normalization, tokenizer/base parameters and
checkpoint paths explicitly. Start a server using that family's serving
guide. Checkpoint identity checks must remain enabled. Three diagnostic updates
and in-memory inference do not create a production checkpoint.

## 6. Evaluate an explicitly selected policy

Keep the policy server in its model environment. Run the evaluator from the
simulator environment and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/inference_forcebench_vectorized.py \
  --task weight_sort --seed_start 300 --num_envs 2 --env_spacing 6 \
  --policy_host 127.0.0.1 --policy_port 8002 --policy_adapter openpi_ws \
  --execute_chunk 50 --max_steps 1500 \
  --output_dir "$FORCEBENCH_DATA_ROOT/evaluations/weight-sort-selected-policy" \
  --headless --enable_cameras --device cuda:0
```

Choose the adapter documented by the selected baseline. Temporal methods need
dense control-rate history; ForceFlow uses `forceflow_ws` and 64 predicted/32
executed actions. Preserve the adapter's observation and history contract.

For a paper-scale evaluation, use the full configured seed range and preserve
the task-specific model/data/norm/asset identities. The two-environment example
above establishes neither paper coverage nor model quality. See
[validation_scope.md](validation_scope.md) before interpreting results.

New Faucet Direction runs use the published one-sided signed angle threshold.
Use the explicit historical band scorer only when recounting the corresponding
archived records; do not replace the frozen table with newly scored runs.
For Stove Press expert collection, the stored trajectory and collection-quality
checks use the same per-clone completion state. This software correction does
not certify a newly generated 200-episode expert dataset.
