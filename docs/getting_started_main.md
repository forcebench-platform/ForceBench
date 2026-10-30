# Getting started

Use three separate environment types: a lightweight Python 3.11+ environment
for repository tools, Python 3.12 for Isaac Sim/Isaac Lab, and an independent
Python 3.11 environment for each model runtime family. Installing all baseline
variants into one environment would mix incompatible `openpi` packages.

## 1. Inspect the benchmark without a GPU

From the repository root:

```bash
python3.12 -m venv .venv-tools
source .venv-tools/bin/activate
python -m pip install -e ".[assets]"
python -m forcebench list-tasks --json
python tools/verify_paper_reference.py
```

Use Python 3.12.13 for asset builds: that is the interpreter verified with
`usd-core==25.11`. Basic repository inspection still supports Python 3.11+.

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
  torch==2.11.0 numpy==2.3.1 usd-core==25.11 \
  --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com
python -m pip install opencv-python-headless==4.13.0.90 h5py==3.16.0 \
  msgpack==1.2.1 websockets==12.0
cd /path/to/ForceBench
python -m pip install -e .
export FORCEBENCH_SIM_PYTHON="$(command -v python)"
```

This prerelease is hosted by NVIDIA; it may not appear on the default PyPI
index alone. The verified direct wheel is
[isaaclab-3.0.0b2.post1](https://pypi.nvidia.cn/isaaclab/isaaclab-3.0.0b2.post1-py3-none-any.whl).
If an index mirror cannot resolve it, use the explicit, hash-pinned URL:

```bash
python -m pip install \
  'isaaclab[isaacsim] @ https://pypi.nvidia.cn/isaaclab/isaaclab-3.0.0b2.post1-py3-none-any.whl#sha256=dd32886588479ffd70f7348019aeac1582eb9ad16c40244f41d9f458f96f73c3' \
  --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com
```

此预发行版本来自 NVIDIA 软件源，不能只用默认 PyPI 的版本列表判断是否存在。
上述固定直链及 SHA256 已实际下载核验。

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

On minimal Linux images, Isaac Sim also needs X11 native libraries even for
headless rendering. For Ubuntu 22.04, install them with:

```bash
sudo apt-get update
sudo apt-get install libxt6 libsm6 libice6
```

On Ubuntu 24.04, use `libxt6t64` instead of `libxt6`. These are platform packages,
not Python packages; ask the machine administrator to install them if needed.
See [Ubuntu's package index](https://packages.ubuntu.com/search?keywords=libxt6).

A `CXXABI_1.3.15` error means the process selected an older `libstdc++` than
the native extensions require. This symbol belongs to the GCC 14 ABI family
([GCC ABI reference](https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html)).
In the activated simulator Conda environment, obtain a compatible library and
check the actual file before selecting it for this simulator shell:

```bash
conda install -c conda-forge 'libstdcxx-ng>=14'
python -c 'import os; from pathlib import Path; p=Path(os.environ["CONDA_PREFIX"])/"lib/libstdc++.so.6"; assert b"CXXABI_1.3.15" in p.read_bytes(), str(p); print(p)'
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
```

Use this override only when resolving the ABI error, not as a system-wide
setting. The [latest execution receipt](../validation/asset_acquisition_20261027.json)
records reuse of compatible native libraries, `LD_PRELOAD` and an extracted
`libXt6` directory; a new OS and complete fresh NVIDIA dependency installation
have not been validated.
Review NVIDIA's terms when installing and running its software.

## 3. Set storage, acquire and import assets

Keep run data outside the checkout:

```bash
export FORCEBENCH_DATA_ROOT=/path/to/forcebench-data
export FORCEBENCH_SCRATCH_ROOT=/path/to/forcebench-scratch
export FORCEBENCH_ARCHIVE_ROOT=/path/to/forcebench-archive
```

These storage roots must be distinct. Prepare all default task assets in the
lightweight tools environment (`pip install -e ".[assets]"`):

```bash
python tools/download_assets.py \
  --source-dir /path/to/asset-source \
  --cache-dir /path/to/download-cache --workers 4
```

Repeat `--task TASK_ID` to select a subset, or omit it for all eight. The
[asset guide](assets.md) documents the exact archives, direct URLs, five build
recipes, directory mappings and upstream terms. It covers all 283 external
or derived payloads. Builds require the public `usd-core==25.11` wheel; another
USD writer may produce different binary hashes and is rejected.

Then switch to the simulator environment and import the downloaded assets
plus the 17 authored wrappers shipped with this checkout:

```bash
conda activate forcebench-sim
python tools/prepare_assets.py \
  --source-dir /path/to/asset-source \
  --asset-dir /path/to/forcebench-assets --check-only
python tools/prepare_assets.py \
  --source-dir /path/to/asset-source \
  --asset-dir /path/to/forcebench-assets
export FORCEBENCH_ASSET_DIR=/path/to/forcebench-assets

python -m forcebench doctor --asset-dir "$FORCEBENCH_ASSET_DIR"
python tools/prepare_nut_assets.py --asset-dir "$FORCEBENCH_ASSET_DIR" --check-only
python tools/verify_nut_asset_composition.py --asset-dir "$FORCEBENCH_ASSET_DIR"
```

The importer verifies 300 files and the installed renderer materials. It
preserves verified existing files and refuses unknown alternatives. The
robot, Nut and Bolt have explicitly audited official export variants; this
is not a general relaxation of checksum checks. Supply
`--runtime-material-dir /path/to/isaacsim/kit/mdl/core/Base` only if discovery
of the supported installed runtime fails.

Download failures leave previously verified assets reusable. For network
restrictions, configure the environment's `HTTPS_PROXY` or use the verified
archive cache described in the asset guide. Do not rename a different asset
to satisfy a missing filename.

统一入口按任务下载并校验固定版本；重跑可续用已验证文件。随后在模拟器环境中完成
300 个文件与运行时材质的检查、导入和 Nut 几何检查，再进行 GPU 启动验收。

## 4. Run a bounded simulator check

Run the checker with a fresh output directory. It initializes real simulator
environments, steps hold actions, and checks finite observations and RGB images:

First read NVIDIA's [license terms and first-run instructions](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/setup/installation/isaaclab_pip_installation.html).
Only if you accept those terms, set the corresponding variables in your
simulator shell. Then perform one interactive warm-up so that any first-run
messages stay visible:

```bash
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACSIM_ACCEPT_EULA=Y
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/inference_forcebench_vectorized.py \
  --task weight_sort --seed_start 300 --num_envs 1 --max_steps 1 \
  --smoke_hold --output_dir "$FORCEBENCH_SCRATCH_ROOT/validation/interactive-warmup" \
  --headless --enable_cameras --device cuda:0
```

The first startup downloads NVIDIA extensions and may take over ten minutes.
Wait for it to exit successfully before launching the checker, which redirects
each task's output to a log. Use a new warm-up/output directory on rerun.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validate_simulation.py \
  --python "$(command -v python)" \
  --assets "$FORCEBENCH_ASSET_DIR" \
  --output "$FORCEBENCH_SCRATCH_ROOT/validation/simulator-first-check" \
  --tasks weight_sort --steps 10 --timeout 1200
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
