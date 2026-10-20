# ForceBench — Research Release

**ForceBench: Benchmarking Force Integration for Robotic Manipulation**

[Project page](https://forcebench-platform.github.io/) ·
[Leaderboard](https://forcebench-platform.github.io/leaderboard.html) ·
[Installation](docs/getting_started_main.md) ·
[Validation scope](docs/validation_scope.md)

ForceBench compares how robotic policies use six-axis wrist force and torque
feedback. Eight simulation tasks cover physical-property inference,
resistance-dependent interaction and contact-rich assembly. This research
release provides task definitions, simulator code, data conversion, evaluation,
policy adapters and ten system implementations within the submitted-paper scope.

The [public validation report](validation/public_report.json) records exactly
which source, assets, environments and execution checks passed. All ten systems
passed bounded GPU model execution checks with real data and parameters.
All eight simulator tasks passed two-environment GPU checks; Weight Sort and
Nut Thread also passed single-environment checks. These
software checks do not establish reproduction of the paper's success rates.

## Start here

The registry and inspection tools need Python 3.11+ and no simulator:

```bash
git clone https://github.com/forcebench-platform/ForceBench.git
cd ForceBench
python3.11 -m venv .venv-tools
source .venv-tools/bin/activate
python -m pip install -e .
python -m forcebench list-tasks
```

Follow [the installation guide](docs/getting_started_main.md) to create the
separate Python 3.12 simulator environment and Python 3.11 model environments.
The source checkout does not install Isaac Sim, download weights or acquire
third-party assets automatically.

After obtaining assets under their original terms, verify and import them
with `tools/prepare_assets.py`, then run `python -m forcebench doctor` in the
simulator environment. The [asset guide](docs/assets.md) documents the pinned
300-file tree, authored wrappers, renderer requirements and reviewed dependency
scope. Source models and textures are supplied separately.

## Eight tasks

| Category | Paper task | Runtime ID |
|---|---|---|
| Physical-property inference | Weight Sort | `weight_sort` |
| Physical-property inference | CoM Sort | `rattle_sort` |
| Resistance-dependent interaction | Faucet Turn | `faucet_turn` |
| Resistance-dependent interaction | Faucet Direction | `faucet_force_turn` |
| Resistance-dependent interaction | Stove Turn | `stove_turn_on` |
| Resistance-dependent interaction | Stove Press | `stove_press_turn` |
| Resistance-dependent interaction | Drawer Pull | `radmanso_drawer_sort_table` |
| Contact-rich assembly | Nut Thread | `nut_thread` |

Use these IDs for collection, policy serving recipes and evaluation. A task's
registered name establishes its interface, while the public validation report
establishes the tested execution coverage.

## Ten systems

| Runtime package | Paper systems |
|---|---|
| [pi0.5](baselines/pi05/README.md) | No Force (F0), Token Fusion (F1), Prompt Fusion (F2), Cross-Attention (F3), Token Fusion + Pred (F4) |
| [Temporal](baselines/temporal/README.md) | Temporal Prompt, Dual-Path Fusion |
| [pi0](baselines/pi0/README.md) | pi0 |
| [ForceVLA](baselines/forcevla/README.md) | ForceVLA |
| [ForceFlow](baselines/forceflow/README.md) | ForceFlow |

Install runtime families in separate environments. Several contain different
implementations of the same `openpi` namespace and cannot share one Python
environment. Their READMEs describe input, training and serving recipes; the
public report is the authority for measured GPU coverage. Nine OpenPI systems
ran three actual optimizer updates plus recorded-observation inference and
loopback websocket checks. ForceFlow ran two-update diagnostics, native-server
inference, and a new-checkpoint reload check. Only the controlled pi0.5/F4
environment received the separate fresh-install check.

The common evaluation interface uses two RGB images, seven-dimensional robot
state and seven-dimensional absolute TCP actions. Force-aware adapters receive
six-axis wrist wrench; temporal adapters maintain history at every 15 Hz
control step. The reference physics rate is 60 Hz. ForceFlow predicts 64
actions and executes 32; the other listed recipes use 50/50.

## Paper reference and new experiments

[The frozen paper tables](results/paper_reference.json) and
[protocol reference](configs/paper_protocol.json) retain the submitted-paper
values. Check their integrity with:

```bash
python tools/verify_paper_reference.py
```

The manuscript reports 1,600 demonstrations across the eight simulation tasks.
Datasets, normalization files, base weights and trained checkpoints are
separate inputs and must be selected explicitly. This repository does not
choose a checkpoint by directory name or timestamp.

The released Faucet Direction scorer uses the paper's one-sided signed angle
threshold. The original narrow-band scorer remains explicit for historical
records. New runs under the paper rule do not replace historical table values.
The Stove Press expert collector also keeps trajectory completion and quality
judging at the same per-clone time point; this correction is not a claim that
200 fresh expert episodes have been generated and validated.

Save new evaluation outputs outside the checkout and preserve their source,
asset, model, normalization and seed identities. New software validation or
rollouts do not overwrite the frozen tables. The paper's five-task real-world
results remain reference values; a complete real-robot deployment is outside
the validated default simulation scope.

## Licenses

ForceBench's own code and authored wrappers use [MIT](LICENSE). Bundled
upstream code retains its own license, and third-party assets and model weights
retain their applicable terms. See [NOTICE.md](NOTICE.md). Asset checksums and
successful loading do not establish redistribution permission.
