# ForceBench

**ForceBench: Benchmarking Force Integration for Robotic Manipulation**

[Project page](https://forcebench-platform.github.io/) ·
[Leaderboard](https://forcebench-platform.github.io/leaderboard.html) ·
[Source preview guide](docs/source_preview.md)

ForceBench studies how robotic policies use six-axis wrist force and torque
feedback. Eight simulation tasks cover physical-property inference,
resistance-dependent interaction, and contact-rich assembly.

## Release status

**Source preview — full simulation and paper reproduction are not yet validated.**

This checkout contains task/environment source, data-conversion and evaluation
code, policy adapters, and candidate baseline implementations. It does not
bundle datasets, model weights or third-party simulator assets. Start with
[the supported entry points and limitations](docs/source_preview.md) and
[the recorded checks](VALIDATION.md).

The reference scope is eight simulation tasks and ten systems. InternVLA and
later experimental extensions are not included in the default benchmark.

## Quick start

```bash
python -m pip install -e .
python -m forcebench list-tasks
python -m forcebench doctor --asset-dir /path/to/authorized/assets
```

The first two commands need Python 3.11+ and no simulator. The last command
reports missing prerequisites and is expected to return nonzero until they are
provided. It does not launch a simulator or download assets.

| Category | Task | Runtime ID |
| --- | --- | --- |
| Physical-property inference | Weight Sort | `weight_sort` |
| Physical-property inference | CoM Sort | `rattle_sort` |
| Resistance-dependent interaction | Faucet Turn | `faucet_turn` |
| Resistance-dependent interaction | Faucet Direction | `faucet_force_turn` |
| Resistance-dependent interaction | Stove Turn | `stove_turn_on` |
| Resistance-dependent interaction | Stove Press | `stove_press_turn` |
| Resistance-dependent interaction | Drawer Pull | `radmanso_drawer_sort_table` |
| Contact-rich assembly | Nut Thread | `nut_thread` |

## Baselines

| Package | Systems |
| --- | --- |
| [pi05](baselines/pi05/README.md) | No Force, Token Fusion, Prompt Fusion, Cross-Attention, Token Fusion + Pred |
| [temporal](baselines/temporal/README.md) | Temporal Prompt, Dual-Path Fusion |
| [pi0](baselines/pi0/README.md) | pi0 |
| [forcevla](baselines/forcevla/README.md) | ForceVLA |
| [forceflow](baselines/forceflow/README.md) | ForceFlow |

Use separate environments for these runtime variants. Their packaged recipes
and CPU checks are not evidence that every paper checkpoint has been loaded or
every result reproduced.

## Paper reference

The unchanged [reported tables](results/paper_reference.json) and
[protocol reference](configs/paper_protocol.json) describe the paper target.
Prompt Fusion's reported mean simulation success is 35.31%, compared with
23.25% for No Force. These are manuscript values, not new results from this
preview.

The paper also reports eight systems on five real-world tasks. A complete
real-robot runtime is outside this source preview.

## Distribution

See [LICENSE](LICENSE), [NOTICE](NOTICE.md), and the license files preserved in
each baseline. This package excludes third-party asset payloads and internal
handoff material. `SOURCE_PREVIEW_MANIFEST.json` records the exact files and
hashes included in this snapshot.
