# ForceBench

> **Status: v0.1 initial scaffold.** Code and tests are here and pass
> (102/109 without Isaac Sim, see `KNOWN_ISSUES.md`); data hosting, per-task
> docs, the eval-protocol writeup, and baseline results are not yet in this
> repo. See `CHANGELOG.md` for the v1.0 checklist. Do not cite this as a
> finished benchmark yet.

**Does force/torque sensing actually help a manipulation policy, and how much
of it does the policy need?**

ForceBench is a simulation benchmark, built on Isaac Lab, for measuring
whether — and how much — 6-axis force/torque feedback improves vision-language-
action (VLA) manipulation policies. Every task is solvable in principle from
vision and proprioception alone in terms of geometry, but the object property
that determines *correct behavior* (its weight, its internal balance, a
mechanism's resistance) is not visually observable: the policy must use force
to succeed above chance, or it must not need force at all — and ForceBench is
built to tell you which, per task, per policy.

Eight tasks, one shared robot/controller/observation contract, one policy
interface. Swap the policy, not the harness.

**Five force-classification tasks** — the object property the policy must
infer is not visually observable at all, and force is the only channel that
carries it:

| Task | What the policy must infer from force | Physically varies |
|---|---|---|
| `weight_sort` | light vs. heavy box (0.06 kg / 0.18 kg, visually identical) | box mass |
| `rattle_sort` | balanced vs. off-center parcel contents (fixed 0.18 kg total) | center-of-mass offset |
| `stove_turn_on` | when the knob detent has been crossed | nonlinear detent torque, peak 0.12 N·m |
| `faucet_turn` | same, on a faucet lever | nonlinear detent torque, peak 0.20 N·m (fixed) |
| `radmanso_drawer_sort_table` | when to stop pulling as drawer resistance rises | breakaway resistance, randomized 5.5-9.0 N |

**Three contact-rich assembly tasks** — included for breadth, but flagged
honestly: as of v1, their success predicates grade final pose only
(centered/seated/aligned/released/retracted) and never consult the wrench
signal, and `peg_insert` in particular has no force/mass constant in its spec
at all. Their MANIFEST descriptions call them "force-sensitive," which
overstates what the current implementation actually requires or measures —
treat that language as aspirational until/unless a force-dependent success
check is added. They are useful contact-rich manipulation tasks; they are not
yet evidence for or against force-sensing the way the five above are.

| Task | What actually varies | Force role today |
|---|---|---|
| `peg_insert` | insertion alignment/contact | none — pure geometry |
| `gear_mesh` | gear seating/mesh | mass scaled for contact realism only, not graded |
| `nut_thread` | thread engagement | not graded |

Per-task documentation cards are not written yet (tracked in
`CHANGELOG.md`); until then, `spec.py`/`runtime.py` under
`coribench/tasks/<name>/` are the source of truth, and
[`docs/PROJECT_GUIDE.md`](./docs/PROJECT_GUIDE.md) covers the shared
simulator/controller/data contract all eight share.

## Why not just use ForceVLA's own dataset?

[ForceVLA](https://arxiv.org/abs/2505.22159) (NeurIPS 2025) introduced a
force-fusion MoE architecture and a 5-task real-robot dataset to validate it.
ForceBench is not a competing architecture paper — it's the harness a
*method-agnostic* comparison needs: one fixed simulator contract, a graded
ladder of how much force signal reaches the policy (proprio-only through raw
6-axis wrench through architecture-level fusion), and a shared eval protocol,
so that "force helped" is a benchmark result rather than a claim specific to
one paper's setup.

## Quickstart

Data hosting and an asset-download script aren't wired up yet (v0.1 -- see
the status note above), so for now this repo is for reading the task
definitions and running the parts that don't need real assets:

```bash
git clone https://github.com/pkuhanxuan/ForceBench
cd ForceBench
pip install -r requirements-dev.txt

# registry/contract tests -- no Isaac Sim needed
python -m pytest tests/

# everything below needs a real Isaac Lab 2.3.2 / Isaac Sim 5.1 environment
# and the assets in place (see NOTICE.md for where each task's assets come
# from -- there is no download script yet, this is a v1.0 tracked item):
CORIBENCH_TASK=weight_sort ./run_native.sh smoke
CORIBENCH_POLICY_ADAPTER=openpi_ws \
CORIBENCH_TASK=weight_sort ./run_native.sh policy 200 600 \
  ./results/weight_sort/my_model/seed_200
```

See [`docs/PROJECT_GUIDE.md`](./docs/PROJECT_GUIDE.md) for the full command
reference (collection, expert generation, replay) and
[`docs/BENCHMARK_STANDARDIZATION.md`](./docs/BENCHMARK_STANDARDIZATION.md)
for the fixed simulator contract every task shares.

## Evaluation protocol

Not written up yet (tracked for v1.0). The intent -- report Clopper-Pearson
confidence intervals alongside raw success rate, not a bare percentage, given
the small-n regime real-robot and even many sim-robot manipulation papers
report in -- is documented for now only in this project's internal planning
notes; treat any success-rate number from this repo as provisional until that
doc exists.

## Adding a policy

One `PolicyObservation` in, one `PolicyActionChunk` out (15 Hz, absolute
world-frame TCP pose + gripper) -- see
[`coribench/policy_adapters/README.md`](./coribench/policy_adapters/README.md)
for the exact contract. Built-in adapters: `openpi_ws` (pi0/pi0.5),
`forcevla_ws` (force-fusion MoE), `coribench_ws` (native).

## Data & assets

No formal episode data ships in this repo yet -- HuggingFace Dataset hosting
is a tracked v1.0 item, not yet set up. Several tasks depend on third-party
3D assets (NVIDIA Isaac/Omniverse SimReady content, Lightwheel Kitchen assets
under CC BY-NC 4.0, and others not yet fully traced) that ForceBench does not
itself redistribute; see [`NOTICE.md`](./NOTICE.md) for the exact per-task
provenance as currently understood, including two items still marked
unconfirmed.

## Reproducing the data from scratch

The formal datasets were generated on Isaac Lab 2.3.2 / Isaac Sim 5.1
(pinned container image digest `sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2`).
A step-by-step reproduction doc is a tracked v1.0 item; for now, the
expert-collection commands are in
[`docs/PROJECT_GUIDE.md`](./docs/PROJECT_GUIDE.md) section 6.

## License

Code: see [`LICENSE`](./LICENSE). Third-party assets: see
[`NOTICE.md`](./NOTICE.md) — several tasks are restricted to non-commercial
research use by their upstream asset license; this repo as a whole should be
treated as **non-commercial research use** until every task's asset chain is
independently cleared (tracked in `NOTICE.md`'s open items).

## Citation

```bibtex
@misc{forcebench2026,
  title  = {ForceBench: A Benchmark for Force-Aware Manipulation Policies},
  author = {Han, Xuan and others},
  year   = {2026},
  url    = {https://github.com/pkuhanxuan/ForceBench}
}
```
