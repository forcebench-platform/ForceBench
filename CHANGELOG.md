# Changelog

## v0.1.0 (2026-08-26) -- initial scaffold, not yet a tagged benchmark release

First public commit. Extracted and cleaned from the internal ForceBench
codebase, scoped to ForceBench's 8 in-scope tasks
(`weight_sort, rattle_sort, stove_turn_on, radmanso_drawer_sort_table,
peg_insert, faucet_turn, gear_mesh, nut_thread`).

What's here: task definitions, shared simulator/controller contract, policy
adapters, expert-collection and evaluation scripts, a passing test suite
(102/109 tests green without Isaac Sim -- see `KNOWN_ISSUES.md`), and
provenance/licensing documentation (`NOTICE.md`).

What's not here yet, tracked for v1.0:
- Formal episode data hosted on HuggingFace Datasets (`scripts/download_datasets.py`
  is not yet implemented).
- `scripts/download_assets.sh` pointing at each vendor's official asset
  channel (not yet implemented).
- Per-task documentation cards under `docs/tasks/`.
- `docs/eval_protocol.md` and `docs/policy_interface.md`.
- Baseline results for at least one external policy.
- Resolution of the two `NOTICE.md` open licensing items (ArtVIP, peg_insert
  asset chain) and the four `KNOWN_ISSUES.md` test failures.
