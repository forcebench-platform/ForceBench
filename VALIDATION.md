# Validation

Observed 2026-10-21: **150 passed, 3 failed, 3 skipped** in the complete CPU suite (Windows, Python 3.13.1).

Command: `python -m pytest tests -q`. Platform-specific skips are reported by pytest.

Outstanding failures remain enabled:

- `tests.test_forge_task_specs::test_normalized_thread_assets_preserve_source_sdf_collisions`
- `tests.test_outcomes::test_rattle_sort_has_no_stationary_inspection_success_gate`
- `tests.test_stove_turn_on::test_stove_layout_keeps_geometry_fixed_and_randomizes_only_the_load`

The Nut check requires external normalized assets. CoM scoring and Stove geometry differences remain unresolved; no success threshold or geometry was changed to make these tests pass.

The lightweight installation, task listing and missing-asset diagnostics are checked separately. GPU initialization, expert rollouts, complete model train/serve/evaluate and paper-score reproduction are not verified.
