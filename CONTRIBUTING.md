# Contributing

ForceBench follows ForceBench's task-plugin contract (documented in
`docs/PROJECT_GUIDE.md` section 8): a new task is a package under
`forcebench/tasks/<name>/` exporting one `TaskManifest`, with its own
`spec.py` (immutable geometry/randomization/success constants) and
`runtime.py` (scene entities, reset, instruction, privileged observations,
success). The task registry (`forcebench/registry.py`) discovers it
automatically -- do not add task switches to the shared environment.

Before opening a PR:

1. `python -m pytest tests/` should pass (see `requirements-dev.txt`; most of
   the suite does not need Isaac Sim). New tasks should add a
   `tests/test_<task>.py` following the existing per-task test files.
2. If your task depends on third-party 3D assets, add a row to the
   `NOTICE.md` provenance table with the exact source and license *before*
   the PR is merged, and add the file's hash to `assets/SHA256SUMS`. Do not
   commit the binary asset itself unless you have redistribution rights --
   see `NOTICE.md` for how existing tasks handle this (wrapper code +
   pointer to the vendor's official channel).
3. If your task's success predicate depends on force/wrench sensing, say so
   explicitly and show which line reads it -- ForceBench's whole point is
   being honest about which tasks genuinely require force and which don't
   (see the "three contact-rich assembly tasks" caveat in `README.md`).

Scope questions (should a new task go into a v-next core set or stay
optional) are project-owner calls; open an issue before a large PR.
