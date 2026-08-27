# Hardcoded machine-specific paths found in copied code

Found by grepping the full staged tree for `/home/pc/ForceBench*`, `/home/ubuntu/`,
`/data/hanxuan`, `/scratch/` and similar patterns across every copied `.py`
and `.sh` file. Not fixed here — flagging so whoever wires up the public repo
can decide the right abstraction (env var already exists in most cases below;
question is really "what should the *default* be when the env var is unset").

## Live runtime defaults (env-var overridable, but default to L20-specific paths)

These three are the important ones — every one of them is already gated by an
environment variable, so ForceBench works out of the box on another machine
*if* the operator sets the env var. The risk is only that the unset default
silently points at a path (`/home/pc/ForceBench...`) that won't exist off L20,
so a misconfigured run fails with a confusing "path not found" rather than a
clear "set this env var" message.

- `forcebench/paths.py:11` — `DATA_ROOT = Path(os.environ.get("FORCEBENCH_DATA_ROOT", "/home/pc/ForceBench_data")).resolve()`
- `forcebench/paths.py:13` — `SCRATCH_ROOT` default `"/home/pc/ForceBench_scratch"` (via `FORCEBENCH_SCRATCH_ROOT`)
- `forcebench/paths.py:16` — `ARCHIVE_ROOT` default `"/home/pc/ForceBench_archive"` (via `FORCEBENCH_ARCHIVE_ROOT`)
- `forcebench/env_cfg.py:27` — `ASSET_DIR = Path(os.environ.get("FORCEBENCH_ASSET_DIR", "/home/pc/ForceBench/assets")).resolve()`
- `run_native.sh:6` — `data_root="${FORCEBENCH_DATA_ROOT:-/home/pc/ForceBench_data}"`
- `run_native.sh:7` — `scratch_root="${FORCEBENCH_SCRATCH_ROOT:-/home/pc/ForceBench_scratch}"`

Note `forcebench/paths.py` also self-validates that `DATA_ROOT`/`SCRATCH_ROOT`/
`ARCHIVE_ROOT` are distinct from each other and from `CODE_ROOT` (see
`_validate_roots()`), so whatever default is picked must keep that invariant.

Suggested direction for the person picking this up: change the unset-default
behavior to either (a) require the env vars explicitly and raise a clear error
naming the missing variable, or (b) default relative to the repo checkout
(e.g. `<repo>/../forcebench_data`) instead of an absolute `/home/pc/...` path
that only exists on the original L20 dev box. Not done here since that's a
product decision about ForceBench's expected directory layout, not a
mechanical fix.

## Inert / metadata-only reference (not a live runtime dependency)

- `scripts/expert/generate_parallel_peg_insert.py:91-95` — a module-level
  dict, `L20_MIGRATION_SOURCE`, records provenance for where this generator's
  logic was migrated from:
  - `"collector": "/home/ubuntu/forcevla_peg/code/tasks/blind_peg_insertion/isaac_collect_dataset.py"`
  - `"isaac_env": "/home/ubuntu/forcevla_peg/code/tasks/blind_peg_insertion/isaac_peg_env.py"`

  These two paths are never opened/read at runtime — the dict (with its two
  SHA-256 fields) is only ever attached as descriptive metadata on written
  episodes (`l20_migration_source` key, lines 667 and 698). Safe to leave as
  historical documentation-in-code, but flagged since it's technically an
  absolute path baked into shipped code and references a sibling repo
  (`forcevla_peg`) that ForceBench does not include and the public won't have.

## Not found / already clean

No other absolute machine paths (no `/scratch/`, no `/data/hanxuan/...`, no
internal IP literals) were found anywhere in the copied `forcebench/`,
`scripts/expert/`, `scripts/evaluation/`, or `tests/` trees. The policy
adapters (`forcebench/policy_adapters/*.py`) take host/port as constructor
arguments with no hardcoded default host, so no gap there.

Separately (not a path, but related): the four dated one-off scripts under
`scripts/evaluation/run_*_20260825.sh` in the source tree (not copied into
this staging tree at all — see final report) hardcode an internal host IP
(`172.31.202.28`) and absolute checkpoint paths under `/data/hanxuan/...`.
Since those files were excluded from the extraction entirely, they aren't
listed as a fix-needed item here, but the reviewer should know why they're
absent.
