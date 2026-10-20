#!/usr/bin/env bash
set -euo pipefail
forceflow_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$forceflow_repo_root"
forceflow_python="${FORCEFLOW_PYTHON:-python}"
export PYTHONPATH="$forceflow_repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$forceflow_python" -m coribench_forceflow.train "$@"
