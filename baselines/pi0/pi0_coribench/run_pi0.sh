#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PI0_PYTHON:?Set PI0_PYTHON to the verified Python environment; no automatic installation}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88
export NCCL_NVLS_ENABLE=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$PI0_PYTHON" "$SCRIPT_DIR/run.py" "$@"
