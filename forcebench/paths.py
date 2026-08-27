"""Canonical storage roots for code, formal artifacts and disposable work."""

from __future__ import annotations

import os
import re
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FORCEBENCH_DATA_ROOT", "/home/pc/ForceBench_data")).resolve()
SCRATCH_ROOT = Path(
    os.environ.get("FORCEBENCH_SCRATCH_ROOT", "/home/pc/ForceBench_scratch")
).resolve()
ARCHIVE_ROOT = Path(
    os.environ.get("FORCEBENCH_ARCHIVE_ROOT", "/home/pc/ForceBench_archive")
).resolve()

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _validate_roots() -> None:
    named = {
        "code": CODE_ROOT,
        "formal data": DATA_ROOT,
        "scratch": SCRATCH_ROOT,
        "archive": ARCHIVE_ROOT,
    }
    values = list(named.values())
    if len(set(values)) != len(values):
        raise ValueError(f"ForceBench storage roots must be distinct: {named}")
    for label, path in named.items():
        if label != "code" and path.is_relative_to(CODE_ROOT):
            raise ValueError(f"{label} root must be outside the code repository: {path}")


_validate_roots()


def _child(root: Path, category: str, name: str | None = None) -> Path:
    if not _SAFE_NAME.fullmatch(category):
        raise ValueError(f"invalid storage category: {category!r}")
    path = root / category
    if name is not None:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"invalid storage name: {name!r}")
        path /= name
    return path


def dataset_dir(task: str) -> Path:
    """Return the formal HDF5 dataset directory for one registered task."""
    return _child(DATA_ROOT, "datasets", task)


def checkpoint_dir(name: str | None = None) -> Path:
    return _child(DATA_ROOT, "checkpoints", name)


def evaluation_dir(name: str) -> Path:
    return _child(DATA_ROOT, "evaluations", name)


def scratch_dir(category: str, name: str | None = None) -> Path:
    """Return a disposable output path, always outside formal data storage."""
    return _child(SCRATCH_ROOT, category, name)
