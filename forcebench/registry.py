"""Automatic discovery and lazy loading of ForceBench task plugins."""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from pathlib import Path

from .scene_profiles import get_scene_profile
from .task_api import TaskManifest, TaskRuntime


LEGACY_TASK_NAMES = frozenset({"gear_mesh", "peg_insert"})


@lru_cache(maxsize=2)
def task_manifests(*, include_legacy: bool = False) -> dict[str, TaskManifest]:
    """Discover tasks; keep prototype Gear/Peg out of default paper choices."""
    from . import tasks as task_namespace

    discovered: dict[str, TaskManifest] = {}
    for module_info in pkgutil.iter_modules(task_namespace.__path__):
        if not module_info.ispkg or module_info.name.startswith("_"):
            continue
        package = importlib.import_module(f"{task_namespace.__name__}.{module_info.name}")
        manifest = getattr(package, "MANIFEST", None)
        if not isinstance(manifest, TaskManifest):
            raise TypeError(f"task package {package.__name__} does not expose a TaskManifest named MANIFEST")
        manifest.validate()
        get_scene_profile(manifest.scene_profile)
        if manifest.name in discovered:
            raise ValueError(f"duplicate ForceBench task: {manifest.name}")
        if any(item.dataset_subdir == manifest.dataset_subdir for item in discovered.values()):
            raise ValueError(f"duplicate ForceBench dataset_subdir: {manifest.dataset_subdir}")
        discovered[manifest.name] = manifest
    if not discovered:
        raise RuntimeError("no ForceBench task plugins were discovered")
    return {
        name: manifest for name, manifest in sorted(discovered.items())
        if include_legacy or name not in LEGACY_TASK_NAMES
    }


def task_names(*, include_legacy: bool = False) -> tuple[str, ...]:
    return tuple(task_manifests(include_legacy=include_legacy))


def get_task_manifest(name: str) -> TaskManifest:
    try:
        # Explicit lookups still support historical tooling; ordinary CLI
        # choices and the release inventory use the paper task set.
        return task_manifests(include_legacy=True)[name]
    except KeyError as exc:
        raise ValueError(f"unknown ForceBench task: {name!r}; available={task_names()}") from exc


def load_task(name: str, asset_dir: str | Path) -> TaskRuntime:
    manifest = get_task_manifest(name)
    module_name, class_name = manifest.runtime_class.split(":", 1)
    runtime_type = getattr(importlib.import_module(module_name), class_name)
    runtime = runtime_type(asset_dir)
    if not isinstance(runtime, TaskRuntime):
        raise TypeError(f"{manifest.runtime_class} must inherit TaskRuntime")
    if runtime.manifest != manifest:
        raise ValueError(f"{manifest.runtime_class} exposes a manifest inconsistent with its package")
    runtime.validate_assets()
    return runtime
