"""Automatic discovery and lazy loading of CoRiBench task plugins."""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from pathlib import Path

from .scene_profiles import get_scene_profile
from .task_api import TaskManifest, TaskRuntime


@lru_cache(maxsize=1)
def task_manifests() -> dict[str, TaskManifest]:
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
            raise ValueError(f"duplicate CoRiBench task: {manifest.name}")
        if any(item.dataset_subdir == manifest.dataset_subdir for item in discovered.values()):
            raise ValueError(f"duplicate CoRiBench dataset_subdir: {manifest.dataset_subdir}")
        discovered[manifest.name] = manifest
    if not discovered:
        raise RuntimeError("no CoRiBench task plugins were discovered")
    return dict(sorted(discovered.items()))


def task_names() -> tuple[str, ...]:
    return tuple(task_manifests())


def get_task_manifest(name: str) -> TaskManifest:
    try:
        return task_manifests()[name]
    except KeyError as exc:
        raise ValueError(f"unknown CoRiBench task: {name!r}; available={task_names()}") from exc


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
