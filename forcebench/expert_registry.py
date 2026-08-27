"""Automatic discovery and lazy loading of expert-data backends."""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from typing import Any

from .expert_api import ExpertBackend, ExpertBackendManifest
from .registry import task_names


@lru_cache(maxsize=1)
def all_expert_backend_manifests() -> dict[tuple[str, str], ExpertBackendManifest]:
    from . import expert_adapters as adapter_namespace

    registered_tasks = set(task_names())
    discovered: dict[tuple[str, str], ExpertBackendManifest] = {}
    for module_info in pkgutil.iter_modules(adapter_namespace.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{adapter_namespace.__name__}.{module_info.name}")
        manifest = getattr(module, "MANIFEST", None)
        if not isinstance(manifest, ExpertBackendManifest):
            raise TypeError(
                f"expert adapter module {module.__name__} does not expose an "
                "ExpertBackendManifest named MANIFEST"
            )
        manifest.validate()
        if manifest.task not in registered_tasks:
            raise ValueError(
                f"expert backend {manifest.name!r} references unknown task: "
                f"{manifest.task!r}"
            )
        key = (manifest.task, manifest.name)
        if key in discovered:
            raise ValueError(f"duplicate expert backend registration: {key}")
        discovered[key] = manifest
    if not discovered:
        raise RuntimeError("no ForceBench expert backends were discovered")
    return dict(sorted(discovered.items()))


def expert_backend_manifests() -> dict[tuple[str, str], ExpertBackendManifest]:
    """Return only collection-ready backends.

    Incomplete task adapters remain discoverable through
    :func:`all_expert_backend_manifests`, but cannot accidentally be selected
    by the collection supervisor.
    """

    return {
        key: manifest
        for key, manifest in all_expert_backend_manifests().items()
        if manifest.enabled
    }


def expert_backend_names(task: str) -> tuple[str, ...]:
    return tuple(name for (registered_task, name) in expert_backend_manifests() if registered_task == task)


def default_expert_backend_name(task: str) -> str:
    names = expert_backend_names(task)
    if not names:
        raise ValueError(f"task {task!r} has no automated expert backend")
    return "scripted" if "scripted" in names else names[0]


def get_expert_backend_manifest(task: str, name: str) -> ExpertBackendManifest:
    try:
        return expert_backend_manifests()[(task, name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown expert backend {name!r} for task {task!r}; "
            f"available={expert_backend_names(task)}"
        ) from exc


def load_expert_backend(task: str, name: str, **kwargs: Any) -> ExpertBackend:
    manifest = get_expert_backend_manifest(task, name)
    module_name, class_name = manifest.backend_class.split(":", 1)
    backend_type = getattr(importlib.import_module(module_name), class_name)
    backend = backend_type(**kwargs)
    if not isinstance(backend, ExpertBackend):
        raise TypeError(f"{manifest.backend_class} must inherit ExpertBackend")
    if backend.manifest != manifest:
        raise ValueError(f"{manifest.backend_class} exposes an inconsistent manifest")
    return backend
