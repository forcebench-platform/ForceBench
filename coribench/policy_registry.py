"""Automatic discovery and lazy loading of CoRiBench policy adapters."""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from typing import Any, Mapping

from .policy_api import PolicyAdapter, PolicyAdapterManifest


@lru_cache(maxsize=1)
def policy_adapter_manifests() -> dict[str, PolicyAdapterManifest]:
    from . import policy_adapters as adapter_namespace

    discovered: dict[str, PolicyAdapterManifest] = {}
    for module_info in pkgutil.iter_modules(adapter_namespace.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{adapter_namespace.__name__}.{module_info.name}")
        manifest = getattr(module, "MANIFEST", None)
        if not isinstance(manifest, PolicyAdapterManifest):
            raise TypeError(
                f"policy adapter module {module.__name__} does not expose "
                "a PolicyAdapterManifest named MANIFEST"
            )
        manifest.validate()
        if manifest.name in discovered:
            raise ValueError(f"duplicate CoRiBench policy adapter: {manifest.name}")
        discovered[manifest.name] = manifest
    if not discovered:
        raise RuntimeError("no CoRiBench policy adapters were discovered")
    return dict(sorted(discovered.items()))


def policy_adapter_names() -> tuple[str, ...]:
    return tuple(policy_adapter_manifests())


def get_policy_adapter_manifest(name: str) -> PolicyAdapterManifest:
    try:
        return policy_adapter_manifests()[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown CoRiBench policy adapter: {name!r}; available={policy_adapter_names()}"
        ) from exc


def load_policy_adapter(
    name: str,
    *,
    host: str,
    port: int,
    options: Mapping[str, Any] | None = None,
) -> PolicyAdapter:
    manifest = get_policy_adapter_manifest(name)
    module_name, class_name = manifest.adapter_class.split(":", 1)
    adapter_type = getattr(importlib.import_module(module_name), class_name)
    adapter = adapter_type(host=host, port=port, options=dict(options or {}))
    if not isinstance(adapter, PolicyAdapter):
        raise TypeError(f"{manifest.adapter_class} must inherit PolicyAdapter")
    if adapter.manifest != manifest:
        raise ValueError(f"{manifest.adapter_class} exposes an inconsistent manifest")
    return adapter
