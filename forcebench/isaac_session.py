"""Process-local ownership of a shared Isaac ``SimulationApp``.

Normal expert generators own and close their application.  The persistent
worker registers one application here so the same generators can reuse it
across multiple isolated episodes without changing their trajectory logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_SHARED_SIMULATION_APP: Any | None = None
_SHARED_ISAAC_RESOURCES: dict[str, tuple[Any, Callable[[Any], None]]] = {}


class PersistentWorkerRecycleRequired(RuntimeError):
    """Request a clean Isaac process before attempting another episode."""


def shared_simulation_app() -> Any | None:
    return _SHARED_SIMULATION_APP


def register_shared_simulation_app(app: Any) -> None:
    global _SHARED_SIMULATION_APP
    if app is None:
        raise ValueError("shared Isaac application cannot be None")
    if _SHARED_SIMULATION_APP is not None and _SHARED_SIMULATION_APP is not app:
        raise RuntimeError("another shared Isaac application is already registered")
    _SHARED_SIMULATION_APP = app


def clear_shared_simulation_app(app: Any) -> None:
    global _SHARED_SIMULATION_APP
    if _SHARED_SIMULATION_APP is not app:
        raise RuntimeError("cannot clear an Isaac application owned by another session")
    _SHARED_SIMULATION_APP = None


def shared_isaac_resource(name: str) -> Any | None:
    """Return one process-local resource owned by the shared Isaac app."""

    entry = _SHARED_ISAAC_RESOURCES.get(name)
    return None if entry is None else entry[0]


def register_shared_isaac_resource(
    name: str,
    resource: Any,
    *,
    close: Callable[[Any], None] | None = None,
) -> None:
    """Keep an environment or sensor resource alive for the worker lifetime."""

    if _SHARED_SIMULATION_APP is None:
        raise RuntimeError("shared Isaac resources require a registered application")
    if not name or not name.replace("_", "").isalnum():
        raise ValueError(f"invalid shared Isaac resource name: {name!r}")
    if resource is None:
        raise ValueError("shared Isaac resource cannot be None")
    if name in _SHARED_ISAAC_RESOURCES:
        raise RuntimeError(f"shared Isaac resource {name!r} is already registered")
    if close is None:
        close_method = getattr(resource, "close", None)
        if not callable(close_method):
            raise TypeError("shared Isaac resource needs a close callback or close() method")
        close = lambda value: value.close()
    _SHARED_ISAAC_RESOURCES[name] = (resource, close)


def close_shared_isaac_resources() -> None:
    """Close long-lived resources in reverse registration order."""

    errors: list[str] = []
    while _SHARED_ISAAC_RESOURCES:
        name = next(reversed(_SHARED_ISAAC_RESOURCES))
        resource, close = _SHARED_ISAAC_RESOURCES.pop(name)
        try:
            close(resource)
        except Exception as error:  # cleanup must continue for the remaining resources
            errors.append(f"{name}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError("failed to close shared Isaac resources: " + "; ".join(errors))
