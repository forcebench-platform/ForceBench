"""Simulator-free contract for ForceBench expert-data backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .recording import CANONICAL_RUNTIME_PROFILE


EXPERT_COLLECTION_CONTRACT = "forcebench_expert_collection"


@dataclass(frozen=True)
class ExpertBackendManifest:
    """Lightweight description used to discover one expert backend."""

    name: str
    task: str
    backend_class: str
    description: str
    requires_checkpoint: bool = False
    enabled: bool = True
    disabled_reason: str | None = None

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid expert backend name: {self.name!r}")
        if not self.task or not self.task.replace("_", "").isalnum():
            raise ValueError(f"invalid expert backend task: {self.task!r}")
        if ":" not in self.backend_class:
            raise ValueError("backend_class must use 'module:ClassName' syntax")
        if not self.description:
            raise ValueError("expert backend description cannot be empty")
        if self.enabled and self.disabled_reason is not None:
            raise ValueError("enabled expert backend cannot have a disabled reason")
        if not self.enabled and not self.disabled_reason:
            raise ValueError("disabled expert backend must explain why it is unavailable")


@dataclass(frozen=True)
class ExpertRequest:
    """Canonical request passed from the supervisor to one backend."""

    task: str
    backend: str
    seed: int
    attempt: int
    code_root: Path
    isaaclab_root: Path
    dataset_dir: Path
    artifact_dir: Path
    device: str = "cuda:0"
    main_image_size: int = 224
    wrist_image_size: int = 224
    physics_rate_hz: int = 60
    solver_position_iterations: int = 64
    record_video: bool = False
    checkpoint: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.task or not self.backend:
            raise ValueError("expert task and backend cannot be empty")
        if self.seed < 0 or self.attempt < 1:
            raise ValueError("expert seed must be non-negative and attempt must be positive")
        if not self.device:
            raise ValueError("expert device cannot be empty")
        if self.main_image_size < 16 or self.wrist_image_size < 16:
            raise ValueError("expert camera sizes must be at least 16 pixels")
        expected_physics = int(CANONICAL_RUNTIME_PROFILE["physics_rate_hz"])
        expected_solver = int(CANONICAL_RUNTIME_PROFILE["solver_position_iterations"])
        if self.physics_rate_hz != expected_physics:
            raise ValueError(f"expert physics rate must be canonical ({expected_physics} Hz)")
        if self.solver_position_iterations != expected_solver:
            raise ValueError(
                "expert solver position iterations must be canonical "
                f"({expected_solver})"
            )
        for label, path in (
            ("code root", self.code_root),
            ("Isaac Lab root", self.isaaclab_root),
            ("dataset directory", self.dataset_dir),
            ("artifact directory", self.artifact_dir),
        ):
            if not path.is_absolute():
                raise ValueError(f"expert {label} must be absolute: {path}")
        if not isinstance(self.options, Mapping):
            raise TypeError("expert backend options must be a mapping")
        if not isinstance(self.record_video, bool):
            raise TypeError("expert record_video must be a boolean")


class ExpertBackend(ABC):
    """Translate a canonical expert request into one isolated Isaac process."""

    manifest: ExpertBackendManifest

    @abstractmethod
    def build_command(self, request: ExpertRequest) -> list[str]:
        """Return an argv vector that attempts to write one successful episode."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "contract": EXPERT_COLLECTION_CONTRACT,
            "backend": self.manifest.name,
            "task": self.manifest.task,
        }
