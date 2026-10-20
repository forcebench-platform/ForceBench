"""Public task-plugin contract for ForceBench.

This module is deliberately simulator-free so task manifests can be discovered
before Isaac Sim is launched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TaskManifest:
    """Lightweight metadata used for discovery and command-line selection."""

    name: str
    prompt: str
    runtime_class: str
    dataset_subdir: str
    scene_profile: str = "contact_lab"
    description: str = ""
    evaluation_max_steps: int = 600
    # None keeps the existing environment drive configuration unchanged.
    # Recovered tasks declare their historical gains explicitly.
    arm_actuator_stiffness: float | None = None
    arm_actuator_damping: float | None = None
    gripper_actuator_stiffness: float | None = None
    gripper_actuator_damping: float | None = None

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid task name: {self.name!r}")
        if ":" not in self.runtime_class:
            raise ValueError("runtime_class must use 'module:ClassName' syntax")
        dataset_path = Path(self.dataset_subdir)
        if (
            not self.dataset_subdir
            or dataset_path.is_absolute()
            or any(part in {"", ".", ".."} for part in dataset_path.parts)
        ):
            raise ValueError("dataset_subdir must be a non-empty relative path")
        if not self.scene_profile or not self.scene_profile.replace("_", "").isalnum():
            raise ValueError(f"invalid scene profile name: {self.scene_profile!r}")
        if type(self.evaluation_max_steps) is not int or self.evaluation_max_steps < 1:
            raise ValueError("evaluation_max_steps must be a positive integer")
        for field_name in (
            "arm_actuator_stiffness", "arm_actuator_damping",
            "gripper_actuator_stiffness", "gripper_actuator_damping",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) not in (int, float) or not 0.0 < float(value) < float("inf")
            ):
                raise ValueError(f"{field_name} must be a positive finite number or None")


@dataclass(frozen=True)
class CameraProfile:
    """Immutable camera contract; tasks may override it at class level."""

    # User-tuned centered workspace overview, promoted to the benchmark-wide
    # main-camera contract on 2026-08-21. Wrist vision remains unchanged and
    # supplies the contact close-up.
    table_position_m: tuple[float, float, float] = (1.30, 0.00, 0.69)
    table_target_m: tuple[float, float, float] | None = (
        0.55732250,
        0.00,
        0.02035074,
    )
    table_rotation_wxyz: tuple[float, float, float, float] = (
        0.64607454,
        0.28738073,
        0.28738073,
        0.64607454,
    )
    table_focal_length_mm: float = 24.0
    wrist_position_in_hand_m: tuple[float, float, float] = (0.060, 0.0, 0.025)
    wrist_rotation_wxyz: tuple[float, float, float, float] = (
        0.18149164,
        0.68341846,
        0.68341846,
        0.18149164,
    )
    wrist_focal_length_mm: float = 18.0
    wrist_clipping_range_m: tuple[float, float] = (0.05, 2.0)
    wrist_image_rotation_deg: int = 0
    wrist_eye_offset_from_fingertip_world_m: tuple[float, float, float] | None = None
    wrist_target_offset_from_fingertip_world_m: tuple[float, float, float] | None = None


# Default camera contract used by tasks without an explicit class-level view.
CANONICAL_CAMERA_PROFILE = CameraProfile()


@dataclass(frozen=True)
class SceneEntitySpec:
    """One named simulator entity supplied by a task plugin."""

    role: str
    kind: str
    config: Any
    friction: float | None = None

    def validate(self) -> None:
        if not self.role or not self.role.replace("_", "").isalnum():
            raise ValueError(f"invalid scene role: {self.role!r}")
        if self.kind not in {"articulation", "rigid_object"}:
            raise ValueError(f"unsupported entity kind: {self.kind!r}")


class TaskRuntime(ABC):
    """Runtime interface implemented by every ForceBench demo."""

    manifest: TaskManifest

    @property
    def camera_profile(self) -> CameraProfile:
        """Return the default camera contract unless the task class overrides it."""
        return CANONICAL_CAMERA_PROFILE

    def __setattr__(self, name: str, value: object) -> None:
        if name == "camera_profile":
            raise AttributeError("camera_profile must be an immutable class-level profile")
        super().__setattr__(name, value)

    def __init__(self, asset_dir: str | Path):
        self.asset_dir = Path(asset_dir).resolve()
        self.manifest.validate()
        if not isinstance(self.camera_profile, CameraProfile):
            raise TypeError("task camera_profile must be a CameraProfile")

    @abstractmethod
    def validate_assets(self) -> None:
        """Raise when a required task asset is missing."""

    @abstractmethod
    def scene_entities(self) -> Sequence[SceneEntitySpec]:
        """Return all task-owned scene entities and their semantic roles."""

    @abstractmethod
    def sample_initial_state(self, seed: int) -> dict[str, object]:
        """Return a deterministic, JSON-serializable initial layout."""

    @abstractmethod
    def reset_scene(self, env: Any, env_ids: Any, layout: Mapping[str, object]) -> None:
        """Apply a sampled layout to the simulator."""

    @abstractmethod
    def evaluate_final_state(self, env: Any) -> dict[str, object]:
        """Evaluate success only when explicitly invoked after submission."""

    @abstractmethod
    def privileged_observations(self, env: Any) -> Mapping[str, Any]:
        """Return named finite vectors to store at each control frame."""

    @abstractmethod
    def wrench_reference_position(self, env: Any) -> Any:
        """Return the environment-local point used as wrench moment origin."""

    def episode_metadata(self) -> dict[str, object]:
        return {}

    def instruction(self, layout: Mapping[str, object]) -> str:
        """Return the language instruction for a sampled layout."""
        return self.manifest.prompt

    def spawn_scene_prims(self, env: Any) -> None:
        """Optional hook for task-specific static geometry before cloning."""

    def after_scene_setup(self, env: Any) -> None:
        """Optional hook after all simulator handles are initialized."""

    def on_control_step(self, env: Any) -> None:
        """Optional 15 Hz callback for task-specific history latching."""

    def failure_reasons(self, report: Mapping[str, object]) -> list[str]:
        return [key for key, value in report.items() if key != "success" and isinstance(value, bool) and not value]

    def debug_state(self, env: Any) -> Mapping[str, object]:
        return {}
