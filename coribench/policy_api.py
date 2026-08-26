"""Stable model-facing contracts used by every CoRiBench policy adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


POLICY_CONTRACT = "coribench_policy"
CANONICAL_ACTION_SPACE = "world_tcp_xyz_rpy_gripper_open_minus1_plus1"
STATE_SEMANTICS = "world_tcp_xyz_rpy_gripper_open_minus1_plus1"
WRENCH_SEMANTICS = "panda_hand_joint_parent_on_child_fx_fy_fz_tx_ty_tz"
WRENCH_FILTER = "none"
POLICY_CONTROL_RATE_HZ = 15.0


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite [{size}], got {array.shape}")
    return array.copy()


def _rgb_image(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 [H, W, 3], got {array.shape}, {array.dtype}")
    return array.copy()


def _finite_history(value: Any, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != width:
        raise ValueError(f"{name} must be finite [T, {width}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


def _rgb_history(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be uint8 [T, H, W, 3], got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 [T, H, W, 3], got {array.dtype}")
    return array.copy()


@dataclass(frozen=True)
class PolicyObservation:
    """One canonical 15 Hz observation, independent of model or transport."""

    step: int
    rollout_id: str
    state: np.ndarray
    main_image: np.ndarray
    wrist_image: np.ndarray
    eef_wrench: np.ndarray
    prompt: str
    state_history: np.ndarray | None = None
    main_image_history: np.ndarray | None = None
    wrist_image_history: np.ndarray | None = None
    eef_wrench_history: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("policy observation step must be non-negative")
        if not isinstance(self.rollout_id, str) or not self.rollout_id.strip():
            raise ValueError("policy rollout_id must be a non-empty string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("policy prompt must be a non-empty string")
        object.__setattr__(self, "state", _finite_vector(self.state, 7, "state"))
        object.__setattr__(self, "main_image", _rgb_image(self.main_image, "main_image"))
        object.__setattr__(self, "wrist_image", _rgb_image(self.wrist_image, "wrist_image"))
        object.__setattr__(self, "eef_wrench", _finite_vector(self.eef_wrench, 6, "eef_wrench"))
        histories = (
            self.state_history,
            self.main_image_history,
            self.wrist_image_history,
            self.eef_wrench_history,
        )
        if any(value is not None for value in histories) and not all(
            value is not None for value in histories
        ):
            raise ValueError("explicit policy history must provide state, both images, and wrench")
        if all(value is not None for value in histories):
            state_history = _finite_history(self.state_history, 7, "state_history")
            main_history = _rgb_history(self.main_image_history, "main_image_history")
            wrist_history = _rgb_history(self.wrist_image_history, "wrist_image_history")
            force_history = _finite_history(self.eef_wrench_history, 6, "eef_wrench_history")
            observation_steps = state_history.shape[0]
            if main_history.shape[0] != observation_steps or wrist_history.shape[0] != observation_steps:
                raise ValueError("state and image histories must have the same length")
            if main_history.shape[1:] != self.main_image.shape or wrist_history.shape[1:] != self.wrist_image.shape:
                raise ValueError("history image shapes must match the current images")
            if not np.allclose(state_history[-1], self.state, atol=1e-6):
                raise ValueError("latest state_history frame must equal state")
            if not np.array_equal(main_history[-1], self.main_image):
                raise ValueError("latest main_image_history frame must equal main_image")
            if not np.array_equal(wrist_history[-1], self.wrist_image):
                raise ValueError("latest wrist_image_history frame must equal wrist_image")
            if not np.allclose(force_history[-1], self.eef_wrench, atol=1e-6):
                raise ValueError("latest eef_wrench_history frame must equal eef_wrench")
            object.__setattr__(self, "state_history", state_history)
            object.__setattr__(self, "main_image_history", main_history)
            object.__setattr__(self, "wrist_image_history", wrist_history)
            object.__setattr__(self, "eef_wrench_history", force_history)

    def canonical_request(self) -> dict[str, Any]:
        """Return the standard wire payload for native CoRiBench policy servers."""
        observation = {
            "state": self.state,
            "state_semantics": STATE_SEMANTICS,
            "images": {
                "main": self.main_image,
                "wrist": self.wrist_image,
            },
            "eef_wrench": self.eef_wrench,
            "wrench_semantics": WRENCH_SEMANTICS,
            "wrench_filter": WRENCH_FILTER,
            "prompt": self.prompt,
        }
        if self.state_history is not None:
            observation["history"] = {
                "states": self.state_history,
                "images": {
                    "main": self.main_image_history,
                    "wrist": self.wrist_image_history,
                },
                "eef_wrenches": self.eef_wrench_history,
            }
        return {
            "contract": POLICY_CONTRACT,
            "step": self.step,
            "rollout_id": self.rollout_id,
            "control_rate_hz": POLICY_CONTROL_RATE_HZ,
            "observation": observation,
        }


@dataclass(frozen=True)
class PolicyActionChunk:
    """Canonical absolute TCP targets returned by an adapter."""

    actions: np.ndarray
    action_space: str = CANONICAL_ACTION_SPACE
    control_rate_hz: float = POLICY_CONTROL_RATE_HZ

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7:
            raise ValueError(f"policy actions must have shape [H, 7], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        if self.action_space != CANONICAL_ACTION_SPACE:
            raise ValueError(
                f"adapter returned action space {self.action_space!r}; "
                f"expected {CANONICAL_ACTION_SPACE!r}"
            )
        if not np.isclose(float(self.control_rate_hz), POLICY_CONTROL_RATE_HZ):
            raise ValueError(
                f"adapter returned {self.control_rate_hz} Hz actions; "
                f"expected {POLICY_CONTROL_RATE_HZ} Hz"
            )
        object.__setattr__(self, "actions", actions.copy())


@dataclass(frozen=True)
class PolicyAdapterManifest:
    """Simulator-free metadata used for automatic adapter discovery."""

    name: str
    adapter_class: str
    description: str = ""

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid policy adapter name: {self.name!r}")
        if ":" not in self.adapter_class:
            raise ValueError("adapter_class must use 'module:ClassName' syntax")


class PolicyAdapter(ABC):
    """Translate one model interface into the canonical CoRiBench contract."""

    manifest: PolicyAdapterManifest

    @property
    @abstractmethod
    def endpoint_uri(self) -> str:
        """Human-readable endpoint used in logs and rollout metadata."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "contract": POLICY_CONTRACT,
            "adapter": self.manifest.name,
            "endpoint": self.endpoint_uri,
            "action_space": CANONICAL_ACTION_SPACE,
            "control_rate_hz": POLICY_CONTROL_RATE_HZ,
        }

    @abstractmethod
    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        """Return canonical absolute TCP targets for one observation."""

    @abstractmethod
    def close(self) -> None:
        """Release any transport owned by the adapter."""
