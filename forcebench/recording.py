"""Canonical ForceBench HDF5 episode buffer and atomic writer."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .control import (
    ARM_CONTROL_MODE,
    DEFAULT_ARM_CONTROLLER_PROFILE,
    DEFAULT_SMOOTH_POSITION_CONFIG,
)


# Tag written into every new episode.
HDF5_SCHEMA_VERSION = "forcebench_hdf5"
# The project was called CoRiBench when the currently published expert data was
# collected, so every existing .hdf5 on disk carries the old tag. Readers must
# accept both; only writers use HDF5_SCHEMA_VERSION above. Do not "clean this
# up" by dropping the legacy value — it would make the released converter
# unable to read the released data.
HDF5_SCHEMA_VERSION_LEGACY = "coribench_hdf5"
HDF5_SCHEMA_VERSIONS_ACCEPTED = (HDF5_SCHEMA_VERSION, HDF5_SCHEMA_VERSION_LEGACY)
MODEL_VECTOR_DIM = 7
WRENCH_DIM = 6
CANONICAL_WRENCH_PROFILE = {
    "source": "link_incoming_joint_force",
    "link": "force_sensor",
    "joint": "panda_hand_joint",
    "frame": "incoming_joint",
    "direction": "parent_on_child",
    "component_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
    "policy_filter": {"type": "none"},
    "stored_control_channels": {
        "raw": {"filter": "none"},
        "ema": {"filter": "ema", "alpha": 0.25},
    },
    "coordinate_transform": "none",
    "gravity_compensation": False,
}
CANONICAL_RUNTIME_PROFILE = {
    "control_rate_hz": 15.0,
    "physics_rate_hz": 60.0,
    "solver_position_iterations": 64,
}
CANONICAL_ARM_CONTROLLER_PROFILE = DEFAULT_ARM_CONTROLLER_PROFILE
# 120 Hz remains readable only so legacy datasets can still be audited.  Every
# current collection entry point is fixed to the 60 Hz production contract.
SUPPORTED_PHYSICS_RATES_HZ = (60, 120)


def runtime_profile(physics_rate_hz: int) -> dict[str, float | int]:
    """Return the canonical runtime contract for a supported physics rate."""

    physics_rate_hz = int(physics_rate_hz)
    if physics_rate_hz not in SUPPORTED_PHYSICS_RATES_HZ:
        raise ValueError(
            f"physics_rate_hz must be one of {SUPPORTED_PHYSICS_RATES_HZ}, "
            f"got {physics_rate_hz}"
        )
    return {
        "control_rate_hz": 15.0,
        "physics_rate_hz": float(physics_rate_hz),
        "solver_position_iterations": 64,
    }


def arm_controller_profile(physics_rate_hz: int) -> dict[str, object]:
    """Return the controller contract at the selected physics rate."""

    return DEFAULT_SMOOTH_POSITION_CONFIG.profile(
        physics_rate_hz=int(physics_rate_hz), control_rate_hz=15
    )


def _profiles_from_metadata(
    metadata: dict[str, Any],
) -> tuple[dict[str, float | int], dict[str, object]]:
    raw_rate = metadata.get("physics_rate_hz", CANONICAL_RUNTIME_PROFILE["physics_rate_hz"])
    rate = float(raw_rate)
    if not math.isfinite(rate) or not rate.is_integer():
        raise ValueError(f"metadata physics_rate_hz must be an integer rate, got {raw_rate!r}")
    runtime = runtime_profile(int(rate))
    return runtime, arm_controller_profile(int(rate))

_FIELD_PATH = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
_SUCCESS_EPISODE_STEM = re.compile(r"^seed_(\d+)_episode_(\d+)$")


def success_episode_stem(seed: int, episode_index: int) -> str:
    seed = int(seed)
    episode_index = int(episode_index)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if episode_index < 1:
        raise ValueError("episode_index must be one-based")
    return f"seed_{seed:03d}_episode_{episode_index:02d}"


def parse_success_episode_stem(stem: str) -> tuple[int, int] | None:
    match = _SUCCESS_EPISODE_STEM.fullmatch(stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _as_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = _as_numpy(value).astype(np.float32, copy=False)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


def _as_rgb_image(value: Any, name: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} cannot be empty")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    return array.astype(np.uint8, copy=True)


def wrap_to_interval(value: float, lower: float, upper: float) -> float:
    width = upper - lower
    if width <= 0:
        raise ValueError("upper must be greater than lower")
    return (value - lower) % width + lower


@dataclass
class EpisodeBuffer:
    """In-memory control-rate buffer for the one official episode schema."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    joint_wrenches_raw: list[np.ndarray] = field(default_factory=list)
    joint_wrenches_ema: list[np.ndarray] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    images: list[np.ndarray] = field(default_factory=list)
    wrist_images: list[np.ndarray] = field(default_factory=list)
    joint_positions: list[np.ndarray] = field(default_factory=list)
    joint_velocities: list[np.ndarray] = field(default_factory=list)
    tcp_poses: list[np.ndarray] = field(default_factory=list)
    privileged_fields: dict[str, list[np.ndarray]] = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    terminated_flags: list[bool] = field(default_factory=list)
    truncated_flags: list[bool] = field(default_factory=list)
    phase_ids: list[int] = field(default_factory=list)
    _images_present: bool | None = field(default=None, init=False, repr=False)
    _privileged_keys: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _phase_present: bool | None = field(default=None, init=False, repr=False)

    def append(
        self,
        *,
        state: Any,
        action: Any,
        wrench_raw: Any,
        wrench_ema: Any,
        timestamp: float,
        image: Any | None = None,
        wrist_image: Any | None = None,
        joint_position: Any | None = None,
        joint_velocity: Any | None = None,
        tcp_pose: Any | None = None,
        privileged: dict[str, Any] | None = None,
        phase_id: int | None = None,
    ) -> None:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self.timestamps and timestamp <= self.timestamps[-1]:
            raise ValueError("timestamps must be strictly increasing")
        if (image is None) != (wrist_image is None):
            raise ValueError("image and wrist_image must be provided together")
        images_present = image is not None
        if self._images_present is None:
            self._images_present = images_present
        elif self._images_present != images_present:
            raise ValueError("cannot mix frames with and without RGB images")

        privileged = privileged or {}
        keys = tuple(sorted(privileged))
        if self._privileged_keys is None:
            for key in keys:
                if not _FIELD_PATH.fullmatch(key):
                    raise ValueError(f"invalid privileged field path: {key!r}")
            self._privileged_keys = keys
            self.privileged_fields = {key: [] for key in keys}
        elif keys != self._privileged_keys:
            raise ValueError(
                f"privileged field set changed from {self._privileged_keys} to {keys}"
            )

        self.states.append(_as_vector(state, MODEL_VECTOR_DIM, "state"))
        self.actions.append(_as_vector(action, MODEL_VECTOR_DIM, "action"))
        self.joint_wrenches_raw.append(
            _as_vector(wrench_raw, WRENCH_DIM, "wrench_raw")
        )
        self.joint_wrenches_ema.append(_as_vector(wrench_ema, WRENCH_DIM, "wrench_ema"))
        self.timestamps.append(float(timestamp))
        if images_present:
            self.images.append(_as_rgb_image(image, "image"))
            self.wrist_images.append(_as_rgb_image(wrist_image, "wrist_image"))

        optional_values = (
            (self.joint_positions, joint_position, "joint_position"),
            (self.joint_velocities, joint_velocity, "joint_velocity"),
            (self.tcp_poses, tcp_pose, "tcp_pose"),
        )
        for destination, value, name in optional_values:
            if value is not None:
                array = _as_numpy(value).astype(np.float32, copy=False)
                if array.ndim != 1 or not np.isfinite(array).all():
                    raise ValueError(f"{name} must be a finite vector, got {array.shape}")
                destination.append(array.copy())

        for key in keys:
            array = _as_numpy(privileged[key]).astype(np.float32, copy=False)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError(f"privileged field {key!r} must be a finite vector")
            self.privileged_fields[key].append(array.copy())

        self.rewards.append(0.0)
        self.successes.append(False)
        self.terminated_flags.append(False)
        self.truncated_flags.append(False)
        phase_present = phase_id is not None
        if self._phase_present is None:
            self._phase_present = phase_present
        elif self._phase_present != phase_present:
            raise ValueError("phase_id must be present either on every frame or no frames")
        if phase_present:
            if not 0 <= int(phase_id) <= 255:
                raise ValueError("phase_id must fit in uint8")
            self.phase_ids.append(int(phase_id))

    def set_last_transition_outcome(
        self, *, reward: float, success: bool, terminated: bool, truncated: bool
    ) -> None:
        if not self.timestamps:
            raise ValueError("cannot set an outcome on an empty episode")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        self.rewards[-1] = float(reward)
        self.successes[-1] = bool(success)
        self.terminated_flags[-1] = bool(terminated)
        self.truncated_flags[-1] = bool(truncated)

    def __len__(self) -> int:
        return len(self.timestamps)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.timestamps:
            raise ValueError("cannot serialize an empty episode")
        arrays = {
            "observation.state": np.stack(self.states).astype(np.float32),
            "action": np.stack(self.actions).astype(np.float32),
            "eef_wrench_joint_raw": np.stack(self.joint_wrenches_raw).astype(np.float32),
            "eef_wrench_joint_ema": np.stack(self.joint_wrenches_ema).astype(np.float32),
            "timestamp": np.asarray(self.timestamps, dtype=np.float64),
            "reward": np.asarray(self.rewards, dtype=np.float32),
            "success": np.asarray(self.successes, dtype=np.bool_),
            "terminated": np.asarray(self.terminated_flags, dtype=np.bool_),
            "truncated": np.asarray(self.truncated_flags, dtype=np.bool_),
        }
        if self._images_present:
            arrays["observation.image"] = np.stack(self.images).astype(np.uint8)
            arrays["observation.wrist_image"] = np.stack(self.wrist_images).astype(np.uint8)
        optional_arrays = (
            ("observation.joint_position", self.joint_positions),
            ("observation.joint_velocity", self.joint_velocities),
            ("observation.tcp_pose", self.tcp_poses),
        )
        for key, values in optional_arrays:
            if values:
                if len(values) != len(self):
                    raise ValueError(f"{key} is present for only {len(values)}/{len(self)} frames")
                arrays[key] = np.stack(values).astype(np.float32)
        for key, values in self.privileged_fields.items():
            if len(values) != len(self):
                raise ValueError(f"privileged/{key} is present for only {len(values)}/{len(self)} frames")
            arrays[f"privileged.{key}"] = np.stack(values).astype(np.float32)
        if self.phase_ids:
            if len(self.phase_ids) != len(self):
                raise ValueError("phase_id is not present for every episode frame")
            arrays["phase_id"] = np.asarray(self.phase_ids, dtype=np.uint8)
        return arrays


class HDF5EpisodeWriter:
    """Atomically write one self-contained canonical HDF5 file per episode."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def next_episode_index(self, seed: int) -> int:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        indices = []
        for path in self.output_dir.glob("seed_*_episode_*.hdf5"):
            parsed = parse_success_episode_stem(path.stem)
            if parsed is not None and parsed[0] == seed:
                indices.append(parsed[1])
        return max(indices, default=0) + 1

    def successful_episode_count(self) -> int:
        return sum(
            parse_success_episode_stem(path.stem) is not None
            for path in self.output_dir.glob("*.hdf5")
        )

    @staticmethod
    def _dataset(group, name: str, data: np.ndarray):
        chunks = (1, *data.shape[1:]) if data.ndim > 1 else True
        return group.create_dataset(
            name,
            data=data,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

    @classmethod
    def _nested_dataset(cls, group, path: str, data: np.ndarray):
        parts = path.split("/")
        current = group
        for part in parts[:-1]:
            current = current.require_group(part)
        return cls._dataset(current, parts[-1], data)

    def write(
        self,
        episode: EpisodeBuffer,
        metadata: dict[str, Any],
        *,
        high_rate: dict[str, np.ndarray] | None = None,
    ) -> tuple[Path]:
        import h5py

        metadata = dict(metadata)
        expert_contract = os.environ.get("FORCEBENCH_EXPERT_COLLECTION_CONTRACT")
        if expert_contract is not None:
            if expert_contract != "forcebench_expert_collection":
                raise ValueError(f"unsupported expert collection contract: {expert_contract!r}")
            expert_task = os.environ.get("FORCEBENCH_TASK")
            if expert_task and metadata.get("task") != expert_task:
                raise ValueError(
                    f"expert collector task {expert_task!r} does not match episode "
                    f"task {metadata.get('task')!r}"
                )
            expert_backend = os.environ.get("FORCEBENCH_EXPERT_BACKEND")
            if not expert_backend:
                raise ValueError("standard expert collection is missing its backend identity")
            metadata.update(
                {
                    "expert_collection_contract": expert_contract,
                    "expert_backend": expert_backend,
                    "expert_policy_wrench_field": "forces/control/eef_wrench_joint_raw",
                    "expert_recorded_wrench_fields": [
                        "forces/control/eef_wrench_joint_raw",
                        "forces/control/eef_wrench_joint_ema",
                    ],
                }
            )

        if "seed" not in metadata:
            raise ValueError("successful episode metadata must include seed")
        required_metadata = ("task", "initial_layout", "scene_profile", "control_source")
        missing_metadata = [key for key in required_metadata if key not in metadata]
        if missing_metadata:
            raise ValueError(f"successful episode metadata is missing: {missing_metadata}")
        if not isinstance(metadata["task"], str) or not metadata["task"]:
            raise ValueError("successful episode task must be a non-empty string")
        if not isinstance(metadata["initial_layout"], dict):
            raise ValueError("successful episode initial_layout must be a mapping")
        if not isinstance(metadata["scene_profile"], str) or not metadata["scene_profile"]:
            raise ValueError("successful episode scene_profile must be a non-empty string")
        if metadata.get("success") is not True:
            raise ValueError("the formal HDF5 writer accepts successful episodes only")
        seed = int(metadata["seed"])
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if high_rate is None:
            raise ValueError("canonical HDF5 requires the physics-rate joint wrench stream")
        if set(high_rate) != {"timestamp", "eef_wrench_joint_raw"}:
            raise ValueError(
                "physics-rate stream must contain only timestamp and eef_wrench_joint_raw"
            )
        physics_time = np.asarray(high_rate["timestamp"], dtype=np.float64)
        physics_wrench = np.asarray(high_rate["eef_wrench_joint_raw"], dtype=np.float32)
        if physics_time.ndim != 1 or not len(physics_time):
            raise ValueError("physics timestamps must be a non-empty vector")
        if np.any(~np.isfinite(physics_time)) or np.any(np.diff(physics_time) <= 0):
            raise ValueError("physics timestamps must be finite and strictly increasing")
        if physics_wrench.shape != (len(physics_time), WRENCH_DIM):
            raise ValueError("physics joint wrench must have shape [physics_frames, 6]")
        if not np.isfinite(physics_wrench).all():
            raise ValueError("physics joint wrench contains NaN or Inf")

        arrays = episode.arrays()
        images_present = "observation.image" in arrays
        if bool(metadata.get("images_present", images_present)) != images_present:
            raise ValueError("metadata images_present does not match episode arrays")
        supplied_profile = metadata.get("wrench_profile")
        if supplied_profile is not None and supplied_profile != CANONICAL_WRENCH_PROFILE:
            raise ValueError("metadata wrench_profile does not match the canonical contract")
        arm_control_mode = metadata.get("arm_control_mode", ARM_CONTROL_MODE)
        if arm_control_mode != ARM_CONTROL_MODE:
            raise ValueError(
                f"arm_control_mode must be {ARM_CONTROL_MODE!r}, got {arm_control_mode!r}"
            )
        expected_runtime, expected_controller = _profiles_from_metadata(metadata)
        supplied_controller = metadata.get("arm_controller_profile")
        if supplied_controller is not None and supplied_controller != expected_controller:
            raise ValueError("metadata arm_controller_profile does not match the canonical contract")
        for key, expected in expected_runtime.items():
            supplied = metadata.get(key)
            if supplied is not None and supplied != expected:
                raise ValueError(f"metadata {key}={supplied!r} conflicts with {expected!r}")

        index = self.next_episode_index(seed)
        stem = success_episode_stem(seed, index)
        path = self.output_dir / f"{stem}.hdf5"
        temporary = self.output_dir / f".{stem}.tmp.hdf5"
        if path.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {stem}")

        privileged_shapes = {
            key.removeprefix("privileged."): list(value.shape[1:])
            for key, value in arrays.items()
            if key.startswith("privileged.")
        }
        document = {
            "schema_version": HDF5_SCHEMA_VERSION,
            "episode_index": index,
            "seed_episode_index": index,
            "seed": seed,
            "num_frames": len(episode),
            "images_present": images_present,
            "privileged_fields": privileged_shapes,
            **metadata,
            "wrench_profile": CANONICAL_WRENCH_PROFILE,
            "arm_control_mode": arm_control_mode,
            "arm_controller_profile": expected_controller,
            **expected_runtime,
        }

        try:
            with h5py.File(temporary, "w", libver="latest") as handle:
                handle.attrs["schema_version"] = HDF5_SCHEMA_VERSION
                handle.attrs["episode_index"] = index
                handle.attrs["seed_episode_index"] = index
                handle.attrs["seed"] = seed
                handle.attrs["num_frames"] = len(episode)
                handle.attrs["success"] = True
                handle.attrs["task"] = metadata["task"]
                handle.attrs["metadata_json"] = json.dumps(document, ensure_ascii=False)

                observations = handle.create_group("observations")
                self._dataset(observations, "state", arrays["observation.state"])
                robot_keys = (
                    "observation.joint_position",
                    "observation.joint_velocity",
                    "observation.tcp_pose",
                )
                present_robot_keys = [key in arrays for key in robot_keys]
                if any(present_robot_keys) and not all(present_robot_keys):
                    raise ValueError("joint position, velocity and TCP pose must be recorded together")
                if all(present_robot_keys):
                    self._dataset(observations, "joint_position", arrays[robot_keys[0]])
                    self._dataset(observations, "joint_velocity", arrays[robot_keys[1]])
                    self._dataset(observations, "tcp_pose", arrays[robot_keys[2]])
                if images_present:
                    images = observations.create_group("images")
                    self._dataset(images, "main", arrays["observation.image"])
                    self._dataset(images, "wrist", arrays["observation.wrist_image"])

                actions = handle.create_group("actions")
                self._dataset(actions, "canonical", arrays["action"])

                if "phase_id" in arrays:
                    phases = handle.create_group("phases")
                    self._dataset(phases, "id", arrays["phase_id"])

                forces = handle.create_group("forces")
                control_forces = forces.create_group("control")
                control_raw = self._dataset(
                    control_forces, "eef_wrench_joint_raw", arrays["eef_wrench_joint_raw"]
                )
                control_ema = self._dataset(
                    control_forces, "eef_wrench_joint_ema", arrays["eef_wrench_joint_ema"]
                )
                control_raw.attrs["source"] = "link_incoming_joint_force"
                control_raw.attrs["joint"] = "panda_hand_joint"
                control_raw.attrs["frame"] = "incoming_joint"
                control_raw.attrs["direction"] = "parent_on_child"
                control_raw.attrs["moment_reference"] = "incoming_joint_origin"
                control_raw.attrs["filter"] = "none"
                control_raw.attrs["gravity_compensated"] = False
                control_ema.attrs["source"] = "link_incoming_joint_force"
                control_ema.attrs["joint"] = "panda_hand_joint"
                control_ema.attrs["frame"] = "incoming_joint"
                control_ema.attrs["direction"] = "parent_on_child"
                control_ema.attrs["moment_reference"] = "incoming_joint_origin"
                control_ema.attrs["filter"] = "ema_alpha_0.25"
                control_ema.attrs["gravity_compensated"] = False

                physics_forces = forces.create_group("physics")
                physics_raw = self._dataset(
                    physics_forces, "eef_wrench_joint_raw", high_rate["eef_wrench_joint_raw"]
                )
                physics_raw.attrs["source"] = "link_incoming_joint_force"
                physics_raw.attrs["joint"] = "panda_hand_joint"
                physics_raw.attrs["frame"] = "incoming_joint"
                physics_raw.attrs["direction"] = "parent_on_child"
                physics_raw.attrs["moment_reference"] = "incoming_joint_origin"
                physics_raw.attrs["filter"] = "none"
                physics_raw.attrs["gravity_compensated"] = False

                timestamps = handle.create_group("timestamps")
                self._dataset(timestamps, "control", arrays["timestamp"])
                self._dataset(timestamps, "physics", high_rate["timestamp"])

                privileged_arrays = {
                    key.removeprefix("privileged."): value
                    for key, value in arrays.items()
                    if key.startswith("privileged.")
                }
                if privileged_arrays:
                    privileged = handle.create_group("privileged")
                    for field_path, value in privileged_arrays.items():
                        self._nested_dataset(privileged, field_path, value)

                terminal = np.zeros(len(episode), dtype=np.bool_)
                terminal[-1] = True
                self._dataset(handle, "is_terminal", terminal)
                self._dataset(handle, "rewards", arrays["reward"])
                self._dataset(handle, "success", arrays["success"])
                self._dataset(handle, "terminated", arrays["terminated"])
                self._dataset(handle, "truncated", arrays["truncated"])
                handle.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return (path,)


def validate_hdf5_episode(path: str | Path) -> dict[str, Any]:
    """Validate the one supported ForceBench HDF5 schema."""

    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        schema = str(handle.attrs.get("schema_version", ""))
        if schema not in HDF5_SCHEMA_VERSIONS_ACCEPTED:
            raise ValueError(
                f"unsupported HDF5 schema: {schema!r}; "
                f"expected one of {HDF5_SCHEMA_VERSIONS_ACCEPTED!r}"
            )
        frames = int(handle.attrs["num_frames"])
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        if metadata.get("wrench_profile") != CANONICAL_WRENCH_PROFILE:
            raise ValueError("canonical HDF5 has an invalid wrench profile")
        arm_control_mode = metadata.get("arm_control_mode", ARM_CONTROL_MODE)
        if arm_control_mode != ARM_CONTROL_MODE:
            raise ValueError(f"canonical HDF5 has invalid arm_control_mode: {arm_control_mode!r}")
        expected_runtime, expected_controller = _profiles_from_metadata(metadata)
        if metadata.get("arm_controller_profile") != expected_controller:
            raise ValueError("canonical HDF5 has an invalid arm controller profile")
        for key, expected in expected_runtime.items():
            if metadata.get(key) != expected:
                raise ValueError(f"canonical HDF5 has invalid {key}: {metadata.get(key)!r}")
        if metadata.get("success") is not True or not bool(handle.attrs["success"]):
            raise ValueError("formal ForceBench HDF5 episodes must be successful")
        privileged_fields = metadata.get("privileged_fields")
        if not isinstance(privileged_fields, dict) or not privileged_fields:
            raise ValueError("ForceBench episode has no declared privileged fields")

        required: dict[str, tuple[int, ...] | None] = {
            "observations/state": (frames, 7),
            "observations/joint_position": (frames, 9),
            "observations/joint_velocity": (frames, 9),
            "observations/tcp_pose": (frames, 7),
            "observations/images/main": None,
            "observations/images/wrist": None,
            "actions/canonical": (frames, 7),
            "forces/control/eef_wrench_joint_raw": (frames, 6),
            "forces/control/eef_wrench_joint_ema": (frames, 6),
            "forces/physics/eef_wrench_joint_raw": None,
            "timestamps/control": (frames,),
            "timestamps/physics": None,
            "rewards": (frames,),
            "success": (frames,),
            "terminated": (frames,),
            "truncated": (frames,),
            "is_terminal": (frames,),
        }
        for field_path, trailing_shape in privileged_fields.items():
            if not isinstance(field_path, str) or not _FIELD_PATH.fullmatch(field_path):
                raise ValueError(f"invalid privileged field declaration: {field_path!r}")
            if not isinstance(trailing_shape, list) or not all(
                isinstance(value, int) for value in trailing_shape
            ):
                raise ValueError(f"invalid privileged field shape for {field_path!r}")
            required[f"privileged/{field_path}"] = (frames, *trailing_shape)

        for key, shape in required.items():
            if key not in handle:
                raise ValueError(f"missing HDF5 dataset: {key}")
            value = handle[key]
            if shape is not None and value.shape != shape:
                raise ValueError(f"{key} has shape {value.shape}, expected {shape}")
            if value.dtype.kind in "fc" and not np.isfinite(value[...]).all():
                raise ValueError(f"{key} contains NaN or Inf")

        if set(handle["forces/control"].keys()) != {
            "eef_wrench_joint_raw",
            "eef_wrench_joint_ema",
        }:
            raise ValueError("control-rate force group contains non-canonical datasets")
        if set(handle["forces/physics"].keys()) != {"eef_wrench_joint_raw"}:
            raise ValueError("physics-rate force group contains non-canonical datasets")

        if metadata.get("seed") is None or not isinstance(metadata.get("initial_layout"), dict):
            raise ValueError("ForceBench episode is missing its seed or realized initial layout")
        if metadata.get("expert_collection_contract") is not None:
            if metadata.get("expert_collection_contract") != "forcebench_expert_collection":
                raise ValueError("canonical HDF5 has an invalid expert collection contract")
            if metadata.get("expert_policy_wrench_field") != (
                "forces/control/eef_wrench_joint_raw"
            ):
                raise ValueError("expert policy wrench must be the raw joint reaction")
            if metadata.get("expert_recorded_wrench_fields") != [
                "forces/control/eef_wrench_joint_raw",
                "forces/control/eef_wrench_joint_ema",
            ]:
                raise ValueError("expert episode must declare both raw and EMA wrench channels")
        if "teleop" in handle:
            raise ValueError("canonical HDF5 forbids device-specific teleoperation datasets")
        if metadata.get("control_source") == "gello" or any(
            str(key).startswith("gello") for key in metadata
        ):
            raise ValueError("canonical HDF5 forbids device-specific GELLO metadata")

        phase_schema = metadata.get("phase_schema")
        if phase_schema is not None:
            if not isinstance(phase_schema, dict) or not phase_schema:
                raise ValueError("phase_schema must be a non-empty mapping")
            if "phases/id" not in handle or handle["phases/id"].shape != (frames,):
                raise ValueError("phased episode is missing phases/id")
            declared = {int(value) for value in phase_schema.values()}
            actual = {int(value) for value in np.unique(handle["phases/id"][...])}
            if not actual.issubset(declared):
                raise ValueError("episode contains a phase id not declared in phase_schema")

        for key in ("observations/images/main", "observations/images/wrist"):
            image = handle[key]
            if image.shape[0] != frames or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{key} has invalid image layout: {image.shape}, {image.dtype}")
        control_time = handle["timestamps/control"][...]
        physics_time = handle["timestamps/physics"][...]
        if np.any(np.diff(control_time) <= 0) or np.any(np.diff(physics_time) <= 0):
            raise ValueError("HDF5 timestamps must be strictly increasing")
        for name, values, rate in (
            ("control", control_time, float(metadata["control_rate_hz"])),
            ("physics", physics_time, float(metadata["physics_rate_hz"])),
        ):
            if len(values) > 1 and not np.allclose(
                np.diff(values), 1.0 / rate, rtol=0.0, atol=1.0e-9
            ):
                raise ValueError(
                    f"{name} timestamps do not match declared {rate:g} Hz rate"
                )
        physics_count = len(physics_time)
        if physics_count < frames:
            raise ValueError("physics-rate force stream is shorter than control stream")
        if handle["forces/physics/eef_wrench_joint_raw"].shape != (physics_count, 6):
            raise ValueError("physics joint-frame wrench has invalid shape")

        return {
            "path": str(path),
            "schema_version": schema,
            "episode_index": int(handle.attrs["episode_index"]),
            "task": str(handle.attrs["task"]),
            "success": bool(handle.attrs["success"]),
            "control_frames": frames,
            "physics_frames": physics_count,
            "image_shape": list(handle["observations/images/main"].shape[1:]),
            "seed": metadata.get("seed"),
        }
