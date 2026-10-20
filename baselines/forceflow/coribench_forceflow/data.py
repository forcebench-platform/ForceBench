from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from .video_decode import decode_video_frames
from torch.utils.data import Dataset


@dataclass
class Episode:
    index: int
    state: np.ndarray
    action_pose: np.ndarray
    gripper: np.ndarray
    force: np.ndarray
    wrist_video: Path
    main_video: Path
    fps: int

    @property
    def length(self) -> int:
        return len(self.state)


class MinMax:
    def __init__(self, minimum: np.ndarray, maximum: np.ndarray):
        self.minimum = np.asarray(minimum, dtype=np.float32)
        self.maximum = np.asarray(maximum, dtype=np.float32)
        self.scale = np.maximum(self.maximum - self.minimum, 1e-6)

    @classmethod
    def from_values(cls, values: list[np.ndarray]) -> "MinMax":
        return cls(
            np.minimum.reduce([value.min(axis=0) for value in values]),
            np.maximum.reduce([value.max(axis=0) for value in values]),
        )

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (2.0 * (value - self.minimum) / self.scale - 1.0).astype(np.float32)

    def as_dict(self) -> dict[str, list[float]]:
        return {"min": self.minimum.tolist(), "max": self.maximum.tolist()}


def shortest_delta(action_pose: np.ndarray, state_pose: np.ndarray) -> np.ndarray:
    delta = np.asarray(action_pose - state_pose, dtype=np.float32)
    delta[..., 3:6] = (delta[..., 3:6] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def action_chunk_deltas(action_pose: np.ndarray, current_state_pose: np.ndarray) -> np.ndarray:
    """Encode absolute targets as one state-relative step followed by target increments."""
    action_pose = np.asarray(action_pose, dtype=np.float32)
    current_state_pose = np.asarray(current_state_pose, dtype=np.float32)
    if action_pose.ndim != 2 or action_pose.shape[1] != 6 or current_state_pose.shape != (6,):
        raise ValueError(
            f"expected action_pose [T,6] and current_state_pose [6], got "
            f"{action_pose.shape} and {current_state_pose.shape}"
        )
    references = np.concatenate([current_state_pose[None], action_pose[:-1]], axis=0)
    return shortest_delta(action_pose, references)


class CoRiBenchLeRobotDataset(Dataset):
    """Read the canonical CoRiBench LeRobot v2.1 dataset without a Zarr copy."""

    def __init__(
        self,
        root: str | Path,
        observation_horizon: int = 2,
        action_horizon: int = 64,
        force_history: int = 10,
        control_frequency_hz: float = 15.0,
        image_size: int = 224,
        episode_limit: int | None = None,
    ):
        self.root = Path(root).resolve()
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(info["fps"])
        self.control_frequency_hz = float(control_frequency_hz)
        if not np.isclose(self.fps, self.control_frequency_hz):
            raise ValueError(
                f"dataset fps {self.fps:g} does not match configured "
                f"control_frequency_hz {self.control_frequency_hz:g}"
            )
        self.observation_horizon = int(observation_horizon)
        self.action_horizon = int(action_horizon)
        self.force_history = int(force_history)
        self.image_size = int(image_size)
        count = int(info["total_episodes"])
        if episode_limit is not None:
            count = min(count, int(episode_limit))
        self.episodes = [self._load_episode(index, info) for index in range(count)]
        self.ends = np.cumsum([episode.length for episode in self.episodes]).tolist()
        self.pos_stats = MinMax.from_values([episode.state[:, :6] for episode in self.episodes])
        action_values = []
        for episode in self.episodes:
            # Every frame can be the first row of a training chunk, while later
            # rows are increments between consecutive absolute targets.
            action_values.append(shortest_delta(episode.action_pose, episode.state[:, :6]))
            if episode.length > 1:
                action_values.append(shortest_delta(episode.action_pose[1:], episode.action_pose[:-1]))
        self.action_stats = MinMax.from_values(action_values)
        self.force_stats = MinMax.from_values([episode.force for episode in self.episodes])

    def _load_episode(self, index: int, info: dict[str, Any]) -> Episode:
        chunk = index // int(info["chunks_size"])
        values = {"episode_chunk": chunk, "episode_index": index}
        data_path = self.root / str(info["data_path"]).format(**values)
        table = pq.read_table(data_path, columns=["state", "actions", "eef_wrench"])
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        force = np.asarray(table["eef_wrench"].to_pylist(), dtype=np.float32)
        if state.shape[1:] != (7,) or action.shape != state.shape or force.shape != (len(state), 6):
            raise ValueError(f"invalid episode {index} shapes: {state.shape}, {action.shape}, {force.shape}")
        video_pattern = str(info["video_path"])
        main = self.root / video_pattern.format(video_key="image", **values)
        wrist = self.root / video_pattern.format(video_key="wrist_image", **values)
        if not main.is_file() or not wrist.is_file() or not np.isfinite(state).all():
            raise ValueError(f"invalid episode assets: {index}")
        return Episode(
            index=index,
            state=state,
            action_pose=action[:, :6],
            gripper=action[:, 6:7],
            force=force,
            wrist_video=wrist,
            main_video=main,
            fps=int(round(self.fps)),
        )

    def __len__(self) -> int:
        return int(self.ends[-1])

    def _locate(self, index: int) -> tuple[Episode, int]:
        if index < 0:
            index += len(self)
        episode_index = bisect.bisect_right(self.ends, index)
        start = 0 if episode_index == 0 else self.ends[episode_index - 1]
        return self.episodes[episode_index], index - start

    @staticmethod
    def _history_indices(frame: int, count: int, length: int) -> np.ndarray:
        return np.clip(np.arange(frame - count + 1, frame + 1), 0, length - 1)

    def _images(self, episode: Episode, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        timestamps = (indices.astype(np.float64) / episode.fps).tolist()
        decoded = []
        for video in (episode.wrist_video, episode.main_video):
            frames = decode_video_frames(
                video, timestamps, tolerance_s=0.5 / episode.fps + 1e-4, backend="pyav"
            )
            frames = F.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            decoded.append((frames - 0.5) / 0.5)
        return decoded[0], decoded[1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode, frame = self._locate(index)
        obs_indices = self._history_indices(frame, self.observation_horizon, episode.length)
        force_indices = self._history_indices(frame, self.force_history, episode.length)
        action_indices = np.clip(
            np.arange(frame, frame + self.action_horizon), 0, episode.length - 1
        )
        next_force_indices = np.clip(action_indices + 1, 0, episode.length - 1)
        arm, fixed = self._images(episode, obs_indices)
        state = np.concatenate(
            [
                self.pos_stats.normalize(episode.state[obs_indices, :6]),
                episode.state[obs_indices, 6:7],
            ],
            axis=-1,
        )
        action_delta = action_chunk_deltas(
            episode.action_pose[action_indices], episode.state[frame, :6]
        )
        action = np.concatenate(
            [
                self.action_stats.normalize(action_delta),
                episode.gripper[action_indices],
                self.force_stats.normalize(episode.force[next_force_indices]),
            ],
            axis=-1,
        )
        return {
            "x0": torch.from_numpy(action),
            "condition_cfg": {
                "image_arm": arm,
                "image_fix": fixed,
                "lowdim": torch.from_numpy(state.astype(np.float32)),
                "force": torch.from_numpy(self.force_stats.normalize(episode.force[force_indices])),
            },
        }

    def normalizer_dict(self) -> dict[str, Any]:
        return {
            "pos": self.pos_stats.as_dict(),
            "action": self.action_stats.as_dict(),
            "force": self.force_stats.as_dict(),
            "source": str(self.root),
            "episodes": len(self.episodes),
            "frames": len(self),
            "fps": self.fps,
            "control_frequency_hz": self.control_frequency_hz,
        }
