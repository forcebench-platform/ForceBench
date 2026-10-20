import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AutoBioInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    force_enabled: bool = False
    force_dim: int = 12
    force_history_len: int = 25
    force_mode: str = "real"
    shuffle_seed: int = 42
    force_mean: tuple[float, ...] | None = None
    force_std: tuple[float, ...] | None = None
    # FORTE R5-lite: collapse the force history window to per-channel mean/std and
    # concatenate [wrench_now, mean, std] into the state instead of emitting a
    # separate force_history input. Model architecture stays untouched.
    force_stats_in_state: bool = False
    # FSD variant B: emit BOTH the stats-augmented state and the raw force_history window.
    force_history_also: bool = False

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        wrist_image_2_raw = data.get("observation/wrist_image_2")
        wrist_image_2_mask = wrist_image_2_raw is not None
        wrist_image_2 = _parse_image(wrist_image_2_raw) if wrist_image_2_mask else np.zeros_like(base_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": wrist_image_2,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": wrist_image_2_mask if mask_padding else np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if self.force_enabled:
            force_history = data.get("force_history", data.get("eef_wrench"))
            if force_history is None:
                raise ValueError("ForceBio requires force_history or eef_wrench in every observation")
            force_history = np.asarray(force_history, dtype=np.float32)
            if force_history.ndim == 1:
                force_history = np.repeat(force_history[None, :], self.force_history_len, axis=0)
            if self.force_stats_in_state:
                # Fail loud on a single-frame fallback: repeating the current wrench
                # would freeze window std at 0 and silently degrade to chance.
                if force_history.shape[0] != self.force_history_len or force_history.ndim != 2:
                    raise ValueError(
                        f"force_stats_in_state requires a real {self.force_history_len}-frame "
                        f"history, got {force_history.shape}"
                    )
            elif force_history.shape != (self.force_history_len, self.force_dim):
                raise ValueError(
                    f"Expected force history {(self.force_history_len, self.force_dim)}, got {force_history.shape}"
                )
            if (self.force_mean is None) != (self.force_std is None):
                raise ValueError("force_mean and force_std must be provided together")
            if self.force_mean is not None:
                mean = np.asarray(self.force_mean, dtype=np.float32)
                std = np.asarray(self.force_std, dtype=np.float32)
                if mean.shape != (self.force_dim,) or std.shape != (self.force_dim,):
                    raise ValueError(f"Expected force normalization stats [{self.force_dim}]")
                force_history = (force_history - mean) / (std + 1e-6)
            is_pad = np.asarray(
                data.get("force_history_is_pad", np.zeros(self.force_history_len, dtype=np.bool_)), dtype=np.bool_
            )
            if self.force_mode == "zero":
                force_history = np.zeros_like(force_history)
            elif self.force_mode == "shuffle":
                if "episode_index" not in data or "frame_index" not in data:
                    raise ValueError("shuffle force mode requires episode_index and frame_index")
                episode_index = int(np.asarray(data["episode_index"]).reshape(-1)[0])
                frame_index = int(np.asarray(data["frame_index"]).reshape(-1)[0])
                sample_seed = (
                    self.shuffle_seed + episode_index * 1_000_003 + frame_index * 9_176
                ) % (2**32)
                permutation = np.random.default_rng(sample_seed).permutation(self.force_history_len)
                force_history = force_history[permutation]
                is_pad = is_pad[permutation]
            elif self.force_mode != "real":
                raise ValueError(f"Unknown force_mode: {self.force_mode}")
            if self.force_stats_in_state:
                raw_state = np.asarray(data["observation/state"], dtype=np.float32).reshape(-1)
                w = np.asarray(force_history, dtype=np.float32)
                inputs["state"] = transforms.pad_to_dim(
                    np.concatenate([raw_state, w[-1], w.mean(axis=0), w.std(axis=0)]),
                    self.action_dim,
                )
            if (not self.force_stats_in_state) or self.force_history_also:
                inputs["force_history"] = force_history
                inputs["force_history_mask"] = np.logical_not(is_pad)
            if "future_wrench" in data:
                future_wrench = np.asarray(data["future_wrench"], dtype=np.float32)
                if future_wrench.ndim != 2 or future_wrench.shape[1] != self.force_dim:
                    raise ValueError(f"Expected future_wrench [Hf, {self.force_dim}], got {future_wrench.shape}")
                if self.force_mean is not None:
                    future_wrench = (future_wrench - mean) / (std + 1e-6)
                future_is_pad = np.asarray(
                    data.get("future_wrench_is_pad", np.zeros(future_wrench.shape[0], dtype=np.bool_)),
                    dtype=np.bool_,
                )
                inputs["future_wrench"] = future_wrench
                inputs["future_wrench_mask"] = np.logical_not(future_is_pad)
        return inputs


@dataclasses.dataclass(frozen=True)
class AutoBioOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class WrapAngularActionDeltas(transforms.DataTransformFn):
    """Wrap selected delta-action angles to their shortest equivalent."""

    indices: tuple[int, ...] = (3, 4, 5)

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"]).copy()
        actions[..., self.indices] = (
            actions[..., self.indices] + np.pi
        ) % (2.0 * np.pi) - np.pi
        return {**data, "actions": actions}


@dataclasses.dataclass(frozen=True)
class ClipForceHistory(transforms.DataTransformFn):
    """Clip already-normalized force values to limit contact outliers."""

    limit: float = 5.0

    def __call__(self, data: dict) -> dict:
        if "force_history" not in data:
            return data
        result = {**data, "force_history": np.clip(data["force_history"], -self.limit, self.limit)}
        if "future_wrench" in result:
            result["future_wrench"] = np.clip(result["future_wrench"], -self.limit, self.limit)
        return result
