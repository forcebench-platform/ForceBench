"""ForceBench's two RGB views, Cartesian state, and optional current wrench.

This is a dataset transform for the upstream Policy, not a custom policy engine.
The camera slots follow upstream PI05 DroidInputs; the state/actions remain
ForceBench Cartesian commands (they must not be interpreted as DROID joints).
"""
import dataclasses
import numpy as np
from openpi import transforms
from openpi.policies.droid_policy import _parse_image


@dataclasses.dataclass(frozen=True)
class ForceBenchInputs(transforms.DataTransformFn):
    method: str = "F0"
    action_dim: int = 32
    training: bool = False
    future_horizon: int = 10

    def __call__(self, data):
        if self.method not in {"F0", "F1", "F2", "F3", "F4"}:
            raise ValueError(f"Unknown ForceBench method: {self.method}")
        if not self.training and any(k in data for k in ("future_wrench", "wrench_window")):
            raise ValueError("Future wrench labels are forbidden in policy observations")
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite Cartesian state [7], got {state.shape}")
        if data.get("observation/wrist_image_2") is not None:
            raise ValueError("This ForceBench version has exactly two real cameras")
        base = _parse_image(data["observation/image"])
        wrist = _parse_image(data["observation/wrist_image"])
        for name, value in (("base", base), ("wrist", wrist)):
            if value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"Expected {name} RGB [H,W,3], got {value.shape}")
        result = {
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        if self.method != "F0":
            wrench = np.asarray(data["eef_wrench"], dtype=np.float32)
            expected = 1 + self.future_horizon if self.training and self.method == "F4" else 1
            if wrench.shape == (6,):
                wrench = wrench[None, :]
            if wrench.shape != (expected, 6) or not np.isfinite(wrench).all():
                raise ValueError(f"Expected wrench [{expected},6], got {wrench.shape}")
            padding = np.asarray(data.get("eef_wrench_is_pad", np.zeros(expected, bool)), dtype=bool)
            if padding.shape != (expected,) or padding[0]:
                raise ValueError("Current wrench must be valid; padding must match temporal query")
            if self.method == "F2":
                state = np.concatenate((state, wrench[0]))
            else:
                result.update(force_history=wrench[:1], force_history_mask=~padding[:1])
                if self.training and self.method == "F4":
                    result.update(future_wrench=wrench[1:], future_wrench_mask=~padding[1:])
        result["state"] = transforms.pad_to_dim(state, self.action_dim)
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
                raise ValueError("Expected finite Cartesian actions [T,7]")
            result["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode() if isinstance(prompt, bytes) else prompt
        return result


@dataclasses.dataclass(frozen=True)
class ForceBenchOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        actions = np.asarray(data["actions"])[..., :7]
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite policy actions [50,7], got {actions.shape}")
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class WrapAngles(transforms.DataTransformFn):
    def __call__(self, data):
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"]).copy()
        actions[..., 3:6] = (actions[..., 3:6] + np.pi) % (2 * np.pi) - np.pi
        return {**data, "actions": actions}


@dataclasses.dataclass(frozen=True)
class ClipWrench(transforms.DataTransformFn):
    def __call__(self, data):
        return {k: np.clip(v, -5.0, 5.0) if k in ("force_history", "future_wrench") else v for k, v in data.items()}
