import dataclasses
from typing import ClassVar
import einops
import numpy as np
from openpi import transforms  
from openpi.models import model as _model

def make_forcevla_example() -> dict:
    """Creates a random input example compatible with Flexiv config."""
    return {
        "state": np.ones((7,)),
        "eef_wrench": np.ones((6,)),
        "image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8), 
        "wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8), 
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Forcevla_inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int
    real_action_dim: int = 7
    proprio_dim: int = 0  # 0=与action同维(兼容); >0时state维度独立
    force_dim: int = 6
    has_wrist_image_2: bool = False

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        proprio = np.asarray(data["state"], dtype=np.float32)
        wrench = np.asarray(data["eef_wrench"], dtype=np.float32)
        _pd = self.proprio_dim if self.proprio_dim else self.real_action_dim
        if proprio.shape[-1] != _pd:
            raise ValueError(f"Expected {_pd}D state, got {proprio.shape}")
        if wrench.shape[-1] != self.force_dim:
            raise ValueError(f"Expected {self.force_dim}D wrench, got {wrench.shape}")
        state = transforms.pad_to_dim(np.concatenate([proprio, wrench], axis=-1), self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["image"])
        left_wrist_image = _parse_image(data["wrist_image"])
        right_wrist_raw = data.get("wrist_image_2")
        right_wrist_image = (
            _parse_image(right_wrist_raw)
            if self.has_wrist_image_2 and right_wrist_raw is not None
            else np.zeros_like(base_image)
        )

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": (
                    np.True_ if self.has_wrist_image_2 else (np.False_ if mask_padding else np.True_)
                ),
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
    
@dataclasses.dataclass(frozen=True)
class Forcevla_outputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    real_action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For forcevla, we only return the first 7 actions (since the rest is padding), xyz  + RPY + gripper
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][..., : self.real_action_dim])}


@dataclasses.dataclass(frozen=True)
class WrapAngularActionDeltas(transforms.DataTransformFn):
    """Wrap RPY delta actions to their shortest equivalent."""

    indices: tuple[int, ...] = (3, 4, 5)

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"]).copy()
        actions[..., self.indices] = (
            actions[..., self.indices] + np.pi
        ) % (2.0 * np.pi) - np.pi
        return {**data, "actions": actions}
