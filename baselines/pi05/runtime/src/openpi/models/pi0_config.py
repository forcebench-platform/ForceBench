import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # ForceBio late-fusion options. These are disabled for all stock pi0/pi0.5
    # configs, so existing checkpoints and policies keep their original shape.
    force_enabled: bool = False
    force_dim: int = 12
    force_history_len: int = 25
    force_num_tokens: int = 4
    force_temporal_channels: int = 128
    force_spectral_channels: int = 128
    force_encoder_mode: str = "temporal_rfft"
    force_stft_window: int = 32
    force_stft_hop: int = 16
    force_fusion_mode: str = "prefix_tokens"
    force_cross_attention_heads: int = 8
    force_cross_attention_gate_init: float = 0.0
    future_force_enabled: bool = False
    future_force_horizon: int = 10
    future_force_hidden: int = 256
    future_force_loss_weight: float = 0.1
    future_force_stop_action_gradient: bool = True
    future_force_input_mode: str = "force_tokens"

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.future_force_enabled and not self.force_enabled:
            raise ValueError("future_force_enabled requires force_enabled")
        if self.force_enabled and self.force_history_len <= 0:
            raise ValueError("force_history_len must be positive")
        if self.force_encoder_mode not in {"mlp", "temporal_rfft", "temporal_only", "temporal_local_stft"}:
            raise ValueError(f"Unsupported force_encoder_mode={self.force_encoder_mode!r}")
        if self.force_enabled and self.force_encoder_mode == "mlp" and self.force_num_tokens <= 0:
            raise ValueError("force_encoder_mode='mlp' requires force_num_tokens>0")
        if self.force_enabled and self.force_encoder_mode != "mlp" and self.force_num_tokens != 4:
            raise ValueError("temporal force encoders require force_num_tokens=4")
        if self.force_enabled and self.force_encoder_mode == "temporal_local_stft":
            if self.force_stft_window <= 1 or self.force_stft_window > self.force_history_len:
                raise ValueError("force_stft_window must be in [2, force_history_len]")
            if self.force_stft_hop <= 0:
                raise ValueError("force_stft_hop must be positive")
        if self.force_fusion_mode not in {"prefix_tokens", "gated_cross_attention", "none"}:
            raise ValueError(f"Unsupported force_fusion_mode={self.force_fusion_mode!r}")
        if not 0 <= self.force_cross_attention_gate_init < 1:
            raise ValueError("Effective force gate initial must be in [0,1)")
        if self.force_cross_attention_heads <= 0:
            raise ValueError("force_cross_attention_heads must be positive")
        if self.future_force_enabled and self.future_force_horizon <= 0:
            raise ValueError("future_force_horizon must be positive")
        if self.future_force_input_mode not in {"force_tokens", "raw_summary"}:
            raise ValueError(f"Unsupported future_force_input_mode={self.future_force_input_mode!r}")
        if self.future_force_enabled and self.force_fusion_mode == "none":
            if self.future_force_input_mode != "raw_summary":
                raise ValueError("force_fusion_mode='none' requires future_force_input_mode='raw_summary'")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                force_history=(
                    jax.ShapeDtypeStruct([batch_size, self.force_history_len, self.force_dim], jnp.float32)
                    if self.force_enabled
                    else None
                ),
                force_history_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.force_history_len], jnp.bool_)
                    if self.force_enabled
                    else None
                ),
                future_wrench=(
                    jax.ShapeDtypeStruct([batch_size, self.future_force_horizon, self.force_dim], jnp.float32)
                    if self.future_force_enabled
                    else None
                ),
                future_wrench_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.future_force_horizon], jnp.bool_)
                    if self.future_force_enabled
                    else None
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
