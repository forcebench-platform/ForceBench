import logging
import os

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class ForceSpec(nnx.Module):
    """Compact spectro-temporal encoder that emits four action-expert tokens."""

    def __init__(self, config: pi0_config.Pi0Config, action_width: int, rngs: nnx.Rngs):
        temporal_width = config.force_temporal_channels
        spectral_width = config.force_spectral_channels

        self.force_dim = config.force_dim
        self.encoder_mode = config.force_encoder_mode
        self.force_num_tokens = config.force_num_tokens
        self.stft_window = config.force_stft_window
        self.stft_hop = config.force_stft_hop
        if self.encoder_mode == "mlp":
            if config.force_num_tokens <= 0:
                raise ValueError("The MLP force encoder requires at least one token")
            self.force_mlp = nnx.Linear(
                config.force_dim,
                config.force_num_tokens * temporal_width,
                rngs=rngs,
            )
            self.token_proj = nnx.Linear(temporal_width, action_width, rngs=rngs)
            self.token_norm = nnx.LayerNorm(action_width, rngs=rngs)
            return

        if config.force_num_tokens != 4:
            raise ValueError("The temporal ForceSpec encoders require force_num_tokens=4")
        if temporal_width != spectral_width:
            raise ValueError("Temporal and spectral widths must match before token projection")
        self.temporal_conv_1 = nnx.Conv(
            config.force_dim, temporal_width, kernel_size=3, padding="SAME", kernel_dilation=1, rngs=rngs
        )
        self.temporal_conv_2 = nnx.Conv(
            temporal_width, temporal_width, kernel_size=3, padding="SAME", kernel_dilation=2, rngs=rngs
        )
        self.temporal_conv_3 = nnx.Conv(
            temporal_width, temporal_width, kernel_size=3, padding="SAME", kernel_dilation=4, rngs=rngs
        )
        if self.encoder_mode == "temporal_rfft":
            self.spectral_proj = nnx.Linear(config.force_dim, spectral_width, rngs=rngs)
        elif self.encoder_mode == "temporal_local_stft":
            self.local_stft_proj = nnx.Linear(
                (self.stft_window // 2 + 1) * config.force_dim,
                spectral_width,
                rngs=rngs,
            )
        self.token_proj = nnx.Linear(temporal_width, action_width, rngs=rngs)
        self.token_norm = nnx.LayerNorm(action_width, rngs=rngs)

    @staticmethod
    def _pool_two(x, mask=None):
        """Pool the first and second halves into two tokens."""
        split = (x.shape[1] + 1) // 2
        chunks = (x[:, :split], x[:, split:])
        if mask is None:
            return jnp.stack([jnp.mean(chunk, axis=1) for chunk in chunks], axis=1)

        mask_chunks = (mask[:, :split], mask[:, split:])
        pooled = []
        for chunk, chunk_mask in zip(chunks, mask_chunks, strict=True):
            weights = chunk_mask[..., None].astype(chunk.dtype)
            pooled.append(jnp.sum(chunk * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0))
        return jnp.stack(pooled, axis=1)

    @staticmethod
    def _pool_four(x, mask):
        """Pool chronological quarters into four explicitly ordered tokens."""
        boundaries = [round(i * x.shape[1] / 4) for i in range(5)]
        pooled = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            chunk = x[:, start:end]
            weights = mask[:, start:end, None].astype(chunk.dtype)
            pooled.append(jnp.sum(chunk * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0))
        return jnp.stack(pooled, axis=1)

    def _local_stft_tokens(self, x, history_mask):
        """Encode local spectra while retaining early/late event location."""
        starts = jnp.arange(0, x.shape[1] - self.stft_window + 1, self.stft_hop)
        offsets = jnp.arange(self.stft_window)
        frame_indices = starts[:, None] + offsets[None, :]
        frames = x[:, frame_indices, :]
        frame_mask = history_mask[:, frame_indices]
        window = jnp.hanning(self.stft_window).astype(frames.dtype)
        frames = frames * window[None, None, :, None]
        spectrum = jnp.log1p(jnp.abs(jnp.fft.rfft(frames, axis=2)))
        spectrum = spectrum.reshape((*spectrum.shape[:2], -1))
        local = jax.nn.gelu(self.local_stft_proj(spectrum))
        valid_frames = jnp.any(frame_mask, axis=2)
        return self._pool_two(local, valid_frames)

    def __call__(self, history, history_mask):
        if history.shape[-1] != self.force_dim:
            raise ValueError(f"Expected force_dim={self.force_dim}, got {history.shape[-1]}")
        if history_mask is None:
            history_mask = jnp.ones(history.shape[:2], dtype=jnp.bool_)

        # Mask before both branches so clamped episode-boundary samples do not
        # masquerade as a long constant force signal.
        x = history.astype(jnp.float32)
        x = jnp.where(history_mask[..., None], x, 0.0)

        if self.encoder_mode == "mlp":
            # The minimal ablation uses only the current normalized 6D wrench.
            # The MLP directly emits six hidden vectors, which become six force
            # tokens after the shared projection. Keeping the length-one history
            # axis reuses the existing training/inference observation schema.
            hidden = jax.nn.gelu(self.force_mlp(x[:, -1]))
            hidden = hidden.reshape((hidden.shape[0], self.force_num_tokens, -1))
            return self.token_norm(self.token_proj(hidden))

        temporal = jax.nn.gelu(self.temporal_conv_1(x))
        temporal = jax.nn.gelu(self.temporal_conv_2(temporal))
        temporal = jax.nn.gelu(self.temporal_conv_3(temporal))
        if self.encoder_mode == "temporal_only":
            tokens = self._pool_four(temporal, history_mask)
        else:
            temporal_tokens = self._pool_two(temporal, history_mask)
            if self.encoder_mode == "temporal_local_stft":
                spectral_tokens = self._local_stft_tokens(x, history_mask)
            else:
                spectrum = jnp.log1p(jnp.abs(jnp.fft.rfft(x, axis=1)))
                spectrum = jax.nn.gelu(self.spectral_proj(spectrum))
                spectral_tokens = self._pool_two(spectrum)
            tokens = jnp.concatenate([temporal_tokens, spectral_tokens], axis=1)
        return self.token_norm(self.token_proj(tokens))


class FutureForceHead(nnx.Module):
    """Training-only future wrench predictor conditioned on force context and clean-action estimate."""

    def __init__(self, config: pi0_config.Pi0Config, action_width: int, rngs: nnx.Rngs):
        hidden = config.future_force_hidden
        self.horizon = config.future_force_horizon
        self.force_dim = config.force_dim
        self.input_mode = config.future_force_input_mode
        context_dim = 3 * config.force_dim if self.input_mode == "raw_summary" else action_width
        self.context_proj = nnx.Linear(context_dim, hidden, rngs=rngs)
        self.action_proj = nnx.Linear(config.action_dim, hidden, rngs=rngs)
        self.mlp_in = nnx.Linear(2 * hidden, 2 * hidden, rngs=rngs)
        self.mlp_out = nnx.Linear(2 * hidden, self.horizon * self.force_dim, rngs=rngs)

    def __call__(self, contextualized_force_tokens, clean_action_estimate):
        context = jnp.mean(contextualized_force_tokens, axis=1)
        context = jax.nn.gelu(self.context_proj(context))
        action = jax.nn.gelu(self.action_proj(clean_action_estimate))
        action = jnp.mean(action, axis=1)
        hidden = jnp.concatenate([context, action], axis=-1)
        hidden = jax.nn.gelu(self.mlp_in(hidden))
        return self.mlp_out(hidden).reshape((-1, self.horizon, self.force_dim))


class GatedForceCrossAttention(nnx.Module):
    """Late force fusion with an explicitly configured residual gate."""

    def __init__(self, width: int, num_heads: int, rngs: nnx.Rngs, gate_initial: float = 0.0):
        if width % num_heads != 0:
            raise ValueError(f"Action width {width} must be divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.query = nnx.Linear(width, width, rngs=rngs)
        self.key = nnx.Linear(width, width, rngs=rngs)
        self.value = nnx.Linear(width, width, rngs=rngs)
        self.output = nnx.Linear(width, width, rngs=rngs)
        # Explicit effective initial gate; ForceBench v2 uses 0.1 so the
        # task loss reaches force encoding and Q/K/V on the first update.
        self.gate = nnx.Param(jnp.full((width,), jnp.arctanh(gate_initial), dtype=jnp.float32))

    def __call__(self, action_hidden, force_tokens):
        query = einops.rearrange(
            self.query(action_hidden), "b t (h d) -> b h t d", h=self.num_heads
        )
        key = einops.rearrange(
            self.key(force_tokens), "b f (h d) -> b h f d", h=self.num_heads
        )
        value = einops.rearrange(
            self.value(force_tokens), "b f (h d) -> b h f d", h=self.num_heads
        )
        logits = jnp.einsum("bhtd,bhfd->bhtf", query, key) / jnp.sqrt(self.head_dim)
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(query.dtype)
        context = jnp.einsum("bhtf,bhfd->bhtd", weights, value)
        context = einops.rearrange(context, "b h t d -> b t (h d)")
        residual = self.output(context)
        gate = jnp.tanh(self.gate.value).astype(residual.dtype)
        return action_hidden + residual * gate[None, None, :]


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.force_enabled = config.force_enabled
        self.force_fusion_mode = config.force_fusion_mode
        self.future_force_enabled = config.future_force_enabled
        self.future_force_loss_weight = config.future_force_loss_weight
        self.future_force_stop_action_gradient = config.future_force_stop_action_gradient
        self.force_num_tokens = config.force_num_tokens
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if self.force_enabled and config.force_fusion_mode != "none":
            self.force_spec = ForceSpec(config, action_expert_config.width, rngs)
            if self.force_fusion_mode == "gated_cross_attention":
                self.force_cross_attn = GatedForceCrossAttention(
                    action_expert_config.width, config.force_cross_attention_heads, rngs, config.force_cross_attention_gate_init
                )
        if self.future_force_enabled:
            self.ff_head = FutureForceHead(config, action_expert_config.width, rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        if self.force_enabled and self.force_fusion_mode == "prefix_tokens":
            if obs.force_history is None:
                raise ValueError("force_enabled=True requires observation.force_history")
            force_tokens = self.force_spec(obs.force_history, obs.force_history_mask)
            tokens.append(force_tokens)
            input_mask.append(jnp.ones(force_tokens.shape[:2], dtype=jnp.bool_))
            # Force tokens form their own suffix block. They may read the VLM
            # prefix and each other, while the later action block may read them.
            ar_mask += [True] + ([False] * (force_tokens.shape[1] - 1))

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def encode_force(self, obs: _model.Observation):
        if not self.force_enabled or self.force_fusion_mode == "none":
            return None
        if obs.force_history is None:
            raise ValueError("force_enabled=True requires observation.force_history")
        return self.force_spec(obs.force_history, obs.force_history_mask)

    def summarize_raw_force(self, obs: _model.Observation):
        """FF-head-only context; this path is never read by the action expert."""
        if obs.force_history is None:
            raise ValueError("raw-summary FF-head requires observation.force_history")
        history = obs.force_history.astype(jnp.float32)
        mask = (
            jnp.ones(history.shape[:2], dtype=jnp.bool_)
            if obs.force_history_mask is None
            else obs.force_history_mask
        )
        weights = mask[..., None].astype(history.dtype)
        mean = jnp.sum(history * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        first_index = jnp.argmax(mask.astype(jnp.int32), axis=1)
        first = jnp.take_along_axis(history, first_index[:, None, None], axis=1)[:, 0]
        last = history[:, -1]
        return jnp.concatenate([mean, last, last - first], axis=-1)[:, None, :]

    def _compute_loss_and_info(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        force_tokens = (
            self.encode_force(observation)
            if self.force_fusion_mode == "gated_cross_attention"
            else None
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        action_hidden = suffix_out[:, -self.action_horizon :]
        fusion_info = {}
        if force_tokens is not None:
            fused = self.force_cross_attn(action_hidden, force_tokens)
            fusion_info["force_residual_ratio"] = jnp.sqrt(jnp.mean((fused - action_hidden) ** 2)) / (jnp.sqrt(jnp.mean(action_hidden ** 2)) + 1e-12)
            action_hidden = fused
        v_t = self.action_out_proj(action_hidden)
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        if not self.future_force_enabled:
            return flow_loss, {"flow_loss": jnp.mean(flow_loss), **fusion_info}
        if observation.future_wrench is None or observation.future_wrench_mask is None:
            raise ValueError("future_force_enabled requires future_wrench and future_wrench_mask during training")

        if self.ff_head.input_mode == "raw_summary":
            contextualized_force = self.summarize_raw_force(observation)
        elif force_tokens is None:
            force_end = suffix_out.shape[1] - self.action_horizon
            force_start = force_end - self.force_num_tokens
            contextualized_force = suffix_out[:, force_start:force_end]
        else:
            contextualized_force = force_tokens
        clean_action_estimate = x_t - time_expanded * v_t
        if self.future_force_stop_action_gradient:
            clean_action_estimate = jax.lax.stop_gradient(clean_action_estimate)
        predicted_future_wrench = self.ff_head(contextualized_force, clean_action_estimate)
        target_future_wrench = observation.future_wrench.astype(jnp.float32)
        future_mask = observation.future_wrench_mask.astype(jnp.float32)
        future_error = jnp.mean(jnp.square(predicted_future_wrench - target_future_wrench), axis=-1)
        future_force_loss = jnp.sum(future_error * future_mask, axis=-1) / jnp.maximum(
            jnp.sum(future_mask, axis=-1), 1.0
        )
        total_loss = flow_loss + self.future_force_loss_weight * future_force_loss[:, None]
        return total_loss, {
            "flow_loss": jnp.mean(flow_loss),
            "future_force_loss": jnp.mean(future_force_loss),
            "weighted_future_force_loss": self.future_force_loss_weight * jnp.mean(future_force_loss),
        }

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self._compute_loss_and_info(rng, observation, actions, train=train)
        return loss

    def compute_loss_with_info(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        return self._compute_loss_and_info(rng, observation, actions, train=train)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        force_tokens = (
            self.encode_force(observation)
            if self.force_fusion_mode == "gated_cross_attention"
            else None
        )


        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            action_hidden = suffix_out[:, -self.action_horizon :]
            if force_tokens is not None:
                action_hidden = self.force_cross_attn(action_hidden, force_tokens)
            v_t = self.action_out_proj(action_hidden)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
