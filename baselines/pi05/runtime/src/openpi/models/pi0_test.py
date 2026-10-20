import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_forcespec_shape_and_padding_mask():
    config = _pi0_config.Pi0Config(
        force_enabled=True,
        force_dim=12,
        force_history_len=25,
        force_temporal_channels=16,
        force_spectral_channels=16,
    )
    encoder = _pi0.ForceSpec(config, action_width=32, rngs=nnx.Rngs(0))
    history = jax.random.normal(jax.random.key(1), (2, 25, 12))
    mask = jnp.ones((2, 25), dtype=jnp.bool_).at[:, :5].set(False)

    expected = encoder(history, mask)
    changed_only_in_padding = history.at[:, :5].set(1e6)
    actual = encoder(changed_only_in_padding, mask)

    assert expected.shape == (2, 4, 32)
    assert np.all(np.isfinite(np.asarray(expected)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_force_and_action_are_separate_attention_blocks():
    # Four force tokens followed by three action tokens. Force queries cannot
    # read future action keys, while action queries can read all force keys.
    input_mask = jnp.ones((1, 7), dtype=jnp.bool_)
    block_starts = jnp.array([True, False, False, False, True, False, False])
    mask = np.asarray(_pi0.make_attn_mask(input_mask, block_starts))[0]

    assert not mask[0, 4]
    assert not mask[3, 6]
    assert mask[4, 0]
    assert mask[6, 3]


def test_future_force_head_shape_and_finite_output():
    config = _pi0_config.Pi0Config(
        force_enabled=True,
        force_dim=12,
        future_force_enabled=True,
        future_force_horizon=10,
        future_force_hidden=32,
    )
    head = _pi0.FutureForceHead(config, action_width=64, rngs=nnx.Rngs(0))
    force_context = jax.random.normal(jax.random.key(1), (2, 4, 64))
    clean_actions = jax.random.normal(jax.random.key(2), (2, 50, 32))

    predicted = head(force_context, clean_actions)

    assert predicted.shape == (2, 10, 12)
    assert np.all(np.isfinite(np.asarray(predicted)))
