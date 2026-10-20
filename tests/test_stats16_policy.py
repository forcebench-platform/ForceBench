"""CPU regression checks for control-rate temporal history and episode isolation."""

import ast
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forcebench.policy_api import PolicyObservation
from forcebench.policy_registry import load_policy_adapter, policy_adapter_names


@pytest.fixture
def adapter_factory(monkeypatch):
    import forcebench.policy_transport as policy_transport

    class FakeTransport:
        def __init__(self, host, port, *, max_wait_seconds):
            self.uri = f"ws://{host}:{port}"
            self.server_metadata = {}
            self.requests = []

        def request(self, payload):
            self.requests.append(payload)
            return {"actions": np.zeros((50, 7), dtype=np.float32)}

        def close(self):
            pass

    monkeypatch.setattr(policy_transport, "MessagePackWebSocketTransport", FakeTransport)

    def make(name="stats16_ws"):
        return load_policy_adapter(name, host="127.0.0.1", port=8002)

    return make


def observation(step, rollout_id="weight_sort:seed=300", value=None):
    return PolicyObservation(
        step=step,
        rollout_id=rollout_id,
        state=np.zeros(7, dtype=np.float32),
        main_image=np.zeros((8, 8, 3), dtype=np.uint8),
        wrist_image=np.zeros((8, 8, 3), dtype=np.uint8),
        eef_wrench=np.full(6, step if value is None else value, dtype=np.float32),
        prompt="test",
    )


def observe(adapter, sample):
    adapter.observe_control_tick(
        step=sample.step, rollout_id=sample.rollout_id, eef_wrench=sample.eef_wrench
    )


def test_chunk_interior_ticks_are_retained_and_initial_padding_is_masked(adapter_factory):
    assert "stats16_ws" in policy_adapter_names()
    adapter = adapter_factory()
    # Fifty actions execute between requests; the second request must contain
    # the latest 16 control ticks, not two repeated inference observations.
    for step in range(51):
        sample = observation(step)
        observe(adapter, sample)
        if step in (0, 50):
            assert adapter.infer(sample).actions.shape == (50, 7)
    initial, next_chunk = adapter._transport.requests
    np.testing.assert_array_equal(initial["force_history"], np.zeros((16, 6)))
    np.testing.assert_array_equal(initial["force_history_is_pad"], [True] * 15 + [False])
    np.testing.assert_array_equal(
        next_chunk["force_history"], np.repeat(np.arange(35, 51)[:, None], 6, axis=1)
    )
    assert not next_chunk["force_history_is_pad"].any()


@pytest.mark.parametrize("new_rollout_id", ["weight_sort:seed=300", "weight_sort:seed=301"])
def test_reset_discards_previous_episode_even_with_repeated_seed(adapter_factory, new_rollout_id):
    adapter = adapter_factory()
    for step in range(20):
        observe(adapter, observation(step))
    first = observation(0, new_rollout_id, value=77)
    observe(adapter, first)
    adapter.infer(first)
    payload = adapter._transport.requests[-1]
    np.testing.assert_array_equal(payload["force_history"], np.full((16, 6), 77))
    assert payload["force_history_is_pad"].sum() == 15


def test_clone_histories_do_not_share_memory_and_copy_input_buffers(adapter_factory):
    left, right = adapter_factory(), adapter_factory()
    for step in range(4):
        sample = observation(step)
        observe(left, sample)
        observe(right, observation(step, "weight_sort:seed=301", value=100 + step))
        sample.eef_wrench[:] = -999  # Simulation buffers may be reused after the tick.
    left.infer(observation(3))
    right.infer(observation(3, "weight_sort:seed=301", value=103))
    np.testing.assert_array_equal(left._transport.requests[-1]["force_history"][-4:, 0], range(4))
    np.testing.assert_array_equal(right._transport.requests[-1]["force_history"][-4:, 0], range(100, 104))


def test_inference_without_tick_fails_before_sending_a_request(adapter_factory):
    adapter = adapter_factory()
    with pytest.raises(RuntimeError, match="observe_control_tick"):
        adapter.infer(observation(0))
    assert adapter._transport.requests == []


def test_nonconsecutive_or_cross_episode_tick_is_rejected(adapter_factory):
    adapter = adapter_factory()
    with pytest.raises(RuntimeError, match="start at step 0"):
        observe(adapter, observation(1))
    observe(adapter, observation(0))
    with pytest.raises(RuntimeError, match="Non-consecutive"):
        observe(adapter, observation(2))
    with pytest.raises(RuntimeError, match="cross-rollout"):
        observe(adapter, observation(1, "weight_sort:seed=301"))
    # Failed observations do not consume a tick or contaminate the history.
    observe(adapter, observation(1))
    with pytest.raises(RuntimeError, match="Non-consecutive"):
        observe(adapter, observation(1))
    adapter.infer(observation(1))
    np.testing.assert_array_equal(adapter._transport.requests[-1]["force_history"][-2:, 0], [0, 1])


@pytest.mark.parametrize("invalid", [np.zeros(5), np.full(6, np.nan), np.full(6, np.inf)])
def test_invalid_wrench_does_not_consume_the_tick(adapter_factory, invalid):
    adapter = adapter_factory()
    observe(adapter, observation(0))
    with pytest.raises(ValueError, match="six finite values"):
        adapter.observe_control_tick(step=1, rollout_id="weight_sort:seed=300", eef_wrench=invalid)
    observe(adapter, observation(1))
    adapter.infer(observation(1))


def test_inference_rejects_stale_or_other_rollout_observation(adapter_factory):
    adapter = adapter_factory()
    observe(adapter, observation(0))
    observe(adapter, observation(1))
    for sample in (observation(0), observation(1, "weight_sort:seed=301")):
        with pytest.raises(RuntimeError, match="rollout or step mismatch"):
            adapter.infer(sample)
    with pytest.raises(RuntimeError, match="Latest wrench mismatch"):
        adapter.infer(replace(observation(1), eef_wrench=np.zeros(6)))
    assert adapter._transport.requests == []


def test_stateless_adapter_tick_preserves_existing_request_contract(adapter_factory):
    adapter = adapter_factory("openpi_ws")
    sample = observation(0)
    observe(adapter, sample)
    adapter.infer(sample)
    assert "force_history" not in adapter._transport.requests[-1]


def runner_observation_preamble():
    """Execute the real runner's pre-action sampling without launching Isaac.

    The simulator boot is deliberately excluded. Keeping the queued-action
    branch in this harness catches placing the history hook inside inference.
    """
    path = Path(__file__).resolve().parents[1] / "scripts/evaluation/inference_forcebench_vectorized.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "step"
    )
    need_index = next(
        index for index, node in enumerate(loop.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "need" for target in node.targets)
    )
    return compile(ast.Module(body=loop.body[:need_index + 1], type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize("num_envs", [1, 2])
def test_vector_runner_samples_chunk_interiors_and_skips_finished_clones(adapter_factory, num_envs):
    policies = [adapter_factory() for _ in range(num_envs)]
    active = np.ones(num_envs, dtype=bool)
    env = SimpleNamespace(policy_state=np.zeros((num_envs, 7)))
    namespace = {
        "args": SimpleNamespace(num_envs=num_envs, smoke_hold=False, task="weight_sort"),
        "env": env,
        "np": np,
        "as_numpy": lambda value: value,
        "active": active,
        "policies": policies,
        "seeds": list(range(300, 300 + num_envs)),
        "queues": [deque([np.zeros(7)] * 50) for _ in policies],
        "forceflow_interface": False,
    }
    preamble = runner_observation_preamble()
    for step in range(6):
        env.force_sensor_joint_raw = np.array([np.full(6, 100 * clone + step) for clone in range(num_envs)])
        namespace["step"] = step
        if num_envs == 2 and step == 3:
            active[1] = False
        exec(preamble, namespace)
        assert namespace["need"] == []  # No policy inference on these ticks.
    policies[0].infer(observation(5))
    np.testing.assert_array_equal(policies[0]._transport.requests[-1]["force_history"][-6:, 0], range(6))
    if num_envs == 2:
        policies[1].infer(observation(2, "weight_sort:seed=301", value=102))
        assert policies[1]._transport.requests[-1]["force_history_is_pad"].sum() == 13


def test_smoke_hold_preamble_does_not_require_policy_connections():
    namespace = {
        "args": SimpleNamespace(num_envs=1, smoke_hold=True, task="weight_sort"),
        "env": SimpleNamespace(policy_state=np.zeros((1, 7)), force_sensor_joint_raw=np.zeros((1, 6))),
        "np": np,
        "as_numpy": lambda value: value,
        "active": np.ones(1, dtype=bool),
        "policies": [],
        "seeds": [300],
        "queues": [deque()],
        "forceflow_interface": False,
        "step": 0,
    }
    exec(runner_observation_preamble(), namespace)
    assert namespace["need"] == [0]

