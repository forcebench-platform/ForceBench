import sys
from dataclasses import replace

import numpy as np
import pytest

from forcebench.policy_adapters import forcevla_ws, openpi_ws
from forcebench.policy_api import (
    CANONICAL_ACTION_SPACE,
    POLICY_CONTRACT,
    PolicyActionChunk,
    PolicyObservation,
)
from forcebench.policy_registry import (
    get_policy_adapter_manifest,
    load_policy_adapter,
    policy_adapter_names,
)


def observation() -> PolicyObservation:
    return PolicyObservation(
        step=12,
        rollout_id="test-task:seed=12",
        state=np.zeros(7, dtype=np.float32),
        main_image=np.zeros((32, 48, 3), dtype=np.uint8),
        wrist_image=np.zeros((24, 24, 3), dtype=np.uint8),
        eef_wrench=np.zeros(6, dtype=np.float32),
        prompt="test task",
    )


def test_policy_adapters_are_auto_discovered_without_model_imports():
    assert {"forcebench_ws", "forcevla_ws", "openpi_ws"} <= set(policy_adapter_names())
    assert "forcebench.policy_transport" not in sys.modules
    assert get_policy_adapter_manifest("openpi_ws").adapter_class.endswith(
        ":OpenPIWebSocketAdapter"
    )


def test_native_policy_contract_is_explicit_and_model_independent():
    request = observation().canonical_request()
    assert request["contract"] == POLICY_CONTRACT
    assert request["rollout_id"] == "test-task:seed=12"
    assert request["control_rate_hz"] == 15.0
    assert set(request["observation"]["images"]) == {"main", "wrist"}
    assert request["observation"]["state_semantics"].startswith("world_tcp")
    assert "panda_hand_joint" in request["observation"]["wrench_semantics"]
    assert request["observation"]["wrench_filter"] == "none"


def test_native_policy_contract_serializes_dense_history():
    current = observation()
    sample = PolicyObservation(
        step=current.step,
        rollout_id=current.rollout_id,
        state=current.state,
        main_image=current.main_image,
        wrist_image=current.wrist_image,
        eef_wrench=current.eef_wrench,
        prompt=current.prompt,
        state_history=np.stack([current.state, current.state]),
        main_image_history=np.stack([current.main_image, current.main_image]),
        wrist_image_history=np.stack([current.wrist_image, current.wrist_image]),
        eef_wrench_history=np.stack([current.eef_wrench] * 10),
    )
    history = sample.canonical_request()["observation"]["history"]
    assert history["states"].shape == (2, 7)
    assert history["images"]["main"].shape == (2, 32, 48, 3)
    assert history["images"]["wrist"].shape == (2, 24, 24, 3)
    assert history["eef_wrenches"].shape == (10, 6)


def test_explicit_history_rejects_missing_or_stale_latest_frame():
    current = observation()
    with pytest.raises(ValueError, match="must provide"):
        replace(
            current,
            state_history=np.stack([current.state, current.state]),
        )
    stale = np.stack([current.state, current.state])
    stale[-1, 0] = 1.0
    with pytest.raises(ValueError, match="latest state_history"):
        replace(
            current,
            state_history=stale,
            main_image_history=np.stack([current.main_image, current.main_image]),
            wrist_image_history=np.stack([current.wrist_image, current.wrist_image]),
            eef_wrench_history=np.stack([current.eef_wrench] * 10),
        )


def test_legacy_repacking_is_confined_to_adapters():
    sample = observation()
    assert set(openpi_ws.encode_observation(sample)) == {
        "observation/state",
        "observation/image",
        "observation/wrist_image",
        "eef_wrench",
        "prompt",
    }
    assert set(forcevla_ws.encode_observation(sample)) == {
        "state",
        "image",
        "wrist_image",
        "eef_wrench",
        "prompt",
    }


def test_policy_action_chunk_rejects_noncanonical_outputs():
    chunk = PolicyActionChunk(np.zeros((50, 7), dtype=np.float32))
    assert chunk.action_space == CANONICAL_ACTION_SPACE
    with pytest.raises(ValueError, match="shape"):
        PolicyActionChunk(np.zeros((50, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="action space"):
        PolicyActionChunk(np.zeros((1, 7), dtype=np.float32), action_space="joint_position")
    with pytest.raises(ValueError, match="NaN"):
        PolicyActionChunk(np.full((1, 7), np.nan, dtype=np.float32))


def test_policy_observation_rejects_ambiguous_images_and_state():
    with pytest.raises(ValueError, match="state"):
        PolicyObservation(
            step=0,
            rollout_id="test-task:seed=0",
            state=np.zeros(13),
            main_image=np.zeros((8, 8, 3), dtype=np.uint8),
            wrist_image=np.zeros((8, 8, 3), dtype=np.uint8),
            eef_wrench=np.zeros(6),
            prompt="test",
        )
    with pytest.raises(ValueError, match="uint8"):
        PolicyObservation(
            step=0,
            rollout_id="test-task:seed=0",
            state=np.zeros(7),
            main_image=np.zeros((8, 8, 3), dtype=np.float32),
            wrist_image=np.zeros((8, 8, 3), dtype=np.uint8),
            eef_wrench=np.zeros(6),
            prompt="test",
        )


def test_policy_observation_rejects_empty_rollout_id():
    with pytest.raises(ValueError, match="rollout_id"):
        PolicyObservation(
            step=0,
            rollout_id="",
            state=np.zeros(7),
            main_image=np.zeros((8, 8, 3), dtype=np.uint8),
            wrist_image=np.zeros((8, 8, 3), dtype=np.uint8),
            eef_wrench=np.zeros(6),
            prompt="test",
        )


def test_registered_adapter_owns_transport_repacking(monkeypatch):
    import forcebench.policy_transport as policy_transport

    class FakeTransport:
        def __init__(self, host, port, *, max_wait_seconds):
            self.uri = f"ws://{host}:{port}"
            self.server_metadata = {}
            self.payload = None
            self.closed = False

        def request(self, payload):
            self.payload = payload
            return {"actions": np.zeros((4, 7), dtype=np.float32)}

        def close(self):
            self.closed = True

    monkeypatch.setattr(policy_transport, "MessagePackWebSocketTransport", FakeTransport)
    adapter = load_policy_adapter(
        "openpi_ws", host="127.0.0.1", port=8002, options={"max_wait_seconds": 1}
    )
    chunk = adapter.infer(observation())
    assert chunk.actions.shape == (4, 7)
    assert set(adapter._transport.payload) == {
        "observation/state",
        "observation/image",
        "observation/wrist_image",
        "eef_wrench",
        "prompt",
    }
    adapter.close()
    assert adapter._transport.closed
