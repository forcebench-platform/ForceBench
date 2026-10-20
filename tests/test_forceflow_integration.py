"""Exercise preserved server codecs and the actual runner queue without a GPU.

AST isolation excludes simulator boot and model inference, not the wire parser,
output conversion, adapter, or queueing logic under test.
"""

import ast
from collections import deque
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import msgpack
import numpy as np
import pytest

from forcebench.policy_api import PolicyObservation
from forcebench.policy_adapters.forceflow_ws import ForceFlowWebSocketAdapter

ROOT = Path(__file__).resolve().parents[1]


def preserved_codecs():
    path = ROOT / "baselines/forceflow/coribench_forceflow/server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Constant)]
    nodes.append(next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "ObservationHistory"))
    nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"pack_numpy", "unpack_numpy"})
    policy = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "ForceFlowPolicy")
    infer = next(node for node in policy.body if isinstance(node, ast.FunctionDef)
                 and node.name == "infer_batch")
    start = next(index for index, node in enumerate(infer.body)
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "normalized" for target in node.targets))
    # Execute the unchanged post-inference statements with a supplied prediction.
    infer.body = infer.body[start:]
    nodes.append(infer)
    namespace = {"np": np, "deque": deque, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    client_path = ROOT / "forcebench/policy_transport.py"
    client_tree = ast.parse(client_path.read_text(encoding="utf-8"))
    client_nodes = [node for node in client_tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"_pack_numpy", "_unpack_numpy"}]
    namespace["_IMAGE_TRANSPORT"] = "raw"
    exec(compile(ast.Module(body=client_nodes, type_ignores=[]), str(client_path), "exec"), namespace)
    return namespace


class PredictionArray:
    """Only the tensor-to-NumPy boundary; this fixture does not simulate a model."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return PredictionArray(self.value[key])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value.astype(np.float32)


def sample():
    return PolicyObservation(
        step=32, rollout_id="nut_thread:seed=300", state=np.zeros(7),
        main_image=np.zeros((8, 8, 3), dtype=np.uint8),
        wrist_image=np.zeros((8, 8, 3), dtype=np.uint8), eef_wrench=np.full(6, 32), prompt="test",
        state_history=np.zeros((2, 7)), main_image_history=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        wrist_image_history=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        eef_wrench_history=np.repeat(np.arange(23, 33)[:, None], 6, axis=1),
    )


def queue_response(actions, adapter="forceflow_ws"):
    path = ROOT / "scripts/evaluation/inference_forcebench_vectorized.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"FORCEFLOW_EXECUTE_CHUNK", "forceflow_interface", "effective_execute_chunk"}
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    nodes.append(next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                      and isinstance(node.target, ast.Tuple)
                      and [getattr(item, "id", None) for item in node.target.elts]
                      == ["env_id", "predicted", "latency"]))
    namespace = {
        "args": SimpleNamespace(policy_adapter=adapter, execute_chunk=50),
        "results": [(0, actions, 0.1)], "queues": [deque()], "latencies": [[]],
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["queues"][0]


def test_archived_server_response_reaches_all_32_runner_actions():
    codecs = preserved_codecs()
    history = codecs["ObservationHistory"](2, 10)
    # Supply 64 known predictions; only the server's first 32 should execute.
    prediction = np.zeros((1, 64, 13), dtype=np.float32)
    prediction[:, :32, :6] = -0.9  # 0.1 delta, then cumulative absolute actions.
    prediction[:, 32:, :6] = 1.0
    codecs["prediction"] = PredictionArray(prediction)
    policy = SimpleNamespace(execution_steps=codecs["EXECUTE_CHUNK"],
                             action_scale=np.full(6, 2.0), action_min=np.zeros(6))
    parsed = []

    def request(payload):
        encoded = msgpack.packb(payload, default=codecs["_pack_numpy"])
        decoded = msgpack.unpackb(encoded, object_hook=codecs["unpack_numpy"])
        item = history.append(decoded, require_explicit_history=True)
        parsed.append(item)
        response = codecs["infer_batch"](policy, [item])[0]
        wire = msgpack.packb(response, default=codecs["pack_numpy"])
        return msgpack.unpackb(wire, object_hook=codecs["_unpack_numpy"])

    adapter = object.__new__(ForceFlowWebSocketAdapter)
    adapter._request = request
    actions = adapter.infer(sample()).actions
    queued = np.stack(queue_response(actions))
    assert queued.shape == (32, 7)
    np.testing.assert_allclose(queued[:, 0], np.arange(1, 33) * 0.1, atol=2e-6)
    np.testing.assert_array_equal(parsed[0]["force"][:, 0], np.arange(23, 33))


@pytest.mark.parametrize("length", [16, 50, 64])
def test_runner_rejects_incompatible_forceflow_lengths(length):
    with pytest.raises(RuntimeError, match="requires exactly"):
        queue_response(np.zeros((length, 7)))


@pytest.mark.parametrize("adapter", ["openpi_ws", "forcebench_ws", "stats16_ws"])
def test_other_adapters_keep_their_50_step_execution(adapter):
    actions = np.arange(64 * 7).reshape(64, 7)
    np.testing.assert_array_equal(np.stack(queue_response(actions, adapter)), actions[:50])


def test_jpeg_is_rejected_before_attempting_connection():
    code = (
        "from forcebench.policy_adapters.forceflow_ws import ForceFlowWebSocketAdapter\n"
        "try:\n"
        "    ForceFlowWebSocketAdapter(host='127.0.0.1', port=1, options={'max_wait_seconds': 0})\n"
        "except ValueError as exc:\n"
        "    assert 'FORCEBENCH_IMAGE_TRANSPORT=raw' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('JPEG unexpectedly accepted')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                            env={**os.environ, "FORCEBENCH_IMAGE_TRANSPORT": "jpeg"}, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_adapter_rejects_missing_history_before_network():
    adapter = object.__new__(ForceFlowWebSocketAdapter)
    no_history = replace(sample(), state_history=None, main_image_history=None,
                         wrist_image_history=None, eef_wrench_history=None)
    with pytest.raises(ValueError, match="dense"):
        adapter.infer(no_history)


@pytest.mark.parametrize("field,value", [
    ("contract", "forcebench_policy"), ("control_rate_hz", 30),
    ("action_space", "joint_position"), ("actions", np.zeros((16, 7))),
])
def test_adapter_rejects_wrong_server_contract(field, value):
    codecs = preserved_codecs()
    response = {"contract": codecs["CONTRACT"], "action_space": codecs["ACTION_SPACE"],
                "control_rate_hz": codecs["CONTROL_RATE_HZ"], "actions": np.zeros((32, 7))}
    response[field] = value
    adapter = object.__new__(ForceFlowWebSocketAdapter)
    adapter._request = lambda payload: response
    with pytest.raises(ValueError):
        adapter.infer(sample())
