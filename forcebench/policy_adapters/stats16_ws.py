"""Consecutive 15 Hz raw wrench history for the temporal OpenPI policies.

Ported from the September 14 evaluation adapter preserved in the September 28
handoff (SHA-256 e466f8b4e55bb6d0c7721703d3eb04e34b6401b2919c60b543f2afdb53f2310a).
The history encoding is unchanged; lifecycle checks also reject inference from
a different rollout and provide an explicit error when the tick hook is absent.
"""

from __future__ import annotations

from collections import deque
from numbers import Integral
from typing import Any, Mapping

import numpy as np

from forcebench.policy_api import PolicyActionChunk, PolicyAdapterManifest, PolicyObservation
from forcebench.policy_adapters._websocket import WebSocketPolicyAdapter
from forcebench.policy_adapters.openpi_ws import encode_observation


HISTORY_LENGTH = 16
MANIFEST = PolicyAdapterManifest(
    name="stats16_ws",
    adapter_class="forcebench.policy_adapters.stats16_ws:Stats16WebSocketAdapter",
    description="OpenPI with per-control-tick 16-frame raw wrench history.",
)


class Stats16WebSocketAdapter(WebSocketPolicyAdapter):
    manifest = MANIFEST

    def __init__(self, *, host: str, port: int, options: Mapping[str, Any]):
        self._history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self._rollout_id: str | None = None
        self._last_step = -1
        super().__init__(host=host, port=port, options=options)

    def observe_control_tick(self, *, step: int, rollout_id: str, eef_wrench: np.ndarray) -> None:
        """Observe every control tick, including ticks within an action chunk.

        A tick with step zero begins a new episode, even when replaying the same
        seed. Each environment must own a separate adapter instance.
        """
        if not isinstance(step, Integral) or isinstance(step, bool) or step < 0:
            raise ValueError("control step must be a non-negative integer")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError("rollout_id must be a non-empty string")
        wrench = np.asarray(eef_wrench, dtype=np.float32)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError("raw wrench must contain six finite values")
        if step == 0:
            self._history.clear()
            self._rollout_id = rollout_id
            self._last_step = -1
        if rollout_id != self._rollout_id or step != self._last_step + 1:
            raise RuntimeError("Non-consecutive or cross-rollout wrench history; start at step 0")
        self._history.append(wrench.copy())
        self._last_step = int(step)

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        if not self._history:
            raise RuntimeError("observe_control_tick must run before stats16 inference")
        if observation.rollout_id != self._rollout_id or observation.step != self._last_step:
            raise RuntimeError("Inference/control-tick rollout or step mismatch")
        if not np.array_equal(self._history[-1], observation.eef_wrench):
            raise RuntimeError("Latest wrench mismatch")
        pad = HISTORY_LENGTH - len(self._history)
        payload = encode_observation(observation)
        payload["force_history"] = np.stack([self._history[0]] * pad + list(self._history))
        payload["force_history_is_pad"] = np.arange(HISTORY_LENGTH) < pad
        return self._legacy_action_chunk(self._request(payload))
