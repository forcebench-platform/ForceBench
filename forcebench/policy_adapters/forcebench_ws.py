"""Native adapter for servers implementing the canonical ForceBench contract."""

from __future__ import annotations

from forcebench.policy_api import (
    CANONICAL_ACTION_SPACE,
    POLICY_CONTRACT,
    POLICY_CONTROL_RATE_HZ,
    PolicyActionChunk,
    PolicyAdapterManifest,
    PolicyObservation,
)
from forcebench.policy_adapters._websocket import WebSocketPolicyAdapter


MANIFEST = PolicyAdapterManifest(
    name="forcebench_ws",
    adapter_class="forcebench.policy_adapters.forcebench_ws:ForceBenchWebSocketAdapter",
    description="Native canonical ForceBench MessagePack WebSocket contract.",
)


class ForceBenchWebSocketAdapter(WebSocketPolicyAdapter):
    manifest = MANIFEST

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        result = self._request(observation.canonical_request())
        if result.get("contract") != POLICY_CONTRACT:
            raise ValueError(f"policy response has invalid contract: {result.get('contract')!r}")
        if "actions" not in result:
            raise ValueError(f"policy response has no actions: {sorted(result)}")
        return PolicyActionChunk(
            result["actions"],
            action_space=str(result.get("action_space", "")),
            control_rate_hz=float(result.get("control_rate_hz", 0.0)),
        )

    @property
    def metadata(self):
        metadata = dict(super().metadata)
        metadata["native_contract"] = True
        metadata["required_response"] = {
            "contract": POLICY_CONTRACT,
            "action_space": CANONICAL_ACTION_SPACE,
            "control_rate_hz": POLICY_CONTROL_RATE_HZ,
        }
        return metadata
