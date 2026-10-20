"""Bridge the archived ForceFlow wire name without changing its numeric server."""

from __future__ import annotations

from typing import Any, Mapping

from forcebench.policy_api import PolicyActionChunk, PolicyAdapterManifest, PolicyObservation
from forcebench.policy_adapters._websocket import WebSocketPolicyAdapter


WIRE_CONTRACT = "coribench_policy"
EXECUTE_CHUNK = 32
MANIFEST = PolicyAdapterManifest(
    name="forceflow_ws",
    adapter_class="forcebench.policy_adapters.forceflow_ws:ForceFlowWebSocketAdapter",
    description="Archived ForceFlow server: dense history and 32 absolute actions.",
)


class ForceFlowWebSocketAdapter(WebSocketPolicyAdapter):
    manifest = MANIFEST

    def __init__(self, *, host: str, port: int, options: Mapping[str, Any]):
        # Read the transport's effective mode (cached at import), before it
        # opens a socket. The preserved server does not decode JPEG envelopes.
        from forcebench.policy_transport import _IMAGE_TRANSPORT

        if _IMAGE_TRANSPORT != "raw":
            raise ValueError("Archived ForceFlow requires FORCEBENCH_IMAGE_TRANSPORT=raw")
        super().__init__(host=host, port=port, options=options)

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        if observation.state_history is None:
            raise ValueError("ForceFlow requires dense state/image/wrench histories on every query")
        if observation.state_history.shape != (2, 7) or observation.eef_wrench_history.shape != (10, 6):
            raise ValueError("ForceFlow requires two state/image frames and ten wrench frames")
        request = observation.canonical_request()
        request["contract"] = WIRE_CONTRACT
        response = self._request(request)
        if response.get("contract") != WIRE_CONTRACT:
            raise ValueError(f"ForceFlow response has invalid contract: {response.get('contract')!r}")
        if "actions" not in response:
            raise ValueError("ForceFlow response has no actions")
        chunk = PolicyActionChunk(
            response["actions"],
            action_space=str(response.get("action_space", "")),
            control_rate_hz=float(response.get("control_rate_hz", 0.0)),
        )
        if chunk.actions.shape != (EXECUTE_CHUNK, 7):
            raise ValueError(f"ForceFlow requires [32, 7] returned actions, got {chunk.actions.shape}")
        return chunk

    @property
    def metadata(self):
        return {
            **super().metadata,
            "server_wire_contract": WIRE_CONTRACT,
            "action_horizon": 64,
            "execute_chunk": EXECUTE_CHUNK,
            "observation_horizon": 2,
            "wrench_history_frames": 10,
        }
