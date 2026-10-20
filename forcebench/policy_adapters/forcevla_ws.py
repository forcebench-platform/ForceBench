"""Adapter for the existing ForceVLA observation-key contract."""

from __future__ import annotations

from forcebench.policy_api import PolicyActionChunk, PolicyAdapterManifest, PolicyObservation
from forcebench.policy_adapters._websocket import WebSocketPolicyAdapter


MANIFEST = PolicyAdapterManifest(
    name="forcevla_ws",
    adapter_class="forcebench.policy_adapters.forcevla_ws:ForceVLAWebSocketAdapter",
    description="Existing ForceVLA MessagePack WebSocket server.",
)


def encode_observation(observation: PolicyObservation) -> dict:
    return {
        "state": observation.state,
        "image": observation.main_image,
        "wrist_image": observation.wrist_image,
        "eef_wrench": observation.eef_wrench,
        "prompt": observation.prompt,
    }


class ForceVLAWebSocketAdapter(WebSocketPolicyAdapter):
    manifest = MANIFEST

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        return self._legacy_action_chunk(self._request(encode_observation(observation)))
