"""Adapter for the existing OpenPI observation-key contract."""

from __future__ import annotations

from forcebench.policy_api import PolicyActionChunk, PolicyAdapterManifest, PolicyObservation
from forcebench.policy_adapters._websocket import WebSocketPolicyAdapter


MANIFEST = PolicyAdapterManifest(
    name="openpi_ws",
    adapter_class="forcebench.policy_adapters.openpi_ws:OpenPIWebSocketAdapter",
    description="Existing OpenPI MessagePack WebSocket server.",
)


def encode_observation(observation: PolicyObservation) -> dict:
    return {
        "observation/state": observation.state,
        "observation/image": observation.main_image,
        "observation/wrist_image": observation.wrist_image,
        "eef_wrench": observation.eef_wrench,
        "prompt": observation.prompt,
    }


class OpenPIWebSocketAdapter(WebSocketPolicyAdapter):
    manifest = MANIFEST

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        return self._legacy_action_chunk(self._request(encode_observation(observation)))
