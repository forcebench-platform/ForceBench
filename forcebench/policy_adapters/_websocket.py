"""Shared implementation for MessagePack WebSocket policy adapters."""

from __future__ import annotations

from typing import Any, Mapping

from forcebench.policy_api import PolicyActionChunk, PolicyAdapter


class WebSocketPolicyAdapter(PolicyAdapter):
    def __init__(self, *, host: str, port: int, options: Mapping[str, Any]):
        # Import transport dependencies only after Isaac Sim has launched. The
        # adapter registry itself must remain safe to discover during CLI setup.
        from forcebench.policy_transport import MessagePackWebSocketTransport

        unknown = set(options) - {"max_wait_seconds"}
        if unknown:
            raise ValueError(f"unsupported adapter options: {sorted(unknown)}")
        self._transport = MessagePackWebSocketTransport(
            host,
            port,
            max_wait_seconds=float(options.get("max_wait_seconds", 180.0)),
        )

    @property
    def endpoint_uri(self) -> str:
        return self._transport.uri

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._transport.request(payload)

    @staticmethod
    def _legacy_action_chunk(result: Mapping[str, Any]) -> PolicyActionChunk:
        if "actions" not in result:
            raise ValueError(f"policy response has no actions: {sorted(result)}")
        return PolicyActionChunk(result["actions"])

    def close(self) -> None:
        self._transport.close()
