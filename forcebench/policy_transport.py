"""Transport primitives shared by model-specific policy adapters."""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

import cv2
import msgpack
import numpy as np
import websockets.sync.client


_IMAGE_TRANSPORT = os.environ.get("FORCEBENCH_IMAGE_TRANSPORT", "raw").strip().lower()
_JPEG_QUALITY = int(os.environ.get("FORCEBENCH_JPEG_QUALITY", "90"))
if _IMAGE_TRANSPORT not in ("raw", "jpeg"):
    raise ValueError(f"unsupported image transport: {_IMAGE_TRANSPORT}")
if not 1 <= _JPEG_QUALITY <= 100:
    raise ValueError(f"JPEG quality must be between 1 and 100: {_JPEG_QUALITY}")


def _pack_numpy(value: Any):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
        if (
            _IMAGE_TRANSPORT == "jpeg"
            and value.dtype == np.uint8
            and value.ndim == 3
            and value.shape[-1] == 3
        ):
            bgr = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
            )
            if not ok:
                raise RuntimeError(f"failed to JPEG-encode image with shape {value.shape}")
            return {
                b"__jpeg_rgb__": True,
                b"data": encoded.tobytes(),
                b"shape": value.shape,
            }
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot serialize {type(value)!r}")


def _unpack_numpy(value: dict[bytes, Any]):
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"]
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class MessagePackWebSocketTransport:
    """Request/response transport compatible with OpenPI's policy server."""

    def __init__(self, host: str, port: int, *, max_wait_seconds: float = 180.0):
        if not host:
            raise ValueError("policy host must be non-empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"invalid policy port: {port}")
        self.uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        compression_name = os.environ.get("FORCEBENCH_WEBSOCKET_COMPRESSION", "none").strip().lower()
        if compression_name in ("", "none", "off"):
            compression = None
        elif compression_name == "deflate":
            compression = "deflate"
        else:
            raise ValueError(f"unsupported WebSocket compression: {compression_name}")
        deadline = time.monotonic() + max_wait_seconds
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                self._socket = websockets.sync.client.connect(
                    self.uri,
                    compression=compression,
                    max_size=None,
                    open_timeout=10,
                )
                metadata = self._socket.recv()
                if isinstance(metadata, str):
                    raise RuntimeError(f"policy server handshake failed: {metadata}")
                self.server_metadata = msgpack.unpackb(metadata, object_hook=_unpack_numpy)
                return
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                last_error = exc
                time.sleep(2.0)
        raise TimeoutError(f"policy server did not become ready at {self.uri}: {last_error}")

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self._socket.send(msgpack.packb(dict(payload), default=_pack_numpy))
        response = self._socket.recv()
        if isinstance(response, str):
            raise RuntimeError(f"policy server error: {response}")
        result = msgpack.unpackb(response, object_hook=_unpack_numpy)
        if not isinstance(result, Mapping):
            raise ValueError(f"policy response must be a mapping, got {type(result)!r}")
        return result

    def close(self) -> None:
        self._socket.close()
