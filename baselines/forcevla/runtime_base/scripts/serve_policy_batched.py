#!/usr/bin/env python3
"""Serve a JAX OpenPI-compatible checkpoint with bounded request batching."""

from __future__ import annotations

import asyncio
import dataclasses
import http
import logging
import socket
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy
import tyro
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from openpi.models import model as model_api
from openpi.policies import policy_config
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    config: str
    checkpoint_dir: str
    port: int = 8002
    max_batch_size: int = 4
    batch_window_ms: float = 8.0
    default_prompt: str | None = None
    expected_clients: int | None = None
    exit_after_idle_s: float | None = None


def infer_batch(policy, observations: list[dict]) -> list[dict]:
    """Run the existing per-sample transforms around one batched model call."""

    transformed = []
    for observation in observations:
        copied = jax.tree.map(lambda value: value, observation)
        transformed.append(policy._input_transform(copied))
    inputs = jax.tree.map(
        lambda *values: jnp.stack([jnp.asarray(value) for value in values], axis=0),
        *transformed,
    )
    policy._rng, sample_rng = jax.random.split(policy._rng)
    started = time.monotonic()
    outputs = {
        "state": inputs["state"],
        "actions": policy._sample_actions(
            sample_rng,
            model_api.Observation.from_dict(inputs),
            **policy._sample_kwargs,
        ),
    }
    outputs = jax.tree.map(np.asarray, outputs)
    model_time_ms = (time.monotonic() - started) * 1000.0
    results = []
    for index in range(len(observations)):
        one = jax.tree.map(lambda value: value[index], outputs)
        one = policy._output_transform(one)
        one["policy_timing"] = {
            "infer_ms": model_time_ms,
            "batch_size": len(observations),
        }
        results.append(one)
    return results


class BatchedWebsocketPolicyServer:
    def __init__(self, policy, args: Args):
        if args.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if args.batch_window_ms < 0.0:
            raise ValueError("batch_window_ms must be non-negative")
        self.policy = policy
        self.args = args
        self.queue = None
        self.shutdown = None
        self.total_clients = 0
        self.active_clients = 0
        self.last_activity = time.monotonic()
        self.metadata = dict(policy.metadata)
        self.metadata.update(
            {
                "batched_server": True,
                "max_batch_size": args.max_batch_size,
                "batch_window_ms": args.batch_window_ms,
                "expected_clients": args.expected_clients,
            }
        )

    async def run(self) -> None:
        self.queue = asyncio.Queue()
        self.shutdown = asyncio.Event()
        worker = asyncio.create_task(self._batch_worker())
        idle_watcher = asyncio.create_task(self._idle_watcher())
        try:
            async with websocket_server.serve(
                self._handler,
                "0.0.0.0",
                self.args.port,
                compression=None,
                max_size=None,
                ping_interval=None,
                process_request=self._health_check,
            ) as server:
                serve_task = asyncio.create_task(server.serve_forever())
                await self.shutdown.wait()
                serve_task.cancel()
                await asyncio.gather(serve_task, return_exceptions=True)
        finally:
            worker.cancel()
            idle_watcher.cancel()
            await asyncio.gather(worker, idle_watcher, return_exceptions=True)

    async def _idle_watcher(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if (
                self.args.exit_after_idle_s is not None
                and self.active_clients == 0
                and time.monotonic() - self.last_activity >= self.args.exit_after_idle_s
            ):
                assert self.shutdown is not None
                self.shutdown.set()
                return

    async def _batch_worker(self) -> None:
        assert self.queue is not None
        while True:
            first = await self.queue.get()
            batch = [first]
            if self.args.batch_window_ms:
                await asyncio.sleep(self.args.batch_window_ms / 1000.0)
            while len(batch) < self.args.max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            observations = [item[0] for item in batch]
            real_size = len(observations)
            observations.extend([observations[-1]] * (self.args.max_batch_size - real_size))
            try:
                results = await asyncio.to_thread(infer_batch, self.policy, observations)
                for (_, future), result in zip(batch, results[:real_size], strict=True):
                    if not future.cancelled():
                        future.set_result(result)
            except BaseException as exc:
                for _, future in batch:
                    if not future.cancelled():
                        future.set_exception(exc)

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        assert self.queue is not None
        assert self.shutdown is not None
        self.total_clients += 1
        self.active_clients += 1
        self.last_activity = time.monotonic()
        packer = msgpack_numpy.Packer()
        try:
            await websocket.send(packer.pack(self.metadata))
            while True:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                future = asyncio.get_running_loop().create_future()
                await self.queue.put((observation, future))
                started = time.monotonic()
                action = await future
                action["server_timing"] = {
                    "total_ms": (time.monotonic() - started) * 1000.0,
                }
                await websocket.send(packer.pack(action))
        except websockets.ConnectionClosed:
            pass
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal error in batched policy server",
            )
            raise
        finally:
            self.active_clients -= 1
            self.last_activity = time.monotonic()
            if (
                self.args.expected_clients is not None
                and self.total_clients >= self.args.expected_clients
                and self.active_clients == 0
            ):
                self.shutdown.set()

    @staticmethod
    def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


def main(args: Args) -> None:
    if args.expected_clients is not None and args.expected_clients < 1:
        raise ValueError("expected_clients must be positive")
    if args.exit_after_idle_s is not None and args.exit_after_idle_s <= 0:
        raise ValueError("exit_after_idle_s must be positive")
    policy = policy_config.create_trained_policy(
        training_config.get_config(args.config),
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    logging.info(
        "Serving batched policy on %s:%d with max_batch_size=%d",
        socket.gethostname(),
        args.port,
        args.max_batch_size,
    )
    asyncio.run(BatchedWebsocketPolicyServer(policy, args).run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
