from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import torch.nn.functional as F
import websockets.asyncio.server
import websockets.frames
import yaml

from .model import build_policy

CONTRACT = "coribench_policy"
ACTION_SPACE = "world_tcp_xyz_rpy_gripper_open_minus1_plus1"
CONTROL_RATE_HZ = 15.0
ACTION_HORIZON = 64
EXECUTE_CHUNK = 32
FORCE_HISTORY = 10


def pack_numpy(value: Any):
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot serialize {type(value)!r}")


def unpack_numpy(value: dict[Any, Any]):
    ndarray = value.get(b"__ndarray__", value.get("__ndarray__"))
    if ndarray:
        data = value.get(b"data", value.get("data"))
        dtype = value.get(b"dtype", value.get("dtype"))
        shape = value.get(b"shape", value.get("shape"))
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=shape).copy()
    npgeneric = value.get(b"__npgeneric__", value.get("__npgeneric__"))
    if npgeneric:
        return np.dtype(value.get(b"dtype", value.get("dtype"))).type(
            value.get(b"data", value.get("data"))
        )
    return value


class ObservationHistory:
    def __init__(self, observation_horizon: int, force_history: int):
        self.states: deque[np.ndarray] = deque(maxlen=observation_horizon)
        self.arm_images: deque[np.ndarray] = deque(maxlen=observation_horizon)
        self.fixed_images: deque[np.ndarray] = deque(maxlen=observation_horizon)
        self.forces: deque[np.ndarray] = deque(maxlen=force_history)
        self.observation_horizon = observation_horizon
        self.force_history = force_history
        self.last_step = -1
        self.rollout_id: str | None = None

    def append(
        self, request: dict[str, Any], *, require_explicit_history: bool = False
    ) -> dict[str, Any]:
        if request.get("contract") != CONTRACT:
            raise ValueError(f"invalid contract: {request.get('contract')!r}")
        if not np.isclose(float(request.get("control_rate_hz", 0.0)), CONTROL_RATE_HZ):
            raise ValueError("ForceFlow server requires canonical 15 Hz observations")
        step = int(request["step"])
        rollout_id = request.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id.strip():
            raise ValueError("rollout_id must be a non-empty string")
        observation = request["observation"]
        state = self._array(observation["state"], (7,), np.float32, "state")
        force = self._array(observation["eef_wrench"], (6,), np.float32, "eef_wrench")
        main = self._image(observation["images"]["main"], "main")
        wrist = self._image(observation["images"]["wrist"], "wrist")
        prompt = str(observation["prompt"])
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if step == 0 or step <= self.last_step:
            self.states.clear()
            self.arm_images.clear()
            self.fixed_images.clear()
            self.forces.clear()
            self.rollout_id = rollout_id
        elif self.rollout_id is None:
            self.rollout_id = rollout_id
        elif rollout_id != self.rollout_id:
            raise ValueError(
                f"rollout_id changed within one history: {self.rollout_id!r} -> {rollout_id!r}"
            )
        explicit = observation.get("history")
        if explicit is None:
            if require_explicit_history:
                raise ValueError(
                    "execute_chunk > 1 requires dense observation.history from every 15 Hz control frame"
                )
            self.states.append(state)
            self.arm_images.append(wrist)
            self.fixed_images.append(main)
            self.forces.append(force)
            self._pad(self.states, self.observation_horizon)
            self._pad(self.arm_images, self.observation_horizon)
            self._pad(self.fixed_images, self.observation_horizon)
            self._pad(self.forces, self.force_history)
        else:
            if not isinstance(explicit, dict):
                raise ValueError("observation.history must be an object")
            states = self._array(
                explicit["states"], (self.observation_horizon, 7), np.float32, "history.states"
            )
            forces = self._array(
                explicit["eef_wrenches"],
                (self.force_history, 6),
                np.float32,
                "history.eef_wrenches",
            )
            fixed_images = self._image_history(
                explicit["images"]["main"], self.observation_horizon, "history.images.main"
            )
            arm_images = self._image_history(
                explicit["images"]["wrist"], self.observation_horizon, "history.images.wrist"
            )
            if not np.allclose(states[-1], state, atol=1e-6):
                raise ValueError("latest history state does not match observation.state")
            if not np.allclose(forces[-1], force, atol=1e-6):
                raise ValueError("latest history wrench does not match observation.eef_wrench")
            if not np.array_equal(fixed_images[-1], main):
                raise ValueError("latest main image history does not match current image")
            if not np.array_equal(arm_images[-1], wrist):
                raise ValueError("latest wrist image history does not match current image")
            self.states.clear()
            self.arm_images.clear()
            self.fixed_images.clear()
            self.forces.clear()
            self.states.extend(states)
            self.arm_images.extend(arm_images)
            self.fixed_images.extend(fixed_images)
            self.forces.extend(forces)
        self.last_step = step
        return {
            "step": step,
            "rollout_id": rollout_id,
            "prompt": prompt,
            "state": np.stack(self.states),
            "image_arm": np.stack(self.arm_images),
            "image_fix": np.stack(self.fixed_images),
            "force": np.stack(self.forces),
        }

    @staticmethod
    def _pad(values: deque[np.ndarray], count: int) -> None:
        while len(values) < count:
            values.appendleft(values[0].copy())

    @staticmethod
    def _array(value: Any, shape: tuple[int, ...], dtype, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=dtype)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite {shape}, got {result.shape}")
        return result.copy()

    @staticmethod
    def _image(value: Any, name: str) -> np.ndarray:
        result = np.asarray(value)
        if result.ndim != 3 or result.shape[-1] != 3 or result.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 [H,W,3], got {result.shape}/{result.dtype}")
        return result.copy()

    @staticmethod
    def _image_history(value: Any, count: int, name: str) -> np.ndarray:
        result = np.asarray(value)
        if (
            result.ndim != 4
            or result.shape[0] != count
            or result.shape[-1] != 3
            or result.dtype != np.uint8
        ):
            raise ValueError(
                f"{name} must be uint8 [{count},H,W,3], got {result.shape}/{result.dtype}"
            )
        return result.copy()


class ForceFlowPolicy:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: str):
        self.config = yaml.safe_load(config_path.read_text())
        self.device = torch.device(device)
        temporal = self.config["temporal"]
        self.control_frequency_hz = float(temporal["control_frequency_hz"])
        if not np.isclose(self.control_frequency_hz, CONTROL_RATE_HZ):
            raise ValueError(
                f"ForceFlow CoRiBench interface requires {CONTROL_RATE_HZ:g} Hz, "
                f"got {self.control_frequency_hz:g}"
            )
        self.observation_horizon = int(temporal["observation_horizon"])
        self.force_history = int(temporal["force_history"])
        self.action_horizon = int(temporal["action_horizon"])
        self.execution_steps = int(temporal.get("evaluation_execute_chunk", 1))
        required = {
            "action_horizon": (self.action_horizon, ACTION_HORIZON),
            "evaluation_execute_chunk": (self.execution_steps, EXECUTE_CHUNK),
            "force_history": (self.force_history, FORCE_HISTORY),
        }
        for name, (actual, expected) in required.items():
            if actual != expected:
                raise ValueError(
                    f"ForceFlow CoRiBench interface requires {name}={expected}, got {actual}"
                )
        if not 1 <= self.execution_steps <= self.action_horizon:
            raise ValueError("evaluation_execute_chunk is outside the action horizon")
        self.sample_steps = int(self.config.get("evaluation", {}).get("ode_steps", 20))
        self.use_ema = bool(self.config.get("evaluation", {}).get("use_ema", False))
        self.image_size = int(self.config["image"]["size"])
        self.policy = build_policy(self.config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(checkpoint["state_dict"], strict=True)
        self.policy.to(self.device).eval()

        stats_path = self.config.get("normalizer_path")
        if stats_path is None:
            stats_path = Path(self.config["output_dir"]) / "norm_stats.json"
        stats = json.loads(Path(stats_path).read_text())
        self.pos_min, self.pos_scale = self._stats(stats["pos"])
        self.action_min, self.action_scale = self._stats(stats["action"])
        self.force_min, self.force_scale = self._stats(stats["force"])

    @staticmethod
    def _stats(value: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
        minimum = np.asarray(value["min"], dtype=np.float32)
        maximum = np.asarray(value["max"], dtype=np.float32)
        return minimum, np.maximum(maximum - minimum, 1e-6)

    @staticmethod
    def _normalize(value: np.ndarray, minimum: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (2.0 * (value - minimum) / scale - 1.0).astype(np.float32)

    @staticmethod
    def _seed(item: dict[str, Any]) -> int:
        identity = (
            f"{item['rollout_id']}\0{item['prompt']}\0{int(item['step'])}".encode("utf-8")
        )
        digest = hashlib.sha256(identity).digest()
        return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF

    def infer_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        states = np.stack([item["state"] for item in items])
        normalized_pose = self._normalize(states[..., :6], self.pos_min, self.pos_scale)
        normalized_state = np.concatenate([normalized_pose, states[..., 6:7]], axis=-1)
        force = self._normalize(
            np.stack([item["force"] for item in items]), self.force_min, self.force_scale
        )
        condition = {
            "image_arm": self._images(items, "image_arm"),
            "image_fix": self._images(items, "image_fix"),
            "lowdim": torch.from_numpy(normalized_state).to(self.device),
            "force": torch.from_numpy(force).to(self.device),
        }
        shape = (len(items), self.action_horizon, 13)
        prior = torch.zeros(shape, device=self.device, dtype=torch.float32)
        source_rows = []
        for item in items:
            generator = torch.Generator(device=self.device).manual_seed(self._seed(item))
            source_rows.append(torch.randn(shape[1:], generator=generator, device=self.device))
        source = torch.stack(source_rows)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            prediction, _ = self.policy.sample(
                prior,
                x1=source,
                solver="euler",
                sample_steps=self.sample_steps,
                use_ema=self.use_ema,
                condition_cfg=condition,
                w_cfg=1.0,
            )
        normalized = prediction[:, : self.execution_steps, :7].float().cpu().numpy()
        delta = (np.clip(normalized[..., :6], -1, 1) + 1) * self.action_scale / 2
        delta += self.action_min
        gripper = np.clip(normalized[..., 6:7], -1, 1)
        results = []
        for index, item in enumerate(items):
            pose = item["state"][-1, :6] + np.cumsum(delta[index], axis=0)
            pose[:, 3:6] = item["state"][-1, 3:6] + (
                pose[:, 3:6] - item["state"][-1, 3:6] + np.pi
            ) % (2.0 * np.pi) - np.pi
            actions = np.concatenate([pose, gripper[index]], axis=-1).astype(np.float32)
            results.append(
                {
                    "contract": CONTRACT,
                    "action_space": ACTION_SPACE,
                    "control_rate_hz": CONTROL_RATE_HZ,
                    "actions": actions,
                }
            )
        return results

    def _images(self, items: list[dict[str, Any]], key: str) -> torch.Tensor:
        values = np.stack([item[key] for item in items])
        tensor = torch.from_numpy(values).permute(0, 1, 4, 2, 3).float().div_(255.0)
        flat = tensor.reshape(-1, *tensor.shape[-3:])
        flat = F.interpolate(
            flat,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return ((flat.reshape(*tensor.shape[:2], 3, self.image_size, self.image_size) - 0.5) / 0.5).to(
            self.device
        )


@dataclass
class WorkItem:
    observation: dict[str, Any]
    future: asyncio.Future


class InferenceBatcher:
    def __init__(self, policy: ForceFlowPolicy, max_batch_size: int, window_ms: float):
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.window_seconds = window_ms / 1000.0
        self.queue: asyncio.Queue[WorkItem] = asyncio.Queue()

    async def submit(self, observation: dict[str, Any]) -> dict[str, Any]:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(WorkItem(observation, future))
        return await future

    async def run(self) -> None:
        while True:
            batch = [await self.queue.get()]
            await asyncio.sleep(self.window_seconds)
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                results = self.policy.infer_batch([item.observation for item in batch])
                for item, result in zip(batch, results):
                    item.future.set_result(result)
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)


async def serve(policy: ForceFlowPolicy, args: argparse.Namespace) -> None:
    batcher = InferenceBatcher(policy, args.max_batch_size, args.batch_window_ms)
    batch_task = asyncio.create_task(batcher.run())
    metadata = msgpack.packb(
        {
            "contract": CONTRACT,
            "model": "ForceFlow",
            "action_space": ACTION_SPACE,
            "control_rate_hz": CONTROL_RATE_HZ,
            "action_horizon": policy.action_horizon,
            "execution_steps": policy.execution_steps,
            "observation_horizon": policy.observation_horizon,
            "force_history": policy.force_history,
            "requires_explicit_history": policy.execution_steps > 1,
            "rollout_identity": "required top-level rollout_id",
        }
    )

    async def handler(websocket: websockets.asyncio.server.ServerConnection):
        history = ObservationHistory(policy.observation_horizon, policy.force_history)
        await websocket.send(metadata)
        while True:
            try:
                payload = await websocket.recv()
                request = msgpack.unpackb(payload, raw=False, object_hook=unpack_numpy)
                result = await batcher.submit(
                    history.append(request, require_explicit_history=policy.execution_steps > 1)
                )
                await websocket.send(msgpack.packb(result, default=pack_numpy))
            except websockets.ConnectionClosed:
                break
            except Exception:
                logging.exception("ForceFlow request failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="ForceFlow inference failed",
                )
                break

    try:
        async with websockets.asyncio.server.serve(
            handler, args.host, args.port, compression=None, max_size=None
        ) as server:
            logging.info("ForceFlow ready on %s:%d", args.host, args.port)
            await server.serve_forever()
    finally:
        batch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await batch_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-batch-size", type=int, default=10)
    parser.add_argument("--batch-window-ms", type=float, default=12.0)
    parser.add_argument("--expected-clients", type=int, default=100)
    parser.add_argument("--exit-after-idle-s", type=float, default=7200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = ForceFlowPolicy(args.config, args.checkpoint, args.device)
    asyncio.run(serve(policy, args))


if __name__ == "__main__":
    main()
