"""CPU PyAV decoding with an explicit decoder-thread limit.

The normal call signature matches LeRobot decode_video_frames. The PyAV path
retains installed LeRobot/torchvision seeking, timestamp selection, RGB conversion,
and float32 /255 normalization. Only codec_context.thread_count is constrained.
Non-PyAV backends are delegated unchanged to an optional installed LeRobot
implementation. The formal PyAV path does not require LeRobot to be installed.

Reviewed baseline:
  lerobot/common/datasets/video_utils.py SHA256
    0bcc946fb5e176aa0a8f547db81f6ad725746aa799f481a7c66160b2fce10e1b
  torchvision/io/video_reader.py SHA256
    57bbe017618d322925521c2f9e42c92d65776a4fa494a90bec4048846b8d983d
"""
from pathlib import Path

import torch
import torchvision


def decode_video_frames(video_path: Path | str, timestamps: list[float],
                        tolerance_s: float, backend: str | None = None, *,
                        decoder_threads: int = 1) -> torch.Tensor:
    if backend != "pyav":
        from lerobot.common.datasets.video_utils import decode_video_frames as _original_decode

        return _original_decode(video_path, timestamps, tolerance_s, backend)
    if not isinstance(decoder_threads, int) or decoder_threads < 1:
        raise ValueError("decoder_threads must be a positive integer")
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(video_path), "video")
    # VideoReader(num_threads=...) is ignored by its PyAV implementation. Set
    # the actual codec context before seek/first decode opens the decoder.
    loaded_frames, loaded_ts = [], []
    try:
        first_ts, last_ts = min(timestamps), max(timestamps)
        reader.container.streams.video[0].codec_context.thread_count = decoder_threads
        reader.seek(first_ts, keyframes_only=True)
        for frame in reader:
            current_ts = frame["pts"]
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
    finally:
        reader.container.close()
    reader = None
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    minimum, argmin = dist.min(1)
    assert (minimum < tolerance_s).all(), (
        f"One or several query timestamps violate tolerance: {minimum=} > {tolerance_s=}; "
        f"{timestamps=}; {loaded_ts=}; {video_path=}"
    )
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin])
    closest_frames = closest_frames.type(torch.float32) / 255
    assert len(timestamps) == len(closest_frames)
    return closest_frames
