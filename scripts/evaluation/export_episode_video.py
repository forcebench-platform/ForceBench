#!/usr/bin/env python3
"""Export a recorded CoRiBench HDF5 episode as a main+wrist MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("episode", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists():
    parser.error(f"output already exists: {args.output}")

with h5py.File(args.episode, "r") as handle:
    metadata = json.loads(str(handle.attrs["metadata_json"]))
    main = handle["observations/images/main"]
    wrist = handle["observations/images/wrist"]
    states = handle["observations/state"]
    if len(main) != len(wrist) or len(main) != len(states):
        raise ValueError("main/wrist/state frame counts differ")
    fps = float(metadata.get("control_rate_hz", 15.0))
    seed = int(metadata["seed"])
    total = len(main)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (768, 512),
    )
    if not writer.isOpened():
        raise OSError(f"failed to open video writer: {args.output}")
    try:
        for index in range(total):
            main_bgr = cv2.cvtColor(np.asarray(main[index]), cv2.COLOR_RGB2BGR)
            wrist_bgr = cv2.cvtColor(np.asarray(wrist[index]), cv2.COLOR_RGB2BGR)
            main_bgr = cv2.resize(main_bgr, (512, 512), interpolation=cv2.INTER_AREA)
            wrist_bgr = cv2.resize(wrist_bgr, (256, 256), interpolation=cv2.INTER_AREA)
            canvas = np.full((512, 768, 3), 242, dtype=np.uint8)
            canvas[:, :512] = main_bgr
            canvas[:256, 512:] = wrist_bgr
            cv2.putText(canvas, "WRIST", (520, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
            lines = (
                "EXPERT DEMONSTRATION",
                f"seed: {seed}",
                f"frame: {index + 1}/{total}",
                f"time: {(index + 1) / fps:.1f}s",
                f"gripper: {(float(states[index, 6]) + 1.0) * 0.5:.2f}",
            )
            for line_index, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (520, 292 + 40 * line_index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (30, 30, 30),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(canvas)
    finally:
        writer.release()

result = {
    "episode": str(args.episode.resolve()),
    "output": str(args.output.resolve()),
    "seed": seed,
    "frames": total,
    "fps": fps,
    "duration_s": total / fps,
}
args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("CORIBENCH_EPISODE_VIDEO=" + json.dumps(result, separators=(",", ":")))
