#!/usr/bin/env python3
"""Convert a validated ForceBench HDF5 snapshot to LeRobot v2.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from forcebench.recording import validate_hdf5_episode
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resized_rgb(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an RGB image, got {image.shape}")
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def unwrap_action_rpy(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Represent an absolute RPY target by its shortest offset from state."""
    action = np.asarray(action, dtype=np.float32).copy()
    angular_delta = action[3:6].astype(np.float64) - state[3:6].astype(np.float64)
    angular_delta = (angular_delta + np.pi) % (2.0 * np.pi) - np.pi
    action[3:6] = state[3:6] + angular_delta.astype(np.float32)
    return action


def main() -> None:
    args = parse_args()
    if args.image_size < 16:
        raise ValueError("image-size must be at least 16")
    manifest_path = args.snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if len(records) != int(manifest.get("episodes", -1)) or not records:
        raise ValueError("snapshot manifest has an invalid episode list")

    output_path = args.output_root / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = {
        "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "eef_wrench": {"dtype": "float32", "shape": (6,), "names": ["eef_wrench"]},
        "image": {
            "dtype": "video",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_path,
        robot_type="franka",
        fps=int(manifest.get("control_rate_hz", 15)),
        features=features,
        image_writer_threads=8,
        image_writer_processes=0,
    )

    converted_frames = 0
    prompts: set[str] = set()
    for episode_index, record in enumerate(records):
        path = args.snapshot / "raw" / str(record["file"])
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"size mismatch: {path}")
        if args.verify_sha256 and sha256(path) != str(record["sha256"]):
            raise ValueError(f"SHA-256 mismatch: {path}")
        validate_hdf5_episode(path)
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("schema_version", "")) != "forcebench_hdf5":
                raise ValueError(f"unsupported non-canonical HDF5 schema: {path}")
            metadata = json.loads(str(handle.attrs["metadata_json"]))
            if not bool(handle.attrs["success"]) or not bool(metadata.get("success")):
                raise ValueError(f"snapshot contains a non-successful episode: {path}")
            frames = int(handle.attrs["num_frames"])
            if frames != int(record["frames"]):
                raise ValueError(f"frame-count mismatch: {path}")
            prompt = str(metadata["prompt"])
            prompts.add(prompt)
            state = handle["observations/state"]
            actions = handle["actions/canonical"]
            wrench = handle["forces/control/eef_wrench_joint_raw"]
            main_images = handle["observations/images/main"]
            wrist_images = handle["observations/images/wrist"]
            expected_shapes = ((frames, 7), (frames, 7), (frames, 6))
            if (state.shape, actions.shape, wrench.shape) != expected_shapes:
                raise ValueError(f"invalid state/action/wrench shapes in {path}")

            for frame_index in range(frames):
                state_frame = np.asarray(state[frame_index], dtype=np.float32)
                action_frame = unwrap_action_rpy(actions[frame_index], state_frame)
                wrench_frame = np.asarray(wrench[frame_index], dtype=np.float32)
                if not all(np.isfinite(value).all() for value in (state_frame, action_frame, wrench_frame)):
                    raise ValueError(f"non-finite frame {frame_index} in {path}")
                dataset.add_frame(
                    {
                        "state": state_frame,
                        "actions": action_frame,
                        "eef_wrench": wrench_frame,
                        "image": resized_rgb(main_images[frame_index], args.image_size),
                        "wrist_image": resized_rgb(wrist_images[frame_index], args.image_size),
                        "task": prompt,
                    }
                )
            dataset.save_episode()
            converted_frames += frames
            print(
                f"FORCEBENCH_LEROBOT_EPISODE={episode_index + 1}/{len(records)},"
                f"file={path.name},frames={frames}",
                flush=True,
            )

    if converted_frames != int(manifest["frames"]):
        raise ValueError(f"converted {converted_frames} frames, expected {manifest['frames']}")
    conversion = {
        "source_snapshot": str(args.snapshot.resolve()),
        "repo_id": args.repo_id,
        "episodes": len(records),
        "frames": converted_frames,
        "fps": int(manifest.get("control_rate_hz", 15)),
        "training_image_shape": [args.image_size, args.image_size, 3],
        "prompts": sorted(prompts),
        "source_manifest": manifest,
    }
    (output_path / "forcebench_conversion.json").write_text(
        json.dumps(conversion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("FORCEBENCH_LEROBOT_COMPLETE=" + json.dumps(conversion, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
