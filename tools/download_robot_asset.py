#!/usr/bin/env python3
"""Download and verify the audited NVIDIA Factory robot export.

The asset retains its upstream NVIDIA terms; ForceBench's MIT license does not
apply to it. Network access occurs only when download_robot() or this CLI runs.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import tempfile
import urllib.request


SOURCE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/IsaacLab/Factory/franka_mimic.usd"
)
OFFICIAL_BYTES = 7_075_031
OFFICIAL_SHA256 = "86d0017a093669b222a8c640bf581cc3e3715035aa7c596600f478da011f6c8e"
HISTORICAL_BYTES = 7_073_847
HISTORICAL_SHA256 = "6ca52983cbede63f4caf1557896213e732f14fd8da9234658a6981ec2515b1bf"


def _existing_identity(destination: Path) -> dict | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    if destination.is_file():
        size = destination.stat().st_size
        for export, expected_size, expected_sha256 in (
            ("official", OFFICIAL_BYTES, OFFICIAL_SHA256),
            ("historical", HISTORICAL_BYTES, HISTORICAL_SHA256),
        ):
            if size == expected_size:
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                if digest == expected_sha256:
                    return {"bytes": size, "sha256": digest, "export": export}
    raise ValueError(f"Existing robot asset does not match either audited export; preserved: {destination}")


def download_robot(source_dir: Path, *, timeout: float = 60.0) -> dict:
    source_dir = source_dir.expanduser().resolve()
    destination = source_dir / "franka_mimic.usd"
    existing = _existing_identity(destination)
    report = {
        "asset": "franka_mimic.usd", "path": str(destination), "source_url": SOURCE_URL,
        "terms": "NVIDIA asset terms apply; this asset is not covered by ForceBench's MIT license.",
    }
    if existing is not None:
        return {**report, "status": "already_present", **existing}

    source_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".franka_mimic-", suffix=".part", dir=source_dir,
                                         delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(SOURCE_URL, timeout=timeout) as response:
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None and int(declared_length) != OFFICIAL_BYTES:
                    raise ValueError(f"Unexpected Content-Length: {declared_length}; expected {OFFICIAL_BYTES}")
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > OFFICIAL_BYTES:
                        raise ValueError(f"Robot download exceeds expected {OFFICIAL_BYTES} bytes")
                    digest.update(block)
                    output.write(block)
            actual_sha256 = digest.hexdigest()
            if size != OFFICIAL_BYTES or actual_sha256 != OFFICIAL_SHA256:
                raise ValueError(
                    f"Robot download failed verification: bytes={size}, sha256={actual_sha256}; "
                    f"expected bytes={OFFICIAL_BYTES}, sha256={OFFICIAL_SHA256}"
                )

        # Both paths share a directory/filesystem. A hard link publishes the
        # verified bytes atomically and cannot replace a concurrent user file.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _existing_identity(destination)
            if existing is None:
                raise
            return {**report, "status": "already_present", **existing}
        return {**report, "status": "downloaded", "export": "official",
                "bytes": size, "sha256": actual_sha256}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Place franka_mimic.usd here for tools/prepare_assets.py")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    args = parser.parse_args()
    try:
        report = download_robot(args.source_dir, timeout=args.timeout)
    except (OSError, ValueError, http.client.HTTPException) as error:
        print(json.dumps({"status": "failed", "asset": "franka_mimic.usd",
                          "source_dir": str(args.source_dir), "source_url": SOURCE_URL,
                          "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
