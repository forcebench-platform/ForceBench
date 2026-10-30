#!/usr/bin/env python3
"""Restore two historical Lightwheel layers from their pinned public originals.

The public originals are ZIP members from Lightwheel Kitchen release commit
10b0f7a86135a27e8c2fba9c690f6bf5c4f06bb1 (ZIP SHA256
721fb9e47d09b2c43a7db396c1d63c659e987d5925db0a5e78cf4fd40cd0b0b0).
The 2026-10-27 typed USD audit found only the UV edits below for Stovetop012,
and removal of two metrics-assembler sublayers for KitchenRoom. All other
authored fields match. Source assets retain Lightwheel's CC BY-NC 4.0 terms.

Requires usd-core==25.11. No simulator or GPU context is started. The original
crate is copied, edited and saved incrementally; Export or Flatten would change
its representation and fail the historical output SHA256 check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ASSET_PINS = {
    "stove": {
        "source_bytes": 2690367,
        "source_sha256": "2eaea2bc322e428d72043887519cbb124f64c4b7983fcb39b3b03494d058e0b4",
        "output_bytes": 2932733,
        "output_sha256": "a18e9b9e2cc214516a11e6a4f8403c23e356790b907ff9a77ed4980b69b9dc1c",
    },
    "kitchen": {
        "source_bytes": 177380414,
        "source_sha256": "a38b267437a516e0c7be1306cb66dd1bc0b644d60d6a07fae94e90062a0cad1f",
        "output_bytes": 177385431,
        "output_sha256": "2c9d4031b7af47eac53ac2ab77ad041e1fd3069b5a31eb8e2abeadc8fb10c6b7",
    },
}
STOVE_UV_PATH = "/root/Stovetop012_Body001.primvars:st"
# Half-open indices into the unchanged, 69,702-element face-varying UV array.
STOVE_UV_INDEX_RANGES = ((67185, 67896), (68253, 69072), (69126, 69408), (69498, 69675))
KITCHEN_SUBLAYERS = (
    "./UnitsAdjust-7f2457e887141c40f0b8d9b67696ec8f.metricsAssembler",
    "./UnitsAdjust-1d4c359600146ac00d02ca8405786de0.metricsAssembler",
)


def _identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return path.stat().st_size, digest.hexdigest()


def _matches(path: Path, pin: dict, prefix: str) -> bool:
    return path.is_file() and _identity(path) == (pin[f"{prefix}_bytes"], pin[f"{prefix}_sha256"])


def _patch_layer(path: Path, asset: str) -> dict:
    from pxr import Gf, Sdf, Usd, Vt

    if Usd.GetVersion() != (0, 25, 11):
        raise ValueError("Exact Lightwheel restoration requires usd-core==25.11")
    layer = Sdf.Layer.FindOrOpen(str(path))
    if layer is None:
        raise ValueError("Public Lightwheel layer cannot be opened")
    if asset == "stove":
        uv = layer.GetAttributeAtPath(STOVE_UV_PATH)
        indices = layer.GetAttributeAtPath(STOVE_UV_PATH + ":indices")
        if uv is None or indices is None or len(uv.default) != 17664 or len(indices.default) != 69702:
            raise ValueError("Unexpected Stovetop012 UV arrays")
        uv.default = Vt.Vec2fArray([*uv.default, Gf.Vec2f(0.25, 0.5)])
        patched = list(indices.default)
        for start, end in STOVE_UV_INDEX_RANGES:
            patched[start:end] = [17664] * (end - start)
        indices.default = Vt.IntArray(patched)
        changes = {"uv_added": [0.25, 0.5], "changed_uv_indices": 1989}
    else:
        if tuple(layer.subLayerPaths) != KITCHEN_SUBLAYERS:
            raise ValueError("Unexpected KitchenRoom sublayers")
        layer.subLayerPaths = []
        changes = {"removed_sublayers": list(KITCHEN_SUBLAYERS)}
    layer.Save()
    return changes


def restore(source: Path, output: Path, *, asset: str) -> dict:
    if asset not in ASSET_PINS:
        raise ValueError(f"Unknown Lightwheel layer: {asset}")
    pin = ASSET_PINS[asset]
    source = Path(source).expanduser().resolve()
    # Do not resolve the final output symlink: an existing link is user content.
    output = Path(output).expanduser().absolute()
    if not _matches(source, pin, "source"):
        raise ValueError(f"Public {asset} source differs from the pinned ZIP member")
    report = {"asset": asset, "source": str(source), "output": str(output),
              "bytes": pin["output_bytes"], "sha256": pin["output_sha256"]}
    if output.exists() or output.is_symlink():
        if not output.is_symlink() and _matches(output, pin, "output"):
            return {**report, "status": "already_present"}
        raise FileExistsError(f"Existing output is different; preserved: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".lightwheel-", suffix=".usd", dir=output.parent,
                                     delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        shutil.copyfile(source, pending)
        changes = _patch_layer(pending, asset)
        if not _matches(pending, pin, "output"):
            raise ValueError(f"Restored {asset} layer differs from the historical byte pin")
        # Publish verified bytes without replacing a file created concurrently.
        try:
            os.link(pending, output)
        except FileExistsError:
            if not output.is_symlink() and _matches(output, pin, "output"):
                return {**report, "status": "already_present"}
            raise FileExistsError(f"Concurrent output is different; preserved: {output}") from None
    finally:
        pending.unlink(missing_ok=True)
    return {**report, "status": "restored", "changes": changes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset", required=True, choices=sorted(ASSET_PINS))
    args = parser.parse_args()
    try:
        report = restore(args.source, args.output, asset=args.asset)
    except (ImportError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
