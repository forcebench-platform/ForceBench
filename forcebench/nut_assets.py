"""Lightweight checks for the external M16 geometry used by Nut Thread.

The two NVIDIA USD files are intentionally acquired separately.  Merely
finding the authored wrapper is insufficient: a missing or changed reference
would otherwise fail only after starting Isaac Sim.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


SOURCE_SHA256 = {
    "nut_m16.usd": "8ff4fa10680ab09b780e9ba1f468eadc06ab3b3a8557c196bc2fa02f799677d4",
    "bolt_m16.usd": "e822eb329d3ab2aad9bae57a405c60539ecea12bfa4d1af5d3e3609f1fe1f654",
}
WRAPPER_SOURCES = {"nut.usda": "nut_m16.usd", "bolt.usda": "bolt_m16.usd"}
WRAPPER_SHA256 = {
    "nut.usda": "17934b50528c06f9a58bbb438d0fd239d752f269bb8fb3683961cbadbc264600",
    "bolt.usda": "dde98ce014c5837b58c282a3634ca0212168e8e5f407d761382f9c319024a7ac",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_nut_assets(asset_dir: Path) -> dict:
    """Check exact source bytes and transparent wrappers without importing USD.

    This does not validate USD composition or contact physics.  Those checks
    must be performed in the supported Isaac Sim environment.
    """
    root = Path(asset_dir).expanduser().resolve()
    issues = []
    sources = {}
    wrappers = {}
    for name, expected in SOURCE_SHA256.items():
        source = root / name
        actual = sha256_file(source) if source.is_file() else None
        sources[name] = {"expected_sha256": expected, "actual_sha256": actual}
        if actual is None:
            issues.append(f"missing external source: {name}")
        elif actual != expected:
            issues.append(f"source SHA256 mismatch: {name}")
    for name, source_name in WRAPPER_SOURCES.items():
        relative = f"normalized/nut_thread/{name}"
        wrapper = root / relative
        if not wrapper.is_file():
            issues.append(f"missing authored wrapper: {relative}")
            wrappers[relative] = {"present": False}
            continue
        content = wrapper.read_text(encoding="utf-8")
        wrapper_sha256 = sha256_file(wrapper)
        references = re.findall(r"@([^@]+)@", content)
        expected_reference = f"../../{source_name}"
        wrappers[relative] = {"present": True, "references": references, "sha256": wrapper_sha256}
        if wrapper_sha256 != WRAPPER_SHA256[name]:
            issues.append(f"authored wrapper SHA256 mismatch: {relative}")
        if expected_reference not in references:
            issues.append(f"wrapper must reference {expected_reference}: {relative}")
        if "convexDecomposition" in content:
            issues.append(f"wrapper overrides source SDF with convex decomposition: {relative}")
        for reference in references:
            if not (wrapper.parent / reference).resolve().is_relative_to(root):
                issues.append(f"wrapper reference leaves asset directory: {relative}")
    return {
        "asset_dir": str(root),
        "ready": not issues,
        "sources": sources,
        "wrappers": wrappers,
        "issues": issues,
        "scope": "Source SHA256 and wrapper references only; GPU composition and physics require separate validation.",
    }


def validate_nut_assets(asset_dir: Path) -> None:
    report = inspect_nut_assets(asset_dir)
    if report["issues"]:
        raise FileNotFoundError(
            "Nut Thread assets are incomplete or differ from the pinned source: "
            + "; ".join(report["issues"])
            + ". Install your separately obtained NVIDIA source files with "
            "python tools/prepare_nut_assets.py --source-dir SOURCE --asset-dir ASSETS. "
            "See docs/nut_assets.md."
        )
