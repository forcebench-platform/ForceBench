#!/usr/bin/env python3
"""Verify composed Nut USD structure and unchanged source SDF meshes.

Run with a Python environment providing pxr (for example Isaac Sim Python).
No simulator, GPU context or physical scene is started by this check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forcebench.nut_assets import validate_nut_assets


def verify(asset_dir: Path) -> dict:
    validate_nut_assets(asset_dir)
    from pxr import Usd, UsdGeom, UsdPhysics

    root = asset_dir.expanduser().resolve()
    rows = {}

    def collisions(stage):
        found = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError(f"Unexpected non-mesh collision: {prim.GetPath()}")
            mesh = UsdGeom.Mesh(prim)
            topology = {
                "points": repr(list(mesh.GetPointsAttr().Get())),
                "face_vertex_counts": list(mesh.GetFaceVertexCountsAttr().Get()),
                "face_vertex_indices": list(mesh.GetFaceVertexIndicesAttr().Get()),
            }
            found.append({
                "path": str(prim.GetPath()),
                "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                "vertices": len(mesh.GetPointsAttr().Get()),
                "geometry_sha256": hashlib.sha256(json.dumps(topology, sort_keys=True).encode()).hexdigest(),
            })
        return found

    for name, source_name, root_name, z_offset, grasp_z, mate_z in (
        ("nut.usda", "nut_m16.usd", "Nut", -0.010, 0.0065, 0.0),
        ("bolt.usda", "bolt_m16.usd", "Bolt", 0.0, 0.0225, 0.032),
    ):
        stage = Usd.Stage.Open(str(root / "normalized/nut_thread" / name))
        source_stage = Usd.Stage.Open(str(root / source_name))
        if not stage or not source_stage:
            raise ValueError(f"USD stage cannot be opened: {name}")
        default = stage.GetDefaultPrim()
        if str(default.GetPath()) != f"/{root_name}":
            raise ValueError(f"Unexpected default prim in {name}")
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
            raise ValueError(f"Unexpected units or axis in {name}")
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        if bodies != [default] or joints:
            raise ValueError(f"Expected one root rigid body and no active joints: {name}")
        source = stage.GetPrimAtPath(f"/{root_name}/Source")
        offset = tuple(source.GetAttribute("xformOp:translate:coribenchNormalization").Get())
        if offset != (0.0, 0.0, z_offset):
            raise ValueError(f"Unexpected support-plane normalization in {name}: {offset}")
        for frame, expected_z in (("Support", 0.0), ("Grasp", grasp_z), ("Mate", mate_z)):
            prim = stage.GetPrimAtPath(f"/{root_name}/Frames/{frame}")
            if not prim or tuple(prim.GetAttribute("xformOp:translate").Get()) != (0.0, 0.0, expected_z):
                raise ValueError(f"Missing or changed {frame} frame in {name}")
        actual, expected = collisions(stage), collisions(source_stage)
        if not actual or any(p["approximation"] != "sdf" for p in actual + expected):
            raise ValueError(f"Expected source SDF collision meshes in {name}")
        fingerprint = lambda rows: sorted(row["geometry_sha256"] for row in rows)
        if fingerprint(actual) != fingerprint(expected):
            raise ValueError(f"Wrapper changes collision mesh points/topology in {name}")
        rows[name] = {"root_rigid_body": str(default.GetPath()), "active_joints": 0,
                      "source_translation_m": list(offset), "collisions": actual}
    return {"passed": True, "wrappers": rows,
            "scope": "USD composition and source-mesh equivalence; does not verify GPU physics or episode success."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.asset_dir)
    except ImportError:
        print("pxr is required; run this check with the supported Isaac Sim Python environment.", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
