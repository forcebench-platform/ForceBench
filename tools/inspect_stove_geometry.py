"""Inspect disconnected stove mesh components without importing task constants.

Requires the ``pxr`` module supplied by the Isaac Sim / USD Python environment.
The output contains measured bounds and hashes, not the USD asset itself.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


def inspect_geometry(usd_path: Path) -> dict[str, object]:
    stage = Usd.Stage.Open(str(usd_path.resolve()))
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {usd_path}")
    cache = UsdGeom.XformCache()
    unit = UsdGeom.GetStageMetersPerUnit(stage)
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        if not points or not indices:
            continue
        parent = list(range(len(points)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + count]
            cursor += count
            for index in face[1:]:
                left, right = find(face[0]), find(index)
                if left != right:
                    parent[right] = left
        groups = defaultdict(list)
        for index in set(indices):
            groups[find(index)].append(index)
        transform = cache.GetLocalToWorldTransform(prim)
        components = []
        for vertex_ids in groups.values():
            transformed = [
                transform.Transform(Gf.Vec3d(*points[index])) * unit
                for index in vertex_ids
            ]
            low = [min(point[axis] for point in transformed) for axis in range(3)]
            high = [max(point[axis] for point in transformed) for axis in range(3)]
            size = [end - start for start, end in zip(low, high)]
            center = [(start + end) / 2.0 for start, end in zip(low, high)]
            # Inspect the original single-burner crop. This filter does not use
            # the task's burner center or choose a component by proximity to it.
            if abs(center[0]) < 0.145 and size[0] > 0.01 and size[1] > 0.01:
                components.append({
                    "vertices": len(vertex_ids),
                    "min_m": low,
                    "max_m": high,
                    "center_m": center,
                    "size_m": size,
                })
        meshes.append({
            "path": str(prim.GetPath()),
            "vertices": len(points),
            "faces": len(counts),
            "components": sorted(components, key=lambda item: item["center_m"][1]),
        })
    return {
        "file": str(usd_path),
        "sha256": hashlib.sha256(usd_path.read_bytes()).hexdigest(),
        "meters_per_unit": unit,
        "meshes": meshes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, help="single-center stove USD asset")
    args = parser.parse_args()
    print(json.dumps(inspect_geometry(args.usd), indent=2))


if __name__ == "__main__":
    main()
