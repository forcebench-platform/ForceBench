#!/usr/bin/env python3
"""Build a visual-only, single-burner derivative of Lightwheel Stovetop012.

The source asset stores the black glass slab and all five burner assemblies in
one mesh.  This builder keeps only disconnected mesh components belonging to
the center burner and center control markings, then replaces the shared slab
with a task-owned charcoal glass cuboid.  The source USD remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


SOURCE_SHA256 = "a18e9b9e2cc214516a11e6a4f8403c23e356790b907ff9a77ed4980b69b9dc1c"
SOURCE_BODY_PATH = "/root/Stovetop012_Body001"
SOURCE_LOOKS_PATH = "/root/Looks"
OUTPUT_NAME = "Stovetop012_single_center.usd"
OUTPUT_ROOT_PATH = "/Stovetop012SingleCenter"

# The center grate reaches x ~= +/-0.121 m in source coordinates.  A 0.145 m
# half-width leaves a realistic 24 mm total shoulder without retaining either
# side burner.  The original front/back depth and thickness are preserved.
CROP_HALF_WIDTH_M = 0.145
KNOB_BANK_Y_MAX_M = -0.18
CENTER_MARKING_HALF_WIDTH_M = 0.035


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connected_face_components(
    face_counts: list[int], face_indices: list[int]
) -> list[list[int]]:
    parent = list(range(len(face_counts)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    vertex_faces: dict[int, list[int]] = defaultdict(list)
    cursor = 0
    for face_index, count in enumerate(face_counts):
        for vertex_index in face_indices[cursor : cursor + count]:
            vertex_faces[int(vertex_index)].append(face_index)
        cursor += count
    for faces in vertex_faces.values():
        for face_index in faces[1:]:
            union(faces[0], face_index)

    components: dict[int, list[int]] = defaultdict(list)
    for face_index in range(len(face_counts)):
        components[find(face_index)].append(face_index)
    return sorted(components.values(), key=len, reverse=True)


def _component_bounds(
    faces: list[int],
    face_counts: list[int],
    face_offsets: list[int],
    face_indices: list[int],
    points,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    vertices: set[int] = set()
    for face_index in faces:
        start = face_offsets[face_index]
        count = face_counts[face_index]
        vertices.update(int(value) for value in face_indices[start : start + count])
    component_points = [points[index] for index in vertices]
    minimum = tuple(
        min(float(point[axis]) for point in component_points) for axis in range(3)
    )
    maximum = tuple(
        max(float(point[axis]) for point in component_points) for axis in range(3)
    )
    return minimum, maximum


def _keep_component(minimum: tuple[float, ...], maximum: tuple[float, ...]) -> bool:
    if minimum[0] < -CROP_HALF_WIDTH_M or maximum[0] > CROP_HALF_WIDTH_M:
        return False
    if minimum[1] < KNOB_BANK_Y_MAX_M:
        return (
            minimum[0] >= -CENTER_MARKING_HALF_WIDTH_M
            and maximum[0] <= CENTER_MARKING_HALF_WIDTH_M
        )
    return True


def _copy_face_varying_primvar(
    source_mesh: UsdGeom.Mesh,
    output_mesh: UsdGeom.Mesh,
    name: str,
    selected_corner_indices: list[int],
) -> None:
    source = UsdGeom.PrimvarsAPI(source_mesh).GetPrimvar(name)
    if not source or not source.HasValue():
        return
    output = UsdGeom.PrimvarsAPI(output_mesh).CreatePrimvar(
        name,
        source.GetTypeName(),
        source.GetInterpolation(),
        source.GetElementSize(),
    )
    output.Set(source.Get())
    if source.IsIndexed():
        source_indices = source.GetIndices()
        output.SetIndices([source_indices[index] for index in selected_corner_indices])


def _create_glass_slab(
    stage: Usd.Stage,
    source_minimum: tuple[float, ...],
    source_maximum: tuple[float, ...],
) -> None:
    material = UsdShade.Material.Define(stage, f"{OUTPUT_ROOT_PATH}/Looks/CharcoalGlass")
    shader = UsdShade.Shader.Define(
        stage, f"{OUTPUT_ROOT_PATH}/Looks/CharcoalGlass/PreviewSurface"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.025, 0.032, 0.040)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.18)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.16)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    cube = UsdGeom.Cube.Define(stage, f"{OUTPUT_ROOT_PATH}/CharcoalGlassSlab")
    cube.CreateSizeAttr(1.0)
    center_y = 0.5 * (source_minimum[1] + source_maximum[1])
    center_z = 0.5 * (source_minimum[2] + source_maximum[2])
    depth = source_maximum[1] - source_minimum[1]
    height = source_maximum[2] - source_minimum[2]
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, center_y, center_z))
    xform.AddScaleOp().Set(Gf.Vec3f(2.0 * CROP_HALF_WIDTH_M, depth, height))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def build(source_path: Path, output_path: Path) -> dict[str, object]:
    actual_sha256 = _sha256(source_path)
    if actual_sha256 != SOURCE_SHA256:
        raise RuntimeError(
            f"unexpected Stovetop012 source hash {actual_sha256}; "
            f"expected {SOURCE_SHA256}"
        )

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise RuntimeError(f"could not open {source_path}")
    source_mesh = UsdGeom.Mesh(source_stage.GetPrimAtPath(SOURCE_BODY_PATH))
    if not source_mesh:
        raise RuntimeError(f"missing source mesh {SOURCE_BODY_PATH}")

    points = source_mesh.GetPointsAttr().Get()
    face_counts = list(source_mesh.GetFaceVertexCountsAttr().Get())
    face_indices = list(source_mesh.GetFaceVertexIndicesAttr().Get())
    face_offsets: list[int] = []
    cursor = 0
    for count in face_counts:
        face_offsets.append(cursor)
        cursor += int(count)

    components = _connected_face_components(face_counts, face_indices)
    selected_faces: list[int] = []
    selected_components: list[dict[str, object]] = []
    slab_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    slab_width = -1.0
    for component in components:
        minimum, maximum = _component_bounds(
            component, face_counts, face_offsets, face_indices, points
        )
        component_width = maximum[0] - minimum[0]
        if component_width > slab_width:
            slab_bounds = (minimum, maximum)
            slab_width = component_width
        if not _keep_component(minimum, maximum):
            continue
        selected_faces.extend(component)
        selected_components.append(
            {"faces": len(component), "minimum": minimum, "maximum": maximum}
        )
    selected_faces.sort()
    if not selected_faces:
        raise RuntimeError("center-component selection unexpectedly produced no faces")

    selected_vertices: set[int] = set()
    selected_corner_indices: list[int] = []
    output_counts: list[int] = []
    for face_index in selected_faces:
        start = face_offsets[face_index]
        count = int(face_counts[face_index])
        selected_corner_indices.extend(range(start, start + count))
        output_counts.append(count)
        selected_vertices.update(int(value) for value in face_indices[start : start + count])
    sorted_vertices = sorted(selected_vertices)
    vertex_remap = {old: new for new, old in enumerate(sorted_vertices)}
    output_points = [points[index] for index in sorted_vertices]
    output_indices = [vertex_remap[int(face_indices[index])] for index in selected_corner_indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    output_stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(output_stage, 1.0)
    UsdGeom.SetStageUpAxis(output_stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(output_stage, OUTPUT_ROOT_PATH)
    output_stage.SetDefaultPrim(root.GetPrim())
    output_stage.SetMetadata(
        "customLayerData",
        {
            "assetVersion": "stovetop-single-center-v1",
            "sourceAsset": "Lightwheel Kitchen v1.1 Stovetop012",
            "sourceSha256": SOURCE_SHA256,
            "cropHalfWidthM": str(CROP_HALF_WIDTH_M),
            "purpose": "visual-only center burner derivative",
        },
    )

    UsdGeom.Scope.Define(output_stage, f"{OUTPUT_ROOT_PATH}/Looks")
    Sdf.CopySpec(
        source_stage.GetRootLayer(),
        SOURCE_LOOKS_PATH,
        output_stage.GetRootLayer(),
        f"{OUTPUT_ROOT_PATH}/Looks/Source",
    )
    output_mesh = UsdGeom.Mesh.Define(output_stage, f"{OUTPUT_ROOT_PATH}/CenterComponents")
    output_mesh.CreatePointsAttr(output_points)
    output_mesh.CreateFaceVertexCountsAttr(output_counts)
    output_mesh.CreateFaceVertexIndicesAttr(output_indices)
    output_mesh.CreateSubdivisionSchemeAttr(
        source_mesh.GetSubdivisionSchemeAttr().Get() or UsdGeom.Tokens.none
    )
    output_mesh.CreateDoubleSidedAttr(source_mesh.GetDoubleSidedAttr().Get())
    output_mesh.CreateOrientationAttr(source_mesh.GetOrientationAttr().Get())
    output_mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(output_points))
    _copy_face_varying_primvar(source_mesh, output_mesh, "st", selected_corner_indices)
    _copy_face_varying_primvar(source_mesh, output_mesh, "normals", selected_corner_indices)
    source_material = UsdShade.Material(
        output_stage.GetPrimAtPath(
            f"{OUTPUT_ROOT_PATH}/Looks/Source/MI_Stovetop012_001"
        )
    )
    if not source_material:
        raise RuntimeError("copied source material is missing")
    UsdShade.MaterialBindingAPI.Apply(output_mesh.GetPrim()).Bind(source_material)

    if slab_bounds is None:
        raise RuntimeError("could not identify the shared source slab component")
    _create_glass_slab(output_stage, *slab_bounds)
    output_stage.GetRootLayer().Save()

    return {
        "output": str(output_path),
        "source_faces": len(face_counts),
        "selected_faces": len(selected_faces),
        "selected_points": len(output_points),
        "selected_components": len(selected_components),
        "crop_half_width_m": CROP_HALF_WIDTH_M,
        "output_sha256": _sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "asset_dir",
        type=Path,
        help="Path to assets/lightwheel_kitchen/Stovetop012",
    )
    args = parser.parse_args()
    result = build(args.asset_dir / "Stovetop012.usd", args.asset_dir / OUTPUT_NAME)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
