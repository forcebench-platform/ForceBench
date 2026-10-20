#!/usr/bin/env python3
"""Normalize NVIDIA ConveyorBelt_A04 into a self-contained visual asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


SOURCE_PRIM = Sdf.Path("/World")
OUTPUT_PRIM = Sdf.Path("/ConveyorA04")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_bounds(stage: Usd.Stage, prim_path: Sdf.Path) -> tuple[list[float], list[float]]:
    prim = stage.GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return [float(x) for x in extent.GetMin()], [float(x) for x in extent.GetMax()]


def strip_external_materials_and_physics(stage: Usd.Stage) -> None:
    looks = [prim.GetPath() for prim in stage.TraverseAll() if prim.GetName() == "Looks"]
    for path in sorted(looks, key=lambda value: len(str(value)), reverse=True):
        stage.RemovePrim(path)
    for prim in stage.TraverseAll():
        for relationship in list(prim.GetRelationships()):
            if relationship.GetName().startswith("material:binding"):
                prim.RemoveProperty(relationship.GetName())
        for api in (
            UsdPhysics.CollisionAPI,
            UsdPhysics.MeshCollisionAPI,
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
        ):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        for prop in list(prim.GetProperties()):
            if prop.GetName().startswith(("physics:", "physx")):
                prim.RemoveProperty(prop.GetName())


def material(stage: Usd.Stage, name: str, color, roughness: float, metallic: float):
    path = f"/ConveyorA04/Materials/{name}"
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def bind(stage: Usd.Stage, prim_path: str, surface: UsdShade.Material) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            surface,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )


def apply_materials(stage: Usd.Stage) -> None:
    frame = material(stage, "Frame", (0.055, 0.13, 0.19), 0.34, 0.62)
    belt = material(stage, "Belt", (0.018, 0.024, 0.030), 0.76, 0.04)
    rollers = material(stage, "Rollers", (0.32, 0.35, 0.37), 0.23, 0.84)
    accent = material(stage, "Accent", (0.93, 0.51, 0.035), 0.42, 0.24)
    bind(stage, str(OUTPUT_PRIM), frame)
    bind(stage, f"{OUTPUT_PRIM}/Belt", belt)
    bind(stage, f"{OUTPUT_PRIM}/Rubberbands", belt)
    bind(stage, f"{OUTPUT_PRIM}/Rollers", rollers)
    bind(stage, f"{OUTPUT_PRIM}/SM_ConveyorBelt_A04_Decal_02", accent)


def build(source: Path, output: Path, metadata_path: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    source_stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    if source_stage is None:
        raise RuntimeError(f"could not open source stage: {source}")
    source_minimum, source_maximum = aligned_bounds(source_stage, SOURCE_PRIM)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, str(OUTPUT_PRIM))
    stage.SetDefaultPrim(root.GetPrim())
    if not Sdf.CopySpec(
        source_stage.GetRootLayer(), SOURCE_PRIM, stage.GetRootLayer(), OUTPUT_PRIM
    ):
        raise RuntimeError(f"failed to copy {SOURCE_PRIM} from {source}")
    strip_external_materials_and_physics(stage)
    apply_materials(stage)
    stage.GetRootLayer().Save()

    output_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadNone)
    output_minimum, output_maximum = aligned_bounds(output_stage, OUTPUT_PRIM)
    metadata = {
        "source": str(source),
        "source_url": (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
            "Assets/Isaac/6.0/Isaac/Props/Conveyors/ConveyorBelt_A04.usd"
        ),
        "source_sha256": sha256(source),
        "source_bounds_m": {"minimum": source_minimum, "maximum": source_maximum},
        "source_size_m": [b - a for a, b in zip(source_minimum, source_maximum)],
        "output": str(output),
        "output_sha256": sha256(output),
        "output_bounds_m": {"minimum": output_minimum, "maximum": output_maximum},
        "output_size_m": [b - a for a, b in zip(output_minimum, output_maximum)],
        "material": "local UsdPreviewSurface set; external materials and physics removed",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    metadata = build(args.source, args.output, args.metadata)
    print("CORIBENCH_CONVEYOR_A04=" + json.dumps(metadata, separators=(",", ":")))


if __name__ == "__main__":
    main()
