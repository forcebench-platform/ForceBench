#!/usr/bin/env python3
"""Extract the rack from NVIDIA's small warehouse digital twin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


SOURCE_PRIM = Sdf.Path("/Lab/Assets/rack")
OUTPUT_PRIM = Sdf.Path("/WarehouseRack/SourceUnits/Rack")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_bounds(stage: Usd.Stage, prim_path: Sdf.Path) -> tuple[list[float], list[float]]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing rack prim: {prim_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = [float(value) for value in extent.GetMin()]
    maximum = [float(value) for value in extent.GetMax()]
    return minimum, maximum


def remove_external_materials(stage: Usd.Stage) -> None:
    material_scopes = [
        prim.GetPath()
        for prim in stage.TraverseAll()
        if prim.GetName() == "Looks"
    ]
    for path in sorted(material_scopes, key=lambda item: len(str(item)), reverse=True):
        stage.RemovePrim(path)

    for prim in stage.TraverseAll():
        for relationship in list(prim.GetRelationships()):
            if relationship.GetName().startswith("material:binding"):
                prim.RemoveProperty(relationship.GetName())


def remove_physics(stage: Usd.Stage) -> None:
    """Keep the extracted rack visual-only; CoRiBench owns its shelf collider."""
    for prim in stage.TraverseAll():
        for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        for prop in list(prim.GetProperties()):
            if prop.GetName().startswith(("physics:", "physx")):
                prim.RemoveProperty(prop.GetName())


def apply_rack_material(stage: Usd.Stage) -> None:
    material = UsdShade.Material.Define(stage, "/WarehouseRack/RackMaterial")
    shader = UsdShade.Shader.Define(stage, "/WarehouseRack/RackMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.12, 0.16, 0.19)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.34)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.72)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    rack = stage.GetPrimAtPath(OUTPUT_PRIM)
    UsdShade.MaterialBindingAPI.Apply(rack).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


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
    root = UsdGeom.Xform.Define(stage, "/WarehouseRack")
    stage.SetDefaultPrim(root.GetPrim())
    # The rack meshes are authored in centimeters while their source ancestors
    # provide the conversion to the source stage's meter units. CopySpec only
    # copies the selected rack branch, so restore that inherited conversion on
    # the extracted asset's root.
    source_units = UsdGeom.Xform.Define(stage, "/WarehouseRack/SourceUnits")
    source_units.AddScaleOp().Set(Gf.Vec3d(100.0, 100.0, 100.0))
    UsdGeom.Xform.Define(stage, str(OUTPUT_PRIM))
    if not Sdf.CopySpec(
        source_stage.GetRootLayer(), SOURCE_PRIM, stage.GetRootLayer(), OUTPUT_PRIM
    ):
        raise RuntimeError(f"failed to copy {SOURCE_PRIM} from {source}")

    remove_external_materials(stage)
    remove_physics(stage)
    apply_rack_material(stage)
    stage.GetRootLayer().Save()

    output_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadNone)
    output_minimum, output_maximum = aligned_bounds(output_stage, OUTPUT_PRIM)
    source_size = [high - low for low, high in zip(source_minimum, source_maximum)]
    output_size = [high - low for low, high in zip(output_minimum, output_maximum)]
    metadata = {
        "source": str(source),
        "source_prim": str(SOURCE_PRIM),
        "source_sha256": sha256(source),
        "source_bounds_m": {"minimum": source_minimum, "maximum": source_maximum},
        "source_size_m": source_size,
        "output": str(output),
        "output_prim": str(OUTPUT_PRIM),
        "output_sha256": sha256(output),
        "output_bounds_m": {"minimum": output_minimum, "maximum": output_maximum},
        "output_size_m": output_size,
        "material": "local UsdPreviewSurface; external warehouse textures removed",
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
    print("CORIBENCH_WAREHOUSE_RACK=" + json.dumps(metadata, separators=(",", ":")))


if __name__ == "__main__":
    main()
