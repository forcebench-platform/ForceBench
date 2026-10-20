import pytest

pytest.importorskip("pxr", reason="USD scene contract is additionally executed in the Isaac environment")
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from forcebench.visual_scene import remove_background_physics


def test_background_physics_is_removed_without_changing_geometry_or_task_body():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,"/Room")
    visual = UsdGeom.Cube.Define(stage,"/Room/FridgeDoor")
    visual.AddTranslateOp().Set(Gf.Vec3d(1,2,3))
    UsdPhysics.RigidBodyAPI.Apply(visual.GetPrim())
    UsdPhysics.CollisionAPI.Apply(visual.GetPrim())
    UsdPhysics.MassAPI.Apply(visual.GetPrim()).CreateMassAttr(5)
    UsdPhysics.DriveAPI.Apply(visual.GetPrim(), "angular")
    joint = UsdPhysics.RevoluteJoint.Define(stage,"/Room/DoorHinge")
    group = UsdPhysics.CollisionGroup.Define(stage,"/Room/DoorCollisionGroup")
    visual.GetPrim().AddAppliedSchema("PhysxLegacyRoomAPI:unused")
    task = UsdGeom.Cube.Define(stage,"/TaskManipuland")
    UsdPhysics.RigidBodyAPI.Apply(task.GetPrim())
    matrix = UsdGeom.Xformable(visual).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    report = remove_background_physics(stage,"/Room")
    assert len(report["removed_apis"]) == 5
    assert "PhysicsDriveAPI:angular" not in visual.GetPrim().GetAppliedSchemas()
    assert not joint.GetPrim().IsActive()
    assert not group.GetPrim().IsActive()
    assert "PhysxLegacyRoomAPI:unused" not in visual.GetPrim().GetMetadata("apiSchemas").GetAppliedItems()
    assert not visual.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert not visual.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Xformable(visual).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) == matrix
    assert task.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
