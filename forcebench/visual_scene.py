"""Keep the surrounding room visual-only before physics replication."""
from __future__ import annotations


def remove_background_physics(stage, root_path: str) -> dict:
    """Remove inherited physics APIs/joints from the decorative room only.

    Disabling collisions alone leaves authored refrigerator rigid bodies for
    PhysX to replicate. The task entities, robot and support collider are
    outside this subtree and retain their physics configuration.
    """
    from pxr import UsdPhysics
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"missing background room prim: {root_path}")
    queue = [root]
    removed = []
    joints = []
    while queue:
        prim = queue.pop()
        if prim.IsInstance():
            prim.SetInstanceable(False)
        if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName().startswith(("Physics", "Physx")):
            joints.append(str(prim.GetPath()))
            prim.SetActive(False)
            continue
        schemas = set(prim.GetAppliedSchemas())
        authored = prim.GetMetadata("apiSchemas")
        if authored is not None:
            # Include legacy schemas that the installed registry does not
            # recognize; their authored tokens must not survive in the room.
            schemas.update(authored.GetAppliedItems())
        for schema in sorted(schemas):
            if schema.startswith(("Physics", "Physx")):
                # Applied-schema tokens can include multiple-apply instance
                # names (e.g. PhysxCookedDataAPI:triangleMesh). RemoveAPI treats
                # that whole token as a schema type; RemoveAppliedSchema
                # removes the exact composed token, including its instance.
                if not prim.RemoveAppliedSchema(schema):
                    raise RuntimeError(f"could not remove decorative physics API {schema}: {prim.GetPath()}")
                removed.append({"prim":str(prim.GetPath()),"schema":schema})
        queue.extend(prim.GetChildren())
    return {"root":root_path,"removed_apis":removed,"disabled_joints":joints}
