# External asset preparation

Start with [the asset guide](../docs/assets.md) and run
`python tools/prepare_assets.py --source-dir SOURCE --asset-dir DESTINATION --check-only`.

The public source contains 17 authored USDA reference/primitive layers. The
300-file runtime manifest includes separately acquired third-party models and
textures; their bytes are not part of this source distribution. See
`asset_manifest.json`, `asset_roots.json` and `asset_dependency_review.json`.

`derived_asset_recipes.json` records input/output/builder hashes for the authored
Stovetop, conveyor and warehouse-rack transformations. The single-center
Stovetop rebuild has been verified against the exact deployed output; the
conveyor and rack historical recipe inputs still need to be acquired for a
fresh rebuild.

`SHA256SUMS` is the historical, broader asset list from the earlier internal
tree, including tasks outside this release. Its old Faucet/Rattle wrapper
entries describe earlier versions. Use `asset_manifest.json` for the current
eight-task import, rather than applying the historical list to a new tree.

Geometry access does not imply redistribution permission. The code license
does not replace NVIDIA, Lightwheel, ArtVIP or other asset terms. No renderer
MDL files or third-party source meshes/textures are bundled here.
