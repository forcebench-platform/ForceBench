# Preparing assets for the eight default tasks

ForceBench source includes authored USD reference wrappers and primitive
geometry. The third-party model files, textures and renderer materials must
be obtained separately under their original terms. The code's MIT license
does not grant rights to those files. The preparation tool neither downloads
assets nor grants redistribution permission.

`assets/asset_manifest.json` pins **300 existing files, 2,300,568,689 bytes**
for the eight task defaults, including the shared `franka_mimic.usd` robot,
task objects and selected room/table profiles. Each file has a relative path,
SHA256, size and dependency list. The 17 authored USDA wrappers total 109,516
bytes and are included with source. Source USD meshes and textures are absent
from the Git distribution.

## Obtain the external files

Acquire the following materials from their applicable upstream distribution,
retaining the directory layout recorded in the manifest. Asset names alone
are insufficient: the importer requires the recorded bytes and reports a
different export or revision instead of silently substituting geometry.

| Source group | Required material | Terms and acquisition status |
|---|---|---|
| NVIDIA Factory/FORGE deployment | `nut_m16.usd`, `bolt_m16.usd` | Recorded Factory/FORGE export from Isaac Lab 2.3.2 / Isaac Sim 5.1. Obtain under the applicable NVIDIA distribution terms; the binaries are not bundled. Source filenames and checksums are detailed in [nut_assets.md](nut_assets.md). |
| NVIDIA Factory shared robot | `franka_mimic.usd` | The official download and the historical binary decode to identical complete authored USD content; both exact hashes are accepted. Follow the [robot download instructions](robot_asset.md). The binary stays external and subject to upstream terms. |
| NVIDIA SimReady / Isaac / ArchVis assets | SimReady `flatbox_a02`, `flatbox_a07`, `container_f27`, `cup09`, conveyor, warehouse rack, shipping background parcels, EastRural table | Use the original asset distributions and preserve their referenced materials/textures. The historical shipping package was `SimReady_Containers_Shipping_01_NVD@10010.zip`. No open redistribution permission is asserted here. |
| Lightwheel Kitchen | `Sink054`, `Stovetop012`, collected KitchenRoom and Pot057 | Recorded release commit `10b0f7a86135a27e8c2fba9c690f6bf5c4f06bb1`; retain the upstream CC BY-NC 4.0 terms and attribution in `assets/lightwheel_kitchen/LICENSE.txt`. Referenced source files remain external. |
| Poly Haven | Ash veneer diffuse, normal and roughness textures used by the precision table | Retain the original source/terms with separately acquired texture files; exact deployed bytes are pinned. |
| ArtVIP | RADMANSO three-drawer `model.usd` and its `resource/` dependencies | The project's historical record has not established redistribution permission. Users must obtain these through an authorized source; ForceBench does not supply the model. |
| Supported Isaac Sim runtime | `OmniPBR.mdl`, `OmniGlass.mdl` and MDL SDK modules | These stay in the installed renderer. The preparation check verifies the recorded `kit/mdl/core/Base` materials; it does not copy them into the asset bundle. |

This is an import recipe for a separately acquired asset tree, rather than an
automated public download service. In particular, an asset availability or
permission gap cannot be fixed by renaming a different model to match the
expected filename. The historical exports remain identifiable by checksum.

### Download the shared robot / 下载共享机器人

```bash
python tools/download_robot_asset.py --source-dir /path/to/legally-acquired-asset-tree
```

The [robot guide](robot_asset.md) gives the fixed NVIDIA URL, upstream hash
and complete content comparison with the historical export. Only those two
exact robot exports are accepted; other assets retain their existing
exact-byte requirements. This command downloads the robot only.

上述命令仅下载共享机器人，保留完整校验；任务物体和场景仍需另行准备。

### Download the default Weight Sort scene / 获取 Weight Sort 默认场景

```bash
python tools/download_weight_sort_assets.py --source-dir /path/to/legally-acquired-asset-tree --workers 4
```

The [Weight Sort acquisition guide](weight_sort_assets.md) pins the historical
official SimReady ZIP and ArchVis source paths, and explains the selected
Range downloads. It fetches all 42 external Weight Sort files; the importer
adds the three repository-authored wrappers and checks runtime materials.

此命令获取 Weight Sort 的全部 42 个外部文件；历史官方包版本、目录映射与校验方法见
[获取指南](weight_sort_assets.md)。其他任务的外部资产需要按各自来源准备。

## Verify and import

Run with the supported Isaac Sim Python environment so the renderer material
directory can be found automatically. Use a new explicit asset directory:

```bash
python tools/prepare_assets.py \
  --source-dir /path/to/legally-acquired-asset-tree \
  --asset-dir /path/to/forcebench-assets \
  --check-only

python tools/prepare_assets.py \
  --source-dir /path/to/legally-acquired-asset-tree \
  --asset-dir /path/to/forcebench-assets

export FORCEBENCH_ASSET_DIR=/path/to/forcebench-assets
```

If Python cannot discover the installed Isaac Sim package, add
`--runtime-material-dir /path/to/isaacsim/kit/mdl/core/Base`. To prepare a
subset, repeat `--task` with the task IDs from `python -m forcebench list-tasks`.

All selected source hashes, dependency reviews and existing destination files
are checked before copying. Different existing files are preserved and make
the check fail. Successful files are copied without changing their bytes,
and the written copies are checked again. The authored wrappers can be read
from this checkout; missing third-party sources cannot. The sole historical
filename alias is `model_coribench.usda` to `model_forcebench.usda` for the
identical authored Drawer overlay.

## Dependency coverage

The manifest records 449 authored references and retains 97 raw dependency
gaps. `assets/asset_dependency_review.json` classifies each gap with source
SHA256 constraints for these exact default scenes:

- Unqualified `OmniPBR.mdl` / `OmniGlass.mdl` and MDL imports are supplied by
  the supported renderer, rather than being local payload files.
- NVIDIA thumbnail payloads are under `/Tagging`, outside the `/RootNode`
  default prim imported by the tasks.
- NVIDIA physics-material references belong to the unselected
  `PhysicsVariant=RigidBody`; the deployed visual defaults select `None` and
  task collision materials are authored separately.
- The old Lightwheel PaintTool cube is under a branch deactivated by the
  authored environment overlay. Task knob visuals reference the handle
  geometry; the kitchen background excludes its duplicate sink.

These observations were checked against the deployed assets on 2026-10-21.
The complete upstream authoring packages are **not** claimed to have closed
dependencies. Changing a variant, scene profile or input hash invalidates
this review and requires a new dependency audit. The importer reports both
the retained raw gaps and the checked default-task scope.

Asset checks establish file identity and scoped dependency coverage. Complete
GPU initialization, physical interaction and paper-score reproduction require
their own validation.

## Derived geometry recipes

`assets/derived_asset_recipes.json` records the original input hashes,
author-written builder hashes and expected output hashes for the compact
Stovetop, conveyor and warehouse-rack exports. The corresponding Python files
are preserved in `tools/asset_builders/`. They read separately acquired inputs
and require `pxr`; they do not fetch source models or grant rights to the
generated geometry.

The Stovetop builder was run independently on 2026-10-21 and reproduced the
deployed output byte for byte. Run it only in a fresh staging directory
containing the pinned `Stovetop012.usd` input and its source dependencies:

```bash
python tools/asset_builders/build_single_stovetop.py /path/to/fresh/Stovetop012
```

The conveyor and warehouse builders accept explicit paths:

```bash
python tools/asset_builders/build_conveyor_a04.py \
  --source /path/to/ConveyorBelt_A04.usd \
  --output /path/to/fresh/conveyor_a04.usd \
  --metadata /path/to/fresh/conveyor_a04.json

python tools/asset_builders/build_small_warehouse_rack.py \
  --source /path/to/small_warehouse_digital_twin.usd \
  --output /path/to/fresh/rack.usd \
  --metadata /path/to/fresh/rack.json
```

Those two original download caches were absent at this audit, so their
recipes are preserved with `rebuild_verified: false`. A fresh build must be
checked against the expected SHA256 before importing. Different bytes require
an explicit geometry/material audit; renaming the output does not make it the
pinned deployment asset.
