# Assets for all eight tasks / 八任务资产准备

## One acquisition entry point / 统一获取入口

From the repository root, in a Python 3.12.13 tools environment (the validated
asset-build interpreter; basic inspection also supports Python 3.11+):

```bash
python -m pip install -e ".[assets]"
python tools/download_assets.py \
  --source-dir /path/to/asset-source \
  --cache-dir /path/to/download-cache --workers 4
```

Omit `--task` for all eight tasks. Repeat it for a subset, for example
`--task faucet_turn --task nut_thread`. The checked-in
[source registry](../assets/asset_sources.json) covers all **283 external or
derived payloads** selected by the default scenes. The importer adds the
17 authored wrappers shipped in Git, making **300 runtime payload files**.
Existing verified files are reused. Unknown, truncated or changed files fail
verification and are preserved; interrupted downloads can be resumed by
rerunning the command.

在仓库根目录安装 `.[assets]` 后运行上述命令，即可按固定来源获取全部八个默认任务的资产。
可用 `--task` 选择任务。下载、解压和构建结果都按长度与 SHA256 校验，不依赖内部服务器。
建议为资产、导入副本和下载缓存预留至少 8 GB；模拟器、运行数据和模型环境另计。

## Task-to-source map / 任务与来源

| Runtime task | Paper task | Required source families |
|---|---|---|
| `weight_sort` | Weight Sort | NVIDIA Shipping @10010, EastRural, shared robot |
| `rattle_sort` | CoM Sort | Shipping @10010, NVIDIA conveyor/warehouse inputs, two CPU builds, shared robot |
| `faucet_turn` | Faucet Turn | Lightwheel Kitchen, exact DOM cup09, shared robot |
| `faucet_force_turn` | Faucet Direction | Lightwheel Kitchen, exact DOM cup09, shared robot |
| `stove_turn_on` | Stove Turn | Lightwheel Kitchen and exact scene/UV/single-burner rebuilds, shared robot |
| `stove_press_turn` | Stove Press | Same Kitchen inputs and rebuilds, shared robot |
| `nut_thread` | Nut Thread | NVIDIA Factory nut/bolt, EastRural, Poly Haven, shared robot |
| `radmanso_drawer_sort_table` | Drawer Pull | Published ArtVIP RADMANSO model, EastRural, shared robot |

The registry contains the full per-file paths, pins and archive-member maps.
The tool downloads only the selected task closure and its build inputs.
Each task's authored wrappers and physical settings remain unchanged.

## Fixed publishers and versions / 固定来源与版本

- **Shared robot:** [NVIDIA Factory robot](robot_asset.md). The historical and
  official binary exports have identical complete authored USD content; both
  exact hashes are accepted.
- **Nut/Bolt:** [NVIDIA Isaac 5.1 Factory sources](nut_assets.md). Only the two
  explicitly audited official export variants and historical exports are
  accepted. SDF collision, vertex counts and task frames remain checked.
- **SimReady:** NVIDIA's [Shipping 01 NVD @10010 archive](https://d4i3qtqj3r0z5.cloudfront.net/SimReady_Containers_Shipping_01_NVD%4010010.zip),
  listed on its [official asset-pack page](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html).
  The archive is 21,907,097,879 bytes, ETag
  `822c58760f11488f10f4c2f0c2bb6bce-2612`. The downloader uses bounded HTTP
  Range requests and ZIP CRC checks for selected members. It never downloads
  the full 21 GB archive. Current same-name S3 object revisions are not substitutes.
- **Lightwheel Kitchen:** [official repository](https://github.com/LightwheelAI/Lightwheel_Kitchen),
  commit `10b0f7a86135a27e8c2fba9c690f6bf5c4f06bb1`. The fixed Git LFS ZIP is
  699,682,909 bytes, SHA256
  `721fb9e47d09b2c43a7db396c1d63c659e987d5925db0a5e78cf4fd40cd0b0b0`.
  This verified GitHub download is used rather than the upstream README's
  currently unavailable GCS URL. Preserve the supplied **CC BY-NC 4.0** terms
  and attribution in `assets/lightwheel_kitchen/LICENSE.txt`.
- **EastRural:** NVIDIA [ArchVis table](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Residential/Furniture/DiningSets/EastRural/EastRural_Table.usd)
  and its six original textures. Exact URLs and hashes are in the registry.
- **Ash veneer:** [Poly Haven](https://polyhaven.com/a/ash_veneer), the pinned
  4K JPG diffuse, OpenGL normal and roughness maps, under its CC0 terms.
- **Drawer:** the [official ArtVIP dataset](https://huggingface.co/datasets/X-Humanoid/ArtVIP),
  commit `d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e`, whose dataset card
  declares Apache-2.0. All six selected payloads match the historical files;
  the published long model filename is copied to local `model.usd`.
- **Cup09:** the exact `objects/cup/cup09.usd` from the explicitly labelled
  [third-party DynamicVLA DOM mirror](https://huggingface.co/datasets/mickeykang/dom-eval-files),
  commit `77dda64f9942cb81d97ed00d4d61b8721d749210`. The mirror ZIP is
  192,600,903 bytes, SHA256
  `02aaa3f87d7efd2e06e16d767cdea81e53299d2531505bea22db861c82c9cd63`.
  The selected cup matches the historical size and SHA256 exactly. This is
  **not an NVIDIA asset or an original-author download**. The original
  [DynamicVLA project](https://github.com/hzxie/DynamicVLA) points to its
  [official object-package request page](https://gateway.infinitescript.com/?f=DOM-3D-Objects),
  which requires email and human verification. No access check is bypassed.
  Use the original publisher's applicable terms; the mirror is not relicensed
  by ForceBench's MIT code license.

全部链接与版本均固定在 registry 中。Cup09 的第三方镜像身份已单独标明，不能把它归为
NVIDIA SimReady，也不能由本仓库的代码许可证推断外部资产许可。
本仓库发布下载与构建工具、校验记录和自有封装，不发布这些第三方二进制素材。

## Five exact CPU builds / 五项精确 CPU 构建

Install **`usd-core==25.11`** through the `.[assets]` extra. The downloader
checks the writer version, source hashes, repository builder hashes and
resulting file hashes. It performs these builds in isolated directories:

| Output | Exact reconstruction |
|---|---|
| `FullKitchen/Collected_KitchenRoom/KitchenRoom.usd` | Copy official layer, remove its two historical UnitsAdjust sublayers, then incremental `Sdf.Layer.Save()` |
| `Stovetop012/Stovetop012.usd` | Copy official layer, restore the recorded UV entry and four index ranges, then incremental save |
| `Stovetop012_single_center.usd` | Run the shipped single-burner selection builder on the restored input |
| `simready/conveyor_a04/conveyor_a04.usd` | Build from the pinned official Isaac 6.0 conveyor source |
| `simready/small_warehouse_digital_twin/rack.usd` | Extract the recorded rack branch from the pinned official warehouse source |

These reproduce the historical output SHA256 values. They do not replace
geometry with approximate objects or alter benchmark success rules. Exporting
or flattening the Kitchen/Stove layers instead of incremental saving changes
the binary serialization and does not satisfy the historical pins. The
complete build dependency order is registered in `asset_sources.json`.

五项构建均已从公共输入重建并命中历史输出哈希；下载器自动处理顺序。
Kitchen 和 Stove 的源层修复必须增量保存，不能自行 Flatten 或重新导出替代。

The [2026-10-27 verification record](../validation/asset_acquisition_20261027.json)
includes all five exact rebuilds, import/composition checks, an offline rerun
with zero network/build calls, and eight real GPU task checks. The tools
environment was fresh; the simulator environment inherited heavy NVIDIA
dependencies. The record lists the actual network route fallbacks and native
library requirements; it is not a fresh-OS or learned-policy result receipt.

## Network restrictions and offline archive cache / 网络与离线包

Configure `HTTPS_PROXY` in the tools environment when a publisher is not
reachable. A failure preserves verified files and removes incomplete temporary
outputs. For the two small fully cached ZIPs, you can download the exact
registry URL with another authorized browser or machine, then import it:

```bash
python tools/download_assets.py \
  --source-dir /path/to/asset-source --cache-dir /path/to/download-cache \
  --archive-file lightwheel_v1=/path/to/Lightwheel_Kitchen.zip \
  --archive-file dom_objects_mirror=/path/to/DOM-3D-Objects.zip
```

The tool verifies each supplied archive's complete length and SHA256 before
using it. It does not accept the Shipping archive through this full-cache
option; Shipping remains a bounded Range source. The cache's contents are
reusable but do not replace per-file checks.

若服务器无法访问 GitHub 或 Hugging Face，可从 registry 给定的同一地址下载原包，
再通过 `--archive-file` 导入。不要修改 ZIP 内容或用别的版本改名替代；完整长度和哈希都会检查。

## Import and start / 导入并启动

Switch to the supported simulator environment from the
[installation guide](getting_started_main.md), including its first-run EULA
and interactive warm-up step before running the non-interactive GPU checker:

```bash
python tools/prepare_assets.py \
  --source-dir /path/to/asset-source --asset-dir /path/to/forcebench-assets --check-only
python tools/prepare_assets.py \
  --source-dir /path/to/asset-source --asset-dir /path/to/forcebench-assets
export FORCEBENCH_ASSET_DIR=/path/to/forcebench-assets
python -m forcebench doctor --asset-dir "$FORCEBENCH_ASSET_DIR"
python tools/verify_nut_asset_composition.py --asset-dir "$FORCEBENCH_ASSET_DIR"

CUDA_VISIBLE_DEVICES=0 python tools/validate_simulation.py \
  --python "$(command -v python)" --assets "$FORCEBENCH_ASSET_DIR" \
  --output /path/to/fresh-validation-output --steps 10 --timeout 1200
```

The importer checks the installed renderer's `OmniPBR.mdl` and `OmniGlass.mdl`;
these stay in Isaac Sim and are not downloaded as dataset payloads. Only use
`--runtime-material-dir` when discovery of the supported runtime fails.

The manifest retains reviewed raw dependency gaps in inactive branches and
unselected variants. The supported scope is the **eight default task scenes**;
changing source hashes, variants or scene profiles requires a new audit.
File checks, environment startup and paper-scale policy results are separate
validation scopes. No training data or model success rate is certified by an
asset download alone.
