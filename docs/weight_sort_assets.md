# Weight Sort asset acquisition / Weight Sort 资产获取

From the repository root, use Python 3.11 or later:

```bash
python tools/download_weight_sort_assets.py \
  --source-dir /path/to/external-source --workers 4
```

This fetches the **42 external files** required by the default Weight Sort
scene: the verified official robot, 34 historical SimReady files and seven
EastRural files. They total **373,667,636 bytes** with the official robot.
Every file is checked against its recorded size and SHA256 before publication
in the source tree. Existing verified files are reused, so rerunning the
command resumes a partial download. Unrecognized existing files are preserved
and cause a failure; move them aside deliberately before retrying.

从仓库根目录执行上述命令，可以获取默认 Weight Sort 场景所需的 42 个外部文件，
包括机器人、箱体、容器、桌面及相关材质和纹理。每个文件均校验长度和 SHA256；
中断后可重跑，已经验证的文件不会重复下载，同名但不匹配的文件不会被覆盖。

## Exact official sources / 确切官方来源

| Local tree | Official source | Acquisition |
|---|---|---|
| `franka_mimic.usd` | [Isaac 5.1 Factory robot](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/franka_mimic.usd) | Fixed URL and exact official hash; see [robot comparison](robot_asset.md). |
| `simready/flatbox_a02/` | [NVIDIA Shipping 01 NVD @10010 ZIP](https://d4i3qtqj3r0z5.cloudfront.net/SimReady_Containers_Shipping_01_NVD%4010010.zip) | 13 entries under `Assets/simready_content/common_assets/props/flatbox_a02/`. |
| `simready/container_f27/` | Same historical ZIP | 21 entries under `Assets/simready_content/common_assets/props/container_f27/`. |
| `residential/east_rural/` | [ArchVis EastRural table](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Residential/Furniture/DiningSets/EastRural/EastRural_Table.usd) and its `Textures/` directory | The table USD and six recorded textures, preserving their relative paths. |

The Shipping ZIP is listed on NVIDIA's [Downloadable Asset Packs page](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html).
Its full size is **21,907,097,879 bytes**, ETag
`822c58760f11488f10f4c2f0c2bb6bce-2612`. The downloader uses validated HTTP
Range requests to read its directory and selected members. It does **not**
download the full 21 GB archive. If a server ignores Range requests, changes
the archive identity or returns incorrect file content, the command fails.

Use this historical `@10010` package. The current S3 SimReady directories
contain different USD revisions and are not accepted substitutes for these
objects. Unlike the robot's separately verified equivalent export, these
object files retain their original exact-byte pins.

箱体和容器使用官方历史包 `@10010`，不要用当前在线目录中的其他同名版本替代。
下载器只按需读取 ZIP 中需要的成员，不下载整个 21 GB 包；每个成员还会校验 ZIP CRC、
文件长度和 SHA256。准确的来源映射及校验记录见
[source verification](../assets/weight_sort_asset_sources_20261027.json)。

The six table textures are:

```text
Textures/ER_TableTop_BaseColor.png
Textures/ER_TableTop_Normal.png
Textures/ER_TableTop_Roughness.png
Textures/ER_Table_WhiteWood_BaseColor.png
Textures/ER_Table_WhiteWood_Normal.png
Textures/ER_Table_WhiteWood_Roughness.png
```

All downloaded assets retain their applicable upstream terms. The repository
distributes the acquisition tools and authored wrappers, not these payloads.

## Import into the runtime tree / 导入运行目录

Use the supported Isaac Sim environment from [the installation guide](getting_started_main.md)
to verify the renderer materials and import the payloads plus the three
ForceBench-authored wrappers shipped with the repository:

```bash
python tools/prepare_assets.py \
  --source-dir /path/to/external-source \
  --asset-dir /path/to/forcebench-assets \
  --task weight_sort --check-only

python tools/prepare_assets.py \
  --source-dir /path/to/external-source \
  --asset-dir /path/to/forcebench-assets \
  --task weight_sort

export FORCEBENCH_ASSET_DIR=/path/to/forcebench-assets
python -m forcebench doctor
```

The downloader does not install Isaac Sim or its renderer materials. If their
automatic discovery fails, give `prepare_assets.py` the supported runtime's
`--runtime-material-dir /path/to/isaacsim/kit/mdl/core/Base`.

下载完成后，使用安装指南指定的 Isaac Sim 环境执行完整校验和导入。
三个 ForceBench 自有封装文件由仓库提供；渲染器材质由独立安装的 Isaac Sim 提供。
资产获取与内容校验不等于训练或论文成功率已经复现。
