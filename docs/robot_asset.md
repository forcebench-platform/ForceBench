# Shared Franka robot / 共享 Franka 机器人

## Official download / 官方下载

Download the [NVIDIA Isaac 5.1 Factory robot asset](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/franka_mimic.usd).
The verified file is **7,075,031 bytes**, SHA256:

```text
86d0017a093669b222a8c640bf581cc3e3715035aa7c596600f478da011f6c8e
```

From the repository root, download it into the external source tree used by
the asset importer:

```bash
python tools/download_robot_asset.py --source-dir /path/to/external-source
```

The downloader checks the complete byte count and SHA256. It preserves an
existing verified historical or official robot, and refuses to overwrite an
unrecognized file. An incomplete or changed upstream download is rejected.
The file comes directly from NVIDIA; the repository's MIT license does not
cover it. Its applicable upstream terms remain in force.

从仓库根目录运行上面的命令，即可从 NVIDIA 官方地址下载机器人到外部资产源目录。
下载器校验完整文件长度和 SHA256；已有且通过验证的历史文件或官方文件会被保留，
其他同名文件不会被覆盖。下载资产适用其上游条款，不属于本仓库的 MIT 代码许可。

## Why the historical hash differs / 历史哈希为何不同

The original deployment inventory records a **7,073,847-byte** export:

```text
6ca52983cbede63f4caf1557896213e732f14fd8da9234658a6981ec2515b1bf
```

On 2026-10-27 we independently read the historical and official files with
OpenUSD 22.11. All **1,082 authored specs and 4,022 authored fields** matched,
including geometry, joints, physics attributes, materials, relationships and
layer metadata. Their complete `Sdf.Layer.ExportToString()` outputs were
identical, without stripping or changing fields. That text's SHA256 was
`51e87379b0e3f1b87c8230ac5753ebe0747eeb71194b3fcf3e09519e1dd98058`.
The checked Isaac 4.2, 4.5, 5.0 and 5.1 asset endpoints returned the same
official bytes above.

The files have different binary representations but identical decoded USD
content in this comparison. The exact historical serialization tool/version
is still unidentified. The manifest retains the historical pin and registers
this single verified official export as an allowed alternative. Other
same-name files remain rejected. See the [comparison record](../assets/robot_asset_equivalence_20261027.json).

历史文件与官方文件的二进制哈希不同，但逐字段核对和完整 USD 文本比较均未发现内容差异。
安装器同时接受这两个确切版本，并保留论文实验所用历史文件的校验记录。
哈希校验仍然启用，其他同名文件不能直接替换。

## Complete task setup / 完整任务准备

This command downloads the shared robot only. Weight Sort users can instead
use the [complete task asset downloader](weight_sort_assets.md), which also
obtains its object, scene and texture files. All eight tasks can use the [unified asset downloader](assets.md). After those
files are in the source tree, verify and import the selected task:

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
```

The static robot comparison does not by itself verify a complete task
installation or reproduce a paper success rate. Task-specific acquisition
recipes need their own file and version checks.

此命令只获取共享机器人。任务物体、场景和纹理仍需另外准备；
机器人内容一致不能直接推出完整任务安装成功或论文结果已经复现。
