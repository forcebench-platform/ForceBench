# `gear_mesh` — 齿轮啮合装配

| 项 | 值 |
|---|---|
| 任务类别 | **接触密集**（当前实现下并不检验力感知，详见下文） |
| 场景 profile | `gear_assembly_cell` |
| 评测步数上限 | 1500 控制帧（15 Hz，约 100 s） |
| 向量化评测批大小 | 100 环境 |
| 专家数据 | ❌ **无**（正式数据集已不在源机器上，见下文说明） |
| 代码真值 | [`forcebench/tasks/gear_mesh/spec.py`](../../forcebench/tasks/gear_mesh/spec.py)、[`outcome.py`](../../forcebench/tasks/gear_mesh/outcome.py) |

## ⚠️ 关于"力敏感"措辞的诚实说明

本任务的 `MANIFEST.description` 字面写的是
"Force-sensitive gear alignment, meshing, release, and retraction"。
**该措辞在当前实现下并不成立，应当被视为前瞻性的（aspirational）而非已实现的：**

- 成功判定（`evaluate_final_state` + `outcome.py`）**完全只检查最终位姿**——居中、
  就位高度、直立、释放、退回——**从不读取任何 wrench 信号**。
- `spec.py` 中确实有一个质量相关常量 `mass_scale = 8.0`，但它的作用是让放大 2 倍的
  齿轮组保持合理的接触动力学，**这个量不参与判分**，也不随 seed 变化。

因此：**`gear_mesh` 是一个有用的接触密集装配任务，但它在当前实现下既不能证明也不能
证伪"力感知是否有帮助"。** 本 benchmark 关于力感知的实际证据来自五个力分类任务
（见 [`docs/tasks/README.md`](./README.md) 的总表）。

在成功判定中加入依赖力的检查，是 v1.0 的待办项。

## 任务目标

**中文：** 拿起中号齿轮，把它放到中央轴上，使其与两侧齿轮都啮合，松手释放，
然后退回夹爪。

**原文（模型收到的 prompt）：**

> Pick up the medium gear, place it on the center shaft so that it meshes with
> both flanking gears, release it, and retract the gripper.

## 场景

整套 Factory Gear Mesh 装配被统一放大 2 倍。取件位固定在 `(0.40, 0.0)`，放着待装的
中号齿轮；目标底板位置随机，其上有中央轴以及小齿轮、大齿轮两个两侧齿轮。两侧齿轮
在各自轴上仍可自由转动（与源任务一致）。

## 物理量与随机化范围

| 量 | 取值 |
|---|---|
| 整体资产缩放 | `(2.0, 2.0, 2.0)` |
| 质量缩放 | `8.0`（仅用于接触真实感，**不参与判分**） |
| 目标底板 X | 均匀采样 `[0.64, 0.70] m` |
| 目标底板 Y | 均匀采样 `[-0.08, 0.08] m` |
| 目标底板 yaw | 固定 `0.0 rad` |
| 取件点 | 固定 `(0.40, 0.0)`，齿轮根部 Z `0.010 m` |
| 最小初始间距 | `0.28 m`（拒绝采样，最多 128 次） |
| 中号齿轮初始 yaw | `-4.5°` 基准 + 随机抖动 |
| **齿相位抖动幅值** | **每个 seed 在 `[2.0, 3.5]°` 内随机，符号随机** |
| 抓取时夹爪开度 | `0.70` |
| 中号齿轮目标偏移 | `(0.04050, 0.0, 0.010) m` |
| 小 / 大齿轮目标偏移 | `(0.10150, 0.0, 0.010)` / `(-0.06050, 0.0, 0.010) m` |
| 夹爪 / 齿轮 / 夹具摩擦 | `1.5` / `0.75` / `0.25`，combine 模式 `min` |

齿相位抖动的设计意图是：中号齿轮有 40 齿（`9°` 齿距），齿顶中心位于 `4.5 + 9k` 度，
而两侧齿轮沿中心连线呈现齿谷，因此 `-4.5°` 的偏航恰好把齿顶送进齿谷。抖动确保每次
不会从"精确兼容的啮合相位"开始，同时又保持取件姿态接近该相位——这是一个**几何**上的
难度来源，不是力学上的。

## 成功判定条件

判据来自 `outcome.py` 的五条 `OutcomeCriterion`，五条全部满足才算成功：

| 判据字段 | 失败码 | 含义 |
|---|---|---|
| `centered` | `gear_not_centered` | 中号齿轮对准其轴 |
| `seated` | `gear_not_seated` | 中号齿轮落到啮合高度 |
| `upright` | `gear_tilted` | 中号齿轮轴线保持竖直 |
| `released` | `gear_not_released` | 夹爪已释放齿轮 |
| `retracted` | `gripper_not_retracted` | 释放后夹爪已退回 |

具体阈值（`spec.py: evaluate_final_state`）：

- **居中**：齿轮中心与目标轴的 XY 距离 `<= 0.005 m`。
- **已就位**：高度差绝对值 `<= 0.006 m`。
- **直立**：齿轮轴线与世界 Z 轴夹角 `<= 12.0°`。
- **已释放**：`gripper_open_fraction >= 0.65`。
- **已退回**：末端相对齿轮的水平距离 `>= 0.10 m` **或**高度差 `>= 0.09 m`（满足其一即可）。

**全部五条都是纯几何量。** 没有一条读取 `eef_wrench`。特别注意：判据里并**没有**
检查齿轮是否真的与两侧齿轮啮合（例如齿相位是否正确）——只检查了中心、高度和倾角。

## ⚠️ 本任务当前没有可用的专家数据

2026-08-27 在源机器 L20 清点确认：`CoRiBench_data/datasets/` 下**只有 7 个任务**，
`gear_mesh` 不在其中。该任务仅在 `CoRiBench_scratch/expert/gear_mesh` 下残留
**1 条** episode（145 MB），属于调试残留，不构成可发布的数据集。

早前的记录称本任务有 62 条数据，该数字**已过时**，本文档以实测清点为准。数据消失的
原因未在可查材料中记录——可能是被清理、归档或迁移，但在源机器上找不到对应去向。

采集本身的难度是已知的：[项目指南](../developer/project_guide.md)第 6 节给出的示例
命令对 `gear_mesh` 用的是 `--seeds 0-19 --workers 2`（而不是其他任务的 `0-199`），
说明这个任务走的是小批量、任务绑定的特权位姿教师路径，吞吐远低于 32 克隆路径。

**因此：ForceBench v0.1 的数据集只覆盖 7 个任务，本任务的代码和任务定义齐全但无数据。**
重新采集是 v1.0 待办项。

## 依赖的第三方资产

| 资产 | 来源 | 许可证 |
|---|---|---|
| `factory_gear_{base,medium,small,large}.usd` | NVIDIA Isaac Lab 2.3.2 FORGE 部署（L20 上的 `isaac-forge`），镜像摘要 `sha256:388dbc8...` | NVIDIA Isaac Sim/Omniverse EULA（部署时通过 `ACCEPT_EULA=Y` 接受；**不是**开放再分发许可） |

ForceBench 只分发自有的包装文件（`assets/normalized/gear_mesh/*.usda`）。使用者必须
自行运行 FORGE / Isaac Lab 部署来获得源文件。这四个文件的部署哈希已记录在
`assets/SHA256SUMS` 中。完整说明见 [`NOTICE.md`](../../NOTICE.md)。
