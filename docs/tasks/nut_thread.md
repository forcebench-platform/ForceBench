# `nut_thread` — 螺母拧入螺栓

| 项 | 值 |
|---|---|
| 任务类别 | **接触密集**（当前实现下并不检验力感知，详见下文） |
| 场景 profile | `precision_ash_veneer` |
| 评测步数上限 | 1500 控制帧（15 Hz，约 100 s） |
| 向量化评测批大小 | 100 环境 |
| 专家数据 | **200** episode（seed 0–199） |
| 代码真值 | [`forcebench/tasks/nut_thread/spec.py`](../../forcebench/tasks/nut_thread/spec.py)、[`outcome.py`](../../forcebench/tasks/nut_thread/outcome.py) |

## ⚠️ 关于"力敏感"措辞的诚实说明

本任务的 `MANIFEST.description` 字面写的是
"Force-sensitive nut engagement and threading with release and retraction"。
**该措辞在当前实现下并不成立，应当被视为前瞻性的（aspirational）而非已实现的：**

- 成功判定（`evaluate_final_state` + `outcome.py`）**完全只检查最终位姿**——居中、
  螺纹高度、直立、释放、退回——**从不读取任何 wrench 信号**。
- `spec.py` 中没有任何力或质量常量；随机化的只有螺栓的平面位置。

因此：**`nut_thread` 是一个有用的接触密集装配任务，但它在当前实现下既不能证明也不能
证伪"力感知是否有帮助"。** 本 benchmark 关于力感知的实际证据来自五个力分类任务
（见 [`docs/tasks/README.md`](./README.md) 的总表）。

在成功判定中加入依赖力的检查，是 v1.0 的待办项。

> 值得注意的是，本任务的螺纹高度判据（下文）是三个装配任务里**唯一**一个不对称的、
> 明确针对"是否真的旋入了"而设计的判据。它仍然是一个位置量，但它是这三个任务中离
> "检验接触结果"最近的一条。

## 任务目标

**中文：** 拿起 M16 螺母，把它拧到螺栓上，松手释放，然后退回夹爪。

**原文（模型收到的 prompt）：**

> Pick up the M16 nut, thread it onto the bolt, release it, and retract the
> gripper.

## 场景

取件位固定在 `(0.49, 0.0)`，上面放着一个 M16 螺母（置于 `0.018 x 0.018 x 0.010 m`
的小支撑块上）；目标 M16 螺栓的平面位置随机。

## 物理量与随机化范围

| 量 | 取值 |
|---|---|
| 目标螺栓 X | 均匀采样 `[0.58, 0.68] m` |
| 目标螺栓 Y | 均匀采样 `[-0.11, 0.11] m` |
| 取件点 | 固定 `(0.49, 0.0)`，螺母根部 Z `0.010 m` |
| 最小初始间距 | `0.14 m`（拒绝采样，最多 128 次） |
| 初始 yaw | 固定 `0.0 rad` |
| 螺栓底座高度 | `0.010 m` |
| 螺栓杆部高度 | `0.025 m` |
| 螺纹螺距 | `0.002 m` |
| 目标螺纹圈数偏移 | `1.5` 个螺距 |
| 夹爪 / 螺母 / 螺栓摩擦 | `1.5` / `0.75` / `0.25`，combine 模式 `min` |

目标螺母底面高度由 `底座高度 + 杆部高度 - 螺距 × 1.5 = 0.010 + 0.025 - 0.003 = 0.032 m`
（相对螺栓根部）确定，即要求螺母旋入 1.5 个螺距。

## 成功判定条件

判据来自 `outcome.py` 的五条 `OutcomeCriterion`，五条全部满足才算成功：

| 判据字段 | 失败码 | 含义 |
|---|---|---|
| `centered` | `nut_not_centered` | 螺母轴线对准螺栓 |
| `threaded` | `nut_not_threaded` | 螺母达到旋入后的目标高度 |
| `upright` | `nut_tilted` | 螺母轴线保持竖直 |
| `released` | `nut_not_released` | 夹爪已释放螺母 |
| `retracted` | `gripper_not_retracted` | 释放后夹爪已退回 |

具体阈值（`spec.py: evaluate_final_state`）：

- **居中**：螺母与目标的 XY 距离 `<= 0.0025 m`（三个装配任务里最严的居中容差）。
- **已旋入**：高度差落在 **`[-0.002 m, +0.001 m]` 这个不对称区间**内。上界只有半个
  螺距，用意是容忍已真正啮合的螺母那种亚毫米级接触沉降变化，同时排除一个只是搁在
  2 mm 螺距螺栓顶端、并未旋入的螺母。
- **直立**：螺母轴线与世界 Z 轴夹角 `<= 10.0°`（比另两个装配任务的 12° 更严）。
- **已释放**：`gripper_open_fraction >= 0.65`。
- **已退回**：末端相对螺母的水平距离 `>= 0.07 m` **或**高度差 `>= 0.055 m`（满足其一即可）。

**全部五条都是纯几何量。** 没有一条读取 `eef_wrench`。

## 依赖的第三方资产

| 资产 | 来源 | 许可证 |
|---|---|---|
| `factory_nut_m16.usd`、`factory_bolt_m16.usd` | NVIDIA Isaac Lab 2.3.2 FORGE 部署（与 `gear_mesh` 同一部署），镜像摘要 `sha256:388dbc8...` | NVIDIA Isaac Sim/Omniverse EULA（部署时通过 `ACCEPT_EULA=Y` 接受；**不是**开放再分发许可） |

ForceBench 只分发自有的包装文件（`assets/normalized/nut_thread/*.usda`）。导入的螺母
和螺栓碰撞网格存在开放边界环，因此改用凸分解烘焙而不是无效的非水密 SDF。这两个文件的
部署哈希已记录在 `assets/SHA256SUMS` 中。完整说明见 [`NOTICE.md`](../../NOTICE.md)。

> 注：`tests/test_forge_task_specs.py::test_normalized_thread_assets_preserve_source_sdf_collisions`
> 是 `KNOWN_ISSUES.md` 中记录的 5 个已知失败测试之一——它需要真实的 FORGE 派生 `.usd`
> 二进制才能检查 SDF 碰撞数据，而这些二进制不在纯代码的公开提取范围内。这不是代码缺陷。
