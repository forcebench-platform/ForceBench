# `peg_insert` — 销轴插孔

| 项 | 值 |
|---|---|
| 任务类别 | **接触密集**（当前实现下并不检验力感知，详见下文） |
| 场景 profile | `precision_ash_veneer` |
| 评测步数上限 | 1500 控制帧（15 Hz，约 100 s） |
| 向量化评测批大小 | 100 环境 |
| 专家数据 | 200 episode（44 GB，2026-08-27 清点确认已补齐） |
| 代码真值 | [`forcebench/tasks/peg_insert/spec.py`](../../forcebench/tasks/peg_insert/spec.py)、[`outcome.py`](../../forcebench/tasks/peg_insert/outcome.py) |

## ⚠️ 关于"力敏感"措辞的诚实说明

本任务的 `MANIFEST.description` 把自己描述为接触密集的插装任务，而 ForceBench 的
早期材料曾把这三个装配任务统称为 "force-sensitive"。**该措辞在当前实现下并不成立，
应当被视为前瞻性的（aspirational）而非已实现的：**

- 成功判定（`evaluate_final_state` + `outcome.py`）**完全只检查最终位姿**——居中、
  插入深度、直立、释放、退回——**从不读取任何 wrench 信号**。
- `spec.py` 中**没有任何力或质量常量**。整个任务规格里不存在一个以牛顿或千克为单位
  的量。这一点在三个装配任务中尤为突出。

因此：**`peg_insert` 是一个有用的接触密集操作任务，但它在当前实现下既不能证明也不能
证伪"力感知是否有帮助"。** 与之相对，五个力分类任务
（[`weight_sort`](./weight_sort.md)、[`rattle_sort`](./rattle_sort.md)、
[`stove_turn_on`](./stove_turn_on.md)、[`faucet_turn`](./faucet_turn.md)、
[`radmanso_drawer_sort_table`](./radmanso_drawer_sort_table.md)）才是本 benchmark
关于力感知的实际证据来源。

在成功判定中加入依赖力的检查，是 v1.0 的待办项。

## 任务目标

**中文：** 从蓝色夹具上拿起银色杆件，把它插进古铜色目标孔，松手释放，然后退回夹爪。

**原文（模型收到的 prompt）：**

> Pick up the silver rod from the blue fixture, insert it into the bronze target
> hole, release it, and retract the gripper.

## 场景

桌面上有一个固定的取件夹具（中心 `(0.50, 0.0)`）上立着一根 8 mm 销轴，以及一个位置
随机的目标孔。初始 TCP 位置固定在 `(0.50, 0.0, 0.18)`。

## 物理量与随机化范围

| 量 | 取值 |
|---|---|
| 目标孔 X | 均匀采样 `[0.58, 0.68] m` |
| 目标孔 Y | 均匀采样 `[-0.12, 0.12] m` |
| 取件点 | 固定 `(0.50, 0.0)` |
| 最小初始间距 | `0.14 m`（拒绝采样，最多 128 次） |
| 销轴初始根部 Z | `0.015 m` |
| 目标孔根部 Z / 配合高度 | `0.0 m` / `0.003 m` |
| 初始 yaw | 固定 `0.0 rad` |
| 夹爪摩擦 | `1.5` |
| 销轴摩擦 | `3.0` |
| 孔摩擦 | `0.01` |
| 摩擦 combine 模式 | `min` |

**注意这张表里没有一行是力或质量。** 随机化的只有目标孔的平面位置。

## 成功判定条件

判据来自 `outcome.py` 的五条 `OutcomeCriterion`，五条全部满足才算成功：

| 判据字段 | 失败码 | 含义 |
|---|---|---|
| `centered` | `peg_not_centered` | 销轴轴线对准目标孔中心 |
| `inserted` | `peg_not_inserted` | 销轴达到目标插入深度 |
| `upright` | `peg_tilted` | 插入后的销轴保持直立 |
| `released` | `peg_not_released` | 夹爪已释放销轴 |
| `retracted` | `gripper_not_retracted` | 释放后夹爪已退回 |

具体阈值（`spec.py: evaluate_final_state`）：

- **居中**：销轴与目标孔的 XY 距离 `<= 0.003 m`。
- **已插入**：销轴与目标孔的高度差绝对值 `<= 0.006 m`。
- **直立**：销轴轴线与世界 Z 轴夹角 `<= 12.0°`。
- **已释放**：`gripper_open_fraction >= 0.65`。
- **已退回**：末端相对销轴的水平距离 `>= 0.07 m` **或**高度差 `>= 0.055 m`（满足其一即可）。

**全部五条都是纯几何量。** 没有一条读取 `eef_wrench`。

### 释放前的稳定性门限

`spec.py` 另外定义了一组比最终判据**更严格**的释放前门限
（`RELEASE_GATE_MAX_CENTER_ERROR_M = 0.0015`、`RELEASE_GATE_MAX_HEIGHT_ERROR_M = 0.0055`、
`RELEASE_GATE_MAX_TILT_DEG = 8.0`）。它在手指仍然夹住销轴时生效，留出时间在初现打滑
演变成不可挽回的弹出之前重新闭合并重新对中。该门限用于采集与专家控制，**不是**最终
成功判定的一部分。

## 专家数据不足 200 的说明

本任务的专家数据**已补齐到 200 条**（2026-08-27 在源机器清点确认，44 GB）。
此前记录为 189 条，该数字已过时。以下关于采集门控的说明保留作为背景；
[项目指南](../developer/project_guide.md)第 8 节指出 `peg_insert` 的正式采集
"在其 32/32 释放稳定性 smoke 通过之前保持关闭"，此前的 189/200 与这一状态一致，但本文档
不对具体缺失原因下结论。使用这批数据时请注意 seed 覆盖不完整。

## 依赖的第三方资产

| 资产 | 来源 | 许可证 |
|---|---|---|
| `peg_8mm.usd`、`hole_8mm.usd`、`franka_mimic.usd` | 内部溯源记录为"`franka2` 上已经存在的自包含 factory 资产" | ⚠️ **UNCONFIRMED** |

> **未确认项。** 根据 `factory_*` 的源命名，这些文件**很可能**派生自 NVIDIA
> Factory/Isaac，但它们抵达 "franka2" 这台机器之前的保管链**没有被独立追溯过**。
> "在 franka2 上"是一个位置，不是一条溯源链。在确认之前不要再分发其二进制文件。
> 这是 [`NOTICE.md`](../../NOTICE.md) 中两个待办的许可证问题之一。
>
> `peg_8mm_convex.usd` 是 ForceBench 自有的派生资产（把细杆的 SDF 碰撞替换为凸包），
> 但它派生自上述来源未确认的源文件。
