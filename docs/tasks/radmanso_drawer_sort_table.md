# `radmanso_drawer_sort_table` — 拉开抽屉并在阻力上升处停手

| 项 | 值 |
|---|---|
| 任务类别 | **力分类**（决定停手时机的阻力曲线视觉上不可观测） |
| 场景 profile | `simple_room_sort_table` |
| 评测步数上限 | 600 控制帧（15 Hz，约 40 s） |
| 向量化评测批大小 | 100 环境 |
| 专家数据 | **200** episode（seed 0–199） |
| 代码真值 | [`forcebench/tasks/radmanso_drawer_sort_table/spec.py`](../../forcebench/tasks/radmanso_drawer_sort_table/spec.py)、[`outcome.py`](../../forcebench/tasks/radmanso_drawer_sort_table/outcome.py) |

## 任务目标

**中文：** 把胡桃木 RADMANSO 柜子的中间抽屉朝机器人方向拉出，直到阻力上升为止，
然后停手并保持抽屉打开，不要把它拉出来。

**原文（模型收到的 prompt）：**

> Pull the middle drawer of the walnut RADMANSO cabinet toward the robot until
> resistance rises, then stop and hold the drawer open without pulling it out.

## 场景

一个 ArtVIP RADMANSO 三抽屉柜直立放在分拣桌上，缩放 `0.45`，位于 `(0.620, -0.275)`，
偏航 `180°`——柜体侧边与桌沿平行/垂直，局部 -Y 开启方向对齐世界 +Y，正对机器人。
三个把手 Xform 都被统一放大 2.5 倍，因此在 `0.45` 缩放下把手仍从抽屉面板凸出
约 11.25 mm。**生产任务固定为中间抽屉**（`SELECTABLE_DRAWERS = ("middle",)`），
上抽屉在运行时被锁定。

## 力相关性

**这是一个真正需要力感知的任务，而且是五个力分类任务中随机化最强的一个。**
抽屉的阻力随开度变化：先是一段自由滑动区（阻力 `0.35 N`），进入一段短的上升阻力带，
如果整条带被完全越过，闩扣会**不可逆地释放**（任务失败）。

关键在于**起步阻力位置、阻力带长度和峰值阻力这三者每个 seed 都不同**（尤其是峰值
阻力在 `5.5–9.0 N` 之间随机），而柜子的外观和几何完全不变。策略无法从视觉上判断
"这一次该在哪里停手"——它必须一边拉一边读腕部力，在阻力升到目标水平时立刻停住。
拉过头会崩掉闩扣，拉不够则没进入阻力区。

判分器读取的是环境内部的抽屉关节位置和**设计阻力曲线**沿轨迹的峰值
（`SPEC.resistance_force_n(...)` 的解析值），**不读取腕部力/力矩传感器数值**。
策略看不到这条隐藏曲线，因此只能靠力感知来定位停手点。详见
[评测协议](../eval_protocol.md#关于判分与力信号的关系)。

## 物理量与随机化范围

| 量 | 取值 |
|---|---|
| 自由滑动阻力 | `0.35 N`（固定） |
| **阻力起始开度** | **每个 seed 在 `[0.045, 0.090] m` 内均匀随机** |
| **阻力带长度** | **每个 seed 在 `[0.012, 0.030] m` 内均匀随机** |
| **闩扣崩开阻力（峰值）** | **每个 seed 在 `[5.5, 9.0] N` 内均匀随机** |
| 目标停手阻力 | `2.35 N`（固定；成功开度由此反解阻力曲线得到） |
| 闩扣崩开后阻力 | `0.0 N` |
| 闩扣崩开保持步数 | `2` |
| 成功开度容差 | `±0.0025 m` |
| 成功稳定帧数 | `15` 帧 |
| 安全停止开度 | `0.135 m`（独立保护，远小于资产授权的 `0.30 m` 限位） |
| 阻力曲线形状 | 三次 smoothstep（避免求解器在阻力带起点产生冲击） |
| 柜体位置 / 缩放 / 朝向 | **固定，不随 seed 变化** |

**成功开度的定义方式值得注意：** 它不是一个固定距离，而是在该 seed 采样出的阻力
曲线上**反解**出 `阻力 = 2.35 N` 的那个开度（`target_opening_for_force()` 用 48 步
二分法求得）。由于起始位置、带长和峰值都随 seed 变化，同一个"停手阻力"对应的物理
开度每次都不同。这正是"必须看力、不能看位置"的具体体现。

## 成功判定条件

判据来自 `outcome.py` 的五条 `OutcomeCriterion`，五条全部满足才算成功。
注意最后一条的 `expected=False`：

| 判据字段 | 失败码 | 期望值 | 含义 |
|---|---|---|---|
| `drawer_open_enough` | `drawer_not_in_resistance` | `True` | 抽屉进入了阻力区 |
| `within_target_window` | `drawer_missed_target` | `True` | 抽屉停在目标窗口内 |
| `stable_at_target` | `drawer_not_stable` | `True` | 抽屉在目标处保持稳定 |
| `entered_high_resistance` | `resistance_not_detected` | `True` | 轨迹达到了设计的高阻力带 |
| `latch_broken` | `drawer_latch_broken` | **`False`** | 崩开闩扣即判失败 |

具体阈值（`spec.py: evaluate_final_state`）：

- **目标窗口**：`|实际开度 - 该 seed 的成功开度| <= 0.0025 m`。这是本 benchmark 中最
  严格的位置容差之一（2.5 mm）。
- **稳定**：在目标处保持 `>= 15` 个控制帧（15 Hz 下 1 秒）。
- **进入高阻力**：轨迹上出现过的设计阻力峰值 `> 0.35 + 0.25 = 0.60 N`。
- **闩扣未崩**：`latch_broken == False`。一旦整条阻力带被越过，闩扣不可逆地释放，
  该 episode 必定失败。

## 依赖的第三方资产

| 资产 | 来源 | 许可证 |
|---|---|---|
| RADMANSO 三抽屉柜 | ArtVIP 资产包 | ⚠️ **UNCONFIRMED — 本项目尚未核实其许可证** |

> **未确认项。** 该资产的许可证**尚未被本项目独立核实**。在确认之前不要再分发其
> 二进制文件。这是 [`NOTICE.md`](../../NOTICE.md) 中两个待办的许可证问题之一，
> 在 v1.0 之前必须解决。
