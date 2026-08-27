# `stove_turn_on` — 旋转灶台旋钮越过点火卡扣

| 项 | 值 |
|---|---|
| 任务类别 | **力分类**（决定停手时机的机构状态视觉上不可观测） |
| 场景 profile | `kitchen_stove` |
| 评测步数上限 | 600 控制帧（15 Hz，约 40 s） |
| 向量化评测批大小 | 10 环境 |
| 专家数据 | **200** episode（seed 0–199） |
| 代码真值 | [`forcebench/tasks/stove_turn_on/spec.py`](../../forcebench/tasks/stove_turn_on/spec.py)、[`outcome.py`](../../forcebench/tasks/stove_turn_on/outcome.py) |

> ⚠️ **已知问题：** `KNOWN_ISSUES.md` 第 4 条记录了本任务的一个未解决的几何常量
> 分歧——`SPEC.burner_center_m` 的 Y 值与测试期望相差约 2.6 cm、Z 值相差约 0.8 cm。
> 使用本任务前请先阅读该条目。

## 任务目标

**中文：** 转动中间的灶台旋钮，越过卡扣点火。

**原文（模型收到的 prompt）：**

> Turn the center stove knob past the detent to ignite the burner.

## 场景

Lightwheel Kitchen 的 `Stovetop012` 灶台（经过 ForceBench 自有的"单灶头"视觉派生
处理），前面板上有一个可旋转的中央旋钮。旋钮是一个独立的固定基座旋转 articulation，
在运行时接收非线性卡扣力矩。点火后可见的火焰是任务自有的、无碰撞的发光几何。

## 力相关性

**这是一个真正需要力感知的任务。** 旋钮的阻力曲线在角度上是非线性的：从基础摩擦
`0.025 N·m` 开始，沿一段长而平缓的上升区爬升到峰值 `0.12 N·m`，随后在极窄的
`0.5°` 区间内**急剧泄载**回到基础值，再在其后 `4°` 内重新升到 `0.08 N·m`。

关键在于：**卡扣峰值出现的角度每个 seed 都不同**（在 `[60°, 90°]` 内均匀采样），
而**峰值大小固定为 `0.12 N·m` 不随 seed 变化**。这一设计是刻意的——如果强度也随
seed 变化，seed 就变成了物理属性的捷径。旋钮外观在所有角度、所有 seed 下都一样，
策略无法从视觉上判断"卡扣在哪、有没有越过"，只能从腕部力矩上感知那次骤降。

判分器读取的是环境内部的旋钮关节角与卡扣状态，不读取力传感器数值；策略看不到
隐藏的卡扣角度，因此只能靠力感知来定位它。详见
[评测协议](../eval_protocol.md#关于判分与力信号的关系)。

## 物理量与随机化范围

| 量 | 取值 |
|---|---|
| **卡扣峰值角度** | **每个 seed 在 `[60°, 90°]` 内均匀随机** |
| 卡扣峰值力矩 | **固定 `0.12 N·m`**（不随 seed 变化） |
| 基础摩擦力矩 | `0.025 N·m` |
| 上升区宽度 | `detent_center - closed_angle`（即从 0° 一路爬升到峰值） |
| 泄载区宽度 | `0.5°` |
| 泄载后低载区宽度 | `0.0°`（最小值是单个转折点，不是平台） |
| 泄载后再上升宽度 | `4.0°` |
| 泄载后阻力 | `0.08 N·m` |
| 粘性阻尼 | `0.055 N·m·s/rad` |
| 摩擦过渡速度 | `0.08 rad/s` |
| 最大力矩限幅 | `0.35 N·m` |
| 旋钮摩擦 | `2.2`，combine 模式 `max` |
| 旋钮角度范围 | `0°` 到 `120°` |
| 成功角 | `卡扣峰值角 + 0.5°` |
| 成功角容差 | `±3.0°` |
| 场景几何 | **固定，不随 seed 变化** |

## 成功判定条件

判据来自 `outcome.py` 的三条 `OutcomeCriterion`，三条全部满足才算成功：

| 判据字段 | 失败码 | 含义 |
|---|---|---|
| `ignition_detent_crossed` | `ignition_detent_not_crossed` | 旋钮越过了隐藏的点火卡扣 |
| `within_final_angle_tolerance` | `knob_not_at_ignition_angle` | 旋钮停在采样得到的点火角附近 |
| `flame_visible` | `burner_not_lit` | 提交时灶头火焰仍然可见 |

判定细节：

- `ignition_detent_crossed` 是**粘性标志**：episode 过程中任一时刻满足
  `knob_angle >= success_angle` 就永久置位，即使之后又转回去也保持为真。
- `within_final_angle_tolerance` 只看**最终**角度：
  `|最终角 - success_angle| <= 3.0°`。这一条要求策略停在泄载后的最小值附近——
  仅仅越过卡扣然后一路转到底是**不够的**。
- `flame_visible` 反映**提交时**的即时状态：火焰在 `angle >= detent_center` 时可见，
  转回卡扣之前则熄灭。

这三条组合起来就是"越过卡扣 → 立刻停手 → 保持住"这一完整行为，而不只是"转得够远"。

## 依赖的第三方资产

| 资产 | 来源 | 许可证 |
|---|---|---|
| `Stovetop012`（灶台，含贴图） | Lightwheel Kitchen，commit `10b0f7a8...` | **CC BY-NC 4.0（禁止商用）** |

ForceBench 分发的是自有的包装文件和派生的单灶头网格
（`assets/lightwheel_kitchen/Stovetop012/stove_detent_knob.usda` 等）；Lightwheel
版本的 `LICENSE.txt` 已逐字包含在 `assets/lightwheel_kitchen/` 中。**本任务受
CC BY-NC 4.0 约束，不得用于商业用途。** 完整说明见 [`NOTICE.md`](../../NOTICE.md)。
