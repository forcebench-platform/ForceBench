# `faucet_turn` — 扳动龙头把手越过卡扣

| 项 | 值 |
|---|---|
| 任务类别 | **力分类**（决定停手时机的机构状态视觉上不可观测） |
| 场景 profile | `kitchen_sink` |
| 评测步数上限 | 600 控制帧（15 Hz，约 40 s） |
| 向量化评测批大小 | 10 环境 |
| 专家数据 | **200** episode（seed 0–199） |
| 代码真值 | [`forcebench/tasks/faucet_turn/spec.py`](../../forcebench/tasks/faucet_turn/spec.py)、[`outcome.py`](../../forcebench/tasks/faucet_turn/outcome.py) |

## 任务目标

**中文：** 转动龙头把手越过卡扣，把水打开。

**原文（模型收到的 prompt）：**

> Rotate the faucet knob past the detent to turn on the water.

## 场景

Lightwheel Kitchen 的 `Sink054` 水槽，原始把手被隐藏，取而代之的是一个任务自有的
旋转手柄：一根固定的水平轴从源龙头一侧的安装座伸出，手柄在竖直的 X-Z 平面内绕该轴
的 Y 轴转动，视觉上引用未经修改的源 `Sink054_handle` 网格。完整的 Lightwheel
`KitchenRoom.usd` 提供周围房间的视觉背景（背景物理全部移除）。水流是任务自有的
可见几何，开度随阀门角度变化。

## 力相关性

**这是一个真正需要力感知的任务**，机制与 `stove_turn_on` 同构，但曲线形状不同：
阻力从基础摩擦 `0.045 N·m` 先平缓上升到一个"肩部"，再在 `10°` 的接近区内快速爬升
到峰值 `0.20 N·m`，随后在 `0.6°` 的窄区内急剧泄载回基础值；紧接着是一段 `1.5°` 的
低载捕获区，再在 `4°` 内重新升到 `0.14 N·m` —— 这样期望的停止位置在两侧都有界。

与 `stove_turn_on` 完全一致的关键设计：**卡扣角度随 seed 变化（`[20°, 30°]` 均匀
采样），而峰值强度固定为 `0.20 N·m` 不随 seed 变化**，且**场景几何在所有 seed 上
完全固定**（`geometry_randomized: False`）。手柄外观不因卡扣位置改变，策略只能从
腕部力矩上感知那次骤降。

判分器读取的是阀门关节角与卡扣状态，不读取力传感器数值；策略看不到隐藏的卡扣角度。
详见[评测协议](../eval_protocol.md#关于判分与力信号的关系)。

## 物理量与随机化范围

| 量 | 取值 |
|---|---|
| **卡扣峰值角度** | **每个 seed 在 `[20°, 30°]` 内均匀随机** |
| 卡扣峰值力矩 | **固定 `0.20 N·m`**（不随 seed 变化） |
| 基础摩擦力矩 | `0.045 N·m` |
| 上升（接近）区宽度 | `10.0°` |
| 泄载区宽度 | `0.6°` |
| 泄载后低载捕获区宽度 | `1.5°` |
| 泄载后再上升宽度 | `4.0°` |
| 泄载后阻力 | `0.14 N·m` |
| 粘性阻尼 | `0.055 N·m·s/rad` |
| 摩擦过渡速度 | `0.08 rad/s` |
| 最大力矩限幅 | `0.55 N·m` |
| 阀门摩擦 | `1.4` |
| 阀门角度范围 | `0°` 到 `110°` |
| 成功角 | `卡扣峰值角 + 0.95°` |
| 场景几何 / 水槽位置 | **固定，不随 seed 变化** |

## 成功判定条件

判据来自 `outcome.py` 的两条 `OutcomeCriterion`，两条都满足才算成功：

| 判据字段 | 失败码 | 含义 |
|---|---|---|
| `detent_crossed` | `detent_not_crossed` | 阀门越过了隐藏的卡扣 |
| `water_visible` | `water_not_flowing` | 提交时水仍在可见地流动 |

判定细节：

- `detent_crossed` 是**粘性标志**：episode 过程中任一时刻满足
  `valve_angle >= success_angle` 就永久置位。
- `water_visible` 反映**提交时**的即时状态：水流在 `angle >= detent_center` 时可见，
  转回卡扣之前则消失。因此"越过后又转回去"会因这一条而失败。

> **与 `stove_turn_on` 的差异：** 本任务**没有**最终角容差判据。`stove_turn_on`
> 要求停在点火角 `±3°` 内，而 `faucet_turn` 只要求"越过卡扣且水还在流"。
> `spec.py` 中虽有一整组 `expert_backoff_*` 常量（回退目标 `峰值 + 0.65°`、容差
> `0.8°`、停止速度 `0.08 rad/s`），运行时也会把 `expert_post_peak_stop` 写进报告，
> 但该字段**不在** `outcome.py` 的判据列表中，因此**不影响成功判定**——它只被专家
> 生成器和离线分析使用。

## 依赖的第三方资产

| 资产 | 来源 | 许可证 |
|---|---|---|
| `Sink054`（水槽，含贴图） | Lightwheel Kitchen，与 `stove_turn_on` 同一版本 | **CC BY-NC 4.0（禁止商用）** |
| `KitchenRoom.usd`（背景房间） | 同上 | **CC BY-NC 4.0（禁止商用）** |

ForceBench 只分发自有的包装文件
（`assets/lightwheel_kitchen/Sink054/faucet_detent_valve.usda` 与
`robot_base_counter_extension.usda`）。**本任务受 CC BY-NC 4.0 约束，不得用于商业
用途。** 完整说明见 [`NOTICE.md`](../../NOTICE.md)。
