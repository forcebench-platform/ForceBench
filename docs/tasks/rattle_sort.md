# `rattle_sort` — 按内部配重分拣包裹

| 项 | 值 |
|---|---|
| 任务类别 | **力分类**（决定正确行为的属性视觉上不可观测） |
| 场景 profile | `parcel_sort_station` |
| 评测步数上限 | 600 控制帧（15 Hz，约 40 s） |
| 向量化评测批大小 | 100 环境 |
| 专家数据 | **200** episode（seed 0–199） |
| 代码真值 | [`forcebench/tasks/rattle_sort/spec.py`](../../forcebench/tasks/rattle_sort/spec.py)、[`outcome.py`](../../forcebench/tasks/rattle_sort/outcome.py) |

## 任务目标

**中文：** 拿起包裹并保持静止地检查它，不要摇晃。把配重偏心的包裹放进左侧黑色
篮筐、把配重均匀的包裹放到右侧传送带上，然后松手释放。

**原文（模型收到的 prompt）：**

> Lift and inspect the parcel without shaking it. Put an imbalanced parcel in the
> left black basket or a uniform parcel on the right conveyor, then release it.

## 场景

一个包裹（`0.054 x 0.120 x 0.02328 m`）放在桌面上，两个投放目标分别是：左侧黑色
篮筐（中心 `(0.64, -0.22)`，开口 `0.235 x 0.145 m`）和右侧传送带（可用投放区中心
`(0.69, 0.22)`，可用面积 `0.24 x 0.22 m`）。`parcel_sort_station` 场景 profile 还会
添加无碰撞的工业货架、背景传送带、安全护栏和货架包裹作为视觉背景。

## 力相关性

**这是一个真正需要力感知的任务，而且比 `weight_sort` 更严格。** 两个类别的包裹
**总质量精确相同**，都是 `0.18 kg`：

- **uniform（均匀）**：壳体 `0.18 kg`，质心居中。
- **imbalanced（偏心）**：壳体 `0.12 kg` + 一个 `0.06 kg` 的松动内衬物，
  质心沿包裹局部 Y 轴偏移 `±0.020 m`。

因此连"称重"都不能区分它们——策略必须感知**力矩**。在 `0.18 kg` 下 `20 mm` 的偏移
产生约 `0.035 N·m` 的重力力矩，这一量级明确可测，同时仍允许稳定的中心抓取。

与 `weight_sort` 同理，判分器读取的是隐藏真值 `box_kind` 与最终位置，不读取力传感器
数值；策略看不到该真值且两类外观一致，因此只能靠物理交互分辨。详见
[评测协议](../eval_protocol.md#关于判分与力信号的关系)。

> **诚实性说明：** prompt 要求"不要摇晃"并做静态检查，`spec.py` 中也确实定义了一整组
> 检查相关常量（最小抬起高度 `0.070 m`、要求静止保持 `2.0 s`、控制步长上限
> `0.003 m` / `0.020 rad`、包裹线速度上限 `0.030 m/s` 等）。但这些常量**只被专家
> 生成器使用**，`evaluate_final_state` 与 `outcome.py` 的成功判定**不检查**它们。
> 也就是说，一个"摇晃着"完成分拣的策略在当前实现下仍会被判为成功。

## 物理量与随机化范围

| 量 | 取值 |
|---|---|
| 包裹总质量（两类相同） | `0.18 kg` |
| uniform 壳体质量 | `0.18 kg`（无内衬物） |
| imbalanced 壳体质量 | `0.12 kg` + `0.06 kg` 松动内衬物 |
| 质心前后偏移 | `±0.020 m`（沿局部 Y 轴；`front` 为 `+`、`back` 为 `-`） |
| 由偏移产生的重力力矩 | 约 `0.035 N·m` |
| 包裹摩擦 | `1.5`；角阻尼 `1.0` |
| 抓取点 X | 均匀采样 `[0.40, 0.48] m` |
| 抓取点 Y | 均匀采样 `[-0.025, 0.06] m` |
| 包裹 yaw | 固定 `pi/2 rad` |
| 包裹根部 Z | 固定 `0.002 m` |

**类别分配：** 与 `weight_sort` 的简单奇偶不同，`box_kind_for_seed()` 把 seed 按
**两两成对**分组，在每一对内部用确定性随机排列决定顺序。因此每一对连续 seed
（`2k`, `2k+1`）必定一轻一重各一个，但对内先后顺序由 seed 决定而非固定——这既保证了
类别严格平衡，又避免"偶数=某类"这种可被模型直接记住的规律。偏心包裹的重端朝
`front` 还是 `back` 也由同一 seed 随机决定。

## 成功判定条件

判据来自 `outcome.py` 的三条 `OutcomeCriterion`，三条全部满足才算成功：

| 判据字段 | 失败码 | 含义 |
|---|---|---|
| `box_in_correct_destination` | `parcel_misclassified` | 包裹在与其质量分布匹配的目标处 |
| `box_released` | `parcel_not_released` | 夹爪已释放包裹 |
| `box_settled` | `parcel_not_settled` | 包裹在提交时已静止 |

具体阈值（`spec.py: evaluate_final_state`）：

- **黑色篮筐内**：XY 位于开口矩形内且留 `0.014 m` 余量，根部高度 `0 <= z <= 0.095 m`。
- **传送带上**：XY 位于可用区域内且留 `0.014 m` 余量，根部高度 `0 <= z <= 0.070 m`。
- **目标对应关系**：`imbalanced` → 黑色篮筐；`uniform` → 传送带。
- **已释放**：`gripper_open_fraction >= 0.65`。
- **已静止**：包裹线速度模长 `<= 0.15 m/s`。

放错目标与没放进任何目标同样被区分记录在 `placement_failure_reason`
（`parcel_in_wrong_destination` / `parcel_not_in_any_destination`）。

## 依赖的第三方资产

| 资产 | 来源 | 许可证 |
|---|---|---|
| `flatbox_a07`（包裹壳体视觉） | NVIDIA Omniverse SimReady 目录，`SimReady_Containers_Shipping_01_NVD@10010.zip` | NVIDIA Omniverse/Isaac 资产条款 |
| `container_f27`（黑色篮筐） | NVIDIA Omniverse SimReady 目录 | 同上 |
| `ConveyorBelt_A04`（传送带视觉） | NVIDIA Omniverse Isaac 资产（`Isaac/Props/Conveyors/`） | 同上 |
| 背景包裹若干（`multidepthbox_a04` 等） | NVIDIA Omniverse SimReady 目录 | 同上 |

ForceBench 只分发归一化包装文件 `assets/normalized/rattle_sort/box_shell.usda`。
完整的许可证说明见 [`NOTICE.md`](../../NOTICE.md)。
