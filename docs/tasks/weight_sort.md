# `weight_sort` — 按重量分拣箱子

| 项 | 值 |
|---|---|
| 任务类别 | **力分类**（决定正确行为的属性视觉上不可观测） |
| 场景 profile | `weight_sort_residential` |
| 评测步数上限 | 600 控制帧（15 Hz，约 40 s） |
| 向量化评测批大小 | 100 环境 |
| 专家数据 | **200** episode（seed 0–199） |
| 代码真值 | [`forcebench/tasks/weight_sort/spec.py`](../../forcebench/tasks/weight_sort/spec.py)、[`outcome.py`](../../forcebench/tasks/weight_sort/outcome.py) |

## 任务目标

**中文：** 拿起箱子以判断它是轻还是重。把轻箱放进桌边的黑色容器、把重箱放进蓝色
容器，然后松手释放。

**原文（模型收到的 prompt）：**

> Pick up the box to determine whether it is light or heavy. Put a light box into
> the black container or a heavy box into the blue container at the side of the
> table, then release it.

## 场景

桌面上有一个待抓取的箱子和两个位于桌边的容器：黑色容器在 `(0.52, -0.16)`，
蓝色容器在 `(0.52, 0.16)`（世界坐标，米）。两个容器都是缩放 `0.16` 的
`container_f27`，配任务自有的碰撞外壳，开口有效尺寸 `0.16 x 0.10 m`。

## 力相关性

**这是一个真正需要力感知的任务。** 轻箱与重箱使用**完全相同**的视觉资产、缩放、
材质、摩擦和碰撞几何——从任何相机视角都无法区分。唯一的差别是质量。因此策略必须
先抓起箱子、从交互中感知负载，才能决定往哪个容器放。

需要注意判分逻辑（详见[评测协议](../eval_protocol.md#关于判分与力信号的关系)）：
判分器读取的是环境内部的隐藏真值 `box_kind` 和箱子最终位置，**它本身不读取力传感器
数值**。这不是漏洞——策略看不到 `box_kind`，而两类箱子外观完全一致，所以策略只能
通过物理交互（力）来分辨。判分器的作用是检查"分类结果是否正确"，而不是检查
"策略是否使用了力"。

## 物理量与随机化范围

| 量 | 取值 |
|---|---|
| 轻箱质量 | `0.06 kg` |
| 重箱质量 | `0.18 kg` |
| 箱体视觉缩放 | `0.36`（源资产原生尺寸 `0.1506 x 0.2118 x 0.0836 m`） |
| 箱体摩擦 | `1.5`，combine 模式 `min` |
| 抓取点 X | 均匀采样 `[0.67, 0.72] m` |
| 抓取点 Y | 均匀采样 `[-0.20, 0.08] m` |
| 箱体 yaw | 固定 `pi/2 rad` |
| 箱体根部 Z | 固定 `0.003 m` |

**类别分配：** `box_kind = BOX_KINDS[seed % 2]`，即**偶数 seed 为轻箱、奇数 seed
为重箱**。这保证任何连续偶数条 seed 区间都是两类严格 1:1 平衡。评测时请使用连续
的 seed 区间以保持这一平衡；也请注意，seed 本身泄露类别信息，因此不要把 seed 作为
任何形式的模型输入。

## 成功判定条件

判据来自 `outcome.py` 的三条 `OutcomeCriterion`，三条全部满足才算成功：

| 判据字段 | 失败码 | 含义 |
|---|---|---|
| `box_in_correct_container` | `box_misclassified` | 箱子落在与其隐藏质量匹配的容器中 |
| `box_released` | `box_not_released` | 夹爪已释放箱子 |
| `box_settled` | `box_not_settled` | 箱子在提交时已静止 |

具体阈值（`spec.py: evaluate_final_state`）：

- **在正确容器内**：箱子 XY 位于目标容器开口矩形内，且四周留有
  `placement_center_margin_m = 0.018 m` 的余量（即有效半宽 `0.062 x 0.032 m`）；
  同时箱子根部高度满足 `0 <= z <= 0.060 m`。轻箱 → 黑色容器，重箱 → 蓝色容器。
- **已释放**：`gripper_open_fraction >= 0.65`。
- **已静止**：箱子线速度模长 `<= 0.15 m/s`。

放错容器与完全没放进任何容器会被区分为两个不同的失败原因
（`box_in_wrong_container` / `box_not_in_any_container`），记录在
`placement_failure_reason` 字段里——这正是分析"策略是否学会了分类"的关键信号。

## 依赖的第三方资产

| 资产 | 来源 | 许可证 |
|---|---|---|
| `flatbox_a02`（箱体视觉） | NVIDIA Omniverse SimReady 目录 | NVIDIA Omniverse/Isaac 资产条款（非开源，二进制本身的再分发受限） |
| `container_f27`（两个容器） | NVIDIA Omniverse SimReady 目录 | 同上 |

ForceBench 只分发自己编写的归一化包装文件
`assets/normalized/weight_sort/box.usda`，它按路径**引用**源文件但不嵌入其内容。
二进制必须由使用者自行从 NVIDIA 官方目录获取。完整的许可证说明见
[`NOTICE.md`](../../NOTICE.md)。
