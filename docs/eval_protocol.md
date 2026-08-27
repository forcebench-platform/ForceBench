# 评测协议

本文规定 ForceBench 的评测方式。**`results/leaderboard.json` 的 `eval_protocol`
字段是机器可读的权威来源**，本文是它的人类可读版本；两者不一致时以数据文件为准。

## 1. seed 与 rollout 数

| 项 | 规定 |
|---|---|
| seed 区间 | **固定 `200`–`299`** |
| 每个任务的 rollout 数 | **100** |
| seed 与训练数据的关系 | 专家数据使用 seed `0`–`199`，评测区间与之**不重叠** |

seed 必须是连续区间，不能挑选。原因见各任务卡片：多个任务用 seed 的奇偶或配对结构
来保证两个类别严格 1:1 平衡（如 [`weight_sort`](./tasks/weight_sort.md) 偶数 seed 为
轻箱、[`rattle_sort`](./tasks/rattle_sort.md) 每对 seed 内一类一个）。挑选 seed 会
破坏类别平衡，使成功率不可比。

**seed 本身泄露类别信息，因此绝不能以任何形式作为模型输入。**

## 2. 运行时契约

| 项 | 取值 |
|---|---|
| 物理频率 | 60 Hz |
| 控制 / 策略频率 | 15 Hz |
| 求解器位置迭代 | 64 |
| 每次推理执行的动作块 | **50 帧**（`FORCEBENCH_EXECUTE_CHUNK` 默认值） |
| **克隆环境间距** | **必须为 6 m**（`FORCEBENCH_ENV_SPACING=6.0`） |

### ⚠️ 环境间距必须是 6 m

**这不是性能调参，而是结果有效性的前提。** 间距更小时，克隆环境的相机会拍到相邻
克隆环境的场景，造成视觉串扰（visual crosstalk），策略看到的图像里混入了别的
环境的物体。

**早期的评测结果因此全部作废。** 只有以 `env_spacing=6` 运行、输出目录带
`spacing6` 标记的结果是有效的。复用旧输出目录中的数字会重新引入这个错误。

## 3. 评测步数上限按任务而定

| 任务 | 步数上限 |
|---|---|
| `gear_mesh`、`nut_thread`、`peg_insert`（装配类） | **1500** 控制帧 |
| 其余五个任务 | **600** 控制帧 |

该上限由 `TaskManifest.evaluation_max_steps` 声明，可用
`scripts/evaluation/resolve_task_max_steps.py <task>` 查询；把 `max_steps` 传
`auto` 即自动解析。

### ⚠️ 步数上限本身会翻转结论

这一点必须写进任何报告结果的地方：**`nut_thread` 在 600 帧上限下，两个模型都是
0%；把上限放宽到 1500 帧后，成功率变成 21% / 36%。**

换句话说，一个过紧的步数预算会把"两个模型都不行"这一结论，变成"两个模型都能做、
且有明显差距"。**报告成功率时必须同时报告所用的步数上限**，否则数字不可比、也不可
复现。跨任务或跨论文比较时，请先确认双方的步数预算一致。

## 4. 成功判定

每个任务的成功判定由该任务的 `outcome.py` 声明为一组 `OutcomeCriterion`，
全部满足才算成功。逐任务的判据与阈值见[任务说明](./tasks/README.md)。

评测循环在**每个控制帧之后**检查成功判定，并在判定稳定后终止
（`--terminate_on_success`，可用 `FORCEBENCH_TERMINATE_ON_SUCCESS=0` 关闭）。
终止后还会额外走 `--final_settle_frames`（默认 8）帧让场景沉降，再做最终判定。

### 关于判分与力信号的关系

这是最容易被误读的一点，单独说明。

五个力分类任务的判分依据是**环境内部的隐藏真值**与最终位姿：

| 任务 | 判分读取的隐藏真值 |
|---|---|
| `weight_sort` | `box_kind`（轻 / 重）+ 箱子最终位置 |
| `rattle_sort` | `box_kind`（均匀 / 偏心）+ 包裹最终位置 |
| `stove_turn_on` | 旋钮关节角、卡扣越过标志、火焰可见状态 |
| `faucet_turn` | 阀门关节角、卡扣越过标志、水流可见状态 |
| `radmanso_drawer_sort_table` | 抽屉关节位置 + 设计阻力曲线沿轨迹的解析峰值 |

**判分器本身不读取腕部力/力矩传感器的数值。** 这不是漏洞，而是刻意的设计：

1. 策略**看不到**这些隐藏真值；
2. 两个类别的**视觉外观完全一致**——`weight_sort` 的轻重箱使用同一个视觉资产、
   同样的缩放和材质；`rattle_sort` 更进一步，两类的**总质量都精确相同**
   （`0.18 kg`），连称重都不能区分，必须感知力矩；
3. 因此策略**只能**通过物理交互（力）来分辨。

判分器的职责是检查"分类 / 停手的结果是否正确"，而不是检查"策略内部是否用到了力"。
后者无法从外部黑盒验证；能验证的是"缺少力信息的策略无法稳定超过随机水平"——这正是
把有力 / 无力策略放在同一套判据下对比的意义。

**反过来，三个接触密集任务（`peg_insert`、`gear_mesh`、`nut_thread`）的判据完全是
几何量，不涉及任何隐藏的力学真值**，因此它们在当前实现下不构成关于力感知的证据。
详见[任务说明索引](./tasks/README.md#第二类接触密集任务3-个)。

## 5. 统计报告要求

**成功率必须与 95% Clopper-Pearson 精确置信区间一同报告。** 不接受裸百分比。

理由：本 benchmark 每任务 100 次 rollout，属于小样本区间。在 n=100 下，
一个 30% 的成功率其 95% CI 大约是 `[21%, 40%]`——近 20 个百分点的宽度。仅凭点估计
比较两个模型，极易把噪声当成差异。Clopper-Pearson 是二项比例的精确（保守）区间，
在小 n 和接近 0% / 100% 的边界处比正态近似可靠。

比较两个模型时：

- 逐任务使用 **Fisher 精确检验**（2×2 列联表，小样本下比卡方可靠）；
- 同时比较多个任务时，必须做**多重比较校正**（如 Bonferroni）。8 个任务上各做一次
  检验，在零假设下也有很大概率至少出现一个 p<0.05。

### ⚠️ 不要写总体性结论

当前基线数据里，六组 π0.5 vs ForceVLA 的逐任务对比中：

- 只有 **3 组**达到 `p<0.05`（Fisher 精确检验）；
- 这 3 组里有 **2 组是无力基线（π0.5）更好**；
- 经 **Bonferroni 校正后只剩 1 组**站得住。

因此**任何文档、论文或 README 都不应写"力有帮助"或"力没用"这类总体结论**。
只陈述逐任务事实，并附上区间与校正后的显著性。逐任务的具体数字见
`results/leaderboard.json` 的 `head_to_head` 字段。

## 6. 怎么跑

批量评测一段 seed 区间的入口是
`scripts/evaluation/evaluate_vectorized_seed_range.sh`：

```bash
export FORCEBENCH_POLICY_HOST=127.0.0.1
export FORCEBENCH_POLICY_PORT=8002
export FORCEBENCH_POLICY_ADAPTER=openpi_ws
export FORCEBENCH_ENV_SPACING=6.0        # 必须；见第 2 节

./scripts/evaluation/evaluate_vectorized_seed_range.sh \
  weight_sort 200 299 auto \
  /path/to/evaluations/weight_sort/my_model_spacing6
```

参数依次为：任务、起始 seed、结束 seed、步数上限（`auto` 表示按任务 manifest 解析）、
输出目录、可选的批大小（默认按场景 profile 解析）。

脚本会按批把 seed 区间跑完，然后把逐 seed 的 `summary.json` 汇总成一个
`aggregate.json`，schema 名为 `forcebench_vectorized_policy_evaluation_v1`，
包含 `success_rate`、`completed_rollouts`、`records`（逐 seed 明细，含
`success_report` 里的逐判据结果）等字段。

> **注意：** `./run_native.sh policy`（单环境评测）已被归档，直接调用会以退出码 2
> 提示改用 `policy-vector`。上面的批量脚本是当前受支持的入口。

## 7. 当前基线结果的覆盖状态

**这不是一个完成的 leaderboard。** 截至本文撰写时：

| 模型 | 状态 |
|---|---|
| π0.5 | 覆盖 8 个任务中的 **7 个**（缺 `gear_mesh`、`peg_insert`） |
| ForceVLA | 覆盖 8 个任务中的 **7 个**（缺 `gear_mesh`、`peg_insert`） |
| π0.5 + 力 6D | **只有故障修复前的评测，已作废，尚未重跑** |
| ForceFlow | **尚未训练** |

即 32 个 (任务, 模型) 格子中只有 **12 格**有有效数据。缺失的格子在
`results/leaderboard.json` 中以显式的 `missing` 行呈现，而不是省略——这样覆盖缺口是
可见的，而不需要从空白单元格去推断。

另有一处已修复的评测故障需要知道：一个 **evaluation arming bug** 曾导致
`radmanso_drawer_sort_table` 上所有模型都得到 0/100。该结果已被带 `_fixed_arming`
标记的重跑取代。同样，旧目录里的数字不要复用。

## 8. 尚未实现的部分

保持诚实：以下都是 v1.0 的待办项，**现在还不存在**。

- 自动计算并写入 Clopper-Pearson 区间的脚本——目前 `aggregate.json` 只写
  `success_rate`，区间需要自行计算。
- Fisher 精确检验 / Bonferroni 校正的脚本——当前 `head_to_head` 的数字是人工核算后
  写入 `results/leaderboard.json` 的。
- 提交结果到 leaderboard 的自动流程。
