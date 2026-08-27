# 评测结果

本目录存放 ForceBench 的 leaderboard 数据。

## 这个目录是什么

`leaderboard.json` 是 leaderboard 的**机器可读权威来源**。网站、论文表格和任何
展示层都应当从它读取，而不是各自维护一份数字。它包含：

| 字段 | 内容 |
|---|---|
| `eval_protocol` | 评测协议的机器可读版本（seed 区间、步数上限、环境间距、统计要求等）。`notes_zh` 里是中文说明。**协议的权威定义在这里**，[`docs/eval_protocol.md`](../docs/eval_protocol.md) 是它的人类可读版本 |
| 逐格结果 | 每个 (任务, 模型) 组合的成功率、rollout 数、来源目录 |
| `missing` 行 | 尚无有效结果的格子，**显式列出而不是省略**，这样覆盖缺口是可见的 |
| `head_to_head` | 模型两两对比，含 Fisher 精确检验的 p 值与多重比较校正结果 |

## 结果覆盖状态

**这不是一个完成的 leaderboard。** 32 个 (任务, 模型) 格子中只有 **12 格**
有有效数据：

| 模型 | 状态 |
|---|---|
| π0.5 | 覆盖 8 个任务中的 **6 个**（缺 `gear_mesh`、`peg_insert`） |
| ForceVLA | 覆盖 8 个任务中的 **6 个**（缺 `gear_mesh`、`peg_insert`） |
| π0.5 + 力 6D | **只有故障修复前的评测，已作废，尚未重跑** |
| ForceFlow | **尚未训练** |

6 个任务 × 2 个模型 = 12 格，与 `leaderboard.json` 中 `status:"complete"`
的行数一致。有结果的六个任务是：`weight_sort`、`rattle_sort`、`stove_turn_on`、
`faucet_turn`、`radmanso_drawer_sort_table`、`nut_thread`。

## ⚠️ 两个已修复的评测故障

早期结果因为这两个故障**全部作废**。复用旧输出目录里的数字会重新引入错误：

1. **视觉串扰（visual crosstalk）** —— 克隆环境间距过小，相机拍到了相邻克隆环境的
   场景。**只有 `env_spacing=6` 的运行（输出目录带 `spacing6` 标记）是有效的。**
2. **evaluation arming bug** —— 导致 `radmanso_drawer_sort_table` 上所有模型都得到
   0/100。已被带 `_fixed_arming` 标记的重跑取代。

## ⚠️ 步数上限会翻转结论

`nut_thread` 在 600 帧上限下两个模型都是 **0%**；放宽到 1500 帧后变成
**21% / 36%**。因此任何引用成功率的地方都必须同时说明所用的步数上限。
详见[评测协议第 3 节](../docs/eval_protocol.md#3-评测步数上限按任务而定)。

## ⚠️ 不要从这些数字得出总体结论

六组 π0.5 vs ForceVLA 的逐任务对比中：

- 只有 **3 组**达到 `p<0.05`（Fisher 精确检验）；
- 其中 **2 组是无力基线（π0.5）更好**；
- 经 **Bonferroni 校正后只剩 1 组**站得住。

**因此不要写"力有帮助"或"力没用"这类总体结论。** 只陈述逐任务事实，并附上
95% Clopper-Pearson 区间和校正后的显著性。仅看原始百分比会大幅高估证据强度。

## 数据格式在哪定义

- 协议字段的权威定义：`leaderboard.json` 自身的 `eval_protocol` 字段
- 人类可读的协议说明：[`docs/eval_protocol.md`](../docs/eval_protocol.md)
- 单次评测产出的 `aggregate.json` schema：`forcebench_vectorized_policy_evaluation_v1`，
  由 [`scripts/evaluation/evaluate_vectorized_seed_range.sh`](../scripts/evaluation/evaluate_vectorized_seed_range.sh)
  生成

## 提交自己的结果

自动化的提交流程是 v1.0 的计划项，**尚未实现**。目前请按
[评测协议](../docs/eval_protocol.md)跑完后提 issue 或 PR，附上：

- 完整的 `aggregate.json`
- 所用的步数上限与环境间距
- 模型 / checkpoint 标识
- 成功率的 95% Clopper-Pearson 区间
