# ForceBench

> ## ⚠️ 状态：v0.1 原型（prototype），不是成品
>
> 代码和测试已经在仓库里并且能跑（无 Isaac Sim 环境下 102/109 通过，
> 见 [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md)）。但是：
>
> - **数据托管尚未搭建** —— 没有 episode 数据，没有下载脚本
> - **资产下载脚本尚未实现** —— 七个任务的第三方资产需手工获取
> - **baseline 评测不完整** —— 32 个 (任务, 模型) 格子中只有约 12 格有有效数据
> - **两处第三方资产许可证尚未确认**
>
> v1.0 清单见 [`CHANGELOG.md`](./CHANGELOG.md)。
> **现阶段请不要把它当作一个已完成的 benchmark 来引用。**

**力/力矩感知到底能不能帮到操作策略？需要多少？**

ForceBench 是一个构建在 Isaac Lab 上的仿真 benchmark，用来测量 6 轴力/力矩反馈
是否——以及在多大程度上——能改善视觉-语言-动作（VLA）操作策略。

它的设计核心是：任务在**几何上**从视觉和本体感知出发都是可解的，但决定
**什么才是正确行为**的那个物体属性（轻重、内部配重、机构阻力）**视觉上不可观测**。
于是策略要么必须用力才能超过随机水平，要么根本不需要力——而 ForceBench 的作用就是
逐任务、逐策略地告诉你是哪一种。

八个任务，一套共享的机器人/控制器/观测契约，一个策略接口。**换模型，不换 harness。**

## 八个任务

### 五个力分类任务

策略必须推断的物体属性**完全不可视觉观测**，力是唯一携带该信息的通道：

| 任务 | 中文名 | 策略必须靠力推断什么 | 物理量变化 | 专家数据 |
|---|---|---|---|---|
| [`weight_sort`](./docs/tasks/weight_sort.md) | 按重量分拣箱子 | 箱子轻重（外观完全相同） | 质量 `0.06` / `0.18 kg` | 200 |
| [`rattle_sort`](./docs/tasks/rattle_sort.md) | 按内部配重分拣包裹 | 内部质量是否偏心（**总质量恒为 `0.18 kg`**） | 质心偏移 `±0.020 m` | 200 |
| [`stove_turn_on`](./docs/tasks/stove_turn_on.md) | 旋转灶台旋钮越过点火卡扣 | 旋钮是否已越过卡扣 | 非线性卡扣力矩，峰值 `0.12 N·m` | 200 |
| [`faucet_turn`](./docs/tasks/faucet_turn.md) | 扳动龙头把手越过卡扣 | 同上，龙头把手 | 非线性卡扣力矩，峰值固定 `0.20 N·m` | 200 |
| [`radmanso_drawer_sort_table`](./docs/tasks/radmanso_drawer_sort_table.md) | 拉开抽屉并在阻力上升处停手 | 阻力上升到何时该停手 | 起步阻力随机 `5.5–9.0 N` | 200 |

`rattle_sort` 值得特别注意：两类包裹**总质量精确相同**，连称重都无法区分，
策略必须感知**力矩**。

### 三个接触密集任务

**为了覆盖面而纳入，但必须如实说明：** 这三个任务的 manifest 自称 "force-sensitive"，
**而当前实现的成功判定只检查最终位姿**（居中/就位/对齐/释放/退回），完全不读取
wrench 信号；`peg_insert` 的 `spec.py` 里甚至没有任何力或质量常量。

**该措辞是前瞻性的（aspirational），当前实现下并不检验力感知。**
它们是有用的接触密集操作任务，但它们**不构成**关于力感知有无帮助的证据——
上面五个任务才是。

| 任务 | 中文名 | 实际变化的量 | 当前的力角色 | 专家数据 |
|---|---|---|---|---|
| [`peg_insert`](./docs/tasks/peg_insert.md) | 销轴插孔 | 目标孔平面位置 | **无** —— spec 中无任何力/质量常量 | 200 |
| [`gear_mesh`](./docs/tasks/gear_mesh.md) | 齿轮啮合装配 | 底板位置、齿相位抖动 | 仅质量缩放用于接触真实感，**不参与判分** | ❌ **无数据** |
| [`nut_thread`](./docs/tasks/nut_thread.md) | 螺母拧入螺栓 | 螺栓平面位置 | **不参与判分** | 200 |

### 关于判分逻辑的常见误读

五个力分类任务的判分依据是**环境内部的隐藏真值**（如 `box_kind`）与最终位姿，
**判分器本身不读取力传感器数值**。

**这不是漏洞。** 策略看不到隐藏真值，而两个类别的视觉外观完全一致，所以策略
**只能**通过物理交互（力）来分辨。判分器检查的是"分类结果对不对"，而不是
"策略内部有没有用力"——后者无法从外部黑盒验证。完整讨论见
[评测协议](./docs/eval_protocol.md#关于判分与力信号的关系)。

## 与 ForceVLA 论文的区别

[ForceVLA](https://arxiv.org/abs/2505.22159)（NeurIPS 2025）提出了一个力融合 MoE
架构，并用一份 5 任务的真机数据集来验证它。

**ForceVLA 是"一个方法 + 一份真机数据集"；ForceBench 是"方法无关的评测基础设施 +
力信息强度分级"。** ForceBench 不是一篇竞争性的架构论文，而是做
*方法无关*比较所需要的那个 harness：

- 一套固定的仿真器契约（同样的机器人、控制器、物理、相机）；
- 一个**力信息强度的分级阶梯**（仅本体感知 → 原始 6 维 wrench → 架构级融合）；
- 一套共享的评测协议。

这样一来，"力有帮助"才可能成为一个 benchmark 结果，而不是某一篇论文特定实验设置下
的一个论断。

## 快速开始

```bash
git clone https://github.com/pkuhanxuan/ForceBench
cd ForceBench
pip install -r requirements-dev.txt

# 契约/注册表测试 —— 不需要 Isaac Sim
python -m pytest tests/
# 预期：102 passed, 5 failed, 2 skipped（5 个失败是已知的，见 KNOWN_ISSUES.md）
```

真正跑仿真需要 **Isaac Lab 2.3.2 / Isaac Sim 5.1** 以及就位的第三方资产
（**目前没有下载脚本**，需按 [`NOTICE.md`](./NOTICE.md) 手工获取）：

```bash
export FORCEBENCH_DATA_ROOT=/your/path/forcebench_data   # 必须设置，否则指向原开发机路径
export FORCEBENCH_ENV_SPACING=6.0                        # 必须为 6，否则视觉串扰

# 专家数据采集（一个 Isaac 进程，32 个克隆环境）
./run_native.sh expert --task weight_sort --seeds 0-31

# 策略评测
export FORCEBENCH_POLICY_ADAPTER=openpi_ws
./scripts/evaluation/evaluate_vectorized_seed_range.sh \
  weight_sort 200 299 auto /path/to/output_spacing6
```

完整步骤、环境变量清单和**各入口的可用性**（部分 `run_native.sh` 入口所依赖的脚本
不在本次公开提取范围内）见[快速开始](./docs/getting_started.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [文档索引](./docs/README.md) | 全部文档的入口 |
| [快速开始](./docs/getting_started.md) | 安装、环境要求、跑通第一个任务 |
| [任务说明](./docs/tasks/README.md) | 八份任务卡片 + 两类任务的关键区分 |
| [评测协议](./docs/eval_protocol.md) | seed、步数上限、成功判定、统计要求 |
| [策略接入](./docs/policy_interface.md) | 如何接入自己的模型 |
| [数据说明](./docs/dataset.md) | HDF5 格式、wrench 语义、LeRobot 转换 |
| [复现指南](./docs/reproduce.md) | 从零重新生成数据 |
| [开发者文档](./docs/developer/project_guide.md) | 架构、契约、资产溯源 |

## Leaderboard

机器可读的权威数据文件是 `results/leaderboard.json`，目录说明见
[`results/README.md`](./results/README.md)。
（⚠️ 该数据文件目前**尚未落盘到本仓库**，详见 `results/README.md` 的"当前状态"一节。）

**当前覆盖状态（务必如实理解）：** π0.5 与 ForceVLA 覆盖 8 个任务中的 7 个
（缺 `gear_mesh`、`peg_insert`）；第三个基线 π0.5+力6D 只有故障修复前的评测、已作废、
尚未重跑；第四个基线 ForceFlow 尚未训练。即 32 格中约 12 格有有效数据。

### ⚠️ 三条必须一起读的方法学提醒

1. **环境间距必须为 6 m。** 更小的间距会让相机拍到相邻克隆环境（视觉串扰），
   早期结果因此**全部作废**。只有带 `spacing6` 标记的输出是有效的。
2. **步数上限本身会翻转结论。** `nut_thread` 在 600 帧上限下两个模型都是 **0%**，
   放宽到 1500 帧后变成 **21% / 36%**。引用任何成功率都必须同时说明步数上限。
3. **不要写"力有帮助"或"力没用"这类总体结论。** 六组 π0.5 vs ForceVLA 对比中只有
   3 组达到 `p<0.05`（Fisher 精确检验），其中 **2 组是无力基线更好**，经 Bonferroni
   校正后**只剩 1 组**站得住。请只陈述逐任务事实，并附 95% Clopper-Pearson 区间。

## 数据与资产许可证

**代码：** MIT，见 [`LICENSE`](./LICENSE)。

**第三方资产：** 见 [`NOTICE.md`](./NOTICE.md) 的逐任务表格。ForceBench **不再分发**
第三方资产的原始文件，只分发自己编写的包装/归一化代码，以及校验和与"去哪里拿"的说明。

已知约束：

- `stove_turn_on` / `faucet_turn` 使用 Lightwheel Kitchen 资产，
  **CC BY-NC 4.0，禁止商用**；
- `gear_mesh` / `nut_thread` 使用从 NVIDIA FORGE 部署中提取的资产，
  受 **NVIDIA Omniverse/Isaac EULA** 约束；
- `weight_sort` / `rattle_sort` 使用 NVIDIA Omniverse SimReady 目录的资产；
- `radmanso_drawer_sort_table`（ArtVIP 柜子）和 `peg_insert`（三个 usd 的来源链条）
  的许可证**尚未确认**。

**项目所有者已决定：整体按非商用 / 研究用途发布，风险自担。** 在每个任务的资产链条
被独立厘清之前，请把本仓库整体视为**仅限非商用研究使用**。

## 引用

```bibtex
@misc{forcebench2026,
  title  = {ForceBench: A Benchmark for Force-Aware Manipulation Policies},
  author = {Han, Xuan and others},
  year   = {2026},
  url    = {https://github.com/pkuhanxuan/ForceBench}
}
```

> 再次提醒：当前是 v0.1 原型，数据托管、部分文档和 baseline 评测均未完成。
> 引用前请确认你引用的是哪个状态。
