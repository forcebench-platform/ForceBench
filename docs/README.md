# ForceBench 文档

## 使用者文档

| 文档 | 内容 |
|---|---|
| [快速开始](./getting_started.md) | 安装、环境要求、配置、跑通第一个任务；各入口的可用性 |
| [任务说明索引](./tasks/README.md) | 八个任务总表，以及**两类任务的关键区分** |
| [评测协议](./eval_protocol.md) | seed 策略、rollout 数、步数上限、成功判定、统计报告要求 |
| [策略接入](./policy_interface.md) | `PolicyObservation` / `PolicyActionChunk` 契约，怎么写适配器 |
| [数据说明](./dataset.md) | HDF5 格式、字段含义、wrench 语义、LeRobot 转换 |
| [复现指南](./reproduce.md) | 如何从零重新生成专家数据 |

## 任务说明卡片

先读[索引](./tasks/README.md)——它解释了为什么这八个任务必须分成两类看。

**力分类任务**（力是唯一携带答案的通道）：

- [`weight_sort`](./tasks/weight_sort.md) —— 按重量分拣箱子
- [`rattle_sort`](./tasks/rattle_sort.md) —— 按内部配重分拣包裹
- [`stove_turn_on`](./tasks/stove_turn_on.md) —— 旋转灶台旋钮越过点火卡扣
- [`faucet_turn`](./tasks/faucet_turn.md) —— 扳动龙头把手越过卡扣
- [`radmanso_drawer_sort_table`](./tasks/radmanso_drawer_sort_table.md) —— 拉开抽屉并在阻力上升处停手

**接触密集任务**（当前实现下**不检验**力感知）：

- [`peg_insert`](./tasks/peg_insert.md) —— 销轴插孔
- [`gear_mesh`](./tasks/gear_mesh.md) —— 齿轮啮合装配
- [`nut_thread`](./tasks/nut_thread.md) —— 螺母拧入螺栓

## 开发者 / 架构文档

这些是从内部代码库继承并翻译的文档，记录的是"当时被固定下来的契约"以及提取过程中
发现的问题。

| 文档 | 内容 |
|---|---|
| [项目指南](./developer/project_guide.md) | 存储布局、仿真器契约、控制器、数据契约、任务插件接口 |
| [基准标准化记录](./developer/benchmark_standardization.md) | 当前被固定下来的规范契约 |
| [资产溯源](./developer/asset_provenance.md) | 每个资产从哪来、被怎样加工过 |
| [路径依赖](./developer/path_dependencies.md) | 代码中残留的机器相关硬编码路径 |
| [溯源缺口](./developer/provenance_gaps.md) | 溯源账本的缺口（主要一项已闭合） |
| [旧版 README](./developer/legacy_readme.md) | 历史存档，已被根目录 README 取代 |

## 其他位置的文档

| 文档 | 内容 |
|---|---|
| [`../README.md`](../README.md) | 项目首页 |
| [`../NOTICE.md`](../NOTICE.md) | **第三方资产逐任务许可证表**（含两处未确认项） |
| [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) | 已知问题，含 5 个已知失败的测试 |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | 贡献指南 |
| [`../CHANGELOG.md`](../CHANGELOG.md) | 版本记录与 v1.0 清单 |
| [`../results/README.md`](../results/README.md) | leaderboard 数据目录说明 |
| [`../forcebench/tasks/README.md`](../forcebench/tasks/README.md) | 新增一个任务（面向开发者） |
| [`../forcebench/policy_adapters/README.md`](../forcebench/policy_adapters/README.md) | 新增一个策略适配器 |
| [`../forcebench/expert_adapters/README.md`](../forcebench/expert_adapters/README.md) | 新增一个专家后端 |

## 文档约定

- **文件名用英文（ASCII 小写 + 下划线），文档内容用中文。**
  理由：跨平台和 git 对 ASCII 路径最稳妥，GitHub 链接不会被百分号转义，而且任务
  文档的文件名与代码里的任务 ID（`FORCEBENCH_TASK=weight_sort`）完全一致，
  可以直接从代码标识符 grep 到文档。
- **所有技术事实以代码为准。** 任务卡片里的常量、阈值和判据都直接取自
  `forcebench/tasks/<name>/spec.py` 与 `outcome.py`。若文档与代码不一致，
  以代码为准，并请提 issue。
- **不编造尚不存在的东西。** 没有实现的脚本一律标注"尚未实现（v1.0 计划项）"，
  而不是给出一条跑不通的命令。
