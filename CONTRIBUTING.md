# 贡献指南

ForceBench 遵循一套任务插件契约（详见
[项目指南第 8 节](./docs/developer/project_guide.md#8-新增一个任务--demo-及其专家采集器)）：
一个新任务就是 `forcebench/tasks/<name>/` 下的一个包，导出一个 `TaskManifest`，
并带有自己的：

- `spec.py` —— 不可变的几何 / 随机化 / 成功判定常量
- `outcome.py` —— 把判据组合成权威判定与稳定失败码，且不导入 Isaac Sim
- `runtime.py` —— 场景实体、reset、指令、特权观测、成功判定

任务注册表（`forcebench/registry.py`）会**自动发现**它——
**不要往共享环境里加任务分支。**

面向开发者的最小示例见
[`forcebench/tasks/README.md`](./forcebench/tasks/README.md)。

## 提 PR 之前

### 1. 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

**当前基线是 `102 passed, 5 failed, 2 skipped`。** 这 5 个失败是已知的，记录在
[`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) 中，**不需要你修复**——只要确认你的改动**没有
让失败数增加**即可。大部分测试不需要 Isaac Sim。

新任务应当按现有的逐任务测试文件风格，添加一个 `tests/test_<task>.py`。

### 2. 资产溯源

如果你的任务依赖第三方 3D 资产，**必须在 PR 合并之前**：

- 在 [`NOTICE.md`](./NOTICE.md) 的溯源表中加一行，写明**确切的来源和许可证**；
- 把文件哈希加进 `assets/SHA256SUMS`。

**除非你确实拥有再分发权利，否则不要提交二进制资产本身。** 现有任务的处理方式见
[`NOTICE.md`](./NOTICE.md)：只提交包装代码 + 指向厂商官方渠道的指引。

如果某项来源或许可证你没有真正核实，**就把它标成 `UNCONFIRMED`**，不要写成已确认。
仓库里已经有两处这样的未确认项，它们保持未确认状态是有意为之。

### 3. 关于力的诚实性要求

**这是 ForceBench 特有的一条，请认真对待。**

如果你的任务的成功判定**依赖**力/wrench 感知，请**明确说明**，并指出**是哪一行代码
读取了它**。

如果不依赖，也请同样明确地说明。

理由：ForceBench 的全部意义就在于诚实地区分"哪些任务真正需要力"和"哪些不需要"。
仓库里现有三个任务（`peg_insert`、`gear_mesh`、`nut_thread`）的 manifest 描述自称
"force-sensitive"，而其成功判定实际上只检查最终位姿——这一点已经在
[README](./README.md) 和它们各自的任务卡片中如实标注为**前瞻性措辞**。
**不要再增加这类描述与实现不符的情况。**

### 4. 文档

新任务需要在 `docs/tasks/<task>.md` 添加一份中文说明卡片，内容参照现有卡片：

- 任务目标（中文翻译 + 英文 prompt 原文）
- 场景
- 力相关性（属于力分类任务还是接触密集任务，如实标注）
- 物理量与随机化范围（**以 `spec.py` 常量为准**）
- 成功判定条件（**以 `evaluate_final_state` 和 `outcome.py` 为准**）
- 专家数据条数
- 依赖的第三方资产及其许可证

并在 [`docs/tasks/README.md`](./docs/tasks/README.md) 的总表中加一行。

## 文档约定

- **文件名用英文（ASCII 小写 + 下划线），内容用中文。** 理由见
  [`docs/README.md`](./docs/README.md) 的"文档约定"一节。
- **技术事实一律以代码为准。**
- **不要编造尚不存在的东西。** 没实现的脚本要标注"尚未实现（v1.0 计划项）"，
  而不是给出一条跑不通的命令。这是现有文档已经遵守的标准，请保持。

## 报告评测结果时

请遵守[评测协议](./docs/eval_protocol.md)。特别注意：

- 固定 seed `200`–`299`，环境间距**必须为 6 m**；
- **必须**说明所用的步数上限（它会翻转结论）；
- 成功率**必须**附 95% Clopper-Pearson 区间；
- **不要**写"力有帮助 / 力没用"这类总体结论，只陈述逐任务事实。

## 范围问题

某个新任务应该进入下一版核心任务集、还是保持为可选任务，属于项目所有者的决定。
**大改动之前请先开 issue 讨论。**
