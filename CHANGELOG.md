# 更新日志

## v0.1.0（2026-08-26 起）—— 初始脚手架，尚不是一个正式发布的 benchmark

首次公开提交。从内部 ForceBench 代码库中提取并清理，范围限定为 ForceBench 的
8 个任务（`weight_sort`、`rattle_sort`、`stove_turn_on`、
`radmanso_drawer_sort_table`、`peg_insert`、`faucet_turn`、`gear_mesh`、
`nut_thread`）。

### 已有的内容

- 八个任务的完整定义（`spec.py` / `outcome.py` / `runtime.py`）
- 共享的仿真器 / 控制器契约
- 策略适配器（`openpi_ws`、`forcevla_ws`、`forcebench_ws`）
- 专家采集与评测脚本
- 可运行的测试套件（无 Isaac Sim 环境下 102/109 通过，见 `KNOWN_ISSUES.md`）
- 溯源与授权文档（`NOTICE.md`）

### 2026-08-29 与采集主机代码同步

以 .48 采集主机（内部主本，2026-08-29 15:41 状态)为准做了一次全量代码同步，
只覆盖正典 8 任务范围：

- 刷新 11 个漂移文件：`episode_qc`、`recording`、`scene_profiles`、
  `policy_transport`、faucet_turn/gear_mesh/nut_thread 三个专家适配器，
  以及 gear_mesh / rattle_sort / stove_turn_on 的任务定义
- 新增 `forcebench/sort_expert_path.py`（`generate_parallel_sort.py`
  的新依赖，此前缺失会导致排序类专家采集不可运行）
- 新增 `scripts/expert/generate_parallel_stove.py`，同步 8 个专家/评测脚本
- 移除 9 个主本已淘汰的旧生成器（scripted/privileged/peak/knob 系列与
  `run_persistent_expert_worker`），专家采集统一走 `collect_expert.py` 调度
- 同步内容仅做 `coribench→forcebench` 机械重命名，逐文件经 py_compile 验证；
  完整测试套件需要在带 torch 的机器上复跑

### 2026-08-27 文档重构

- 全部面向用户的文档**改为中文**，文件名统一为英文（ASCII 小写 + 下划线），
  理由见 [`docs/README.md`](./docs/README.md) 的"文档约定"一节。
- 目录结构按 LIBERO 风格重组：
  - `docs/` 下新增 `getting_started.md`、`eval_protocol.md`、`policy_interface.md`、
    `dataset.md`、`reproduce.md` 和 `README.md`（索引）；
  - **新增 `docs/tasks/` 下的八份任务说明卡片**——这是 CHANGELOG 此前记录的 v1.0
    待办项之一，现已完成；
  - 从内部代码库继承的六份英文文档移入 `docs/developer/` 并翻译为中文，技术事实未改；
  - 新增 `results/` 目录及其说明。
- 顶层文档（`README.md`、`NOTICE.md`、`CHANGELOG.md`、`CONTRIBUTING.md`、
  `KNOWN_ISSUES.md`）改写为中文。
- 补充记录了三处此前未在文档中说明的事实：
  - `run_native.sh` 的 `collect`、`smoke` 及各类 `*-smoke` / `*-views` 入口依赖
    `scripts/collection/`、`scripts/smoke/`、`scripts/tuning/`，而这些目录**不在**
    本次公开提取范围内；`policy` 入口已归档，直接调用会以退出码 2 提示改用
    `policy-vector`。
  - `assets/SHA256SUMS` 中原先缺失的 6 条 gear/nut/bolt 记录**已经补齐**，
    因此 `PROVENANCE_GAPS` 记录的主要缺口已闭合。
  - 三个接触密集任务（`peg_insert`、`gear_mesh`、`nut_thread`）的 "force-sensitive"
    措辞在当前实现下并不成立，已在 README、任务索引和三份任务卡片中明确标注。

## v1.0 待办清单

### 数据与资产

- [ ] 在 HuggingFace Datasets 上托管正式 episode 数据
      （目标仓库 `pkuhanxuan/ForceBench-datasets` **尚未建立**；
      下载脚本 `scripts/download_datasets.py` 正在补充中，但在仓库建立并上传之前
      仍然无法取到数据）
- [ ] 实现 `scripts/download_assets.sh`，指向各厂商官方资产渠道
      （**尚未实现**）
- [ ] 重新采集 `gear_mesh` 的专家数据（正式数据集已不在源机器上，仅剩 1 条调试残留；`peg_insert` 已于 2026-08-27 前补齐到 200）

### 授权与溯源

- [ ] 确认 ArtVIP RADMANSO 柜子资产的许可证（`NOTICE.md` 未确认项 1）
- [ ] 追溯 `peg_insert` 三个 usd 的实际上游来源与许可证（`NOTICE.md` 未确认项 2）
- [ ] 补上两个展平前的上游源哈希（`provenance_gaps.md` 剩余缺口）
- [ ] 对 `NOTICE.md` 的资产表做一次法律审阅

### 评测

- [ ] 补齐 baseline 覆盖：当前 32 个 (任务, 模型) 格子中只有约 12 格有有效数据。
      π0.5 与 ForceVLA 各覆盖 7/8 个任务（缺 `gear_mesh`、`peg_insert`）；
      π0.5+力6D 只有故障修复前的评测、已作废、尚未重跑；ForceFlow 尚未训练。
- [ ] 实现自动计算 Clopper-Pearson 区间的脚本（当前 `aggregate.json` 只写
      `success_rate`，区间需人工计算）
- [ ] 实现 Fisher 精确检验 / Bonferroni 校正的脚本（当前 `head_to_head` 的数字是
      人工核算后写入的）
- [ ] 实现 leaderboard 结果提交流程
- [ ] 把"力信息强度分级"固化成标准配置，而不是依赖各适配器自行屏蔽

### 代码与测试

- [ ] 解决 `KNOWN_ISSUES.md` 中的 5 个已知失败测试
- [ ] 补齐 `scripts/collection/`、`scripts/smoke/`、`scripts/tuning/`，
      使 `run_native.sh` 的全部入口可用
- [ ] 为三个接触密集任务加入依赖力的成功判定，或者把它们的 "force-sensitive"
      描述改成与实现一致的措辞
- [ ] 决定 `forcebench/paths.py` 未设环境变量时的默认行为
      （见 `docs/developer/path_dependencies.md`）
