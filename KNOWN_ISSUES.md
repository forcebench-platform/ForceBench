# 已知问题

（对应 2026-08-26 的 ForceBench 初始提取）

测试套件在独立环境下运行（无 Isaac Sim / GPU），安装
`pytest numpy h5py msgpack websockets opencv-python` 之后的结果是：

> **102 passed, 5 failed, 2 skipped**（2 个 skip 需要真实的 Isaac Lab 环境）

**这 5 个失败是已知的，不需要贡献者修复。** 提 PR 时只要确认失败数没有增加即可。

## 提取过程中已修复的问题

除了 `ASSET_PROVENANCE.md` 缺失所暴露的三个文档问题和另外两处文档订正之外，
提取过程中还发现并修复了六个测试 / 代码漂移问题：

- `scripts/expert/parallel_sort_collection.py` 导出的是 `DEFAULT_CLONE_COUNT`，
  而 `tests/test_expert_interface.py` 仍在导入 `CLONE_COUNT`。已修复（在导入处改名）。
- `tests/test_architecture.py` 和 `tests/test_expert_interface.py` 都硬编码了旧文件名
  `scripts/evaluation/inference_forcebench.py`，该文件在 8 月 25 日的迁移中已改名为
  `inference_forcebench_vectorized.py`。已修复（两处调用点都已更新）。
- `tests/test_architecture.py` 断言 `peg_insert` 和 `nut_thread` 的 `scene_profile`
  仍是 `"gear_assembly_cell"`，但这两个任务已经改为 `"precision_ash_veneer"`
  （只有 `gear_mesh` 还在 `gear_assembly_cell`）。已按实际 manifest 修复。
- `tests/test_expert_interface.py::test_training_and_inference_use_raw_wrench`
  断言评测脚本中存在字面量 `"env.force_sensor_joint_raw[0]"`，而向量化多环境评测
  循环把它批处理成了 `env.force_sensor_joint_raw`（没有 `[0]`，后续再按环境索引）。
  已修复为检查批处理形式以及 EMA 变量的**不存在**——后者才是这个测试真正想守护的
  不变式（到达策略的是 raw wrench，不是 EMA）。
- `convert_hdf5_to_lerobot.py`（顶层脚本）在最初的提取中缺失——它没有被列在提取清单
  里，但上面那个测试以及任何要转换 episode 用于训练的人都需要它。已从源机器补入。
- 两个 ForceBench 自己编写的 `.usda` 文本包装文件
  （`assets/lightwheel_kitchen/Sink054/faucet_detent_valve.usda`、
  `.../robot_base_counter_extension.usda`，以及
  `Stovetop012/stove_detent_knob.usda`）被提取时"一律不要 `.usd`/`.usda`"的规则误伤。
  该规则本意针对的是厂商的原始二进制，而不是这些 1–6 KB、手工编写的几何 / 物理描述
  文件——`assets/SHA256SUMS` 本来就把它们作为 ForceBench 自有派生内容在追踪。
  已补入；文件大小与既有的 SHA256SUMS 记录一致。

## 未解决的问题（留待后续，不阻塞初始提交）

### 1. `tests/test_expert_interface.py::test_standard_supervisor_validates_and_resumes_by_seed`

在 `expert_collection.execute_expert_collection` 内部更深的位置失败，**未完全追踪**。
需要熟悉专家采集"续采 / 校验"逻辑的人来诊断。

### 2. `tests/test_forge_task_specs.py::test_normalized_thread_assets_preserve_source_sdf_collisions`

需要真实的 FORGE 派生 `.usd` 二进制（含 SDF 碰撞数据）才能检查。这些二进制不在
纯代码提取的范围内（见 [`NOTICE.md`](./NOTICE.md)）；在资产流水线
（`scripts/download_assets.sh`）搭好、并且有人带着真实资产跑一遍测试之前，
这个测试无法通过。**这不是代码缺陷。**

### 3. `tests/test_scene_profiles.py`（两个失败）

实际的 `forcebench/scene_profiles.py` 注册的 profile 比测试里硬编码的期望集合更多，
其中包括 `whiteboard_simple_room`（被 `wipe_board` 使用，而该任务明确不在 ForceBench
的 8 任务范围内）以及 `precision_ash_veneer` / `parcel_sort_station_collection_20260823`。

这引出一个本次提取没有解决的**范围问题**：`scene_profiles.py` 本身应该被裁剪到只保留
8 个在范围内的任务所用的 profile，还是应该保持为与 ForceBench 其余部分共享的超集？
一旦这个决定做出，裁剪它并更新测试是机械性的工作；但**做这个决定不是本次提取的职责**。

### 4. `tests/test_stove_turn_on.py::test_stove_layout_keeps_geometry_fixed_and_randomizes_only_the_load`

实际代码中 `SPEC.burner_center_m` 是 `(0.435843146821, 0.268566305108, 0.047)`，
而测试期望 `(0.435837259, 0.242254671, 0.055)`。X 坐标只差浮点噪声，但 **Y 相差约
2.6 cm、Z 相差约 0.8 cm**——太大，不可能是舍入误差。

这看起来是一次真实的几何修订（可能与 L20 巡检时发现的、同样没有文档记录的
`rattle_sort` 箱体缩放修订有关，即一次资产 / 场景更新落地了但对应的测试没有同步刷新），
但要确认**哪一组数值在物理上是正确的**，需要检查实际的场景 / 资产，而不能只看这两组
数字。因此保持未解决，而不是猜一个。

⚠️ **`stove_turn_on` 的使用者在此问题解决之前应当知晓这一点。**

## 其他已知缺口

### `run_native.sh` 部分入口不可用

`run_native.sh` 是从内部代码库原样带出来的，它引用的一部分脚本目录
（`scripts/collection/`、`scripts/smoke/`、`scripts/tuning/`）**不在本次公开提取范围
内**。因此 `collect`（GELLO 遥操作采集）、`smoke` 以及全部 `*-smoke` / `*-views` /
`faucet-live-tune` 入口会以"文件不存在"失败。

另外，`policy`（单环境评测）入口已被归档，直接调用会以退出码 2 提示改用
`policy-vector`。

**可用的入口：** `expert`、`policy-vector`、`replay`、`policy-replay`。
完整对照表见[快速开始第 6 节](./docs/getting_started.md#6-run_nativesh-各入口的可用性)。

### 专家数据不完整

- `peg_insert`：**189 / 200**
- `gear_mesh`：**62 / 200**

缺口原因未在提取材料中记录。详见各自的任务卡片
（[`peg_insert`](./docs/tasks/peg_insert.md)、[`gear_mesh`](./docs/tasks/gear_mesh.md)）。

### 三个接触密集任务的描述与实现不符

`peg_insert`、`gear_mesh`、`nut_thread` 的 `MANIFEST.description` 自称
"force-sensitive"，但它们的成功判定只检查最终位姿，不读取任何 wrench 信号。
这已在 [README](./README.md)、[任务索引](./docs/tasks/README.md) 和三份任务卡片中
如实标注为**前瞻性措辞**。修正措辞或加入依赖力的判定是 v1.0 待办项。

### 默认路径指向原始开发机

`forcebench/paths.py` 和 `env_cfg.py` 在环境变量未设置时，默认指向 `/home/pc/...`
下只在原始 L20 开发机上存在的路径。详见
[路径依赖](./docs/developer/path_dependencies.md)。

---

**上述四个未解决的测试问题都不影响使用 ForceBench 的 8 个任务做训练 / 评测**——
它们是内部一致性检查（测试套件卫生、一个依赖资产的完整性检查、一个场景 profile 的
范围问题，以及 `stove_turn_on` 特有的一个几何常量分歧）。
