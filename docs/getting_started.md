# 快速开始

> **先读这一句：** ForceBench 当前是 **v0.1 原型**。数据托管和资产下载脚本都还没有
> 搭好，因此"跑通第一个任务"目前**需要你自己准备 Isaac Lab 环境和第三方资产**。
> 只想读任务定义、跑契约测试的话，第 1–2 节就够了，不需要 Isaac Sim。

## 1. 环境要求

分成两档，按你想做什么选：

### A 档：只读代码 / 跑契约测试（不需要 GPU、不需要 Isaac Sim）

- Python 3.10+
- `requirements-dev.txt` 里的依赖：`pytest numpy h5py msgpack websockets opencv-python`

### B 档：真正跑仿真（采集、评测、回放）

- **Isaac Sim 6.0.1 / Isaac Lab 6.1.14**（当前采集所用版本；conda 环境名 `env_isaacsim6`。
  `gear_mesh`/`nut_thread` 的 FORGE 工厂资产历史上提取自 Isaac Lab 2.3.2 容器部署，见 NOTICE.md）
- 一块支持 Isaac Sim 的 NVIDIA GPU
- 各任务依赖的第三方 3D 资产**已就位**（见第 4 节）

## 2. 安装与自检

```bash
git clone https://github.com/pkuhanxuan/ForceBench
cd ForceBench
pip install -r requirements-dev.txt

# 契约 / 注册表测试，不需要 Isaac Sim
python -m pytest tests/
```

**预期结果：`102 passed, 5 failed, 2 skipped`。**

这 5 个失败是**已知的**，记录在 [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) 里，
不是你的环境问题：其中涉及场景 profile 的裁剪范围、`stove_turn_on` 的几何常量分歧，
以及一个需要真实 FORGE 二进制资产才能通过的完整性检查。2 个 skip 需要真实的
Isaac Lab 环境。

确认任务注册表能被发现（同样不需要 Isaac Sim）：

```bash
python -c "from forcebench.registry import task_names; print(sorted(task_names()))"
```

应当输出八个任务名。

## 3. 配置环境变量

ForceBench 的默认路径指向原始开发机（L20）上的绝对路径，在别的机器上**必须**通过
环境变量覆盖，否则会以令人困惑的 "path not found" 失败。详见
[路径依赖](./developer/path_dependencies.md)。

```bash
export FORCEBENCH_DATA_ROOT=/your/path/forcebench_data       # 正式数据集/评测结果
export FORCEBENCH_SCRATCH_ROOT=/your/path/forcebench_scratch # 一次性输出
export FORCEBENCH_ARCHIVE_ROOT=/your/path/forcebench_archive # 归档
export FORCEBENCH_ASSET_DIR=/your/path/ForceBench/assets     # 资产目录

# Isaac Lab 相关（run_native.sh 使用）
export FORCEBENCH_ISAACLAB_ROOT=/your/path/IsaacLab
export FORCEBENCH_CONDA_SH=/your/path/miniforge/etc/profile.d/conda.sh
export FORCEBENCH_CONDA_ENV=env_isaacsim6
```

这四个根目录必须两两不同、且都不等于代码目录——`forcebench/paths.py` 会自校验
这一不变式。

## 4. 准备第三方资产

**目前没有下载脚本。** `scripts/download_assets.sh` 是 v1.0 的计划项，**尚未实现**。

八个任务里有七个依赖第三方 3D 资产，ForceBench **不再分发**这些资产本身，只分发
自己编写的归一化包装文件、校验和，以及"去哪里拿"的说明。你需要按
[`NOTICE.md`](../NOTICE.md) 的逐任务表格，从各厂商的官方渠道自行获取：

| 来源 | 涉及任务 | 备注 |
|---|---|---|
| NVIDIA Omniverse SimReady 目录 | `weight_sort`、`rattle_sort` | 需自行下载资产包 |
| Lightwheel Kitchen | `stove_turn_on`、`faucet_turn` | **CC BY-NC 4.0，禁止商用** |
| NVIDIA Isaac Lab FORGE 部署 | `gear_mesh`、`nut_thread` | 需自行运行 FORGE / Isaac Lab 部署 |
| ArtVIP | `radmanso_drawer_sort_table` | ⚠️ 许可证**未确认** |
| 来源未确认 | `peg_insert` | ⚠️ 保管链**未追溯** |

拿到文件后，用 `assets/SHA256SUMS` 校验你的副本与 ForceBench 构建时所用的是否一致：

```bash
cd assets && sha256sum -c SHA256SUMS
```

## 5. 跑通第一个任务

准备好 B 档环境和资产后，所有 Isaac Lab 入口都通过 `run_native.sh`：

```bash
# 用脚本化专家采集 weight_sort 的 seed 0-31（一个 Isaac 进程，32 个克隆环境）
FORCEBENCH_TASK=weight_sort ./run_native.sh expert --task weight_sort --seeds 0-31

# 列出所有已注册的专家插件
./run_native.sh expert --list

# 回放一个已采集的 episode
./run_native.sh replay $FORCEBENCH_DATA_ROOT/datasets/weight_sort/seed_000_episode_01.hdf5
```

评测一个策略（需要先起一个策略推理服务）：

```bash
export FORCEBENCH_POLICY_HOST=127.0.0.1
export FORCEBENCH_POLICY_PORT=8002
export FORCEBENCH_POLICY_ADAPTER=openpi_ws
export FORCEBENCH_ENV_SPACING=6.0        # 必须为 6，见评测协议

./scripts/evaluation/evaluate_vectorized_seed_range.sh \
  weight_sort 200 299 auto \
  $FORCEBENCH_DATA_ROOT/evaluations/weight_sort/my_model_spacing6
```

完整的评测规定（seed 区间、步数上限、统计要求）见[评测协议](./eval_protocol.md)。

## 6. ⚠️ `run_native.sh` 各入口的可用性

`run_native.sh` 是从内部代码库原样带出来的，它引用的一部分脚本**不在本次公开提取的
范围内**。在补齐之前，这些入口会以"文件不存在"失败。如实列出：

| 入口 | 状态 | 依赖 |
|---|---|---|
| `expert` | ✅ 可用 | `scripts/expert/collect_expert.py` |
| `policy-vector` | ✅ 可用 | `scripts/evaluation/inference_forcebench_vectorized.py` |
| `replay` | ✅ 可用 | `scripts/evaluation/replay_forcebench.py` |
| `policy-replay` | ✅ 可用 | `scripts/evaluation/render_policy_rollout.py` |
| `policy` | ⛔ 已归档 | 直接退出码 2，提示改用 `policy-vector` |
| `collect` | ❌ 不可用 | `scripts/collection/record_forcebench.py`（未提取） |
| `smoke` | ❌ 不可用 | 同上 |
| `grasp-smoke`、`drop-smoke` | ❌ 不可用 | `scripts/smoke/`（未提取） |
| `faucet-detent-smoke`、`faucet-contact-smoke` | ❌ 不可用 | `scripts/smoke/`（未提取） |
| `stove-detent-smoke` | ❌ 不可用 | `scripts/smoke/`（未提取） |
| `faucet-views`、`stove-views`、`rattle-sort-views` | ❌ 不可用 | `scripts/smoke/`（未提取） |
| `faucet-live-tune` | ❌ 不可用 | `scripts/tuning/`（未提取） |

**这意味着 GELLO 遥操作采集和各类 smoke 自检目前无法在公开仓库里跑。** 补齐这些
脚本是 v1.0 的待办项。脚本化专家采集、策略评测和回放这三条主链路不受影响。

## 7. 下一步

- [任务说明索引](./tasks/README.md) —— 八个任务分别在测什么，以及最重要的两类区分
- [评测协议](./eval_protocol.md) —— seed、步数上限、统计报告要求
- [策略接入](./policy_interface.md) —— 如何接入自己的模型
- [数据说明](./dataset.md) —— 专家数据格式与字段含义
- [复现指南](./reproduce.md) —— 如何从零重新生成数据
