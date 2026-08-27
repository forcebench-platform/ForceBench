# 复现指南

从零重新生成 ForceBench 的专家数据。

> **前提：** 你已经完成[快速开始](./getting_started.md)的 B 档环境配置，
> 并且第三方资产已就位。**目前没有资产下载脚本**，这一步需要手工完成。

## 1. 生成环境

正式数据集在下列环境上生成：

| 项 | 值 |
|---|---|
| 仿真器 | **Isaac Lab 2.3.2 / Isaac Sim 5.1** |
| 容器镜像摘要 | `sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2` |
| 物理频率 | 60 Hz |
| 求解器位置迭代 | 64 |
| 控制频率 | 15 Hz（`decimation=4`） |

`configure_runtime()` 会拒绝其他物理或求解器取值，所以这几个数不是建议值而是硬约束。

**不同的 Isaac Sim 版本会产生不同的接触动力学**，因此在别的版本上重新生成的数据
与本文描述的数据集不是逐比特可比的。跨版本比较结果时请说明所用版本。

## 2. 采集拓扑

生产采集使用**一个 Isaac 进程 + 正好 32 个克隆并做过碰撞过滤的工作空间**，
配批量 TiledCamera 渲染。这套路径由 `scripts/expert/collect_expert.py` 统一调度。

五个任务走这条默认的 32 克隆路径：

```bash
./run_native.sh expert --task weight_sort                 --seeds 0-199
./run_native.sh expert --task rattle_sort                 --seeds 0-199
./run_native.sh expert --task stove_turn_on               --seeds 0-199
./run_native.sh expert --task radmanso_drawer_sort_table  --seeds 0-199
./run_native.sh expert --task peg_insert                  --seeds 0-199
```

三个任务走各自任务绑定的适配器：

```bash
./run_native.sh expert --task faucet_turn --seeds 0-199
./run_native.sh expert --task gear_mesh   --seeds 0-19 --workers 2
./run_native.sh expert --task nut_thread  --seeds 0-19 --workers 2
```

注意 `gear_mesh` / `nut_thread` 用的是小批量 + 多 worker 的特权位姿教师路径，
吞吐远低于 32 克隆路径——这也是 `gear_mesh` 迟迟未能采满的直接原因。截至 2026-08-27，该任务的正式数据集已不在源机器上。

查看所有已注册的专家插件及其启用状态：

```bash
./run_native.sh expert --list
```

## 3. 断点续采

监督器会**跳过已经验证通过的 seed**，所以中断后直接重跑同一条命令即可继续，
不会产生重复的 HDF5。最后一个不满 32 的批次会被补齐而不重复写入。

每个批次还会录制一段 32 宫格的马赛克视频，方便快速目检这一批的质量。

## 4. 专家是什么

**没有任何一个任务使用学习式的 checkpoint 作为专家。** 全部是脚本化或特权状态教师：

| 任务 | 专家类型 |
|---|---|
| `weight_sort`、`rattle_sort` | 脚本化，读取隐藏的 `box_kind` 决定投放目标 |
| `stove_turn_on`、`faucet_turn` | 从**测得的腕部力矩**检测卡扣泄载，遵循"加载—泄载—回退"模式 |
| `radmanso_drawer_sort_table` | 按设计阻力曲线拉到目标阻力后停手 |
| `peg_insert`、`gear_mesh`、`nut_thread` | 特权位姿教师，读取物体/夹具的精确位姿 |

**一条硬性约束：** 装配类专家可以读取声明过的特权观测作为教师状态，但**必须**通过
共享的 SmoothPosition 控制器下发指令，让 PhysX 真实地推动场景。
**禁止 teleport 物体、禁止覆写机器人状态、禁止绕过统一控制器。**
这保证了专家数据和策略评测跑在完全相同的物理上。

退役的"学习式力技能专家"路径只存在于外部归档中，不在本仓库内。

## 5. 类别平衡是怎么保证的

几个分类任务用 seed 结构而不是随机采样来保证严格平衡，复现时**必须用连续的 seed
区间**，否则平衡会被破坏：

- **`weight_sort`**：`seed % 2` 直接决定轻/重。任何连续偶数条 seed 都是 1:1。
- **`rattle_sort`**：seed 两两成对，每对内部用确定性随机排列决定顺序。每对必定
  一类一个，但对内顺序不固定——既保证平衡又避免"偶数=某类"被模型记住。
- **`stove_turn_on` / `faucet_turn`**：卡扣**角度**随 seed 变化，
  **强度固定**。这是刻意的：如果强度也随 seed 变，seed 就成了物理属性的捷径。
- **`radmanso_drawer_sort_table`**：起始阻力、带长、峰值阻力三者都随 seed 变化。

## 6. 校验产出

```bash
# 校验单个 episode
python -c "
from forcebench.recording import validate_hdf5_episode
print(validate_hdf5_episode('seed_000_episode_01.hdf5'))
"

# 回放一个 episode 目检
./run_native.sh replay $FORCEBENCH_DATA_ROOT/datasets/weight_sort/seed_000_episode_01.hdf5
```

监督器在写入前就会做 HDF5 校验，并且**只接受成功的 episode**，所以数据集里不会有
失败轨迹。

`scripts/evaluation/audit_expert_replay.py` 可以用来审计专家轨迹的回放一致性。

## 7. 转成训练格式

见[数据说明第 8 节](./dataset.md#8-转换成-lerobot-格式)。

## 8. 已知的复现障碍

如实列出会挡住你的东西：

1. **没有资产下载脚本。** `scripts/download_assets.sh` 尚未实现，七个任务的第三方
   资产要按 [`NOTICE.md`](../NOTICE.md) 手工获取。其中 `radmanso_drawer_sort_table`
   和 `peg_insert` 的资产**来源/许可证尚未确认**。
2. **默认路径指向原始开发机。** 必须设置 `FORCEBENCH_*_ROOT` 系列环境变量，
   见[路径依赖](./developer/path_dependencies.md)。
3. **部分 `run_native.sh` 入口不可用。** `collect`（GELLO 遥操作）和各类 smoke 自检
   依赖未提取的 `scripts/collection/`、`scripts/smoke/`、`scripts/tuning/`。
   见[快速开始第 6 节](./getting_started.md#6-run_nativesh-各入口的可用性)。
4. **`stove_turn_on` 有一个未解决的几何常量分歧**，见
   [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) 第 4 条。在它解决之前，该任务重新生成的
   数据可能与原数据集的场景几何不一致。
5. **`peg_insert` 的正式采集本身是 gated 的**——[项目指南](./developer/project_guide.md)
   第 8 节说明它"在 32/32 释放稳定性 smoke 通过之前保持关闭"。该任务此前只有 189 条
   数据的状态一致。
