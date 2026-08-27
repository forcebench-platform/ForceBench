# 数据说明

代码真值：[`forcebench/recording.py`](../forcebench/recording.py)。

## 1. ⚠️ 数据当前还没有托管

**本仓库不包含任何 episode 数据**（这是仿真 benchmark 的惯例：代码仓库保持轻量，
大数据放 HuggingFace）。

**但 HuggingFace 上的数据集仓库目前还没有建起来**，因此下载这条路暂时走不通。

现在想要数据只有一条路：按[复现指南](./reproduce.md)自己重新生成。

> 下载脚本 `scripts/download_datasets.py` 正在补充中。即使脚本就位，在上述
> HuggingFace 数据集仓库真正建立并上传之前，它也只能报告"仓库不存在"。
> 托管本身是 v1.0 的待办项，见 [`CHANGELOG.md`](../CHANGELOG.md)。

## 2. 数据规模

专家数据使用 seed `0`–`199`（评测使用 `200`–`299`，与之不重叠）。

| 任务 | episode 数 | 说明 |
|---|---|---|
| `weight_sort` | 200 | |
| `rattle_sort` | 200 | |
| `stove_turn_on` | 200 | |
| `faucet_turn` | 200 | |
| `radmanso_drawer_sort_table` | 200 | |
| `nut_thread` | 200 | |
| `peg_insert` | **189** | ⚠️ 不足 200，seed 覆盖不完整 |
| `gear_mesh` | **62** | ⚠️ 远不足 200，样本量很小 |

后两个任务的缺口原因未在提取材料中记录，各自的任务卡片
（[`peg_insert`](./tasks/peg_insert.md)、[`gear_mesh`](./tasks/gear_mesh.md)）
有进一步说明。**使用这两个任务的数据时请注意样本量。**

**正式 writer 只接受成功的 episode。** 失败的 rollout 不会被写进数据集，
所以上面的条数就是成功 episode 数。

## 3. 文件格式

- schema 名：**`forcebench_hdf5`**（`h5py` 文件属性 `schema_version`）
- 一个 episode 一个文件，**原子写入**（先写临时文件再 `os.replace`）
- 文件名：`seed_<seed:03d>_episode_<index:02d>.hdf5`，如 `seed_007_episode_01.hdf5`
- 压缩：gzip level 4 + shuffle

顶层文件属性：`schema_version`、`episode_index`、`seed_episode_index`、`seed`、
`num_frames`、`success`（恒为 `True`）、`task`、`metadata_json`。

## 4. 字段布局

### 控制频率数据（15 Hz，长度 = `num_frames`）

| 路径 | 形状 | 含义 |
|---|---|---|
| `observations/state` | `[N, 7]` | 测得的世界 TCP `[x,y,z,roll,pitch,yaw,gripper]`，RPY 连续 |
| `observations/joint_position` | `[N, ...]` | 测得的关节位置 |
| `observations/joint_velocity` | `[N, ...]` | 测得的关节速度 |
| `observations/tcp_pose` | `[N, ...]` | 测得的 TCP 位姿 |
| `observations/images/main` | `[N, H, W, 3] uint8` | 主相机 RGB |
| `observations/images/wrist` | `[N, H, W, 3] uint8` | 腕部相机 RGB |
| `actions/canonical` | `[N, 7]` | **绝对**目标 TCP `[x,y,z,roll,pitch,yaw,gripper]` |
| `forces/control/eef_wrench_joint_raw` | `[N, 6]` | **raw** wrench，**模型输入用这一路** |
| `forces/control/eef_wrench_joint_ema` | `[N, 6]` | EMA wrench（`alpha=0.25`），仅供分析 |
| `timestamps/control` | `[N]` | 控制帧时间戳 |
| `rewards` / `success` / `terminated` / `truncated` / `is_terminal` | `[N]` | 逐帧标记 |
| `phases/id` | `[N] uint8` | 阶段标识（部分任务有） |
| `privileged/...` | 任务而定 | 特权教师字段，见下 |

`observations/` 里的三个机器人字段（`joint_position`、`joint_velocity`、`tcp_pose`）
**要么全有要么全无**——writer 会拒绝只写其中一部分。

### 物理频率数据（60 Hz）

| 路径 | 形状 | 含义 |
|---|---|---|
| `forces/physics/eef_wrench_joint_raw` | `[M, 6]` | 60 Hz 的 raw wrench |
| `timestamps/physics` | `[M]` | 物理帧时间戳，严格递增 |

这一路是**必需的**——`write()` 在缺少物理频率 wrench 流时会直接报错。它让你能看到
控制频率下被抽稀掉的接触瞬态。

## 5. wrench 字段的确切语义

这是最容易用错的字段，单独说明。`forces/` 下每个 dataset 都带这些属性：

| 属性 | 值 |
|---|---|
| `source` | `link_incoming_joint_force` |
| `joint` | `panda_hand_joint` |
| `frame` | `incoming_joint`（入向关节坐标系） |
| `direction` | `parent_on_child`（父作用于子） |
| `moment_reference` | `incoming_joint_origin` |
| `gravity_compensated` | `False` |
| `filter` | `none`（raw）/ `ema_alpha_0.25`（EMA） |

分量顺序固定为 `[Fx, Fy, Fz, Tx, Ty, Tz]`。

**要点：**

- **不做重力补偿。** 静止时读数里包含手和被抓物体的重力——这对
  [`weight_sort`](./tasks/weight_sort.md) 这类任务恰恰是信号而非噪声。
- **不做坐标变换。** 数值在入向关节坐标系里，不是世界系也不是 TCP 系。
- **训练和推理都用 raw 一路。** EMA 是同步的辅助通道，用于分析和专家控制诊断，
  不要拿它当模型输入——否则你的训练分布和评测时策略实际收到的信号不一致。

## 6. 元数据

`metadata_json` 文件属性里是一个完整的 JSON 文档，包含：

- `task`、`seed`、`num_frames`、`episode_index`、`images_present`
- `initial_layout` —— 该 episode **实际采样出的布局**，包括隐藏真值。
  例如 `weight_sort` 的 `box_kind` / `box_mass_kg`、`stove_turn_on` 的
  `knob_load_profile`（含 `detent_center_deg`）、
  `radmanso_drawer_sort_table` 的 `breakaway_resistance_n` 等。
  **离线分析时这是判断"策略本该做什么"的依据。**
- `scene_profile`、`control_source`、`prompt`
- `wrench_profile` —— 上一节那张表的机器可读版本
- `arm_control_mode`、`arm_controller_profile`
- `control_rate_hz` (15.0)、`physics_rate_hz` (60.0)、`solver_position_iterations` (64)
- `privileged_fields` —— 各特权字段的形状
- 结果判定：`success`、逐判据结果、稳定的失败码
- 专家采集时另有 `expert_collection_contract`、`expert_backend`、
  `expert_policy_wrench_field`、`expert_recorded_wrench_fields`

**刻意不包含的：** 设备相关的遥操作读数。文件保存的是测得的机器人状态和规范的目标
动作，与该动作来自遥操作、脚本化专家还是别的来源无关。

## 7. 校验

```python
from forcebench.recording import validate_hdf5_episode
info = validate_hdf5_episode("seed_000_episode_01.hdf5")
```

会检查 schema 版本、形状一致性、时间戳单调性、有限性等。schema 不是
`forcebench_hdf5` 会直接拒绝。

## 8. 转换成 LeRobot 格式

仓库根目录的 `convert_hdf5_to_lerobot.py` 可以把一个已校验的 HDF5 快照转换成
**LeRobot v2.1** 数据集：

```bash
python convert_hdf5_to_lerobot.py \
  --snapshot /path/to/snapshot_dir \
  --output-root /path/to/lerobot_root \
  --repo-id my_org/forcebench_weight_sort \
  --image-size 224 \
  --verify-sha256
```

输入的 `snapshot_dir` 需要有一个 `manifest.json`（含 `records` 列表、每条记录的
`file`/`bytes`/`sha256`/`frames`）和一个 `raw/` 子目录。

产出的 feature 布局：

| feature | dtype / shape |
|---|---|
| `state` | `float32 [7]` |
| `actions` | `float32 [7]` |
| `eef_wrench` | `float32 [6]` |
| `image` | `video [S, S, 3]` |
| `wrist_image` | `video [S, S, 3]` |

`robot_type` 为 `franka`，`fps` 取自 manifest 的 `control_rate_hz`（15）。

**一个转换细节值得注意：** `unwrap_action_rpy()` 会把绝对 RPY 目标改写成"相对当前
state 的最短角度偏移"再加回 state。也就是说，转换后的动作在数值上仍然是绝对的，
但被表示在离当前测量值最近的那个分支上。这是为了避免下游训练框架在
`±pi` 附近学到巨大的跳变。

> 该脚本 `import lerobot`，需要你自行安装 LeRobot。
