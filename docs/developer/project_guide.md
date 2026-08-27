# ForceBench 项目指南

> 本文由内部代码库的 `PROJECT_GUIDE.md` 翻译而来，技术事实未作改动。
> 文中的绝对路径（`/home/pc/ForceBench*`）是原始开发机（L20）上的布局，
> 公开仓库中这些路径均可由环境变量覆盖，详见
> [路径依赖](./path_dependencies.md)。

## 1. 权威目录位置

| 用途 | 路径 |
| --- | --- |
| 源码与自带运行时 | `/home/pc/ForceBench` |
| 正式数据集 | `/home/pc/ForceBench_data/datasets` |
| 正式 checkpoint | `/home/pc/ForceBench_data/checkpoints` |
| 正式评测结果 | `/home/pc/ForceBench_data/evaluations` |
| 一次性 smoke / 预览 / 日志 | `/home/pc/ForceBench_scratch` |
| 历史归档 | `/home/pc/ForceBench_archive` |

`forcebench/paths.py` 是这些根目录在 Python 侧的唯一权威来源。只能通过
`FORCEBENCH_DATA_ROOT`、`FORCEBENCH_SCRATCH_ROOT` 和 `FORCEBENCH_ARCHIVE_ROOT`
覆盖它们。生成的数据不应放进源码树。

## 2. 固定的仿真器契约

- 物理频率：60 Hz。
- 位置求解器迭代次数：64。
- 策略 / 控制频率：15 Hz（`decimation=4`）。
- 手动采集与评测使用单个环境。生产环境的克隆式专家采集使用同一个 Isaac
  进程内的正好 32 个相互隔离的环境。
- 机器人资产：`assets/franka_mimic.usd`。
- 手臂驱动：隐式关节位置 PD，刚度 80，阻尼 12。
- 夹爪：对两个手指关节使用隐式位置驱动。

`configure_runtime()` 会拒绝其他物理或求解器取值。任务可以定义物体、随机化、
成功判定和相机配置；但不得私自改动整个 benchmark 范围内的机器人 / 控制器契约。

## 3. 唯一的规范手臂控制器

只有一个受支持的手臂控制器，标识为 `smooth_position`：

1. 接收一个世界坐标系下的绝对 TCP 位姿。
2. 归一化其四元数，并把每条指令相对测得 TCP 位姿限制在 25 mm 平移和
   0.12 rad 旋转以内。
3. 把当前位姿、目标位姿以及几何雅可比统一表达到机器人基座坐标系。
4. 用 Isaac Lab 官方的 `SmoothPositionController` 求解，配置为：

   ```text
   command_type="pose"
   use_relative_mode=False
   ik_method="dls"
   lambda_val=0.05
   ```

5. 把求解出的关节变化限制在每条 15 Hz 指令 0.08 rad 以内，再夹到机器人的软关节
   限位，并规划一条同步的五次多项式关节轨迹。
6. 以 60 Hz 把该轨迹下发给隐式关节位置驱动。

环境不会覆写 articulation 状态，也不提供 TSI、PI-hold、阻尼消融或控制器选择接口。

### 角度连续性

环境维护连续的测量关节角和连续的测量 TCP RPY。原始的 `+pi/-pi` 表示切换会先被
转换成最短角度增量，然后才进入 IK、记录状态、策略输入或回放。策略给出的 RPY
目标同样被表示在离当前测量 TCP RPY 最近的分支上。

## 4. 观测、动作与力/力矩

七维策略状态在物理步进之后测得：

```text
[tcp_x, tcp_y, tcp_z, tcp_roll, tcp_pitch, tcp_yaw, gripper]
```

TCP RPY 是连续的。`gripper` 为 `-1` 表示闭合、`+1` 表示张开。它不是目标状态，
也不是用于状态覆写的量。

七维动作为：

```text
[target_tcp_x, target_tcp_y, target_tcp_z,
 target_tcp_roll, target_tcp_pitch, target_tcp_yaw, target_gripper]
```

规范的六维 wrench 是通过 `panda_hand_joint` 作用在 `force_sensor` link 上的
入向固定关节反作用力，表达在入向关节坐标系中，顺序为
`[Fx, Fy, Fz, Tx, Ty, Tz]`（父作用于子）。不做任何坐标变换，也不做重力补偿。
raw 与 EMA（`alpha=0.25`）两路信号都会被记录；训练数据转换与策略推理使用 raw 一路。
EMA 作为同步的辅助通道，仅用于分析和专家控制诊断。

## 5. 正式 HDF5 数据

新 episode 使用稳定的 schema 名 `forcebench_hdf5`。正式 writer 只接受成功的
episode，并且是原子写入。每个 episode 包含：

- 测得的 7 维状态和绝对 7 维动作；
- 测得的关节位置、关节速度和 TCP 位姿；
- 规范的主视角 / 腕部 RGB 帧；
- 控制频率下的 raw 与 EMA wrench；
- 60 Hz 物理频率下的 raw wrench；
- 时间戳、任务结果判定（`success`、稳定的失败码以及逐条判据结果）、seed、
  实际布局、控制器 / wrench profile、任务元数据以及声明过的特权教师字段。

设备相关的遥操作读数被刻意排除在外。文件保存的是测得的机器人状态和规范的目标
动作，与该动作来自遥操作、脚本化专家还是其他来源无关。

旧 schema / 控制器属于历史产物。它们保存在归档目录中，不与当前数据集混放。

## 6. 标准命令

所有 Isaac Lab 命令都通过 `run_native.sh` 执行：

```bash
cd /home/pc/ForceBench

# GELLO 采集；默认输出到 /home/pc/ForceBench_data/datasets/<task>
FORCEBENCH_TASK=weight_sort ./run_native.sh collect

# 一次性 smoke 输出；默认落在 /home/pc/ForceBench_scratch 下
FORCEBENCH_TASK=weight_sort ./run_native.sh smoke

# 列出与任务绑定的自动专家插件
./run_native.sh expert --list

# 克隆式专家采集。这是受支持的生产拓扑：一个 Isaac 进程、32 个
# 克隆并做过碰撞过滤的工作空间、批量 TiledCamera RGB。
# 已验证过的 seed 会被跳过；每个批次都会写一段 32 宫格视频。
./run_native.sh expert --task weight_sort --seeds 0-199
./run_native.sh expert --task rattle_sort --seeds 0-199
./run_native.sh expert --task stove_turn_on --seeds 0-199
./run_native.sh expert --task radmanso_drawer_sort_table --seeds 0-199
./run_native.sh expert --task peg_insert --seeds 0-199

# 装配类专家是任务特有的特权位姿教师。它们读取物体 / 夹具的精确位姿，
# 但仍然通过普通的 SmoothPosition 控制器下发指令；它们从不覆写机器人或
# 物体状态，也不需要 checkpoint。
./run_native.sh expert --task gear_mesh --seeds 0-19 --workers 2
./run_native.sh expert --task nut_thread --seeds 0-19 --workers 2

# 策略评测；默认每次推理执行一段 50 帧的动作块。
# 用 FORCEBENCH_POLICY_ADAPTER 选择已注册的适配器。
FORCEBENCH_POLICY_ADAPTER=openpi_ws \
FORCEBENCH_TASK=weight_sort ./run_native.sh policy-vector 200 10 auto \
  /home/pc/ForceBench_data/evaluations/weight_sort/model_name/seed_200

# 回放一个当前规范格式的 episode
./run_native.sh replay /home/pc/ForceBench_data/datasets/<task>/<episode>.hdf5
```

> **公开仓库的偏差说明。** 原文档此处写的是单环境的
> `./run_native.sh policy ...`。在当前代码中该入口已被归档，直接调用会以
> 退出码 2 提示改用 `policy-vector` / `scripts/evaluation/evaluate_vectorized_seed_range.sh`，
> 上面的命令已相应更新。此外，`collect`、`smoke` 以及各类 `*-smoke` / `*-views`
> 入口依赖 `scripts/collection/` 与 `scripts/smoke/` 下的脚本，这些脚本**不在**
> 本次公开提取的范围内，详见[快速开始](../getting_started.md)中的入口可用性表。

评测在每个控制帧之后检查任务是否成功，并在判定稳定后终止。
`FORCEBENCH_EXECUTE_CHUNK` 可以为某个明确声明的实验覆盖默认的 50；不存在任何
隐藏的姿态或控制器覆盖。

## 7. 新增一个模型适配器

策略评测消费一个 `PolicyObservation`，并且只接受规范绝对 TCP 动作空间中的
`PolicyActionChunk`。传输层和模型特有的键布局位于 `forcebench/policy_adapters/`
下面，而不在评测主循环里。

内置适配器为 `openpi_ws`、`forcevla_ws` 和原生的 `forcebench_ws`。新增一个不以
下划线开头的 Python 模块，导出 `MANIFEST` 和一个 `PolicyAdapter` 实现即可被自动
发现。适配器内部可以做相对动作或关节动作的转换，但必须返回 15 Hz 的世界坐标系
绝对 TCP `xyz/rpy/gripper` 目标。适配器特有的 JSON 选项通过
`FORCEBENCH_POLICY_ADAPTER_OPTIONS` 传入。

## 8. 新增一个任务 / demo 及其专家采集器

不要往共享环境里加任务分支。请在 `forcebench/tasks/<task_name>/` 下新增一个包，
包含：

- `__init__.py`：导出一个经过校验、名为 `MANIFEST` 的 `TaskManifest`；
- `spec.py`：不可变的几何 / 随机化常量与终态判据；
- `outcome.py`：把这些判据组合成该任务的权威判定与稳定失败码，且不导入 Isaac Sim；
- `runtime.py`：一个 `TaskRuntime` 实现，负责场景实体、reset、指令、特权观测以及
  最终的结果报告。

任务注册表会自动发现这些包。任务特有的资产放在
`assets/normalized/<task_name>/` 下，记录溯源 / 哈希，并补上针对性的任务契约测试。
脚本化的 demo 生成器属于 `scripts/expert/`，且必须调用共享的 `ForceBenchEnv`
控制器 API。它们不得实现第二套机器人控制器，也不得在规范 writer 之外写入正式数据。

普通任务的自动专家采集是一套与任务绑定的插件契约。在
`forcebench/expert_adapters/` 下新增一个不以下划线开头的模块，导出正好一个
`ExpertBackendManifest` 和一个只服务于该任务的 `ExpertBackend`。其
`build_command()` 把规范请求翻译成针对单个 seed 的一次隔离生成器调用。
`weight_sort`、`rattle_sort`、`stove_turn_on`、`radmanso_drawer_sort_table` 和
`peg_insert` 有意绕过了这套按 seed 的适配器契约：`collect_expert.py` 把这五个
任务统一路由到 `parallel_sort_collection.py`，后者在一个 Isaac 进程里启动正好
32 个克隆工作空间，跳过已验证的 seed，在最后一个不满批时补齐而不会重复写 HDF5，
并录制一段 32 宫格视频。它们退役的单 seed 生成器和常驻重建生成器不再保留。
Stove 的每个克隆环境保留各自独立的、与 seed 相关的卡扣负载模型和火焰状态。
其他后端保持各自任务特有的进程生命周期。参见
`forcebench/expert_adapters/README.md`。

当前采集状态：

- `weight_sort`、`rattle_sort`、`stove_turn_on`、`radmanso_drawer_sort_table`：
  通过默认的 32 克隆路径启用；
- `peg_insert`：仅通过 32 克隆 60 Hz 采集器路由；正式采集在其 32/32 释放稳定性
  smoke 通过之前保持关闭；
- `faucet_turn`、`gear_mesh`、`nut_thread`：通过任务绑定的适配器启用。

`gear_mesh`、`nut_thread` 和 `peg_insert` 使用任务自有的特权状态教师来获得精确的
装配轨迹。退役的"学习式力技能专家"路径属于历史内容，只存在于外部归档中。
对于 `faucet_turn`，场景几何与峰值阻力（0.20 N·m）是固定的；seed 只随机化卡扣角度。
它的专家从测得的腕部力矩检测这一转变，遵循与 `stove_turn_on` 相同的
"加载—泄载"模式。

## 9. 仓库卫生

- 保留可复用的采集、评测、专家、转换和任务接口。
- 一次性的诊断脚本放在 scratch 根目录下；调查结束后删除。
- 不要在仓库里保留带时间戳的源码备份、PID 文件、生成的日志、预览、
  `__pycache__` 或 `.pytest_cache`。
- 被取代的正式数据 / checkpoint 请连同 manifest 一起归档，而不是删除或与当前
  产物混放。
- 改动共享控制器之后，务必跑一遍静态编译、单元测试和一次无渲染的仿真器精度 smoke。
