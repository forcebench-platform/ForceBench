# 旧版 README（历史存档）

> **已废弃。** 这是 ForceBench 在内部代码库中的旧 README，已被仓库根目录的
> [`README.md`](../../README.md) 取代。保留它是因为它用最紧凑的形式记录了提取当时
> 的规范运行时契约，可以作为[基准标准化记录](./benchmark_standardization.md)和
> [项目指南](./project_guide.md)的交叉核对依据。**不要**把它当作当前文档使用。

ForceBench 是一个直接构建在 Isaac Lab 之上的任务插件式 benchmark。每个任务共享同一个
机器人、同一套渲染契约、同一套观测 / 动作契约以及同一个手臂控制器。

规范运行时：

- 物理：60 Hz，PhysX 位置求解器迭代次数：64；
- 策略 / 控制：15 Hz；
- 手臂：Isaac Lab `DifferentialIKController`，绝对位姿，DLS（`lambda=0.05`），
  之后接一段有界的五次多项式关节轨迹和隐式关节位置 PD；
- 状态：测得的 TCP `xyz`、连续的 `rpy`、测得的夹爪开度；
- 动作：绝对目标 TCP `xyz/rpy` 和目标夹爪开度；
- wrench：`panda_hand_joint` 入向约束反作用力在关节坐标系下的 raw 与 EMA 两路，
  训练和推理使用 raw；
- 模型接口：自动发现的策略适配器，共用一套规范观测 / 动作契约；
- 专家接口：与任务绑定、自动发现的采集器，统一藏在 `./run_native.sh expert` 之后；
- 正式 episode 格式：`forcebench_hdf5`，不含设备相关的遥操作字段。

使用 `./run_native.sh` 作为 Isaac Lab 的入口。存储布局、数据契约、评测行为和任务插件
接口详见 `PROJECT_GUIDE.md`（现为[项目指南](./project_guide.md)）。
