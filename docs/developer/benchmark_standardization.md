# 基准标准化记录

> 本文由内部代码库的 `BENCHMARK_STANDARDIZATION.md` 翻译而来，技术事实未作改动。
> 它记录的是"当前被固定下来的契约"，与[项目指南](./project_guide.md)第 2–4 节
> 互为印证。

## 当前的规范契约

- 只有一个手臂控制器：Isaac Lab 的 `DifferentialIKController`，绝对位姿指令，
  DLS（`lambda=0.05`）。
- 控制器消费测得的 TCP 位姿、基座坐标系下的几何雅可比以及连续的测量关节位置。
  其求解结果经过关节步长限制、关节限位夹取，再通过共享的五次多项式轨迹送入隐式
  位置驱动。
- 不存在状态覆写、操作空间阻抗分支、PI hold、控制器选择环境变量，或仅在评测时
  生效的姿态 / 阻尼保护。
- 关节角与测得的 TCP RPY 在 `-pi/+pi` 分支处保持连续。
- 物理固定在 60 Hz，策略控制固定在 15 Hz，求解器位置迭代固定为 64。
- 正式数据使用稳定的 schema 名 `forcebench_hdf5`，且不含设备相关的遥操作字段。
  以前的数据集和 checkpoint 保存在外部归档根目录下，不再作为新的正式输入被接受。
- 专家采集会存储同步的 raw 与 EMA 入向关节 wrench 两路通道。raw 是模型输入，
  EMA 是辅助通道。任务特有的生成器通过自动发现的专家插件以及唯一的
  `run_native.sh expert` 监督器接入。
- Gear mesh、nut thread 和 peg insert 使用直接的特权位姿教师；此前的"学习式力技能"
  采集路径已归档且不再可用。
- Faucet 的几何与卡扣强度（0.20 N·m）在所有 seed 上固定；只有卡扣角度被随机化。
  它的转变检测器消费的是测得的腕部力矩，而不是解析阻力曲线。
- 策略推理使用唯一的规范观测 / 动作契约。模型与传输层的差异被隔离在自动发现的
  策略适配器中。

## 存储分离

- 代码 / 运行时：`/home/pc/ForceBench`
- 正式数据集 / checkpoint / 评测结果：`/home/pc/ForceBench_data`
- smoke 测试、预览和一次性输出：`/home/pc/ForceBench_scratch`
- 不可变的历史产物：`/home/pc/ForceBench_archive`

归档 manifest 记录原始路径、大小和移动时间。新的 smoke 或调试输出绝不能写到正式
的 `datasets/` 目录下。

> 上述绝对路径是原始开发机（L20）上的布局。在公开仓库中它们都可以由环境变量
> 覆盖，详见[路径依赖](./path_dependencies.md)。
