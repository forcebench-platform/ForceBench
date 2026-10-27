# ForceBench — Research Release / 研究代码版

**ForceBench: Benchmarking Force Integration for Robotic Manipulation**  
**面向机器人操作的力信息融合评测基准**

[Project page / 项目主页](https://forcebench-platform.github.io/) ·
[Leaderboard / 排行榜](https://forcebench-platform.github.io/leaderboard.html) ·
[Installation / 安装指南](docs/getting_started_main.md) ·
[Validation scope / 验证范围](docs/validation_scope.md)

ForceBench compares how robotic policies use six-axis wrist force and torque
feedback. Eight simulation tasks cover physical-property inference,
resistance-dependent interaction and contact-rich assembly. This release
provides task definitions, simulator code, data conversion, evaluation,
policy adapters and ten system implementations within the submitted-paper scope.

ForceBench 用于比较机器人策略如何融合六维腕部力/力矩反馈。八个仿真任务覆盖物性推断、阻力相关交互和接触密集型装配。本研究代码版提供与投稿论文范围对应的任务定义、仿真代码、数据转换、评测、策略适配器及十个系统的实现。

The [validation report](validation/public_report.json) records the tested source,
assets, environments and execution checks. All ten systems passed short GPU
model execution checks with real data and parameters. All eight tasks passed
two-environment GPU checks; Weight Sort and Nut Thread also passed
single-environment checks. These checks do not establish reproduction of the
paper's success rates.

[验收报告](validation/public_report.json)记录了已检查的源码、资产、环境及运行范围。十个系统均使用真实数据和模型参数通过了短程 GPU 运行验证；八个任务均通过双环境 GPU 检查，Weight Sort 和 Nut Thread 另通过了单环境检查。论文中的成功率尚未通过这些检查重新复现。

## Getting started / 快速开始

The registry and inspection tools need Python 3.11+ and no simulator.

任务列表和基础检查工具仅需 Python 3.11+，无需安装仿真器。

```bash
git clone https://github.com/forcebench-platform/ForceBench.git
cd ForceBench
python3.11 -m venv .venv-tools
source .venv-tools/bin/activate
python -m pip install -e .
python -m forcebench list-tasks
```

Follow the [installation guide](docs/getting_started_main.md) to create a
separate Python 3.12 simulator environment and Python 3.11 model environments.
The checkout does not automatically install Isaac Sim, download model weights
or acquire third-party assets.

按照[安装指南](docs/getting_started_main.md)，分别配置 Python 3.12 仿真环境和 Python 3.11 模型环境。克隆仓库后，Isaac Sim、模型权重及第三方资产仍需单独准备。

After acquiring assets under their original terms, verify and import them with
`tools/prepare_assets.py`, then run `python -m forcebench doctor` in the simulator
environment. The [asset guide](docs/assets.md) documents the pinned 300-file
tree, authored wrappers, renderer requirements and reviewed dependencies.
Source models and textures are supplied separately.

按照各资产的原有条款取得文件后，使用 `tools/prepare_assets.py` 校验并导入，再在仿真环境中运行 `python -m forcebench doctor`。[资产说明](docs/assets.md)列出了固定版本的 300 个文件、自有封装文件、渲染器要求及已核查的依赖。原始模型与纹理文件需单独取得。

For the shared Franka robot, a [verified NVIDIA download](docs/robot_asset.md)
is available through `python tools/download_robot_asset.py --source-dir /path/to/external-source`.
The official and historical binary hashes differ, but their complete decoded
USD content was verified identical; both exact exports are accepted.

共享 Franka 机器人已提供[官方获取步骤](docs/robot_asset.md)：上面的下载命令会校验已验证的 NVIDIA 文件。
官方与历史文件的二进制哈希不同，但完整 USD 内容已核对一致；两份确切版本均可通过校验。

For Weight Sort, [download all required external scene assets](docs/weight_sort_assets.md)
with `python tools/download_weight_sort_assets.py --source-dir /path/to/external-source --workers 4`,
then use the importer above. The historical SimReady package version is pinned.

Weight Sort 已提供[完整外部场景资产的获取步骤](docs/weight_sort_assets.md)，包括正确历史版本的箱体、容器和桌面；
执行上述任务下载命令后，再用资产导入器完成封装和运行时材质检查。

## Eight tasks / 八个任务

| Category / 类别 | Paper task / 论文任务 | Runtime ID / 运行标识 |
|---|---|---|
| Physical-property inference / 物性推断 | Weight Sort / 重量分类 | `weight_sort` |
| Physical-property inference / 物性推断 | CoM Sort / 质心分类 | `rattle_sort` |
| Resistance-dependent interaction / 阻力相关交互 | Faucet Turn / 水龙头旋转 | `faucet_turn` |
| Resistance-dependent interaction / 阻力相关交互 | Faucet Direction / 水龙头转向选择 | `faucet_force_turn` |
| Resistance-dependent interaction / 阻力相关交互 | Stove Turn / 炉灶旋钮旋转 | `stove_turn_on` |
| Resistance-dependent interaction / 阻力相关交互 | Stove Press / 炉灶按压旋转 | `stove_press_turn` |
| Resistance-dependent interaction / 阻力相关交互 | Drawer Pull / 抽屉拉取 | `radmanso_drawer_sort_table` |
| Contact-rich assembly / 接触密集型装配 | Nut Thread / 螺母旋拧 | `nut_thread` |

Use these runtime IDs for collection, policy serving and evaluation. The registry
defines each task's interface; the validation report records the tested scope.

数据采集、策略服务和评测均使用表中的运行标识。任务注册表定义接口，验收报告说明实际验证过的运行范围。

## Ten systems / 十个系统

| Runtime package / 运行包 | Paper systems / 论文系统 |
|---|---|
| [pi0.5](baselines/pi05/README.md) | No Force（无力反馈，F0）, Token Fusion（Token 融合，F1）, Prompt Fusion（提示融合，F2）, Cross-Attention（交叉注意力，F3）, Token Fusion + Pred（Token 融合与预测，F4） |
| [Temporal](baselines/temporal/README.md) | Temporal Prompt（时序提示）, Dual-Path Fusion（双路径融合） |
| [pi0](baselines/pi0/README.md) | pi0 |
| [ForceVLA](baselines/forcevla/README.md) | ForceVLA |
| [ForceFlow](baselines/forceflow/README.md) | ForceFlow |

Install runtime families in separate environments: several implement the same
`openpi` namespace differently. Their READMEs describe inputs, training and
serving; the validation report records measured GPU coverage.

不同运行包需要使用独立环境，其中多个包包含不同版本的 `openpi` 实现。各包的 README 提供输入、训练和服务启动说明；实际 GPU 验证范围以验收报告为准。

Nine OpenPI systems ran three actual optimizer updates, recorded-observation
inference and loopback WebSocket checks. ForceFlow ran two-update diagnostics,
native-server inference and a new-checkpoint reload check. Only the controlled
pi0.5/F4 environment received the separate fresh-install check.

九个 OpenPI 系统完成了三次真实优化器更新、基于已记录观测的推理及本机 WebSocket 通信检查。ForceFlow 完成了两次更新的诊断、原生服务推理及新检查点重载验证。单独的全新环境安装验证仅覆盖受控的 pi0.5/F4 配置。

The common interface uses two RGB images, seven-dimensional robot state and
seven-dimensional absolute TCP actions. TCP means tool center point.
Force-aware adapters receive six-axis wrist wrench; temporal adapters maintain
history at every 15 Hz control step. Physics runs at 60 Hz. ForceFlow predicts
64 actions and executes 32; the other listed recipes use 50/50.

统一评测接口使用两路 RGB 图像、七维机器人状态和七维绝对 TCP 动作；TCP 指工具中心点。使用力反馈的适配器接收六维腕部力/力矩，时序适配器在每个 15 Hz 控制步维护历史信息。物理仿真频率为 60 Hz。ForceFlow 每次预测 64 步、执行 32 步；其余所列配置为预测 50 步、执行 50 步。

## Paper reference and new experiments / 论文参考结果与新实验

The [frozen paper tables](results/paper_reference.json) and
[protocol reference](configs/paper_protocol.json) retain the submitted-paper
values. Check their integrity with the command below.

[冻结的论文结果表](results/paper_reference.json)和[协议参考](configs/paper_protocol.json)保留投稿版本的数值，可使用以下命令检查一致性。

```bash
python tools/verify_paper_reference.py
```

The manuscript reports 1,600 demonstrations across eight simulation tasks.
Datasets, normalization files, base weights and trained checkpoints are separate
inputs and must be selected explicitly. The repository does not select a
checkpoint by directory name or timestamp.

论文包含八个仿真任务、共 1,600 条示范。数据集、归一化文件、基础模型权重和训练检查点均需单独准备并明确指定，代码不会根据目录名称或时间戳自动选择检查点。

The released Faucet Direction scorer uses the paper's one-sided signed angle
threshold. The original narrow-band scorer remains explicit for historical
records. New runs under the paper rule do not replace historical table values.

公开版 Faucet Direction 判分采用论文规定的单侧有符号角度阈值，同时保留原始窄角度区间判分函数，供历史记录核对使用。按论文规则开展的新评测不会覆盖历史表格数值。

The Stove Press expert collector aligns trajectory completion and quality
judging to the same per-clone time point. This correction does not establish
that 200 new expert episodes have been generated and validated.

Stove Press 专家采集器已将轨迹终止和质量判定统一到每个并行环境自身的同一完成时刻。目前尚未据此重新生成并验证 200 条完整专家示范。

Save new evaluation outputs outside the checkout and preserve their source,
asset, model, normalization and seed identities. New checks and rollouts do not
overwrite frozen tables. The paper's five-task real-world results remain
reference values; complete real-robot deployment is outside the validated
default simulation scope.

请将新评测输出保存在仓库外，并记录对应的源码、资产、模型、归一化文件和随机种子信息。新检查与新轨迹不会覆盖冻结结果表。论文中的五任务真机结果作为参考保留；完整真机部署尚未纳入本代码版默认仿真流程的验证范围。

## Licenses / 许可说明

ForceBench's own code and authored wrappers use [MIT](LICENSE). Bundled upstream
code retains its own license; third-party assets and model weights retain their
applicable terms. See [NOTICE.md](NOTICE.md). File checksums and successful asset
loading do not establish redistribution permission.

ForceBench 自有代码和自有封装文件采用 [MIT 许可](LICENSE)。仓库内的上游代码保留各自许可，第三方资产与模型权重遵循其适用条款，详见 [NOTICE.md](NOTICE.md)。文件校验通过或资产成功加载，并不代表已取得再分发授权。
