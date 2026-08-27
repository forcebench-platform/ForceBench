---
pretty_name: ForceBench
license: other
license_name: forcebench-research-only
license_link: https://github.com/pkuhanxuan/ForceBench/blob/main/NOTICE.md
task_categories:
  - robotics
tags:
  - robotics
  - manipulation
  - force-torque
  - simulation
  - isaac-lab
  - vla
language:
  - en
size_categories:
  - 1K<n<10K
---

# ForceBench 专家数据集

> **本文档的定位**：这是打算发布到 HuggingFace（`pkuhanxuan/ForceBench-datasets`）的
> dataset card 草稿，按 HF 惯例组织（描述 / 体量 / 字段 / 采集方式 / 许可证 / 引用 /
> 局限）。上传时把本文件的正文内容（去掉这一段说明和顶部 YAML 之外）用作该仓库的
> `README.md`。字段/schema 的详尽参考见仓库内 [`docs/dataset.md`](./dataset.md)——
> 本卡片只摘录使用者最容易踩坑的部分，不重复其中的完整字段表。
>
> **写作时的原型阶段状态**：本仓库目前还没有 HuggingFace token，因此**这份数据
> 尚未实际上传**。本文档描述的是上传后的目标状态，并在末尾给出上传本身需要人做的
> 具体步骤。哪些内容已经过端到端验证、哪些还没有，见下面「验证状态」一节，请如实
> 看待，不要假设一切都已经跑通。

## 验证状态（写这份卡片时）

- ✅ `scripts/download_datasets.py`（本次一并提交）已经针对**真实的**
  `pkuhanxuan/ForceBench-datasets` 仓库测试过「仓库尚不存在」这条路径——目前该
  仓库确实不存在，脚本给出清晰中文提示并以非零退出码退出，不会崩溃。
- ❌ **端到端转换验证未完成。** 原计划用 `gear_mesh` 做验证，但该任务的正式数据集
  在源机器上已不存在（详见下一节）。转换链路本身尚未在本次工作中跑通过。
- ✅ 已修复一个会让转换脚本读不了已发布数据的缺陷：磁盘上所有已采集的 HDF5 都带着
  项目改名前的 `schema_version = "coribench_hdf5"`，而改名后的校验只认
  `"forcebench_hdf5"`。读取路径现已同时接受两者（写入仍只用新标签）。
- ⚠️ **本文档中的许可证判断未经法务审查**，与 `NOTICE.md` 的免责声明一致。
- ❌ 尚未实际执行过任何数据上传——本地没有 HuggingFace token。

## 数据集描述

ForceBench 评测**力/力矩感知对 VLA 操作策略是否有帮助**：8 个 Isaac Lab 仿真任务，
每个任务的专家示教数据都在同一套机器人/控制器/观测契约下采集（15 Hz 控制、60 Hz
物理、`smooth_position` 控制器，详见
[`docs/developer/benchmark_standardization.md`](./developer/benchmark_standardization.md)）。

八个任务分两类，这是本数据集最重要的事实：

- **五个力分类任务**（`weight_sort`、`rattle_sort`、`stove_turn_on`、`faucet_turn`、
  `radmanso_drawer_sort_table`）：决定"正确行为"的物体属性视觉上完全不可观测，力是
  唯一携带该信息的通道。ForceBench 关于"力感知是否有帮助"的实际证据来自这五个任务。
- **三个接触密集装配任务**（`peg_insert`、`gear_mesh`、`nut_thread`）：当前实现下
  成功判定只检查最终位姿，**不读取任何力信号**——MANIFEST 里的 "force-sensitive"
  措辞是前瞻性的，不是已实现的。它们是有用的接触密集操作任务，但不能证明或证伪
  力感知的作用。

完整的任务说明卡片见 [`docs/tasks/`](./tasks/README.md)。

## 任务列表与体量

| 任务 | 类别 | episodes | 体量（约） | 第三方资产许可证 |
|---|---|---:|---:|---|
| `weight_sort` | 力分类 | 200 | 19 GB | NVIDIA SimReady（非开放再分发） |
| `rattle_sort` | 力分类 | 200 | 11 GB | NVIDIA SimReady（非开放再分发） |
| `stove_turn_on` | 力分类 | 200 | 11 GB | **CC BY-NC 4.0**（非商用） |
| `faucet_turn` | 力分类 | 200 | 13 GB | **CC BY-NC 4.0**（非商用） |
| `radmanso_drawer_sort_table` | 力分类 | 200 | 16 GB | ⚠️ **未确认**（ArtVIP） |
| `nut_thread` | 接触密集 | 200 | 25 GB | NVIDIA Isaac/Omniverse EULA |
| `peg_insert` | 接触密集 | 200 | 44 GB | ⚠️ **未确认** |
| `gear_mesh` | 接触密集 | ❌ **无正式数据** | — | NVIDIA Isaac/Omniverse EULA |
| **合计** | | **1,400**（7 个任务） | **约 139 GB** | 见下方「许可证」一节 |

以上为 2026-08-27 在源机器 L20 `/home/pc/CoRiBench_data/datasets/` 上逐目录清点
`.hdf5` 文件的实测结果。两点与早前记录不同，以本表为准：

- **`peg_insert` 已补齐到 200 条**（早前记为 189），体量随之从 42 GB 增至 44 GB。
- **`gear_mesh` 的正式数据集已不在源机器上**。该目录下现在只有 7 个任务。
  `gear_mesh` 仅在 `CoRiBench_scratch/expert/gear_mesh` 下残留 1 条 episode（145 MB），
  属于调试残留，不构成可发布的数据集。因此 **ForceBench v0.1 的数据集实际只包含
  7 个任务**，而代码/任务定义仍是 8 个。这个缺口的原因未在可查材料中记录。

权威数字以 HuggingFace 仓库根目录的 `manifest.json` 为准（由上传脚本在上传时生成，
见文末）。

### 数据集组织方式：单仓库、按任务分文件夹（而不是每任务一个仓库）

数据发布为**一个** HuggingFace Dataset 仓库 `pkuhanxuan/ForceBench-datasets`，
内部按 `<task>/*.hdf5` 分文件夹存放，外加一个根目录 `manifest.json`。理由：

1. **多数用户只要一两个任务**，142 GB 全量不是常态需求。`huggingface_hub` 的
   `snapshot_download(..., allow_patterns=[f"{task}/*"])` 能在**单仓库**内做到按
   文件夹精确选择下载，不必为此把仓库拆开——这正是 `scripts/download_datasets.py`
   的 `--tasks` 选项背后的机制。拆成 8 个仓库并不会让"只下一个任务"变得更容易，
   只会让用户多记 8 个仓库名。
2. **参照 LIBERO 等同类仿真 benchmark 的做法**：代码仓库轻量、数据整体托管在一处、
   提供一个下载脚本按需拉取，而不是把每个任务拆成独立的下载单元。
3. **HuggingFace 单仓库本身没有会在 142 GB / ~1,450 个文件这个体量遇到的硬限制**
   （LFS 单文件建议不超过约 50 GB，本数据集单个 HDF5 文件在百 MB 量级，远低于此）。
4. **增量更新更简单**：`gear_mesh` 未来重新采集、
   或者 ForceBench 增加第 9 个任务时，只需要往同一个仓库对应文件夹追加文件、更新
   `manifest.json`，用户用同一个 `--repo-id` 重新运行下载脚本即可增量拿到新文件。
   拆成 8 个仓库的话，每次都要协调 8 个仓库各自的版本、README、issue 追踪，对目前
   单人维护的项目是不必要的负担。
5. **代价是可以接受的**：单仓库意味着任何任务的更新都会体现在同一条 commit
   历史里（不同任务的提交会交错），且仓库级别的许可证标签只能是一个笼统值——本卡片
   用「体量表」里逐任务的许可证列 + 下面完整的「许可证」一节来弥补这一点，而不是
   为了让每个任务拥有独立、精确的仓库级许可证标签就拆分成 8 个仓库。

如果未来某个任务的资产许可证被确认为可以商用、而其他任务仍然是非商用限定，
到那时再把该任务拆出单独仓库仍然可行（HuggingFace 不要求组织方式一开始就是最终
形态）；现在没有理由为一个假设的未来分叉预先支付管理成本。

## 如何下载

```bash
pip install huggingface_hub
python scripts/download_datasets.py --list                       # 看看有哪些任务、多大
python scripts/download_datasets.py --tasks weight_sort rattle_sort # 只下这两个任务
python scripts/download_datasets.py --all --yes                   # 全量下载（约 139 GB）
```

下载目标目录遵循仓库既有的 `FORCEBENCH_DATA_ROOT` 约定（见
[`forcebench/paths.py`](../forcebench/paths.py)），即
`$FORCEBENCH_DATA_ROOT/datasets/<task>/*.hdf5`，可以用 `--dest` 临时覆盖。脚本会在
下载后做完整性校验（文件数量/大小，`--checksum` 时还核对 SHA-256），并在 HF 仓库
还不存在或网络不通时给出清晰提示而不是抛出未处理的异常。

## 字段/Schema 说明

完整字段表、每个 dataset 的形状/dtype、wrench 的坐标系与滤波语义，见
[`docs/dataset.md`](./dataset.md)。这里只强调两个最容易用错、且不会报错、只会让
训练"悄悄"失败的地方。

### ⚠️ 训练用 `actions/canonical`，不要用 `observations/*`

HDF5 里同时存了**两个版本的位置/姿态信息**，含义完全不同：

| 路径 | 含义 | 能不能用作训练的动作标签 |
|---|---|---|
| `observations/state`、`observations/tcp_pose`、`observations/joint_position` 等 | 仿真器**读回的实际值**——物理步进之后测得的机器人状态 | ❌ 不是动作标签 |
| `actions/canonical` | 当时**发给控制器的目标** TCP 位姿（绝对世界系 `[x,y,z,roll,pitch,yaw,gripper]`） | ✅ 训练用这个 |

**训练策略时应当用 `actions/canonical` 作为动作标签，而不是任何 `observations/*`
字段。** 二者数值上通常接近（因为 `smooth_position` 控制器会在一个控制周期内把
误差收敛掉），这正是这个坑隐蔽的原因——用错字段不会报错、不会让 shape 检查失败，
只会让训练目标混入"实际达到的状态"而不是"期望达到的目标"，随 seed、随控制器跟踪
误差悄悄漂移，模型可能仍然收敛到一个能跑但次优、且与评测时策略实际输出的动作空间
不完全一致的结果。`convert_hdf5_to_lerobot.py` 已经正确地只从 `actions/canonical`
取动作（见该脚本 `actions = handle["actions/canonical"]`），如果你自己写转换/加载
代码，请对齐这个选择。

### wrench 用 raw 一路，不用 EMA

同理，`forces/control/eef_wrench_joint_raw`（训练/推理都用这个）和
`forces/control/eef_wrench_joint_ema`（`alpha=0.25`，仅供离线分析）也是两路不同
用途的数据，混用会导致训练分布和评测时策略实际收到的信号不一致。详见
[`docs/dataset.md`](./dataset.md) 第 5 节。

### 每个 episode 一个 HDF5 文件

文件名 `seed_<seed:03d>_episode_<index:02d>.hdf5`（如 `seed_007_episode_01.hdf5`），
`schema_version` 属性标为 `coribench_hdf5` 或 `forcebench_hdf5`。**一个历史命名残留**：项目代码库
在提取时把内部名字 `coribench` 统一改成了 `forcebench`（含这个 schema 版本字符串
本身），但截至本文档写作时，源头机器（L20/`.48`）上实际采集产出的 HDF5 文件里，
`schema_version` 属性仍然是旧值 `coribench_hdf5`（2026-08-27 抽查 `weight_sort` 首个文件确认）。
**该问题已在本次提交中修复**：读取路径（`forcebench/recording.py` 的 `validate_hdf5_episode`
与 `convert_hdf5_to_lerobot.py`）现已同时接受新旧两个标签，写入路径仍只产出新标签。
因此下载下来的现存数据可以直接转换，不需要预先重写属性。

## 采集方式

正式数据集在 Isaac Lab 2.3.2 / Isaac Sim 5.1 上生成，固定容器镜像摘要
`sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2`。
物理频率 60 Hz，控制频率 15 Hz（decimation=4），机械臂用隐式关节位置 PD
（stiffness 80 / damping 12）经 `smooth_position` 控制器（`SmoothPositionController`，
`ik_method="dls"`, `lambda_val=0.05`）跟踪 15 Hz 目标位姿。完整契约见
[`docs/developer/benchmark_standardization.md`](./developer/benchmark_standardization.md)。
`weight_sort`/`rattle_sort`/`stove_turn_on`/`radmanso_drawer_sort_table` 通过 32
克隆环境的向量化采集；`peg_insert`/`gear_mesh`/`nut_thread`/`faucet_turn`
通过任务绑定的特权位姿专家采集，正式 writer 只接受**成功**的 episode（这也是
`gear_mesh` 迟迟采不满的直接原因——失败的 rollout 不落盘，不是采集中断丢失）。

### 端到端转换验证：**尚未完成**（2026-08-27）

原计划用体量最小的 `gear_mesh` 跑一遍完整的 `convert_hdf5_to_lerobot.py` 以确认
转换链路，但清点时发现该任务的正式数据集已不在源机器上（见「任务列表与体量」一节），
验证因此没有进行。**不要把本仓库的转换链路当作已验证。**

清点过程中确实发现并修复了一个会直接阻断转换的缺陷（见「验证状态」一节的
`schema_version` 兼容性修复），但那是静态代码审查的结果，不等于跑通。

另外需要注意：`convert_hdf5_to_lerobot.py` 接受的是 `--snapshot`（由
`scripts/collection/build_dataset_snapshot.py` 生成的数据集快照），**不是**原始的
`datasets/<task>/` 目录。而 `scripts/collection/` 不在本次公开提取范围内
（见 [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md)），因此当前仓库里**没有**从原始 HDF5
走到 LeRobot 格式的完整链路。补齐这条链路是 v1.0 待办项。

## 许可证与第三方资产限制

**ForceBench 自己的代码**（任务定义、控制器、转换脚本等）按仓库 [`LICENSE`](../LICENSE)
（MIT 变体）发布。**本数据集里的 HDF5 文件不受该许可证约束**——它们是几何/物理
仿真轨迹，但场景里的第三方 3D 资产各自带有自己的版权和许可证条款，会通过渲染影响
到 RGB 观测帧（`observations/images/*`）里出现的视觉内容。以下判断**照抄自
[`NOTICE.md`](../NOTICE.md)，未经法务审查**，两项标记为「未确认」的**不要当作已经
澄清**：

| 任务 | 第三方资产 | 来源 | 许可证（现有理解） |
|---|---|---|---|
| `weight_sort` | `flatbox_a02` | NVIDIA Omniverse SimReady 目录 | NVIDIA Omniverse/Isaac 资产条款（非开放再分发） |
| `rattle_sort` | `flatbox_a07` | NVIDIA Omniverse SimReady 目录 | 同上 |
| `stove_turn_on` | `Stovetop012` | Lightwheel Kitchen | **CC BY-NC 4.0（非商用）** |
| `faucet_turn` | `Sink054`、`KitchenRoom.usd` | Lightwheel Kitchen，同一发布 | **CC BY-NC 4.0（非商用）** |
| `radmanso_drawer_sort_table` | ArtVIP 三抽屉柜 | ArtVIP 资产包 | ⚠️ **未确认** |
| `peg_insert` | `peg_8mm.usd`、`hole_8mm.usd`、`franka_mimic.usd` | 据内部记录"已存在于 franka2 机器上的自包含 factory 资产" | ⚠️ **未确认**（来源链未独立追溯） |
| `gear_mesh` | `factory_gear_{base,medium,small,large}.usd` | NVIDIA Isaac Lab 2.3.2 FORGE 部署（镜像摘要同上） | NVIDIA Isaac Sim/Omniverse EULA（`ACCEPT_EULA=Y` 接受，非开放再分发） |
| `nut_thread` | `factory_nut_m16.usd`、`factory_bolt_m16.usd` | 同上 FORGE 部署 | 同上 |

**项目所有者已决定：本数据集整体按非商用、研究用途发布**，与代码仓库的立场
（见 `LICENSE` 末尾条款）保持一致，直到上表两项「未确认」被独立澄清为止。这不是
在假装两项未确认的条目已经解决——是在缺乏更严格许可证信息的情况下，选择对所有
任务都套用最保守的限制，而不是逐任务区别对待、冒着对未确认项判断过于宽松的风险。
如果你打算把本数据集用于商用场景，请先自行完成对应任务的许可证核实，不要依赖本
文档。

## 引用方式

```bibtex
@misc{forcebench2026,
  title  = {ForceBench: A Benchmark for Force-Aware Manipulation Policies},
  author = {Han, Xuan and others},
  year   = {2026},
  url    = {https://github.com/pkuhanxuan/ForceBench}
}
```

## 已知局限

- **`gear_mesh` 无正式数据**（2026-08-27 清点确认），其余七个任务均为 200/200。缺口
  原因未在提取材料中记录；正式 writer 只接受成功 episode，条数不足反映的是当时
  采集吞吐/成功率，不是数据传输丢失。用这两个任务的数据训练或评测时，样本量本身
  就是需要在结果里说明的限制，统计推断应当附带相应的置信区间（见
  [评测协议](./eval_protocol.md)）。
- **本数据集是仿真数据，不是真机数据。** 8 个任务全部在 Isaac Lab / Isaac Sim 里
  采集，没有物理机器人参与；sim-to-real gap 是使用这份数据训练或评测时需要自行
  评估的问题，本数据集本身不包含任何真实世界 rollout。
- **三个接触密集任务（`peg_insert`/`gear_mesh`/`nut_thread`）的成功判定不读取力
  信号**，只检查最终位姿。这三个任务的数据可以用来做接触密集操作任务，但不能
  用来验证"力感知是否有帮助"——这个问题只有五个力分类任务的数据能回答。
- **自行搭建评测环境时，克隆环境间距必须设为 6 m。** 间距更小会导致相机拍到相邻
  克隆环境（视觉串扰），产生不可用的评测结果——这是评测/仿真搭建层面的坑，不是
  数据本身的字段错误，但用这份数据训练完模型后自己起评测环境的人需要知道，否则
  得到的成功率数字不可信。`./scripts/evaluation/evaluate_vectorized_seed_range.sh`
  已经把这个约束硬编码为 `FORCEBENCH_ENV_SPACING=6`，仅当你绕开标准评测脚本、自己
  拼装向量化环境时才需要自己注意这一条。
- **`schema_version` 命名残留**：见上面「字段/Schema 说明」一节，源头 HDF5 文件
  当前仍标记为旧的 `coribench_hdf5`，需要在打包上传前统一改写。
- **本卡片的许可证判断未经法务审查**（见上一节）。
- **体量数字是冻结发布时的参考值**，源头采集机器上部分任务的实时目录可能显示更大
  的正在进行中的采集结果——那是尚未纳入本次发布快照的后续数据，不代表本次发布
  的体量数字有误。

## 给维护者：如何完成上传

这是本文档写作时**尚未完成**的部分，因为本地没有 HuggingFace token、无法实际
上传。按顺序需要人做的事情：

1. **生成一个 HuggingFace token**：登录 `pkuhanxuan` 账号，在
   <https://huggingface.co/settings/tokens> 创建一个 **Write** 权限的 token
   （细粒度 token 只需要对将要创建的这一个 dataset 仓库有写权限即可，不需要
   账号级别的完全写权限）。
2. **创建仓库**（网页操作，或 `huggingface-cli repo create ForceBench-datasets
   --type dataset`）：仓库 id `pkuhanxuan/ForceBench-datasets`，类型 dataset，
   建议先设为 private，人工确认过一遍内容之后再切换成 public。
3. **在有网络能直连 `huggingface.co`（而不是只能连 `hf-mirror.com` 这类国内
   下载镜像）的机器上登录**：`huggingface-cli login` 或设置
   `HF_TOKEN=<token>` 环境变量。**L20 目前不满足这个条件**——`huggingface.co`
   从 L20 直连不通（`hf-mirror.com` 能连但那是只读下载镜像，不支持推送）。可选
   路径：给 L20 配一个能连回 `huggingface.co` 的代理后在 L20 直接跑；或者把数据
   同步到一台确认能连 `huggingface.co` 的机器上再跑上传（本次准备工作确认了运行
   本次调查的 Windows 机器可以直连 `huggingface.co`）。**上传前务必先用
   `curl -I https://huggingface.co` 之类的命令重新确认一遍网络路径**，不要假设
   和本次调查时完全一致。
4. **打包前统一修一次 `schema_version`**：写一个一次性脚本，把每个任务目录下
   HDF5 的顶层属性 `schema_version` 与 `metadata_json.schema_version` 从
   `coribench_hdf5` 改写为 `forcebench_hdf5`（见上面「字段/Schema 说明」一节），
   在**副本**上做，不要改动 L20/`.48` 上的原始文件。
5. **生成 `manifest.json`**，放在仓库根目录，供 `scripts/download_datasets.py`
   做完整性校验：

   ```json
   {
     "generated_at": "2026-xx-xxTxx:xx:xxZ",
     "tasks": {
       "gear_mesh": {
         "episode_count": 200,
         "bytes": 5583457280,
         "files": [
           {"name": "seed_000_episode_01.hdf5", "bytes": 100776152, "sha256": "..."}
         ]
       }
     }
   }
   ```

6. **用 `huggingface_hub.upload_large_folder`（不是普通的 `upload_folder`）上传**
   ——它是专门为几十/上百 GB、断点续传、可中断重试这类场景设计的：

   ```python
   from huggingface_hub import HfApi
   HfApi().upload_large_folder(
       repo_id="pkuhanxuan/ForceBench-datasets",
       repo_type="dataset",
       folder_path="/path/to/staged/datasets",  # 内含 <task>/*.hdf5 + manifest.json
   )
   ```

   建议先只上传体量最小的 `gear_mesh`（约 5–6 GB）跑通一遍，确认仓库结构、
   `manifest.json` 格式、`scripts/download_datasets.py` 能正确下载校验之后，
   再排期上传其余任务——具体带宽实测和分批建议见提交说明/项目维护笔记，此处
   不重复。
7. **把本文件（去掉最上面「本文档的定位」和「验证状态」两段说明性文字，以及本
   节）设为该 dataset 仓库的 `README.md`**，或者直接保留这两段——HuggingFace 会
   如实渲染，两段文字对最终用户也是有用的上下文，不是必须删除的草稿痕迹。
8. **仓库切到 public**，然后回来跑一遍
   `python scripts/download_datasets.py --tasks gear_mesh` 做真实的端到端验收。
