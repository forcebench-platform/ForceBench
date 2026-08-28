# 开源发布计划

**当前策略：代码与官网先行，数据集与 checkpoint 推迟到 2026-09-15 ICRA 投稿之后。**

本文件记录推迟的原因、已经查实的约束、以及届时需要做的决策，避免到时候重新调研一遍。

## 时间线

| 阶段 | 内容 | 状态 |
|---|---|---|
| 现在 – 09-15 | 代码仓库、中文文档、官网与 leaderboard | 进行中 |
| 09-15 ICRA 投稿后 | HuggingFace 数据集与 baseline checkpoint 上传 | 未开始 |

推迟的理由：投稿前把精力放在能被审稿人看到的东西（论文、官网、可复现的评测协议）上；
数据上传是一次性的体力活，且涉及存储费用决策，投稿后再做不影响论文。

## 已查实的约束（2026-08-27 核对 HuggingFace 官方文档）

### 体积：需要付费套餐或申请资助

实测源机器上的体积：

| 内容 | 体积 | 位置 |
|---|---|---|
| π0.5 checkpoint（7 任务） | 250 GB | H200 `/data/hanxuan/coribench_models/pi0.5/` |
| ForceVLA checkpoint（7 任务） | 250 GB | H200 `/data/hanxuan/coribench_models/forcevla/` |
| ForceFlow checkpoint | 22 GB | H200 `/data/hanxuan/coribench_models/forceflow/` |
| 专家数据（7 任务） | 139 GB | L20 `/home/pc/CoRiBench_data/datasets/` |
| **合计** | **约 660 GB** | |

HuggingFace 存储政策：

- **免费账号的公开存储是「best-effort」**，官方文档明确写着「超过最初几个 GB 请负责任地使用……
  需要大量存储就必须升级到 PRO / Team / Enterprise」。660 GB 在免费账号上不现实。
- PRO：约 10 TB 公开存储。
- **学术/高影响力开源项目可申请存储资助**，联系 `datasets@huggingface.co`（数据集）
  或 `models@huggingface.co`（模型），需提供社区影响力证据（下载量、引用、采用情况）。
  论文发表后再申请，材料更有说服力——这也是推迟的一个附带好处。

**可以大幅缩减体积**：单个 checkpoint 的构成是

```
params/       12 GB   ← 推理只需要这个
train_state/  31 GB   ← Adam 优化器状态（mu/nu），只有续训才需要
assets/       20 KB   ← 归一化统计量等
```

**只发布 `params/` + `assets/` 可以砍掉约 74%**：7 任务 × 3 模型 × 12 GB ≈ 250 GB，
加数据 139 GB，约 390 GB。若只发五个力分类任务则更少。

届时需要决定：是否发布 `train_state/`（发了别人能续训，但体积翻四倍）。

### 格式：HDF5 不符合 HF 对大数据集的要求

HuggingFace 对托管大数据集列了明确要求，其中一条是「使用与 HF 生态良好集成的格式」，
推荐 **Parquet / WebDataset**，并要求「避免使用自定义加载脚本」。

我们的原始数据是 **HDF5**，后果：

- **dataset viewer 用不了**（网页上无法预览数据），复用门槛显著提高
- 不满足 HF 对大数据集的托管要求之一

建议上传 **LeRobot 格式（底层是 Parquet）** 而非原始 HDF5。但存在前置缺口，见下。

### 仓库结构限制（我们都不触碰）

| 限制项 | HF 建议 | 我们的情况 |
|---|---|---|
| 每仓库文件数 | < 100k | 约 1,400 ✅ |
| 每目录条目数 | < 10k | 每任务 200 ✅ |
| 单文件大小 | < 200 GB | 百 MB 量级 ✅ |
| 单次 commit 文件数 | < 100 | 用 `upload_folder` 自动分批 ✅ |

## 必须在上传前解决的阻塞项

1. **转换链路缺失**：`convert_hdf5_to_lerobot.py` 吃的是 `--snapshot`（由
   `scripts/collection/build_dataset_snapshot.py` 生成），而 `scripts/collection/`
   不在本次公开提取范围内。**当前仓库没有从原始 HDF5 到 LeRobot/Parquet 的完整路径。**
   见 [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md)。
2. **`gear_mesh` 无数据**：源机器上正式数据集已不存在，只剩 1 条调试残留。
   要么重新采集，要么在发布时明确本任务只有代码没有数据。
3. **两处资产许可证未确认**：`radmanso_drawer_sort_table` 的 ArtVIP 柜子、
   `peg_insert` 的三个 usd 来源链条。见 [`NOTICE.md`](../NOTICE.md)。
4. **π0.5+力6D 基线无有效评测**、**ForceFlow 未评测**——发布时这两列是空的，
   或者在投稿前补跑。

## 届时需要人做的操作（我无法代劳）

1. **建 HuggingFace Organization**（https://huggingface.co/organizations/new）。
   个人账号下的仓库**没有协作者机制**，要让组员有写权限必须用 Organization。
   组织成员角色：`admin`（全权 + 管成员）/ `write`（可读写组织下所有仓库）/
   `contributor`（只能读写自己创建的仓库）/ `read` / `no_access`。
2. **创建两个仓库**：`<org>/ForceBench-datasets`（dataset 类型）、
   `<org>/ForceBench-baselines`（model 类型）。
3. **生成 Write 类型的 access token**（https://huggingface.co/settings/tokens），
   在自己的终端跑 `hf auth login` 存好。**token 不要写进仓库、不要贴进任何对话。**
4. **决定存储方案**：升级 PRO，或向 HF 申请学术开源资助。
5. 开启 GitHub Pages（Settings → Pages），source 必须设为 **`/(root)`** 而非 `/docs`，
   否则官网找不到 `../results/leaderboard.json`。

## 建议的上传顺序

不要一上来就推 390 GB。先用体积最小的任务（`rattle_sort`，11 GB）跑通全流程，
确认权限、格式、dataset viewer、下载脚本都正常，再决定全量怎么传。
