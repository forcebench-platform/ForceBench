# NOTICE（草案）

ForceBench 自有的代码（任务定义、控制器、策略适配器、评测 harness，以及它自己编写的
小型"归一化包装" USD 文件）按 [LICENSE](./LICENSE) 发布，供**非商业研究用途**使用。

有若干任务依赖 3D 资产，这些资产**自身的版权属于第三方**。
**ForceBench 不再分发这些第三方的原始资产文件。** 它只分发：

- (a) 自己派生的包装 / 归一化代码；
- (b) 一份有据可查的来源指引 + 校验和，使得已经通过厂商官方渠道、依照厂商自身条款
  独立获取到该资产的使用者，可以确认自己手上的文件正确无误，并放到预期路径下。

逐任务状态见下表。

> ## ⚠️ 重要
>
> **本表是根据项目内部溯源记录整理的草案，未经法律审阅。**
> 其中**两行标记为 `UNCONFIRMED`（未确认）**，因为提取过程中并未独立核实该来源的
> 许可证。**在这两项被解决之前，不要把本表当作法律层面的许可确认。**

## 逐任务第三方资产许可证

| 任务 | 第三方资产 | 来源 | 许可证（按当前理解） | 状态 |
|---|---|---|---|---|
| `weight_sort` | `flatbox_a02` | NVIDIA Omniverse SimReady 目录 | NVIDIA Omniverse/Isaac 资产条款（非开源；二进制本身的再分发受限） | 仅包装代码；二进制必须由使用者自行从 NVIDIA 官方目录获取 |
| `rattle_sort` | `flatbox_a07`（`SimReady_Containers_Shipping_01_NVD`） | NVIDIA Omniverse SimReady 目录（在 L20 上下载） | 同上 | 仅包装代码 |
| `stove_turn_on` | `Stovetop012` | Lightwheel Kitchen，commit `10b0f7a8...` | **CC BY-NC 4.0（非商业）** | 仅包装文件 + 派生的单灶头网格；Lightwheel 发行版的 `LICENSE.txt` 已逐字包含在 `assets/lightwheel_kitchen/` 中 |
| `faucet_turn` | `Sink054`、`KitchenRoom.usd` | Lightwheel Kitchen，与上同一发行版 | **CC BY-NC 4.0（非商业）** | 仅包装代码 |
| `radmanso_drawer_sort_table` | ArtVIP RADMANSO 三抽屉柜 | ArtVIP 资产包 | ⚠️ **UNCONFIRMED —— 本项目尚未核实其许可证** | 在确认之前不要再分发其二进制 |
| `peg_insert` | `peg_8mm.usd`、`hole_8mm.usd`、`franka_mimic.usd` | 内部溯源记录为"`franka2` 上已经存在的自包含 factory 资产" | ⚠️ **UNCONFIRMED —— 根据 `factory_*` 的源命名，很可能派生自 NVIDIA Factory/Isaac，但它抵达 "franka2" 这台机器之前的保管链未被独立追溯** | 在确认之前不要再分发其二进制 |
| `gear_mesh` | `factory_gear_{base,medium,small,large}.usd` | NVIDIA Isaac Lab 2.3.2 FORGE 部署（L20 上的 `isaac-forge`），镜像摘要 `sha256:388dbc8...` | NVIDIA Isaac Sim/Omniverse EULA（部署时通过 `ACCEPT_EULA=Y` 接受；**不是**开放再分发许可） | 仅包装代码；使用者必须自行运行 FORGE / Isaac Lab 部署来获取源文件 |
| `nut_thread` | `factory_nut_m16.usd`、`factory_bolt_m16.usd` | 与 `gear_mesh` 同一 FORGE 部署 | 同 `gear_mesh` | 仅包装代码 |

> **关于两处 `UNCONFIRMED` 的说明：**
> "在 franka2 上"是一个**位置**，不是一条**溯源链**。把这两项写成"已确认"是不诚实的，
> 因此它们保持未确认状态，直到有人真正把来源和许可证追溯清楚为止。

## ForceBench 针对每个任务实际分发的内容

- **任务逻辑**（`spec.py`、`runtime.py`、成功判定、随机化范围）—— ForceBench 自有代码，
  依 [LICENSE](./LICENSE) 按 MIT 条款发布。
- **`assets/normalized/<task>/*.usda` 包装文件** —— ForceBench 编写的粘合层
  （根刚体、`Support`/`Grasp`/`Mate` 坐标系、缩放归一化），它按路径**引用**第三方源
  文件，但**不嵌入**其内容。
- **`assets/SHA256SUMS`** —— ForceBench 自有派生文件的校验和，以及（在法律上可分发的
  范围内）ForceBench 构建时所依据的那个上游文件的精确校验和，供使用者确认自己独立
  获取的副本是否一致。
- **`scripts/download_assets.sh`** —— 按上表指向各厂商的官方分发渠道，自身不托管也不
  代理任何受限二进制。
  ⚠️ **该脚本尚未实现（v1.0 计划项）。** 当前需要按上表手工获取资产。

## 整体授权立场

**项目所有者已决定：ForceBench 整体按非商用 / 研究用途发布，风险自担。**

在每个任务的资产链条被独立厘清之前，请把本仓库**整体**视为**仅限非商业研究使用**。
特别注意 `stove_turn_on` 和 `faucet_turn` 所依赖的 Lightwheel Kitchen 资产明确采用
**CC BY-NC 4.0**，其非商业条款对这两个任务是硬性约束。

## v1.0 之前的待办项

1. 确认 `radmanso_drawer_sort_table` 所用 ArtVIP RADMANSO 柜子资产的许可证。
2. 追溯 `peg_8mm.usd` / `hole_8mm.usd` / `franka_mimic.usd` 在"曾经在 franka2 上"
   之外的真实上游来源与许可证——目前那是一个位置，不是一条溯源链。
3. ~~把缺失的 6 条 SHA256 记录（gear/nut/bolt 文件）补进 `assets/SHA256SUMS`~~
   —— **已完成。** 这 6 条记录现已在 `assets/SHA256SUMS` 中，哈希值与
   [溯源缺口](./docs/developer/provenance_gaps.md)记录的一致。
   该文档中提到的另一处较小缺口（两个展平前的上游源哈希未在 `SHA256SUMS` 中出现）
   **仍未闭合**。
4. 实现 `scripts/download_assets.sh`。
5. 对本表做一次法律审阅。
