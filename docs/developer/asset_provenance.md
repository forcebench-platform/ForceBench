# 资产溯源

> 本文由内部代码库的 `ASSET_PROVENANCE.md` 翻译而来，技术事实未作改动。
> 面向使用者的**许可证**结论请看仓库根目录的 [`NOTICE.md`](../../NOTICE.md)；
> 本文记录的是"这些文件从哪来、被怎样加工过"的工程细节。

Git 仓库中保存的是这份溯源记录、期望的校验和以及派生资产的构建器。部署用的 USD
二进制文件仍然留在受控的 L20 运行时资产目录中，不在源码管理中复制一份。

`/home/pc/ForceBench/assets` 中最初的三个源 USD 文件是从 `franka2` 上已经存在的
自包含 factory 资产复制过来的：

- `factory_peg_8mm.usd` -> `peg_8mm.usd`
- `factory_hole_8mm.usd` -> `hole_8mm.usd`
- `franka_mimic.usd` -> `franka_mimic.usd`

另外两个接触密集型 demo 使用从 `liaohe-l20` 上的 FORGE 部署中提取的六个可移植 USD：

- `factory_gear_base.usd` -> `gear_base.usd`
- `factory_gear_medium.usd` -> `gear_medium.usd`
- `factory_gear_small.usd` -> `gear_small.usd`
- `factory_gear_large.usd` -> `gear_large.usd`
- `factory_nut_m16.usd` -> `nut_m16.usd`
- `factory_bolt_m16.usd` -> `bolt_m16.usd`

源部署为 NVIDIA Isaac Lab 2.3.2 / Isaac Sim 5.1，由记录在
`/home/ubuntu/isaac-forge/VERSIONS.txt` 中的 FORGE 容器摘要固定。`gear_base`、
`gear_medium`、`nut_m16` 和 `bolt_m16` 是
`/home/ubuntu/isaac-forge/workspace/assets/factory` 下已展平文件的逐字节副本。
官方 FORGE GearMesh 配置还需要小齿轮和大齿轮这两个两侧齿轮；这两个文件在早先的
可移植导出中缺失，后来从同一个 Isaac 5.1 资产根目录取回，并用
`Usd.Stage.Flatten(addSourceFileComment=False)` 展平。它们的原始源文件哈希为
`c3284806005543e88006f10a2ee43454f4ceb4f43c72be810654c41afc46434c`
和 `c0c494b877f3548cc1d74759e14f115c1ba548c53f5000e2337445a7eca01739`。

所有九个借用的 USD 都保留了明确的溯源，其部署哈希记录在 `assets/SHA256SUMS` 中。
ForceBench 只从自己的资产目录加载它们。`gear_mesh` 和 `nut_thread` 插件使用了来自
对应 FORGE demo 的、经过审计的几何常量，但它们并不导入 FORGE 或 Factory 的环境、
控制器、奖励、reset 代码或成功判定。

> **公开仓库的核对说明。** [溯源缺口](./provenance_gaps.md)曾指出上面这句
> "所有九个……记录在 `assets/SHA256SUMS` 中"在提取时并不成立（只有 3 条）。
> 本仓库当前的 `assets/SHA256SUMS` 已经补齐了缺失的 6 条 gear/nut/bolt 记录，
> 哈希值与该文档给出的一致，因此这一条缺口在本仓库中已闭合。
> 该文档中提到的另一处较小的缺口（两个展平前的上游源哈希未在 `SHA256SUMS`
> 中以任何名字出现）仍然存在。

`peg_8mm_convex.usd` 是 ForceBench 自有的派生资产，由
`assets/build_derived_assets.py` 生成。它保留了源杆件几何，但把细杆的 SDF 碰撞
替换为凸包，并设置 0.25 mm 的接触偏移。这样可以避免不稳定的侧向接触和端边卡住，
同时保持原始 `peg_8mm.usd` 未被修改、可供审计。

`assets/build_normalized_task_assets.py` 在不修改任何源 USD 的前提下生成
`assets/normalized/` 下的运行时包装文件。每个单体包装文件都是米制、Z 轴向上、
只有一个根刚体、没有活动关节、根原点位于支撑平面上，并带有 `Support` / `Grasp` /
`Mate` 三个坐标系。Peg 保留其凸包；孔和齿轮保留 SDF 碰撞。导入的螺母和螺栓碰撞网格
存在开放边界环，因此改用凸分解烘焙，而不是无效的非水密 SDF。生成的包装文件哈希记录
在 `assets/SHA256SUMS` 中。

默认的 `contact_lab` 视觉 profile 不引用任何外部房间或桌子 USD 文件。它的墙板、
灯光点缀、工作台框架和桌腿都是参数化的 ForceBench 图元。可见的桌面就是提供任务碰撞
表面的那个立方体，因此两者在 `z=0` 处精确对齐。这消除了原本无法审计的、来自视觉资产
的高度约定，也让默认场景无需联网下载即可启动。

## 逐任务的资产加工细节

### `weight_sort`

`weight_sort` 通过归一化包装文件 `assets/normalized/weight_sort/box.usda` 使用
NVIDIA 的 `flatbox_a02` 视觉资产。每个 episode 生成一个箱子，其质量被固定为
`0.06 kg` 或 `0.18 kg`；视觉资产、缩放、材质、摩擦和碰撞几何在两类之间完全相同。
相邻 seed 交替这两种质量，因此任何完整的 50-seed 全成功数据集都包含 25 个轻箱和
25 个重箱。两个分拣目标都使用缩放为 `0.16` 的 `container_f27`，并配有任务自有的
匹配碰撞外壳。轻箱目标被覆盖为黑色材质，重箱目标被覆盖为蓝色材质。

### `rattle_sort`

`rattle_sort` 使用来自 `SimReady_Containers_Shipping_01_NVD@10010.zip` 的 NVIDIA
`flatbox_a07`，该压缩包是在 L20 上从 NVIDIA Omniverse 官方可下载资产包目录中下载的。
只有完整的 `flatbox_a07` 源目录及其引用的 MDL 材质和贴图被部署到
`assets/simready/flatbox_a07/` 下。原始源尺寸为 `0.25604 x 0.31720 x 0.11038 m`；
`assets/build_rattle_sort_assets.py` 施加 `0.2109023311` 的统一缩放，得到一个可抓取的
`0.05400 x 0.06690 x 0.02328 m` 包裹。它生成的包装文件使用六片不可见的墙体碰撞器，
使得一个独立的 `0.06 kg` 内衬物可以在壳体内移动并撞击壳壁。

空包裹 episode 给壳体分配 `0.18 kg`。装载 episode 给壳体分配 `0.12 kg`，并在其中放置
`0.06 kg` 的松动内衬物，从而在两个类别中都精确保持 `0.18 kg` 的总质量。相邻 seed 交替
空 / 载两种包裹，因此每个完整的 50-episode 全成功数据集各含 25 个。两个分拣目标都使用
缩放为 `0.16` 的 `container_f27`；空包裹目标被覆盖为黑色材质，装载目标保留 NVIDIA 原生
的蓝色材质。`parcel_sort_station` 场景 profile 额外添加无碰撞的工业货架、背景传送带、
安全护栏和货架上的包裹，同时保留全局机器人基座、桌面支撑几何和主相机。

### `stove_turn_on`

`stove_turn_on` 使用 CC BY-NC 4.0 授权的 Lightwheel Kitchen `Stovetop012` 源资产，
commit 为 `10b0f7a86135a27e8c2fba9c690f6bf5c4f06bb1`。部署的源 USD 和贴图仍然存在，
其精确的运行时哈希记录在 `assets/SHA256SUMS` 中。

`Stovetop012_single_center.usd` 是由 `assets/build_single_stovetop.py` 生成的、
ForceBench 自有的纯视觉派生文件。源资产把玻璃面板和全部五个灶头组件合并在同一个
网格里，因此构建器会挑出源坐标系下 0.145 m 半宽范围内的不连通组件，在前面板旋钮区
只保留中间那一组控制标识，并用一块 0.290 m 宽的炭黑玻璃立方体替换共享的全宽面板。
它保留了源资产的进深、高度、中央灶头几何、UV、法线和源材质，同时把面数从 23,234
降到 11,685。整个构建以源文件的精确 SHA-256 为前置条件，且是确定性的。

环境包装加载的是这个单灶头派生文件。另有一个独立的固定基座旋转 articulation 复用
未经修改的源中央旋钮几何，并在运行时接收非线性卡扣力矩；只有它的视觉材质被改成拉丝
不锈钢，以便在炭黑灶面上仍然清晰可辨。旋钮的碰撞、质量、关节轴、限位和摩擦均未改动。
发光的灶头火焰是任务自有的、无碰撞的视觉几何。

### `faucet_turn`

`faucet_turn` 使用同一个 Lightwheel Kitchen 版本中的 `Sink054` 资产，通过一个隐藏其
原始把手的纯视觉包装接入。一根固定的水平轴把源龙头一侧的安装座延伸到一个任务自有的
旋转手柄；原先独立的圆柱形控制件已不存在。旋转手柄在视觉上引用未经修改的源
`/root/Sink054_handle` 网格，绕 Z 轴偏航 -90 度以对齐轴平面，在竖直的 X-Z 平面内绕轴的
Y 轴转动，并在运行时接收连续的非线性卡扣力矩。它简化后的碰撞几何、水流视觉、轴和
articulation 都是任务自有的。完整的 Lightwheel `KitchenRoom.usd` 通过一个生成的包装
文件提供周围的视觉房间；该包装移除了所有背景物理，隐藏了重复的水槽、源灯光以及正对
相机的墙 / 窗，然后把房间对齐到任务自有的水槽，同时不移动机器人和主相机。
