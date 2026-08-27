# 溯源账本缺口

> 本文由代码提取时产生的 `PROVENANCE_GAPS.md` 翻译而来，技术事实未作改动。
>
> **状态更新（文档重构时核对）：** 下面记录的主缺口——6 条缺失的 gear/nut/bolt
> 哈希——在本仓库当前的 `assets/SHA256SUMS` 中**已经补齐**，且哈希值与本文
> "可直接填入"一节给出的完全一致。因此本文的主要部分现在是一份历史记录。
> 文末"另需注意"一节描述的较小缺口仍然存在。

[资产溯源](./asset_provenance.md)（按提取时复制进暂存树的版本）声称
"所有九个借用的 USD 都保留了明确的溯源，其部署哈希记录在 `assets/SHA256SUMS` 中"。
对于源 ForceBench 树中那份 `assets/SHA256SUMS` 的副本（因而也对于本次提取——按范围
规定它逐字复制该账本：只复制文本 / 哈希文件，不复制二进制），这一说法当时是不成立的。

## 文档声称被追踪的内容（"九个借用的 USD"）

1. `peg_8mm.usd`（来自 `factory_peg_8mm.usd`）
2. `hole_8mm.usd`（来自 `factory_hole_8mm.usd`）
3. `franka_mimic.usd`（来自 `franka_mimic.usd`）
4. `gear_base.usd`（来自 `factory_gear_base.usd`）
5. `gear_medium.usd`（来自 `factory_gear_medium.usd`）
6. `gear_small.usd`（来自 `factory_gear_small.usd`）
7. `gear_large.usd`（来自 `factory_gear_large.usd`）
8. `nut_m16.usd`（来自 `factory_nut_m16.usd`）
9. `bolt_m16.usd`（来自 `factory_bolt_m16.usd`）

## 提取时 `assets/SHA256SUMS` 实际包含的内容

上面这份清单中只有 3 个存在：`peg_8mm.usd`、`hole_8mm.usd`、`franka_mimic.usd`。

（注：`assets/SHA256SUMS` 还有一条 `peg_8mm_convex.usd` 的记录，但它是 ForceBench
派生的资产，不属于那九个"借用"文件，因此在这个缺口的计算中既不加分也不减分。）

## 缺失的记录（9 个中的 6 个）——全部是 gear/nut/bolt 这批 FORGE 派生文件

- `gear_base.usd`
- `gear_medium.usd`
- `gear_small.usd`
- `gear_large.usd`
- `nut_m16.usd`
- `bolt_m16.usd`

尽管 `ASSET_PROVENANCE.md` 断言九个都被记录在册，这六个在 `assets/SHA256SUMS` 中
当时无处可寻。

## 六个缺失文件的实测哈希（可直接填入）

这些 `.usd` 二进制文件都很小（每个都远小于 1.5 MB），并且在 L20 的
`/home/pc/ForceBench/assets/` 下可只读直接访问，因此与其只是标记缺口，不如就地对它们
运行了 `sha256sum`（没有任何二进制被复制进暂存树——按范围规定只有代码 / 文本）。
将来搭建这六个文件的资产托管流程的人，可以在二进制真正准备好分发时把下面这几行直接
放进 `assets/SHA256SUMS`：

```
e55ace31759c493a159ae94840bbbb389bfc1eba396c58f77696368e4d5638aa  gear_base.usd
a86713f9cb518216456b1b9ab7fd159bce7cc083fce0afca2374fb40cb739b13  gear_medium.usd
f62c054f1f2ddfe4c6e96d2c6bcb5b9bcbc7cc87cb4c35d5e32d0d7c6e5367bf  gear_small.usd
9e92af584fa258670b7f61c3b81cdda3a74cf8d906fd44640209c6d01be0c377  gear_large.usd
8ff4fa10680ab09b780e9ba1f468eadc06ab3b3a8557c196bc2fa02f799677d4  nut_m16.usd
e822eb329d3ab2aad9bae57a405c60539ecea12bfa4d1af5d3e3609f1fe1f654  bolt_m16.usd
```

（来源：L20 上的
`/home/pc/ForceBench/assets/{gear_base,gear_medium,gear_small,gear_large,nut_m16,bolt_m16}.usd`，
哈希于 2026-08-26。文件大小从约 342 KB 到 1.24 MB 不等——小到就地哈希既安全又便宜；
这些哈希只针对 L20 上当前部署的字节，并未与 `ASSET_PROVENANCE.md` 中引用的原始
FORGE/Factory 源哈希做过校验。）

> **这一步已经完成：** 上面六行现在都在本仓库的 `assets/SHA256SUMS` 中。

## 另需注意

`ASSET_PROVENANCE.md` 还引用了另外两个原始**源**哈希（对应小齿轮和大齿轮，在
ForceBench 自己的展平步骤之前），而它们本身在 `SHA256SUMS` 中并未以任何名字出现——
它们描述的是上游 / 展平前的产物，而不是部署后的文件，因此属于另一个更小的、未计入
上文的缺口（**该缺口至今仍未闭合**）：

- 小齿轮源文件：`c3284806005543e88006f10a2ee43454f4ceb4f43c72be810654c41afc46434c`
- 大齿轮源文件：`c0c494b877f3548cc1d74759e14f115c1ba548c53f5000e2337445a7eca01739`
