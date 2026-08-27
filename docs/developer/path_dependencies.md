# 代码中残留的机器相关硬编码路径

> 本文由代码提取时产生的 `PATH_DEPENDENCIES.md` 翻译而来，技术事实未作改动。

这些路径是在整个暂存树中 grep `/home/pc/ForceBench*`、`/home/ubuntu/`、
`/data/hanxuan`、`/scratch/` 等模式，覆盖所有复制过来的 `.py` 和 `.sh` 文件后
找到的。**这里没有做修复**——只是标记出来，让接手搭建公开仓库的人决定合适的抽象方式
（下面大多数情况已经有环境变量了；真正的问题是"环境变量没设时默认值应该是什么"）。

## 运行时的实际默认值（可由环境变量覆盖，但默认指向 L20 专有路径）

下面这三组是重点——它们每一个都已经由环境变量把关，因此只要操作者设置了对应变量，
ForceBench 在另一台机器上就能开箱运行。风险只在于：未设置时的默认值会静默指向一个
在 L20 之外并不存在的路径（`/home/pc/ForceBench...`），于是配置错误的运行会以一条
令人困惑的 "path not found" 失败，而不是一条清晰的 "请设置这个环境变量"。

- `forcebench/paths.py:11` — `DATA_ROOT = Path(os.environ.get("FORCEBENCH_DATA_ROOT", "/home/pc/ForceBench_data")).resolve()`
- `forcebench/paths.py:13` — `SCRATCH_ROOT` 默认 `"/home/pc/ForceBench_scratch"`（经由 `FORCEBENCH_SCRATCH_ROOT`）
- `forcebench/paths.py:16` — `ARCHIVE_ROOT` 默认 `"/home/pc/ForceBench_archive"`（经由 `FORCEBENCH_ARCHIVE_ROOT`）
- `forcebench/env_cfg.py:27` — `ASSET_DIR = Path(os.environ.get("FORCEBENCH_ASSET_DIR", "/home/pc/ForceBench/assets")).resolve()`
- `run_native.sh:6` — `data_root="${FORCEBENCH_DATA_ROOT:-/home/pc/ForceBench_data}"`
- `run_native.sh:7` — `scratch_root="${FORCEBENCH_SCRATCH_ROOT:-/home/pc/ForceBench_scratch}"`

注意 `forcebench/paths.py` 还会自校验 `DATA_ROOT`/`SCRATCH_ROOT`/`ARCHIVE_ROOT`
彼此互不相同、且都与 `CODE_ROOT` 不同（见 `_validate_roots()`），所以无论选什么默认值，
都必须保持这个不变式。

给接手者的建议方向：把"未设置时"的行为改成 (a) 强制要求显式设置这些环境变量，并抛出
一条点名缺失变量的清晰错误；或者 (b) 相对于仓库 checkout 位置取默认值
（例如 `<repo>/../forcebench_data`），而不是一个只在原始 L20 开发机上存在的绝对路径
`/home/pc/...`。这里没有直接改，是因为这属于关于 ForceBench 预期目录布局的产品决策，
而不是机械性的修补。

## 惰性 / 仅作元数据的引用（不是运行时依赖）

- `scripts/expert/generate_parallel_peg_insert.py:91-95` — 一个模块级字典
  `L20_MIGRATION_SOURCE`，记录该生成器的逻辑是从哪里迁移过来的：
  - `"collector": "/home/ubuntu/forcevla_peg/code/tasks/blind_peg_insertion/isaac_collect_dataset.py"`
  - `"isaac_env": "/home/ubuntu/forcevla_peg/code/tasks/blind_peg_insertion/isaac_peg_env.py"`

  这两个路径在运行时从不会被打开 / 读取——这个字典（连同它的两个 SHA-256 字段）
  只是作为描述性元数据附加在写出的 episode 上（`l20_migration_source` 键，
  第 667 和 698 行）。作为"代码内的历史文档"留着是安全的，但仍然标记出来，因为它
  技术上是被打包进代码的绝对路径，且引用了一个 ForceBench 并不包含、公众也拿不到的
  兄弟仓库（`forcevla_peg`）。

## 未发现 / 本来就干净的部分

在复制过来的 `forcebench/`、`scripts/expert/`、`scripts/evaluation/` 和 `tests/`
目录树中，没有找到其他绝对机器路径（没有 `/scratch/`、没有 `/data/hanxuan/...`、
没有内网 IP 字面量）。策略适配器（`forcebench/policy_adapters/*.py`）以构造参数接收
host/port，没有硬编码的默认 host，因此那里不存在缺口。

另外（不算路径问题，但相关）：源码树中 `scripts/evaluation/run_*_20260825.sh` 那四个
带日期的一次性脚本（它们完全没有被复制进这个暂存树——见最终报告）硬编码了一个内网
主机 IP（`172.31.202.28`）和 `/data/hanxuan/...` 下的绝对 checkpoint 路径。由于这些文件
在提取时就被整体排除了，这里没有把它们列为"待修复"项，但审阅者应当知道它们缺席的原因。
