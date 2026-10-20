#!/usr/bin/env python3
"""从 HuggingFace Datasets 下载 ForceBench 的专家示教 HDF5 数据。

ForceBench 遵循「代码仓库保持轻量、大数据放 HuggingFace」的惯例（与 LIBERO 等仿真
benchmark 的做法一致）：本仓库本身不包含任何 episode 数据，全部 8 个任务的正式 HDF5
数据集托管在一个 HuggingFace Dataset 仓库里，按任务分文件夹存放。本脚本是获取那份
数据的唯一推荐方式。

用法举例：

    # 列出全部任务及其体量，不下载任何东西，不需要网络
    python scripts/download_datasets.py --list

    # 只下载 gear_mesh 和 weight_sort 两个任务（默认行为：必须显式选择任务）
    python scripts/download_datasets.py --tasks gear_mesh weight_sort

    # 下载全部 8 个任务（约 142 GB，会先打印总体量并要求确认）
    python scripts/download_datasets.py --all

    # 非交互式场景（CI / 脚本调用），跳过确认提示
    python scripts/download_datasets.py --all --yes

数据下载到哪里由 ``FORCEBENCH_DATA_ROOT`` 环境变量决定（与仓库其余部分完全一致的
约定，见 ``forcebench/paths.py``）：具体是
``$FORCEBENCH_DATA_ROOT/datasets/<task>/*.hdf5``。未设置该环境变量时默认为
``/home/pc/ForceBench_data``（多数用户应当覆盖它，见 ``docs/getting_started.md``
第 3 节）。也可以用 ``--dest`` 临时覆盖，不受环境变量影响。

依赖：``pip install huggingface_hub``（不在 ``requirements-dev.txt`` 里，因为只有
下载数据这一个场景需要它）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# Windows 终端的默认代码页往往不是 UTF-8，会把中文输出打印成乱码（这不影响 Linux/
# 大多数 Isaac Lab 机器，但本脚本也常被用户在还没配置好环境时单独运行）。这里显式
# 把标准输出/错误切到 UTF-8，失败也不影响脚本主体逻辑。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# 让脚本既能在设置了 PYTHONPATH（如 run_native.sh）的环境下运行，也能被用户直接
# `python scripts/download_datasets.py` 独立调用——后者通常发生在用户还没有搭建
# Isaac Lab 环境、只是想先把数据拉下来的阶段（见 README Quickstart）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from forcebench.paths import DATA_ROOT, dataset_dir
except Exception:  # pragma: no cover - 极端情况下的兜底，不应该发生
    DATA_ROOT = Path(os.environ.get("FORCEBENCH_DATA_ROOT", "/home/pc/ForceBench_data")).resolve()

    def dataset_dir(task: str) -> Path:  # type: ignore[no-redef]
        return DATA_ROOT / "datasets" / task


# ---------------------------------------------------------------------------
# 数据集元信息
# ---------------------------------------------------------------------------

DEFAULT_REPO_ID = "pkuhanxuan/ForceBench-datasets"
DEFAULT_REVISION = "main"

# 这些体量数字是本数据集"冻结发布"时的参考值（与 docs/数据集卡片.md、
# docs/tasks/*.md 保持一致），用于 --list 展示和下载前的确认提示，不需要网络访问。
# 实际以下载时 HuggingFace 仓库根目录的 manifest.json 为准——如果两者不一致，脚本
# 按 manifest.json 校验，这里的数字只影响展示和"要下多少 GB"的预估。
#
# 注意：weight_sort/rattle_sort/stove_turn_on/radmanso_drawer_sort_table/
# faucet_turn/nut_thread 六个任务的采集在源头机器上仍在持续进行，实时体量可能比这
# 里更大；peg_insert（189/200）和 gear_mesh（62/200）的样本量缺口是已知且记录在案
# 的限制，不是下载不完整，见 docs/数据集卡片.md「已知局限」一节。


@dataclass(frozen=True)
class TaskInfo:
    episodes: int
    bytes_approx: int
    license_tag: str  # 简短许可证标签，完整说明见 dataset card「许可证」一节


TASKS: dict[str, TaskInfo] = {
    "weight_sort": TaskInfo(200, 19 * 1024**3, "SimReady(NVIDIA)"),
    "rattle_sort": TaskInfo(200, 11 * 1024**3, "SimReady(NVIDIA)"),
    "stove_turn_on": TaskInfo(200, 11 * 1024**3, "CC BY-NC 4.0"),
    "radmanso_drawer_sort_table": TaskInfo(200, 16 * 1024**3, "待确认(ArtVIP)"),
    "faucet_turn": TaskInfo(200, 13 * 1024**3, "CC BY-NC 4.0"),
    "nut_thread": TaskInfo(200, 25 * 1024**3, "NVIDIA EULA"),
    "peg_insert": TaskInfo(189, 42 * 1024**3, "待确认"),
    "gear_mesh": TaskInfo(62, 5.2 * 1024**3, "NVIDIA EULA"),
}

TOTAL_BYTES = sum(info.bytes_approx for info in TASKS.values())


def human_size(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出全部可用任务及体量/许可证标签，不下载，不需要网络访问，然后退出。",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        choices=sorted(TASKS),
        default=None,
        help="只下载指定任务（可多选，空格分隔）。八个任务名见 --list。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载全部 8 个任务（约 %s）。出于「不要一次拉 142G」的默认原则，"
        "必须显式传入这个参数或 --tasks，脚本不会隐式下载全部数据。"
        % human_size(TOTAL_BYTES),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="覆盖下载目标目录（默认写入各任务的 forcebench.paths.dataset_dir(task)，"
        "即 $FORCEBENCH_DATA_ROOT/datasets/<task>/）。传入后，所有任务都会被下载到"
        "这一个目录下的同名子目录，不再读取 FORCEBENCH_DATA_ROOT。",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace dataset 仓库 id，默认 {DEFAULT_REPO_ID}。",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"HuggingFace 仓库的分支/tag/commit，默认 {DEFAULT_REVISION}。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token。多数情况下不需要——数据集是公开的话，匿名也能下载；"
        "也可以用 HF_TOKEN 环境变量代替这个参数。",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过下载后的完整性校验（文件数量/大小，以及可用时的 SHA-256）。",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="完整性校验时额外核对 SHA-256（默认只核对文件数量与大小，更快；"
        "远端 manifest.json 里没有某个文件的 sha256 时会跳过该文件的哈希核对）。",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="跳过下载前的大小确认提示，直接开始下载（非交互式场景使用）。",
    )
    args = parser.parse_args(argv)

    if not args.list:
        if args.all and args.tasks:
            parser.error("--all 与 --tasks 不能同时使用，两者选一个。")
        if not args.all and not args.tasks:
            parser.error(
                "必须显式指定 --tasks <任务名...> 或 --all，脚本不会隐式下载全部 "
                f"{human_size(TOTAL_BYTES)} 数据。先用 --list 看看有哪些任务可选。"
            )
    return args


# ---------------------------------------------------------------------------
# 下载 + 校验
# ---------------------------------------------------------------------------


def _confirm(total_bytes: int, task_names: list[str]) -> bool:
    print(f"即将下载 {len(task_names)} 个任务，合计约 {human_size(total_bytes)}：")
    for name in task_names:
        info = TASKS[name]
        print(f"  - {name}: {info.episodes} episodes, 约 {human_size(info.bytes_approx)}")
    if not sys.stdin.isatty():
        print(
            "检测到非交互式终端（stdin 不是 tty），无法读取确认输入。"
            "请加上 --yes 跳过这个确认步骤后重试。",
            file=sys.stderr,
        )
        return False
    reply = input("确认下载？[y/N] ").strip().lower()
    return reply in ("y", "yes")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_remote_manifest(api, repo_id: str, revision: str, token: str | None) -> dict | None:
    """尝试拉取仓库根目录的 manifest.json；不存在则返回 None（不算错误）。"""

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename="manifest.json",
            token=token,
        )
    except EntryNotFoundError:
        print(
            "提示：远端仓库存在，但根目录还没有 manifest.json（可能是数据仍在上传中）。"
            "本次只做「文件数量 + 非空」校验，跳过按内容（大小/SHA-256）的校验。"
        )
        return None
    except RepositoryNotFoundError:
        raise
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_task(
    task: str,
    task_dir: Path,
    remote_manifest: dict | None,
    *,
    do_checksum: bool,
) -> bool:
    local_files = sorted(task_dir.glob("*.hdf5"))
    expected_count = TASKS[task].episodes
    ok = True

    if remote_manifest is not None and task in remote_manifest.get("tasks", {}):
        entry = remote_manifest["tasks"][task]
        expected_count = int(entry.get("episode_count", expected_count))
        by_name = {f["name"]: f for f in entry.get("files", [])}
        local_names = {p.name for p in local_files}
        missing = set(by_name) - local_names
        if missing:
            print(f"  [{task}] 缺少 {len(missing)} 个文件（manifest 中有、本地没有）："
                  f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
            ok = False
        for local_path in local_files:
            record = by_name.get(local_path.name)
            if record is None:
                continue  # 本地多出的文件不算错误，可能是历史下载遗留
            actual_bytes = local_path.stat().st_size
            if "bytes" in record and actual_bytes != int(record["bytes"]):
                print(
                    f"  [{task}] {local_path.name} 大小不符："
                    f"期望 {record['bytes']} 字节，实际 {actual_bytes} 字节"
                )
                ok = False
                continue
            if do_checksum and "sha256" in record:
                actual_sha256 = _sha256(local_path)
                if actual_sha256 != record["sha256"]:
                    print(f"  [{task}] {local_path.name} SHA-256 不匹配，文件可能已损坏")
                    ok = False
    else:
        # 没有可用的远端 manifest：退化为「数量 + 非空」检查。
        if len(local_files) != expected_count:
            print(
                f"  [{task}] 本地有 {len(local_files)} 个 .hdf5 文件，"
                f"期望 {expected_count} 个（这是已知的冻结发布数量，见 dataset card）"
            )
            ok = False
        empty = [p.name for p in local_files if p.stat().st_size == 0]
        if empty:
            print(f"  [{task}] 发现 {len(empty)} 个大小为 0 的文件：{empty[:5]}")
            ok = False

    if ok:
        print(f"  [{task}] 校验通过：{len(local_files)} 个文件")
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        print(f"ForceBench 专家数据集（HuggingFace: {args.repo_id}）")
        print(f"{'任务':<28}{'episodes':>10}{'体量':>10}   许可证")
        for name in sorted(TASKS):
            info = TASKS[name]
            print(f"{name:<28}{info.episodes:>10}{human_size(info.bytes_approx):>10}   {info.license_tag}")
        print(f"\n合计：{len(TASKS)} 个任务，约 {human_size(TOTAL_BYTES)}。")
        print(
            "许可证标签仅为速查，完整的第三方资产条款与「待确认」项说明见 "
            "docs/数据集卡片.md 与 NOTICE.md。"
        )
        return 0

    try:
        import huggingface_hub  # noqa: F401
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        print(
            "缺少 huggingface_hub 依赖。请先运行：\n"
            "    pip install huggingface_hub\n"
            "（这个依赖只有下载数据这一步需要，没有写进 requirements-dev.txt。）",
            file=sys.stderr,
        )
        return 1

    task_names = sorted(TASKS) if args.all else sorted(dict.fromkeys(args.tasks))
    token = args.token or os.environ.get("HF_TOKEN")
    total_bytes = sum(TASKS[name].bytes_approx for name in task_names)

    if not args.yes:
        if not _confirm(total_bytes, task_names):
            print("已取消。")
            return 1

    api = HfApi()
    try:
        api.dataset_info(args.repo_id, revision=args.revision, token=token)
    except RepositoryNotFoundError:
        print(
            f"数据集仓库 {args.repo_id!r} 在 HuggingFace 上还不存在（或你没有访问权限）。\n"
            "这在 ForceBench 当前的原型阶段是预期状态——数据还没有上传。\n"
            "如果你是数据集所有者，需要先创建并上传这个仓库；具体步骤见 "
            "docs/数据集卡片.md 末尾「给维护者：如何完成上传」一节。\n"
            "如果你以为这个仓库应该已经存在，请检查 --repo-id 拼写，或确认你的 "
            "--token / HF_TOKEN 有访问私有仓库的权限。",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - 网络/权限类错误种类很多，统一给出可读提示
        print(
            f"访问 HuggingFace 仓库 {args.repo_id!r} 时出错：{exc}\n"
            "常见原因：网络不通（国内直连 huggingface.co 经常被墙，需要代理），"
            "或 --token / HF_TOKEN 无效。",
            file=sys.stderr,
        )
        return 2

    if args.dest is not None:
        local_root = args.dest
        dest_for_task = {name: local_root / name for name in task_names}
    else:
        local_root = dataset_dir(task_names[0]).parent  # $FORCEBENCH_DATA_ROOT/datasets
        dest_for_task = {name: dataset_dir(name) for name in task_names}
    local_root.mkdir(parents=True, exist_ok=True)

    print(f"下载目标根目录：{local_root}")
    allow_patterns = [f"{name}/*" for name in task_names] + ["manifest.json"]
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_root),
        allow_patterns=allow_patterns,
        token=token,
    )

    if args.no_verify:
        print("已跳过完整性校验（--no-verify）。")
        return 0

    print("开始校验完整性...")
    try:
        remote_manifest = _fetch_remote_manifest(api, args.repo_id, args.revision, token)
    except RepositoryNotFoundError:
        remote_manifest = None

    all_ok = True
    for name in task_names:
        task_dir = dest_for_task[name]
        if not _verify_task(name, task_dir, remote_manifest, do_checksum=args.checksum):
            all_ok = False

    if not all_ok:
        print(
            "\n部分任务的完整性校验未通过，见上面的详情。可以重新运行本脚本"
            "（已下载的文件不会重复下载，huggingface_hub 会跳过校验一致的文件），"
            "或加 --checksum 做更严格的 SHA-256 核对。",
            file=sys.stderr,
        )
        return 3

    print("完整性校验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
