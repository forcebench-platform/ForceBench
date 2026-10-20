"""Explicit inputs and identity checks for the portable controlled candidate."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
TASKS = ("weight_sort", "rattle_sort", "faucet_turn", "faucet_force_turn",
         "stove_turn_on", "stove_press_turn", "radmanso_drawer_sort_table", "nut_thread")
METHODS = ("F0", "F1", "F2", "F3", "F4")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    """Include the portable wrappers and dependency recipe in the source guard."""
    files = [p for directory in (RUNTIME / "src", RUNTIME / "scripts", RUNTIME / "packages")
             for p in directory.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in {".py", ".json", ".toml", ".lock"}]
    files.extend(RUNTIME / name for name in ("UPSTREAM.json", "pyproject.toml", "uv.lock"))
    files.extend(ROOT.glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def add_inputs(parser):
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--lerobot-home", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--norm-file", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)


def inspect_inputs(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", args.repo_id):
        raise ValueError("repo-id must be namespace/name using ASCII letters, numbers, _, - or .")
    if args.repo_id.split("/")[1] in {".", ".."}:
        raise ValueError("Invalid dataset name")
    home = args.lerobot_home.resolve(strict=True)
    dataset = (home / args.repo_id).resolve(strict=True)
    if not dataset.is_relative_to(home):
        raise ValueError("Dataset must remain within the explicit LeRobot root")
    base = args.base_params.resolve(strict=True)
    if not base.is_dir() or not any(base.iterdir()):
        raise ValueError("Base parameter directory is missing or empty")
    run = args.run_dir.resolve()
    if any(run == p or run.is_relative_to(p) or p.is_relative_to(run) for p in (dataset, base, ROOT)):
        raise ValueError("Run directory must be separate from dataset, base parameters and package")
    info_file = dataset / "meta/info.json"
    info = json.loads(info_file.read_text(encoding="utf-8"))
    if info.get("fps") != 15 or info.get("total_episodes") != 200:
        raise ValueError("Expected the selected 200-demonstration, 15 Hz dataset")
    for key, shape in (("state", [7]), ("actions", [7]), ("eef_wrench", [6])):
        if info.get("features", {}).get(key, {}).get("shape") != shape:
            raise ValueError(f"Expected {key} shape {shape}")
    for key in ("image", "wrist_image"):
        if key not in info["features"]:
            raise ValueError(f"Missing dataset feature: {key}")
    norm = args.norm_file.resolve(strict=True)
    payload = json.loads(norm.read_text(encoding="utf-8"))
    stats = payload["norm_stats"]
    dimensions = {"state": 32, "actions": 32}
    if args.method in {"F1", "F3", "F4"}:
        dimensions["force_history"] = 6
    if args.method == "F4":
        dimensions["future_wrench"] = 6
    for key, dim in dimensions.items():
        for statistic in ("mean", "std", "q01", "q99"):
            values = stats[key][statistic]
            if len(values) != dim or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError(f"Expected finite {key}/{statistic} vector of length {dim}")
    identity = {"task": args.task, "method": args.method, "repo_id": args.repo_id,
                "dataset_info_sha256": sha256(info_file), "normalization_sha256": sha256(norm),
                "base_parameter_path": str(base), "training_steps": 10000, "batch_size": 32}
    return {"identity": identity, "dataset": dataset, "base": base, "norm": norm,
            "run": run, "home": home}


def activate(paths):
    os.environ["FORCEBENCH_RUN_ROOT"] = str(paths["run"])
    os.environ["FORCEBENCH_LEROBOT_HOME"] = str(paths["home"])
    sys.path[:0] = [str(ROOT), str(RUNTIME / "scripts"), str(RUNTIME / "src"),
                   str(RUNTIME / "packages/openpi-client/src")]


def preserve_checkpoint_norm(checkpoint, repo_id, norm_file, expected_sha256):
    """Preserve input bytes after OpenPI has serialized equivalent norm values."""
    original = Path(norm_file).read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("Imported normalization changed during training")
    target = Path(checkpoint) / "assets" / repo_id / "norm_stats.json"
    supplied = json.loads(original)["norm_stats"]
    saved = json.loads(target.read_text(encoding="utf-8"))["norm_stats"]
    def numeric_fields(stats):
        return {name: {key: value.get(key) for key in ("mean", "std", "q01", "q99")}
                for name, value in stats.items()}
    if numeric_fields(supplied) != numeric_fields(saved):
        raise ValueError("Checkpoint normalization values differ from imported training assets")
    # OpenPI's serializer can change whitespace or key order without changing
    # values. Retain the exact selected asset for the strict serving hash gate.
    temporary = target.with_suffix(".source.tmp")
    temporary.write_bytes(original)
    temporary.replace(target)


def validate_checkpoint_identity(checkpoint, args, paths, schema):
    receipt = Path(checkpoint) / "forcebench_identity.json"
    if not receipt.is_file():
        raise ValueError("Checkpoint has no identity; no diagnostic bypass is supported")
    identity = json.loads(receipt.read_text(encoding="utf-8"))
    if (identity.get("schema"), identity.get("task"), identity.get("method")) != (schema, args.task, args.method):
        raise ValueError("Checkpoint schema/task/method mismatch; use its frozen runtime")
    if identity.get("source_sha256") != source_identity():
        raise ValueError("Training and serving source hashes differ; use the checkpoint's frozen runtime")
    if identity.get("portable_inputs") != paths["identity"]:
        raise ValueError("Checkpoint input identity differs from the supplied inputs")
    norm = Path(checkpoint) / "assets" / args.repo_id / "norm_stats.json"
    if identity.get("normalization_sha256") != sha256(norm) or sha256(norm) != sha256(paths["norm"]):
        raise ValueError("Checkpoint normalization differs from training or the supplied norm file")
    if identity.get("completed_updates") != 10000:
        raise ValueError("This paper candidate serves only the explicit 10,000-update checkpoint")
    return identity
