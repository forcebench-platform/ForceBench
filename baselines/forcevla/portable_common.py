"""Explicit resources and identities for archived ForceVLA candidate recipes."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
PROFILES = ("base", "stove_sep14", "stovepress_sep13")
CATALOG = json.loads((ROOT / "recipe_catalog.json").read_text(encoding="utf-8"))
SCHEMA = "forcebench_portable_forcevla_candidate_20261014"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(profile):
    runtime = ROOT / ("runtime_" + profile)
    paths = [p for directory in (runtime / "src", runtime / "scripts", runtime / "packages", runtime / "flaxformer")
             for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in {".py", ".json", ".toml", ".lock"}]
    paths.extend(runtime / name for name in ("pyproject.toml", "uv.lock"))
    paths.extend(ROOT.glob("*.py"))
    paths.append(ROOT / "recipe_catalog.json")
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda p: p.relative_to(ROOT).as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def add_inputs(parser):
    parser.add_argument("--source-snapshot", choices=PROFILES, required=True)
    parser.add_argument("--recipe", choices=tuple(CATALOG), required=True)
    parser.add_argument("--lerobot-home", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--norm-file", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)


def inspect_inputs(args):
    if args.recipe == "stovepress_sep13" and args.source_snapshot != "stovepress_sep13":
        raise ValueError("The September 13 Stove Press recipe requires its recorded source snapshot")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", args.repo_id) or args.repo_id.split("/")[1] in {".", ".."}:
        raise ValueError("repo-id must be a namespace/name without path traversal")
    home = args.lerobot_home.resolve(strict=True)
    dataset = (home / args.repo_id).resolve(strict=True)
    if not dataset.is_relative_to(home):
        raise ValueError("Dataset must remain within the explicit LeRobot root")
    base = args.base_params.resolve(strict=True)
    if not base.is_dir() or not any(base.iterdir()):
        raise ValueError("Base parameter directory is missing or empty")
    run = args.run_dir.resolve()
    if any(run == p or run.is_relative_to(p) or p.is_relative_to(run) for p in (dataset, base, ROOT)):
        raise ValueError("Run directory must be separate from source, data and base parameters")
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("total_episodes") != 200 or info.get("fps") != 15:
        raise ValueError("Expected 200 demonstrations at 15 Hz")
    for key, shape in (("state", [7]), ("actions", [7]), ("eef_wrench", [6])):
        if info.get("features", {}).get(key, {}).get("shape") != shape:
            raise ValueError(f"Expected {key} shape {shape}")
    if not {"image", "wrist_image"}.issubset(info.get("features", {})):
        raise ValueError("Expected the two recorded RGB dataset streams")
    norm = args.norm_file.resolve(strict=True)
    stats = json.loads(norm.read_text(encoding="utf-8"))["norm_stats"]
    if set(stats) != {"state", "actions"}:
        raise ValueError("ForceVLA normalizes concatenated state and actions only; unexpected norm keys")
    for key in ("state", "actions"):
        for statistic in ("mean", "std"):
            values = stats[key][statistic]
            if len(values) != 32 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values):
                raise ValueError(f"Expected finite {key}/{statistic} vector of length 32")
        if any(x < 0 for x in stats[key]["std"]):
            raise ValueError("Normalization standard deviation cannot be negative")
    recipe = CATALOG[args.recipe]
    identity = {"schema": SCHEMA, "recipe": args.recipe, "task": recipe["task"],
                "source_snapshot": args.source_snapshot, "repo_id": args.repo_id,
                "dataset_info_sha256": sha256(info_path), "normalization_sha256": sha256(norm),
                "base_parameter_path": str(base), "training_steps": recipe["num_train_steps"],
                "batch_size": recipe["batch_size"], "fsdp_devices": recipe["fsdp_devices"]}
    return {"identity": identity, "home": home, "dataset": dataset, "norm": norm,
            "base": base, "run": run, "runtime": ROOT / ("runtime_" + args.source_snapshot)}


def activate(args, paths):
    runtime, run = paths["runtime"], paths["run"]
    sys.path[:0] = [str(ROOT), str(runtime / "scripts"), str(runtime / "src"),
                   str(runtime / "packages/openpi-client/src"), str(runtime / "flaxformer")]
    sys.dont_write_bytecode = True
    for key, relative in {"HF_HOME": "cache/huggingface", "HF_DATASETS_CACHE": "cache/huggingface/datasets",
                          "HF_HUB_CACHE": "cache/huggingface/hub", "OPENPI_DATA_HOME": "cache/openpi",
                          "JAX_COMPILATION_CACHE_DIR": "cache/jax", "FORCEVLA_JAX_CACHE": "cache/jax",
                          "XDG_CACHE_HOME": "cache/xdg", "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
                          "TORCH_HOME": "cache/torch", "WANDB_DIR": "logs/wandb"}.items():
        path = run / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["HF_LEROBOT_HOME"] = str(paths["home"])
    os.environ["FORCEVLA_CHECKPOINT_LOCK"] = str(run / "checkpoint.lock")
    os.environ["FORCEVLA_IMPORTED_NORM"] = str(paths["norm"])
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled")


def record_checkpoint_identity(checkpoint, config, completed_updates):
    metadata = config.policy_metadata
    identity = metadata["candidate_identity"]
    if source_identity(identity["source_snapshot"]) != metadata["source_sha256"]:
        raise ValueError("Source package changed during training")
    original = Path(os.environ["FORCEVLA_IMPORTED_NORM"]).read_bytes()
    if hashlib.sha256(original).hexdigest() != metadata["normalization_sha256"]:
        raise ValueError("Selected normalization changed during training")
    norm_path = Path(checkpoint) / "assets" / identity["repo_id"] / "norm_stats.json"
    def values(payload):
        return {key: {stat: row.get(stat) for stat in ("mean", "std", "q01", "q99")}
                for key, row in payload["norm_stats"].items()}
    if values(json.loads(original)) != values(json.loads(norm_path.read_text())):
        raise ValueError("Saved checkpoint norm values differ from selected assets")
    temporary = norm_path.with_suffix(".source.tmp")
    temporary.write_bytes(original)
    temporary.replace(norm_path)
    receipt = {**metadata, "completed_updates": completed_updates}
    path = Path(checkpoint) / "forcebench_identity.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_checkpoint(checkpoint, args, paths):
    receipt = Path(checkpoint) / "forcebench_identity.json"
    if not receipt.is_file():
        raise ValueError("Checkpoint identity missing; historical checkpoints need their frozen runtime")
    identity = json.loads(receipt.read_text(encoding="utf-8"))
    if identity.get("candidate_identity") != paths["identity"]:
        raise ValueError("Checkpoint input/recipe/source selection mismatch")
    if identity.get("source_sha256") != source_identity(args.source_snapshot):
        raise ValueError("Checkpoint source identity mismatch")
    if identity.get("completed_updates") != CATALOG[args.recipe]["num_train_steps"]:
        raise ValueError("Checkpoint does not complete the selected recipe")
    norm = Path(checkpoint) / "assets" / args.repo_id / "norm_stats.json"
    if identity.get("normalization_sha256") != sha256(norm) or sha256(norm) != sha256(paths["norm"]):
        raise ValueError("Checkpoint normalization identity mismatch")
    if not (Path(checkpoint) / "params").is_dir():
        raise ValueError("Checkpoint params directory missing")
    return identity
