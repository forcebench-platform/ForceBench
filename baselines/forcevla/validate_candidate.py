"""Dependency-light CLI and train/checkpoint/serve identity checks; no model execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from portable_common import CATALOG, PROFILES, ROOT, inspect_inputs, record_checkpoint_identity, sha256, source_identity, validate_checkpoint


def numeric_arguments(call):
    values = {}
    for kw in call.keywords:
        if kw.arg in {"name", "exp_name", "assets_base_dir", "checkpoint_base_dir", "weight_loader", "policy_metadata"}:
            continue
        node = kw.value
        if kw.arg == "data":
            node = ast.Call(func=node.func, args=node.args, keywords=[k for k in node.keywords if k.arg not in {"repo_id", "assets"}])
        values[kw.arg] = ast.unparse(node)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    syntax = 0
    for runtime in [ROOT / ("runtime_" + p) for p in PROFILES]:
        for directory in (runtime / "src", runtime / "scripts", runtime / "packages", runtime / "flaxformer"):
            for path in directory.rglob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                syntax += 1
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        syntax += 1
    checks = []
    generated = ast.parse((ROOT / "candidate_recipes.py").read_text())
    for key, row in CATALOG.items():
        fn = next(x for x in generated.body if isinstance(x, ast.FunctionDef) and x.name == "_" + key)
        assert numeric_arguments(fn.body[0].value) == row["recorded_numeric_arguments"]
    checks.append("nine_recipe_numeric_arguments_match_catalog")
    original_recipes = 0
    if args.source_root:
        for key, row in CATALOG.items():
            text = (args.source_root / row["recipe_snapshot"] / "src/openpi/training/config.py").read_text()
            tree = ast.parse(text)
            if key == "stovepress_sep13":
                base = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "TrainConfig"
                            and any(k.arg == "name" and isinstance(k.value, ast.Constant) and k.value.value == "coribench_peg_insert_forcevla_lora_200_current_v1" for k in n.keywords))
                replacement = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_stove_minus1mm" for t in n.targets))
                keywords = {k.arg: k for k in base.keywords}
                keywords.update({k.arg: k for k in replacement.keywords if k.arg != "data"})
                base.keywords = list(keywords.values())
            else:
                base = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.lineno == row["source_line"] and isinstance(n.func, ast.Name) and n.func.id == "TrainConfig")
            assert numeric_arguments(base) == row["recorded_numeric_arguments"], key
            original_recipes += 1
        checks.append("nine_recipe_numeric_arguments_match_original_source")
    with tempfile.TemporaryDirectory(prefix="forcebench_forcevla_check_") as temporary:
        temp = Path(temporary)
        home = temp / "data"
        dataset = home / "example/task"
        (dataset / "meta").mkdir(parents=True)
        info = {"fps": 15, "total_episodes": 200, "features": {"state": {"shape": [7]}, "actions": {"shape": [7]}, "eef_wrench": {"shape": [6]}, "image": {}, "wrist_image": {}}}
        (dataset / "meta/info.json").write_text(json.dumps(info))
        base = temp / "base"; base.mkdir(); (base / "fixture.txt").write_text("No model weights")
        norm = temp / "norm_stats.json"
        stats = {"mean": [0.] * 32, "std": [1.] * 32, "q01": [-1.] * 32, "q99": [1.] * 32}
        norm.write_text(json.dumps({"norm_stats": {"state": stats, "actions": stats}}))
        selected = SimpleNamespace(source_snapshot="stovepress_sep13", recipe="stovepress_sep13", lerobot_home=home, repo_id="example/task", norm_file=norm, base_params=base, run_dir=temp / "run")
        paths = inspect_inputs(selected)
        for profile in PROFILES:
            for recipe in CATALOG:
                if recipe == "stovepress_sep13" and profile != "stovepress_sep13":
                    continue
                command = [sys.executable, str(ROOT / "run.py"), "check", "--source-snapshot", profile, "--recipe", recipe,
                           "--lerobot-home", str(home), "--repo-id", selected.repo_id, "--norm-file", str(norm), "--base-params", str(base), "--run-dir", str(selected.run_dir)]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                data = json.loads(result.stdout)
                assert data["weights_loaded"] is False and data["dataset_payload_checksums_checked"] is False
        assert not selected.run_dir.exists()
        checks.append("25_explicit_recipe_snapshot_metadata_cli_combinations")
        for script in ("run.py", "serve.py", "audit_candidate.py"):
            subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, check=True)
        checks.append("cli_help_without_model_imports")
        checkpoint = temp / "checkpoint"
        checkpoint_norm = checkpoint / "assets/example/task/norm_stats.json"
        checkpoint_norm.parent.mkdir(parents=True)
        (checkpoint / "params").mkdir()
        checkpoint_norm.write_text(json.dumps(json.loads(norm.read_text()), sort_keys=True, indent=2))
        metadata = {"candidate_identity": paths["identity"], "source_sha256": source_identity(selected.source_snapshot), "normalization_sha256": sha256(norm)}
        config = SimpleNamespace(policy_metadata=metadata)
        old_norm = os.environ.get("FORCEVLA_IMPORTED_NORM")
        os.environ["FORCEVLA_IMPORTED_NORM"] = str(norm)
        try:
            assert sha256(checkpoint_norm) != sha256(norm)
            record_checkpoint_identity(checkpoint, config, 10000)
            assert checkpoint_norm.read_bytes() == norm.read_bytes()
            validate_checkpoint(checkpoint, selected, paths)
            checks.append("actual_training_receipt_to_serving_gate_with_reserialized_norm")
            receipt = checkpoint / "forcebench_identity.json"
            valid = json.loads(receipt.read_text())
            for key, wrong in (("source_sha256", "wrong"), ("candidate_identity", {}), ("normalization_sha256", "wrong"), ("completed_updates", 9999)):
                receipt.write_text(json.dumps({**valid, key: wrong}))
                try:
                    validate_checkpoint(checkpoint, selected, paths)
                except ValueError:
                    checks.append(key + "_mismatch_rejected")
                else:
                    raise AssertionError(key + " accepted")
            receipt.write_text(json.dumps(valid))
            altered = json.loads(norm.read_text()); altered["norm_stats"]["state"]["mean"][7] = 1.0
            checkpoint_norm.write_text(json.dumps(altered))
            try:
                record_checkpoint_identity(checkpoint, config, 10000)
            except ValueError:
                checks.append("altered_saved_wrench_norm_value_rejected")
            else:
                raise AssertionError("Altered norm accepted")
        finally:
            if old_norm is None:
                os.environ.pop("FORCEVLA_IMPORTED_NORM", None)
            else:
                os.environ["FORCEVLA_IMPORTED_NORM"] = old_norm
        selected.source_snapshot = "base"
        try:
            inspect_inputs(selected)
        except ValueError:
            checks.append("new_stove_recipe_with_wrong_snapshot_rejected")
        else:
            raise AssertionError("Wrong recipe snapshot accepted")
        selected.source_snapshot = "stovepress_sep13"; selected.repo_id = "../escape"
        try:
            inspect_inputs(selected)
        except ValueError:
            checks.append("dataset_path_traversal_rejected")
        else:
            raise AssertionError("Dataset traversal accepted")
    print(json.dumps({"status": "cpu_metadata_and_asset_checks_passed_gpu_unvalidated", "python_files_parsed": syntax,
                      "checks": checks, "original_recipe_constructors_checked": original_recipes,
                      "source_identities": {p: source_identity(p) for p in PROFILES},
                      "training_started": False, "server_started": False, "weights_loaded": False,
                      "limitations": ["Fixtures have no dataset payload or model weights.", "No fresh dependency installation, tokenizer/model import, GPU or simulator was exercised."]}, indent=2))


if __name__ == "__main__":
    main()
