"""CPU checks for package integrity, transforms, explicit inputs and identity gates."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from audit_candidate import check_transforms
from portable_common import METHODS, ROOT, RUNTIME, TASKS, inspect_inputs, preserve_checkpoint_norm, sha256, source_identity, validate_checkpoint_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    syntax = 0
    paths = [p for directory in (RUNTIME / "src", RUNTIME / "scripts", RUNTIME / "packages")
             for p in directory.rglob("*.py")]
    for path in [*paths, *ROOT.glob("*.py")]:
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        syntax += 1
    transforms = check_transforms(RUNTIME)
    with tempfile.TemporaryDirectory(prefix="forcebench_pi05_check_") as tmp:
        temp = Path(tmp)
        home = temp / "datasets"
        dataset = home / "example/task"
        (dataset / "meta").mkdir(parents=True)
        info = {"fps": 15, "total_episodes": 200,
                "features": {"state": {"shape": [7]}, "actions": {"shape": [7]},
                             "eef_wrench": {"shape": [6]}, "image": {}, "wrist_image": {}}}
        (dataset / "meta/info.json").write_text(json.dumps(info))
        base = temp / "base"
        base.mkdir()
        (base / "fixture_only.txt").write_text("Metadata fixture; no weights are loaded.")
        norm = temp / "norm_stats.json"
        def stats(dim):
            return {"mean": [0.] * dim, "std": [1.] * dim,
                    "q01": [-1.] * dim, "q99": [1.] * dim}
        norm.write_text(json.dumps({"norm_stats": {"state": stats(32), "actions": stats(32),
                                                  "force_history": stats(6), "future_wrench": stats(6)}}))
        selected = SimpleNamespace(task=TASKS[0], method="F4", lerobot_home=home,
                                   repo_id="example/task", norm_file=norm, base_params=base,
                                   run_dir=temp / "run")
        checked = inspect_inputs(selected)
        checks = []
        for method in METHODS:
            command = [sys.executable, str(ROOT / "run.py"), "check", "--task", TASKS[0],
                       "--method", method, "--lerobot-home", str(home), "--repo-id", "example/task",
                       "--norm-file", str(norm), "--base-params", str(base), "--run-dir", str(temp / "run")]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            receipt = json.loads(result.stdout)
            assert receipt["weights_loaded"] is False
            assert receipt["dataset_payload_checksums_checked"] is False
            checks.append(method + "_metadata_cli")
        assert not selected.run_dir.exists(), "check mode must not create run outputs"
        for path in ("run.py", "serve.py", "audit_candidate.py"):
            subprocess.run([sys.executable, str(ROOT / path), "--help"], capture_output=True, check=True)
        checks.append("cli_help_without_model_imports")
        checkpoint = temp / "checkpoint"
        checkpoint.mkdir()
        receipt_path = checkpoint / "forcebench_identity.json"
        schema = "forcebench_portable_pi05_candidate_20261014"
        def reject(label):
            try:
                validate_checkpoint_identity(checkpoint, selected, checked, schema)
            except (ValueError, FileNotFoundError):
                checks.append(label)
            else:
                raise AssertionError(label + " was accepted")
        reject("missing_identity_rejected")
        identity = {"schema": schema, "task": selected.task, "method": selected.method,
                    "source_sha256": source_identity(), "portable_inputs": checked["identity"],
                    "normalization_sha256": sha256(norm), "completed_updates": 10000}
        checkpoint_norm = checkpoint / "assets/example/task/norm_stats.json"
        checkpoint_norm.parent.mkdir(parents=True)
        # Exercise the actual asset fix across JSON reserialization, as OpenPI
        # save_state does, without requiring JAX/Orbax or any model weights.
        checkpoint_norm.write_text(json.dumps(json.loads(norm.read_text()), indent=2, sort_keys=True))
        assert sha256(checkpoint_norm) != sha256(norm)
        preserve_checkpoint_norm(checkpoint, selected.repo_id, norm, sha256(norm))
        assert checkpoint_norm.read_bytes() == norm.read_bytes()
        checks.append("checkpoint_reserialization_preserves_exact_selected_norm")
        altered = json.loads(norm.read_text())
        altered["norm_stats"]["actions"]["mean"][0] = 0.25
        checkpoint_norm.write_text(json.dumps(altered))
        try:
            preserve_checkpoint_norm(checkpoint, selected.repo_id, norm, sha256(norm))
        except ValueError:
            checks.append("checkpoint_changed_norm_values_rejected")
        else:
            raise AssertionError("Changed checkpoint normalization values accepted")
        checkpoint_norm.write_bytes(norm.read_bytes())
        try:
            preserve_checkpoint_norm(checkpoint, selected.repo_id, norm, "changed-input-hash")
        except ValueError:
            checks.append("imported_norm_change_during_training_rejected")
        else:
            raise AssertionError("Changed training norm identity accepted")
        for key, wrong in (("schema", "historical_schema"), ("source_sha256", "wrong"),
                           ("normalization_sha256", "wrong"), ("completed_updates", 5000),
                           ("portable_inputs", {})):
            receipt_path.write_text(json.dumps({**identity, key: wrong}))
            reject(key + "_mismatch_rejected")
        receipt_path.write_text(json.dumps(identity))
        validate_checkpoint_identity(checkpoint, selected, checked, schema)
        checks.append("matching_fixture_identity_accepted_without_loading_weights")
        selected.repo_id = "../escape"
        try:
            inspect_inputs(selected)
        except ValueError:
            checks.append("dataset_traversal_rejected")
        else:
            raise AssertionError("Dataset traversal accepted")
    report = {"status": "cpu_checks_passed_gpu_not_validated", "python_files_parsed": syntax,
              "source_sha256": source_identity(), "isolated_transform_checks": transforms,
              "portable_wrapper_checks": checks,
              "training_started": False, "server_started": False, "weights_loaded": False,
              "limitations": ["Fixtures contain no real dataset or model weights.",
                              "No clean dependency installation, JAX model, GPU, tokenizer, optimizer or simulator was exercised.",
                              "Check mode validates metadata and norm structure, not dataset payload identity or base weight content."]}
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as file:
            file.write(output)
    print(output)


if __name__ == "__main__":
    main()
