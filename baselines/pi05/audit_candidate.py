"""Audit supplied controlled-pi05 source snapshots; never train or serve a model.

The optional CPU check runs hash-verified NumPy transform definitions in
isolation. Heavy model, tokenizer, simulator and training imports are not run.
"""
import argparse
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


TASKS = (
    "weight_sort", "rattle_sort", "faucet_turn", "faucet_force_turn",
    "stove_turn_on", "stove_press_turn", "radmanso_drawer_sort_table", "nut_thread",
)
METHODS = {
    "F0": "pi05_no_force", "F1": "token_fusion", "F2": "prompt_fusion",
    "F3": "cross_attention", "F4": "token_fusion_pred",
}


def verify_sources(root, manifest):
    """Check every expected source before evaluating any selected definitions."""
    checked = []
    for row in manifest["files"]:
        relative = Path(row["snapshot_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest path must be relative to the snapshot root")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Source must remain within the supplied snapshot root")
        content = path.read_bytes()
        if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Source identity mismatch: {relative.as_posix()}")
        if path.suffix == ".py":
            ast.parse(content, filename=relative.as_posix())
        checked.append(relative.as_posix())
    return checked


def load_definitions(path, names, namespace):
    """Execute only named, already hash-verified numerical definitions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing requested definitions in {path.name}")
    isolated = ast.Module(body=selected, type_ignores=[])
    exec(compile(isolated, path.name, "exec"), namespace)


def verify_historical_identities(root, manifest):
    result = {}
    for label in manifest["snapshots"]:
        runtime = root / label
        files = [*runtime.joinpath("src").rglob("*.py"),
                 *runtime.joinpath("src").rglob("*.json"),
                 *runtime.joinpath("scripts").glob("*.py"), runtime / "UPSTREAM.json"]
        digest = hashlib.sha256()
        for path in sorted(files, key=lambda item: item.relative_to(runtime).as_posix()):
            digest.update(path.relative_to(runtime).as_posix().encode())
            digest.update(path.read_bytes())
        expected = manifest["historical_source_identities"][label]
        if len(files) != expected["hashed_files"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Historical complete-source identity mismatch: {label}")
        result[label] = {"hashed_files": len(files), "sha256": digest.hexdigest()}
    return result


def check_transforms(runtime):
    import numpy as np

    ns = {"np": np, "dataclasses": dataclasses, "__name__": __name__}
    load_definitions(runtime / "src/openpi/transforms.py", ["pad_to_dim"], ns)
    # Only HWC inputs are used below; no replacement for einops is installed.
    load_definitions(runtime / "src/openpi/policies/droid_policy.py", ["_parse_image"], ns)
    ns["transforms"] = SimpleNamespace(DataTransformFn=object, pad_to_dim=ns["pad_to_dim"])
    load_definitions(runtime / "src/openpi/policies/forcebench_policy.py",
                     ["ForceBenchInputs", "ForceBenchOutputs", "WrapAngles", "ClipWrench"], ns)
    inputs = ns["ForceBenchInputs"]
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.array([0.45, 0.1, 0.25, 3.13, 0.0, 0.2, 0.6], dtype=np.float32)
    wrench = np.array([1, 2, 3, .1, .2, .3], dtype=np.float32)
    actions = np.repeat(state[None], 50, axis=0)
    observation = {"observation/image": image, "observation/wrist_image": image,
                   "observation/state": state, "eef_wrench": wrench, "prompt": "Sort the parcel."}
    checks = []

    def rejects(function, value, label):
        try:
            function(value)
        except ValueError:
            checks.append(label)
        else:
            raise AssertionError(label + " was not rejected")

    for method in METHODS:
        infer = inputs(method=method)
        result = infer(observation)
        assert result["state"].shape == (32,)
        assert list(result["image_mask"].values()) == [True, True, False]
        assert not result["image"]["right_wrist_0_rgb"].any()
        if method == "F2":
            np.testing.assert_array_equal(result["state"][:13], np.concatenate([state, wrench]))
        else:
            np.testing.assert_array_equal(result["state"][:7], state)
        assert ("force_history" in result) == (method in {"F1", "F3", "F4"})
        assert "future_wrench" not in result
        if method in {"F1", "F3", "F4"}:
            np.testing.assert_array_equal(result["force_history"], wrench[None])
            np.testing.assert_array_equal(result["force_history_mask"], [True])
        training = {**observation, "actions": actions}
        if method == "F4":
            training["eef_wrench"] = np.repeat(wrench[None], 11, axis=0)
            training["eef_wrench_is_pad"] = np.array([False] * 9 + [True, True])
        trained = inputs(method=method, training=True)(training)
        np.testing.assert_array_equal(trained["state"], result["state"])
        np.testing.assert_array_equal(trained["actions"][:, :7], actions)
        if method == "F4":
            assert trained["future_wrench"].shape == (10, 6)
            np.testing.assert_array_equal(trained["future_wrench_mask"], [True] * 8 + [False, False])
        checks.append(method + "_input_camera_and_train_infer_contract")
        for key in ("future_wrench", "wrench_window"):
            rejects(infer, {**observation, key: np.zeros((10, 6))}, method + "_reject_" + key)
    rejects(inputs(method="F1"), {**observation, "eef_wrench_is_pad": [True]}, "reject_invalid_current_force")
    rejects(inputs(method="F2"), {**observation, "eef_wrench": np.full(6, np.nan)}, "reject_nan_force")
    rejects(inputs(method="F0"), {**observation, "observation/state": np.full(7, np.nan)}, "reject_nan_state")
    rejects(inputs(method="F0"), {**observation, "observation/wrist_image_2": image}, "reject_third_camera")
    float_image = ns["_parse_image"](np.full((224, 224, 3), 0.5, np.float32))
    assert float_image.dtype == np.uint8 and np.all(float_image == 127)
    checks.append("hwc_float_image_scaling")
    padded = ns["pad_to_dim"](actions, 32)
    np.testing.assert_array_equal(ns["ForceBenchOutputs"]()({"actions": padded})["actions"], actions)
    rejects(ns["ForceBenchOutputs"](), {"actions": padded[:49]}, "reject_wrong_action_horizon")
    angles = np.array([[1, 2, 3, 3 * np.pi, -3 * np.pi, .2, .8]])
    wrapped = ns["WrapAngles"]()({"actions": angles})["actions"]
    np.testing.assert_allclose(wrapped[0, 3:6], [-np.pi, -np.pi, .2], atol=1e-12)
    np.testing.assert_array_equal(wrapped[0, [0, 1, 2, 6]], angles[0, [0, 1, 2, 6]])
    clipped = ns["ClipWrench"]()({"force_history": np.array([-8., 8.]), "state": np.array([8.])})
    np.testing.assert_array_equal(clipped["force_history"], [-5., 5.])
    np.testing.assert_array_equal(clipped["state"], [8.])
    checks.extend(["absolute_action_output_shape", "angle_wrap", "wrench_only_clipping"])
    return {"numpy_version": np.__version__, "passed_checks": checks,
            "limitations": ["No tokenizer, JAX model, optimizer, dataset loader, checkpoint, or GPU was exercised.",
                            "Definitions were AST-isolated from the verified snapshot; this is not a full runtime import.",
                            "Image checks cover HWC only; CHW/einops handling was not executed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory containing both snapshots named in SOURCE_MANIFEST.json")
    parser.add_argument("--check-transforms", action="store_true", help="Run isolated NumPy checks after hash verification")
    parser.add_argument("--report", type=Path, help="Write a new receipt; refuses to overwrite an existing file")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    source = args.source_root.resolve(strict=True)
    checked = verify_sources(source, manifest)
    report = {"status": "original_snapshots_verified_portable_gpu_unvalidated", "verified_source_files": len(checked),
              "paper_tasks": list(TASKS), "method_mapping": METHODS, "training_or_serving_started": False,
              "remaining_inputs": manifest["remaining_release_inputs"],
              "recovered_companions": manifest["recovered_companions"]}
    report["historical_source_identities"] = verify_historical_identities(source, manifest)
    if args.check_transforms:
        report["isolated_cpu_transform_checks"] = check_transforms(source / "pi05_formal_10k")
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as target:
            target.write(output)
    print(output)


if __name__ == "__main__":
    main()
