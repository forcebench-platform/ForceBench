"""Audit preserved ForceVLA snapshots and optional isolated NumPy transforms.

This command never imports a model, tokenizer, dataset loader or GPU runtime.
Without --source-root, only preserved packaged core files are hash-verified;
verification of all historical snapshot files requires the original sources.
"""
import argparse
import ast
import dataclasses
import enum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
SNAPSHOTS = ("base", "stove_sep14", "stovepress_sep13")
CORE = (
    "src/openpi/models/model.py",
    "src/openpi/models/pi0_force.py",
    "src/openpi/models/limoe_simple.py",
    "src/openpi/policies/forcevla_policy.py",
    "src/openpi/transforms.py",
    "src/openpi/training/optimizer.py",
    "src/openpi/training/weight_loaders.py",
)


def snapshot_label(value):
    label = value.removeprefix("forcevla_")
    if label not in SNAPSHOTS:
        raise ValueError(f"Unexpected ForceVLA snapshot: {value}")
    return label


def checked_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative manifest path: {value}")
    return path


def verify_file(root, relative, row):
    root = root.resolve(strict=True)
    path = (root / checked_relative(relative)).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path escapes source root: {relative}")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if len(content) != row["size"] or digest != row["sha256"]:
        raise ValueError(f"Source identity mismatch: {path}")
    return digest


def manifest_rows(manifest):
    rows = {}
    for row in manifest["files"]:
        key = (snapshot_label(row["snapshot"]), checked_relative(row["relative_path"]).as_posix())
        if key in rows:
            raise ValueError(f"Duplicate source manifest entry: {key}")
        rows[key] = row
    if {key[0] for key in rows} != set(SNAPSHOTS):
        raise ValueError("Source manifest must identify all three ForceVLA snapshots")
    return rows


def verify_originals(source, rows):
    source = source.resolve(strict=True)
    counts = {label: 0 for label in SNAPSHOTS}
    for (label, relative), row in rows.items():
        verify_file(source, Path("forcevla_" + label) / relative, row)
        counts[label] += 1
    return {"status": "all_manifest_files_verified", "verified_files": sum(counts.values()),
            "files_per_snapshot": counts}


def verify_packaged_core(rows):
    result = {}
    for relative in CORE:
        hashes = {}
        for label in SNAPSHOTS:
            row = rows[(label, relative)]
            hashes[label] = verify_file(ROOT / ("runtime_" + label), relative, row)
        identical = len(set(hashes.values())) == 1
        if not identical:
            raise ValueError(f"Expected byte-identical numerical core across snapshots: {relative}")
        result[relative] = {"identical_across_snapshots": identical, "sha256": hashes["base"]}
    return result


def class_default(path, class_name, field_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == field_name:
                        return ast.literal_eval(item.value)
    raise ValueError(f"Missing literal default {class_name}.{field_name}")


def check_declared_defaults():
    expected = {
        ("training/config.py", "DataConfig", "use_quantile_norm"): False,
        ("models/pi0_force.py", "Pi0_GuidanceConfig", "action_dim"): 32,
        ("models/pi0_force.py", "Pi0_GuidanceConfig", "action_horizon"): 50,
        ("models/pi0_force.py", "Pi0_GuidanceConfig", "max_token_len"): 48,
        ("models/pi0_force.py", "Pi0_GuidanceConfig", "proprio_dim"): 7,
        ("models/pi0_force.py", "Pi0_GuidanceConfig", "force_dim"): 6,
        ("training/optimizer.py", "CosineDecaySchedule", "warmup_steps"): 1000,
        ("training/optimizer.py", "CosineDecaySchedule", "decay_steps"): 30000,
        ("training/optimizer.py", "CosineDecaySchedule", "peak_lr"): 2.5e-5,
        ("training/optimizer.py", "CosineDecaySchedule", "decay_lr"): 2.5e-6,
    }
    output = {}
    for (relative, cls, field), value in expected.items():
        for label in SNAPSHOTS:
            actual = class_default(ROOT / ("runtime_" + label) / "src/openpi" / relative, cls, field)
            if actual != value:
                raise ValueError(f"Unexpected {label} {cls}.{field}: {actual!r}")
        output[f"{cls}.{field}"] = value
    return output


def load_definitions(path, names, namespace):
    """Execute unchanged named definitions, with imports and other code omitted."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Missing requested definitions in {path}")
    # Annotations need not import Flax/JAX merely to inspect NumPy transforms.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    isolated = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(isolated, str(path), "exec"), namespace)


def check_transforms(runtime):
    import numpy as np

    ns = {"np": np, "dataclasses": dataclasses, "enum": enum,
          "DataTransformFn": object, "__name__": __name__}
    load_definitions(runtime / "src/openpi/models/model.py", ["ModelType"], ns)
    load_definitions(runtime / "src/openpi/transforms.py",
                     ["pad_to_dim", "make_bool_mask", "DeltaActions", "AbsoluteActions", "Normalize", "Unnormalize"], ns)
    ns["transforms"] = SimpleNamespace(DataTransformFn=object, pad_to_dim=ns["pad_to_dim"])
    ns["_model"] = SimpleNamespace(ModelType=ns["ModelType"])
    load_definitions(runtime / "src/openpi/policies/forcevla_policy.py",
                     ["_parse_image", "Forcevla_inputs", "Forcevla_outputs", "WrapAngularActionDeltas"], ns)
    checks = []
    image = np.full((32, 40, 3), 50, dtype=np.uint8)
    wrist = np.full((32, 40, 3), 90, dtype=np.uint8)
    state = np.array([.45, .1, .25, 3.13, 0., .2, .6], dtype=np.float32)
    wrench = np.array([1., 2., 3., .1, .2, .3], dtype=np.float32)
    actions = np.repeat(state[None], 50, axis=0)
    actions[:, :3] += [.1, .2, .3]
    actions[:, 3] = -3.13
    observation = {"image": image, "wrist_image": wrist, "state": state,
                   "eef_wrench": wrench, "prompt": "Sort the parcel."}
    inputs = ns["Forcevla_inputs"](action_dim=32, real_action_dim=7, proprio_dim=7, force_dim=6)
    infer = inputs(observation)
    train = inputs({**observation, "actions": actions.copy()})
    assert infer["state"].shape == (32,) and train["actions"].shape == (50, 32)
    checks.append("state32_and_action50x32_shapes")
    np.testing.assert_array_equal(infer["state"][:13], np.concatenate((state, wrench)))
    np.testing.assert_array_equal(infer["state"][13:], 0)
    checks.append("proprio7_current_wrench6_order_and_zero_padding")
    np.testing.assert_array_equal(train["state"], infer["state"])
    checks.append("training_and_inference_state_identical")
    np.testing.assert_array_equal(train["actions"][:, :7], actions)
    np.testing.assert_array_equal(train["actions"][:, 7:], 0)
    checks.append("action_padding_preserves_values")
    assert list(infer["image_mask"].values()) == [True, True, False]
    np.testing.assert_array_equal(infer["image"]["base_0_rgb"], image)
    np.testing.assert_array_equal(infer["image"]["left_wrist_0_rgb"], wrist)
    np.testing.assert_array_equal(infer["image"]["right_wrist_0_rgb"], 0)
    checks.append("two_rgb_views_and_masked_zero_third_camera")
    assert infer["prompt"] == observation["prompt"] and "actions" not in infer
    assert "force_history" not in infer and "future_wrench" not in infer
    checks.append("current_force_only_inference_and_prompt_contract")
    for field, value in (("state", np.zeros(8)), ("eef_wrench", np.zeros(5))):
        try:
            inputs({**observation, field: value})
        except ValueError:
            checks.append("reject_wrong_" + field + "_dimension")
        else:
            raise AssertionError(f"Wrong {field} dimension was accepted")
    float_rgb = ns["_parse_image"](np.full((32, 40, 3), .5, dtype=np.float32))
    assert float_rgb.dtype == np.uint8
    np.testing.assert_array_equal(float_rgb, 127)
    checks.append("hwc_float_rgb_to_uint8")
    mask = ns["make_bool_mask"](6, -1)
    assert mask == (True,) * 6 + (False,)
    delta = ns["DeltaActions"](mask)({"state": train["state"], "actions": train["actions"].copy()})
    np.testing.assert_allclose(delta["actions"][:, :6], actions[:, :6] - state[:6])
    np.testing.assert_array_equal(delta["actions"][:, 6], actions[:, 6])
    checks.append("xyz_rpy_delta_and_absolute_gripper")
    recovered = ns["AbsoluteActions"](mask)({"state": train["state"], "actions": delta["actions"].copy()})
    np.testing.assert_allclose(recovered["actions"], train["actions"], atol=1e-6)
    checks.append("delta_absolute_roundtrip")
    wrapped = ns["WrapAngularActionDeltas"]()(delta)
    expected = (actions[:, 3:6] - state[3:6] + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(wrapped["actions"][:, 3:6], expected, atol=1e-6)
    np.testing.assert_array_equal(wrapped["actions"][:, [0, 1, 2, 6]], delta["actions"][:, [0, 1, 2, 6]])
    checks.append("shortest_rpy_wrap_leaves_xyz_gripper_unchanged")
    restored = ns["AbsoluteActions"](mask)(wrapped)
    output = ns["Forcevla_outputs"](real_action_dim=7)(restored)["actions"]
    assert output.shape == (50, 7)
    np.testing.assert_allclose(np.sin(output[:, 3:6]), np.sin(actions[:, 3:6]), atol=1e-6)
    np.testing.assert_allclose(np.cos(output[:, 3:6]), np.cos(actions[:, 3:6]), atol=1e-6)
    checks.append("output50x7_and_wrapped_orientation_equivalence")
    stats = SimpleNamespace(mean=np.arange(32) / 10, std=np.arange(32) / 10 + .25)
    normalizer, inverse = ns["Normalize"](None), ns["Unnormalize"](None)
    assert normalizer.use_quantiles is False and inverse.use_quantiles is False
    normalized = normalizer._normalize(infer["state"], stats)
    np.testing.assert_allclose(normalized, (infer["state"] - stats.mean) / (stats.std + 1e-6))
    np.testing.assert_allclose(inverse._unnormalize(normalized, stats), infer["state"], atol=1e-12)
    checks.append("actual_zscore_methods_and_inverse_include_wrench_dimensions")
    assert ns["DeltaActions"](mask)(infer) is infer
    assert ns["WrapAngularActionDeltas"]()(infer) is infer
    checks.append("action_transforms_noop_for_inference_observation")
    return {"numpy_version": np.__version__, "runtime": "runtime_base",
            "passed_checks": checks,
            "limitations": [
                "Definitions were AST-isolated; full runtime imports were not executed.",
                "No replacement model or tokenizer was used, and neither was imported.",
                "Images cover HWC only; CHW/einops handling was not executed.",
                "Z-score arithmetic methods were executed; nested-tree normalization integration was not.",
                "Dataset loading, optimizer, checkpoint, training, serving and GPU execution remain untested.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        help="Optional directory containing forcevla_base, forcevla_stove_sep14 and forcevla_stovepress_sep13")
    parser.add_argument("--check-transforms", action="store_true", help="Run isolated actual NumPy transform definitions")
    parser.add_argument("--report", type=Path, help="Write a new JSON receipt; existing files are not overwritten")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    rows = manifest_rows(manifest)
    report = {"status": "preserved_core_verified_portable_gpu_unvalidated",
              "training_or_serving_started": False,
              "original_snapshots": (verify_originals(args.source_root, rows) if args.source_root else
                                     {"status": "not_supplied_not_verified"}),
              "preserved_core": verify_packaged_core(rows),
              "declared_defaults": check_declared_defaults(),
              "paper_result_reproduction_verified": False}
    if args.check_transforms:
        report["isolated_cpu_transform_checks"] = check_transforms(ROOT / "runtime_base")
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as file:
            file.write(output)
    print(output)


if __name__ == "__main__":
    main()
