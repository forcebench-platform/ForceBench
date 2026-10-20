"""Offline preparation must verify all inputs before mutating a destination."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


def _tool():
    spec = importlib.util.spec_from_file_location(
        "prepare_assets", Path(__file__).resolve().parents[1] / "tools/prepare_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(files):
    return {"files": [
        {"path": name, "present": True, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dependencies": []}
        for name, data in files.items()
    ], "tasks": {"example": {"files": list(files)}}, "gaps": []}


def _write(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_preparation_copies_pinned_tree_and_is_idempotent(tmp_path):
    files = {"models/part.usd": b"asset bytes", "textures/part.png": b"texture bytes"}
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, files)
    module = _tool()
    report = module.prepare(source, destination, _manifest(files), {})
    assert report["checks_passed"] and report["prepared"] and report["copied_files"] == 2
    for name, data in files.items():
        assert (destination / name).read_bytes() == data
    assert module.prepare(source, destination, _manifest(files), {})["copied_files"] == 0


def test_bad_input_leaves_destination_absent(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    expected = {"first.usd": b"correct", "second.usd": b"expected"}
    _write(source, {"first.usd": b"correct", "second.usd": b"wrong"})
    report = _tool().prepare(source, destination, _manifest(expected), {})
    assert not report["checks_passed"]
    assert report["issues"] == [{"file": "second.usd", "kind": "source_sha256_mismatch"}]
    assert not destination.exists()


def test_different_destination_is_preserved_and_prevents_all_copying(tmp_path):
    files = {"first.usd": b"expected", "second.usd": b"another"}
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, files)
    _write(destination, {"first.usd": b"preserve user content"})
    report = _tool().prepare(source, destination, _manifest(files), {})
    assert not report["checks_passed"]
    assert (destination / "first.usd").read_bytes() == b"preserve user content"
    assert not (destination / "second.usd").exists()


def test_manifest_path_escape_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Non-portable"):
        _tool().prepare(tmp_path / "source", tmp_path / "destination", _manifest({"../outside.usd": b"bad"}), {})


def test_dependency_omission_and_unreviewed_gap_are_rejected(tmp_path):
    files = {"part.usd": b"input"}
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, files)
    manifest = _manifest(files)
    manifest["files"][0]["dependencies"] = ["missing.png"]
    manifest["gaps"] = [{"file": "part.usd", "kind": "outside_asset_root", "reference": "../missing.usd"}]
    report = _tool().prepare(source, destination, manifest, {})
    assert {row["kind"] for row in report["issues"]} == {"dependency_not_selected", "unreviewed_dependency_gap"}
    assert not destination.exists()


def test_documented_review_is_bound_to_its_source_hash(tmp_path):
    files = {"part.usd": b"input"}
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, files)
    manifest = _manifest(files)
    gap = {"file": "part.usd", "kind": "outside_asset_root", "reference": "../thumbnail.usd"}
    manifest["gaps"] = [gap]
    review = {"reviews": [{**gap, "classification": "outside_referenced_default_prim", "owner_sha256": {"part.usd": "0" * 64}}]}
    report = _tool().prepare(source, destination, manifest, review)
    assert report["issues"][0]["kind"] == "stale_dependency_review"
    review["reviews"][0]["owner_sha256"]["part.usd"] = manifest["files"][0]["sha256"]
    report = _tool().prepare(source, destination, manifest, review, check_only=True)
    assert report["checks_passed"] and not report["prepared"]
    assert not destination.exists()


def test_runtime_material_files_are_checked_without_being_redistributed(tmp_path):
    files = {"part.usd": b"input"}
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, files)
    manifest = _manifest(files)
    gap = {"file": "part.usd", "kind": "mdl_runtime_module", "reference": "OmniPBR"}
    manifest["gaps"] = [gap]
    material = b"supported runtime material"
    review = {"reviews": [{**gap, "classification": "runtime_mdl_module", "owner_sha256": {"part.usd": manifest["files"][0]["sha256"]}}],
              "runtime_materials": [{"name": "OmniPBR.mdl", "bytes": len(material), "sha256": hashlib.sha256(material).hexdigest()}]}
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    assert not _tool().prepare(source, destination, manifest, review, runtime_material_dir=runtime)["checks_passed"]
    (runtime / "OmniPBR.mdl").write_bytes(material)
    report = _tool().prepare(source, destination, manifest, review, runtime_material_dir=runtime)
    assert report["checks_passed"]
    assert not (destination / "OmniPBR.mdl").exists()
