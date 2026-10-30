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


def _robot_manifest(historical=b"historical export", official=b"official export"):
    manifest = _manifest({"franka_mimic.usd": historical})
    manifest["files"][0]["verified_equivalent_exports"] = [{
        "bytes": len(official), "sha256": hashlib.sha256(official).hexdigest(),
        "source_url": "https://example.invalid/audited-robot.usd",
        "verification_report": "robot_asset_equivalence_20261027.json",
    }]
    return manifest


def test_audited_robot_export_copies_its_actual_bytes_and_reports_identity(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    official = b"official export"
    _write(source, {"franka_mimic.usd": official})
    manifest = _robot_manifest()
    report = _tool().prepare(source, destination, manifest, {})
    assert report["checks_passed"] and report["copied_files"] == 1
    assert (destination / "franka_mimic.usd").read_bytes() == official
    assert report["verified_bytes"] == len(official)
    identity = report["verified_assets"][0]
    for location in ("source", "destination"):
        assert identity[location]["sha256"] == hashlib.sha256(official).hexdigest()
        assert identity[location]["export"] == "verified_equivalent"
        assert identity[location]["present"]
    assert identity["source"]["verification_report"] == "robot_asset_equivalence_20261027.json"
    assert manifest["files"][0]["sha256"] == hashlib.sha256(b"historical export").hexdigest()
    assert _tool().prepare(source, destination, manifest, {})["copied_files"] == 0


def test_unknown_robot_bytes_are_rejected_even_at_an_allowed_size(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, {"franka_mimic.usd": b"x" * len(b"official export")})
    report = _tool().prepare(source, destination, _robot_manifest(), {})
    assert report["issues"] == [{"file": "franka_mimic.usd", "kind": "source_sha256_mismatch"}]
    assert not destination.exists()


@pytest.mark.parametrize("source_bytes,destination_bytes,source_export,destination_export", [
    (b"official export", b"historical export", "verified_equivalent", "primary"),
    (b"historical export", b"official export", "primary", "verified_equivalent"),
])
def test_either_audited_destination_export_is_kept(tmp_path, source_bytes, destination_bytes,
                                                 source_export, destination_export):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, {"franka_mimic.usd": source_bytes})
    _write(destination, {"franka_mimic.usd": destination_bytes})
    report = _tool().prepare(source, destination, _robot_manifest(), {})
    assert report["checks_passed"] and report["prepared"] and report["copied_files"] == 0
    assert (destination / "franka_mimic.usd").read_bytes() == destination_bytes
    identity = report["verified_assets"][0]
    assert identity["action"] == "keep"
    assert identity["source"]["export"] == source_export
    assert identity["destination"]["export"] == destination_export
    assert identity["source"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert identity["destination"]["sha256"] == hashlib.sha256(destination_bytes).hexdigest()


def test_unknown_robot_destination_is_preserved_and_blocks_other_copying(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    manifest = _robot_manifest()
    manifest["files"].append(_manifest({"other.usd": b"another asset"})["files"][0])
    manifest["tasks"]["example"]["files"].append("other.usd")
    _write(source, {"franka_mimic.usd": b"official export", "other.usd": b"another asset"})
    _write(destination, {"franka_mimic.usd": b"user changes"})
    report = _tool().prepare(source, destination, manifest, {})
    assert not report["checks_passed"]
    assert report["issues"] == [{"file": "franka_mimic.usd", "kind": "different_existing_destination"}]
    assert (destination / "franka_mimic.usd").read_bytes() == b"user changes"
    assert not (destination / "other.usd").exists()
    identity = next(row for row in report["verified_assets"] if row["file"] == "franka_mimic.usd")
    assert identity["action"] == "blocked"
    assert "sha256" not in identity["destination"]


def test_equivalent_exports_cannot_relax_unrelated_asset_pins(tmp_path):
    manifest = _robot_manifest()
    manifest["files"][0]["path"] = "gear_medium.usd"
    manifest["tasks"]["example"]["files"] = ["gear_medium.usd"]
    with pytest.raises(ValueError, match="only supported for the audited robot"):
        _tool().prepare(tmp_path / "source", tmp_path / "destination", manifest, {})


def test_equivalent_robot_keeps_dependency_review_bound_to_historical_pin(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, {"franka_mimic.usd": b"official export"})
    manifest = _robot_manifest()
    gap = {"file": "franka_mimic.usd", "kind": "mdl_runtime_module", "reference": "OmniPBR"}
    manifest["gaps"] = [gap]
    review = {"reviews": [{**gap, "classification": "runtime_mdl_module",
                            "owner_sha256": {"franka_mimic.usd": manifest["files"][0]["sha256"]}}]}
    assert _tool().prepare(source, destination, manifest, review, check_only=True)["checks_passed"]
    assert not destination.exists()
    review["reviews"][0]["owner_sha256"]["franka_mimic.usd"] = hashlib.sha256(b"official export").hexdigest()
    report = _tool().prepare(source, destination, manifest, review, check_only=True)
    assert report["issues"][0]["kind"] == "stale_dependency_review"


def test_copied_robot_is_checked_against_selected_export_and_cleaned_on_failure(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, {"franka_mimic.usd": b"official export"})
    module = _tool()
    monkeypatch.setattr(module.shutil, "copyfileobj", lambda incoming, output, length: output.write(b"corrupted"))
    with pytest.raises(OSError, match="differs from selected export"):
        module.prepare(source, destination, _robot_manifest(), {})
    assert not (destination / "franka_mimic.usd").exists()


def test_unknown_destination_created_after_preflight_is_preserved_and_rejected(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, {"franka_mimic.usd": b"official export"})
    module = _tool()
    def create_concurrent_destination():
        _write(destination, {"franka_mimic.usd": b"concurrent user changes"})
        return None
    monkeypatch.setattr(module, "_runtime_material_dir", create_concurrent_destination)
    with pytest.raises(OSError, match="differs from approved exports; preserved"):
        module.prepare(source, destination, _robot_manifest(), {})
    assert (destination / "franka_mimic.usd").read_bytes() == b"concurrent user changes"


@pytest.mark.parametrize("concurrent_bytes,export", [
    (b"historical export", "primary"), (b"official export", "verified_equivalent"),
])
def test_audited_destination_created_after_preflight_is_kept_with_actual_identity(
        tmp_path, monkeypatch, concurrent_bytes, export):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write(source, {"franka_mimic.usd": b"historical export"})
    module = _tool()
    def create_concurrent_destination():
        _write(destination, {"franka_mimic.usd": concurrent_bytes})
        return None
    monkeypatch.setattr(module, "_runtime_material_dir", create_concurrent_destination)
    report = module.prepare(source, destination, _robot_manifest(), {})
    assert report["checks_passed"] and report["prepared"] and report["copied_files"] == 0
    assert (destination / "franka_mimic.usd").read_bytes() == concurrent_bytes
    identity = report["verified_assets"][0]
    assert identity["source"]["export"] == "primary"
    assert identity["destination"]["present"] and identity["action"] == "keep"
    assert identity["destination"]["export"] == export
    assert identity["destination"]["sha256"] == hashlib.sha256(concurrent_bytes).hexdigest()
