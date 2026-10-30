"""Preflight must reject incomplete/mismatched external geometry before GPU startup."""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import pytest

from forcebench import nut_assets
from forcebench.nut_assets import inspect_nut_assets, validate_nut_assets


def _prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_nut_assets", Path(__file__).resolve().parents[1] / "tools" / "prepare_nut_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nut_preflight_reports_sources_even_when_wrappers_exist(tmp_path):
    wrappers = tmp_path / "normalized/nut_thread"
    wrappers.mkdir(parents=True)
    for wrapper, source in (("nut.usda", "nut_m16.usd"), ("bolt.usda", "bolt_m16.usd")):
        (wrappers / wrapper).write_text(f"#usda 1.0\n# reference @{ '../../' + source }@\n", encoding="utf-8")
    report = inspect_nut_assets(tmp_path)
    assert not report["ready"]
    assert "missing external source: nut_m16.usd" in report["issues"]
    assert "missing external source: bolt_m16.usd" in report["issues"]
    with pytest.raises(FileNotFoundError, match="prepare_nut_assets.py"):
        validate_nut_assets(tmp_path)


def test_nut_preflight_rejects_replaced_geometry(tmp_path):
    (tmp_path / "nut_m16.usd").write_bytes(b"different geometry")
    report = inspect_nut_assets(tmp_path)
    assert "source SHA256 mismatch: nut_m16.usd" in report["issues"]


def test_nut_preflight_rejects_collision_override_and_external_reference(tmp_path):
    wrappers = tmp_path / "normalized/nut_thread"
    wrappers.mkdir(parents=True)
    (wrappers / "nut.usda").write_text(
        "#usda 1.0\n# @../../nut_m16.usd@ @../../../outside.usd@\n"
        'token physics:approximation = "convexDecomposition"\n', encoding="utf-8"
    )
    report = inspect_nut_assets(tmp_path)
    assert any("convex decomposition" in issue for issue in report["issues"])
    assert any("leaves asset directory" in issue for issue in report["issues"])


def test_nut_installer_rejects_bad_source_without_partial_destination(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "assets"
    source.mkdir()
    (source / "nut_m16.usd").write_bytes(b"wrong source release")
    with pytest.raises(ValueError, match="No exact nut_m16.usd"):
        _prepare_module().prepare(source, destination)
    assert not destination.exists()


@pytest.fixture
def audited_exports(monkeypatch):
    """Exercise real file I/O without committing external NVIDIA geometry."""
    payloads = {
        name: {"historical": f"historical {name}".encode(),
               "official_isaac_5_1": f"official raw {name}".encode()}
        for name in ("nut_m16.usd", "bolt_m16.usd")
    }
    exports = {
        name: tuple({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "export": label}
                    for label, data in variants.items())
        for name, variants in payloads.items()
    }
    monkeypatch.setattr(nut_assets, "SOURCE_ALLOWED_EXPORTS", exports)
    monkeypatch.setattr(nut_assets, "SOURCE_SHA256", {
        name: hashlib.sha256(variants["historical"]).hexdigest() for name, variants in payloads.items()
    })
    return payloads


@pytest.mark.parametrize("export", ["historical", "official_isaac_5_1"])
@pytest.mark.parametrize("upstream_names", [False, True])
def test_installer_and_runtime_accept_each_audited_export(tmp_path, audited_exports, export, upstream_names):
    source, destination = tmp_path / "source", tmp_path / "assets"
    source.mkdir()
    for name, variants in audited_exports.items():
        (source / (f"factory_{name}" if upstream_names else name)).write_bytes(variants[export])
    report = _prepare_module().prepare(source, destination)
    assert report["ready"]
    for name, variants in audited_exports.items():
        assert (destination / name).read_bytes() == variants[export]
        assert report["sources"][name]["export"] == export
        assert report["sources"][name]["expected_sha256"] == hashlib.sha256(variants["historical"]).hexdigest()
    validate_nut_assets(destination)


@pytest.mark.parametrize("existing_export,source_export", [
    ("historical", "official_isaac_5_1"), ("official_isaac_5_1", "historical"),
])
def test_existing_audited_destination_is_preserved_across_export_versions(
    tmp_path, audited_exports, existing_export, source_export,
):
    source, destination = tmp_path / "source", tmp_path / "assets"
    source.mkdir()
    destination.mkdir()
    for name, variants in audited_exports.items():
        (source / name).write_bytes(variants[source_export])
        (destination / name).write_bytes(variants[existing_export])
    report = _prepare_module().prepare(source, destination)
    assert report["ready"]
    for name, variants in audited_exports.items():
        assert (destination / name).read_bytes() == variants[existing_export]
        assert report["sources"][name]["export"] == existing_export


def test_same_size_unknown_official_source_blocks_all_copying(tmp_path, audited_exports):
    source, destination = tmp_path / "source", tmp_path / "assets"
    source.mkdir()
    for name, variants in audited_exports.items():
        (source / f"factory_{name}").write_bytes(variants["official_isaac_5_1"])
    (source / "factory_bolt_m16.usd").write_bytes(b"x" * len(audited_exports["bolt_m16.usd"]["official_isaac_5_1"]))
    with pytest.raises(ValueError, match="No exact bolt_m16.usd"):
        _prepare_module().prepare(source, destination)
    assert not destination.exists()


def test_unknown_existing_geometry_is_preserved_before_installing_wrappers(tmp_path, audited_exports):
    source, destination = tmp_path / "source", tmp_path / "assets"
    source.mkdir()
    destination.mkdir()
    for name, variants in audited_exports.items():
        (source / name).write_bytes(variants["official_isaac_5_1"])
    existing = destination / "nut_m16.usd"
    existing.write_bytes(b"x" * len(audited_exports["nut_m16.usd"]["official_isaac_5_1"]))
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        _prepare_module().prepare(source, destination)
    assert existing.read_bytes() == b"x" * len(audited_exports["nut_m16.usd"]["official_isaac_5_1"])
    assert sorted(path.name for path in destination.iterdir()) == ["nut_m16.usd"]


def test_official_exports_do_not_relax_wrapper_collision_checks(tmp_path, audited_exports):
    source, destination = tmp_path / "source", tmp_path / "assets"
    source.mkdir()
    for name, variants in audited_exports.items():
        (source / name).write_bytes(variants["official_isaac_5_1"])
    assert _prepare_module().prepare(source, destination)["ready"]
    wrapper = destination / "normalized/nut_thread/nut.usda"
    wrapper.write_text(wrapper.read_text() + '\ntoken physics:approximation = "convexDecomposition"\n')
    report = inspect_nut_assets(destination)
    assert not report["ready"]
    assert any("convex decomposition" in issue for issue in report["issues"])
    assert any("authored wrapper SHA256 mismatch" in issue for issue in report["issues"])


def _write_audited_sources(source, payloads, export="historical"):
    source.mkdir()
    for name, variants in payloads.items():
        (source / name).write_bytes(variants[export])


@pytest.mark.parametrize("concurrent_export", ["historical", "official_isaac_5_1"])
def test_concurrent_audited_geometry_is_preserved_with_actual_identity(
    tmp_path, audited_exports, monkeypatch, concurrent_export,
):
    source, destination = tmp_path / "source", tmp_path / "assets"
    _write_audited_sources(source, audited_exports)
    module = _prepare_module()
    original_link = module.os.link
    concurrent_bytes = audited_exports["nut_m16.usd"][concurrent_export]
    def competing_link(pending, target):
        if target.name == "nut_m16.usd":
            target.write_bytes(concurrent_bytes)
        original_link(pending, target)
    monkeypatch.setattr(module.os, "link", competing_link)
    report = module.prepare(source, destination)
    assert report["ready"]
    assert (destination / "nut_m16.usd").read_bytes() == concurrent_bytes
    identity = report["sources"]["nut_m16.usd"]
    assert identity["export"] == concurrent_export
    assert identity["actual_bytes"] == len(concurrent_bytes)
    assert identity["actual_sha256"] == hashlib.sha256(concurrent_bytes).hexdigest()
    assert not list(destination.rglob("*.part"))


@pytest.mark.parametrize("target_relative", ["nut_m16.usd", "normalized/nut_thread/nut.usda"])
def test_unknown_concurrent_geometry_or_wrapper_is_preserved(tmp_path, audited_exports, monkeypatch, target_relative):
    source, destination = tmp_path / "source", tmp_path / "assets"
    _write_audited_sources(source, audited_exports)
    module = _prepare_module()
    original_link = module.os.link
    concurrent = destination / target_relative
    def competing_link(pending, target):
        if target == concurrent:
            target.write_bytes(b"concurrent user content")
        original_link(pending, target)
    monkeypatch.setattr(module.os, "link", competing_link)
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        module.prepare(source, destination)
    assert concurrent.read_bytes() == b"concurrent user content"
    assert not list(destination.rglob("*.part"))
    if target_relative.endswith("nut.usda"):
        # Earlier successful publications remain valid after a later conflict.
        for name, variants in audited_exports.items():
            assert (destination / name).read_bytes() == variants["historical"]


def test_concurrent_pinned_wrapper_is_preserved_and_passes_final_validation(tmp_path, audited_exports, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "assets"
    _write_audited_sources(source, audited_exports)
    module = _prepare_module()
    original_link = module.os.link
    wrapper_bytes = (module.ROOT / "assets/normalized/nut_thread/nut.usda").read_bytes()
    def competing_link(pending, target):
        if target.name == "nut.usda":
            target.write_bytes(wrapper_bytes)
        original_link(pending, target)
    monkeypatch.setattr(module.os, "link", competing_link)
    report = module.prepare(source, destination)
    assert report["ready"]
    assert (destination / "normalized/nut_thread/nut.usda").read_bytes() == wrapper_bytes
    assert report["wrappers"]["normalized/nut_thread/nut.usda"]["sha256"] == hashlib.sha256(wrapper_bytes).hexdigest()
    assert not list(destination.rglob("*.part"))


def test_partial_copy_only_cleans_its_own_temporary_file(tmp_path, audited_exports, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "assets"
    _write_audited_sources(source, audited_exports)
    destination.mkdir()
    (destination / "user.txt").write_bytes(b"preserve this")
    module = _prepare_module()
    original_copy = module.shutil.copyfileobj
    calls = []
    def interrupted_copy(incoming, output, length):
        calls.append(incoming.name)
        if len(calls) == 2:
            output.write(b"partial")
            raise OSError("copy interrupted")
        original_copy(incoming, output, length)
    monkeypatch.setattr(module.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        module.prepare(source, destination)
    assert (destination / "nut_m16.usd").read_bytes() == audited_exports["nut_m16.usd"]["historical"]
    assert not (destination / "bolt_m16.usd").exists()
    assert (destination / "user.txt").read_bytes() == b"preserve this"
    assert not list(destination.rglob("*.part"))
    for name, variants in audited_exports.items():
        assert (source / name).read_bytes() == variants["historical"]


def test_copied_bytes_are_verified_before_publication(tmp_path, audited_exports, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "assets"
    _write_audited_sources(source, audited_exports)
    module = _prepare_module()
    def corrupt_copy(incoming, output, length):
        output.write(b"x" * len(incoming.read()))
    monkeypatch.setattr(module.shutil, "copyfileobj", corrupt_copy)
    with pytest.raises(OSError, match="differs from the selected source export"):
        module.prepare(source, destination)
    assert not (destination / "nut_m16.usd").exists()
    assert not list(destination.rglob("*.part"))
