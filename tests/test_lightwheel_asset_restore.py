"""Offline checks for exact Lightwheel restoration and safe output publication."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def restoration_tool():
    spec = importlib.util.spec_from_file_location(
        "restore_lightwheel_layers", Path(__file__).resolve().parents[1]
        / "tools/asset_builders/restore_lightwheel_layers.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["stove", "kitchen"])
def fixture_export(tmp_path, monkeypatch, restoration_tool, request):
    asset = request.param
    source_data, restored_data = b"public layer bytes", b"exact historical layer bytes"
    pin = {"source_bytes": len(source_data), "source_sha256": hashlib.sha256(source_data).hexdigest(),
           "output_bytes": len(restored_data), "output_sha256": hashlib.sha256(restored_data).hexdigest()}
    monkeypatch.setitem(restoration_tool.ASSET_PINS, asset, pin)
    source, output = tmp_path / "public.usd", tmp_path / "restored.usd"
    source.write_bytes(source_data)
    return restoration_tool, asset, source, output, source_data, restored_data


def test_public_bytes_are_checked_before_patching(fixture_export, monkeypatch):
    module, asset, source, output, source_data, _ = fixture_export
    source.write_bytes(b"x" * len(source_data))
    monkeypatch.setattr(module, "_patch_layer", lambda *args: pytest.fail("patching unknown input"))
    with pytest.raises(ValueError, match="pinned ZIP member"):
        module.restore(source, output, asset=asset)
    assert not output.exists()
    assert sorted(path.name for path in source.parent.iterdir()) == ["public.usd"]


def test_exact_restore_keeps_the_original_and_can_resume(fixture_export, monkeypatch):
    module, asset, source, output, source_data, restored_data = fixture_export
    calls = []
    def patch(pending, selected):
        assert pending.read_bytes() == source_data
        calls.append(selected)
        pending.write_bytes(restored_data)
        return {"fixture_patch": True}
    monkeypatch.setattr(module, "_patch_layer", patch)
    report = module.restore(source, output, asset=asset)
    assert report["status"] == "restored" and calls == [asset]
    assert source.read_bytes() == source_data and output.read_bytes() == restored_data
    assert not list(source.parent.glob(".lightwheel-*"))
    monkeypatch.setattr(module, "_patch_layer", lambda *args: pytest.fail("patch on valid resume"))
    assert module.restore(source, output, asset=asset)["status"] == "already_present"


@pytest.mark.parametrize("fails_during_patch", [False, True])
def test_failed_patch_never_publishes_or_leaves_temporary_bytes(fixture_export, monkeypatch, fails_during_patch):
    module, asset, source, output, source_data, _ = fixture_export
    def patch(pending, selected):
        pending.write_bytes(b"partial or incorrect rebuilt bytes")
        if fails_during_patch:
            raise ValueError("interrupted patch")
        return {}
    monkeypatch.setattr(module, "_patch_layer", patch)
    with pytest.raises(ValueError):
        module.restore(source, output, asset=asset)
    assert not output.exists() and not list(source.parent.glob(".lightwheel-*"))
    assert source.read_bytes() == source_data


def test_unknown_existing_output_is_preserved(fixture_export, monkeypatch):
    module, asset, source, output, _, restored_data = fixture_export
    existing = b"x" * len(restored_data)
    output.write_bytes(existing)
    monkeypatch.setattr(module, "_patch_layer", lambda *args: pytest.fail("patch over user output"))
    with pytest.raises(FileExistsError, match="preserved"):
        module.restore(source, output, asset=asset)
    assert output.read_bytes() == existing


def test_concurrent_user_output_is_not_replaced(fixture_export, monkeypatch):
    module, asset, source, output, _, restored_data = fixture_export
    def patch(pending, selected):
        pending.write_bytes(restored_data)
        return {}
    original_link = module.os.link
    def competing_link(pending, destination):
        destination.write_bytes(b"concurrent user content")
        original_link(pending, destination)
    monkeypatch.setattr(module, "_patch_layer", patch)
    monkeypatch.setattr(module.os, "link", competing_link)
    with pytest.raises(FileExistsError, match="preserved"):
        module.restore(source, output, asset=asset)
    assert output.read_bytes() == b"concurrent user content"
    assert not list(source.parent.glob(".lightwheel-*"))
