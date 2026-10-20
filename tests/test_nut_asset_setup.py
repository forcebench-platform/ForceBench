"""Preflight must reject incomplete/mismatched external geometry before GPU startup."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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
