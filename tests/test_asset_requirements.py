"""Doctor's direct-file catalog must exclude source metadata and keep local assets."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import importlib
from importlib.resources import files
import json
from pathlib import PurePosixPath

from forcebench.cli import inspect_assets
from forcebench.registry import task_manifests
from forcebench.scene_profiles import get_scene_profile
from forcebench.tasks.rattle_sort.spec import SPEC as RATTLE_SPEC


def _catalog():
    return json.loads(files("forcebench").joinpath("asset_requirements.json").read_text(encoding="utf-8"))["tasks"]


def test_all_direct_requirements_are_portable_local_paths_without_source_urls():
    catalog = _catalog()
    manifests = task_manifests()
    assert set(catalog) == set(manifests) and len(catalog) == 8
    for task, paths in catalog.items():
        assert paths and len(paths) == len(set(paths))
        for name in paths:
            relative = PurePosixPath(name)
            assert name and not relative.is_absolute() and ".." not in relative.parts
            assert ":" not in name and "\\" not in name and str(relative) == name
        specification = importlib.import_module(f"forcebench.tasks.{task}.spec")
        spec = getattr(specification, "SPEC", None)
        if is_dataclass(spec):
            for field in fields(spec):
                if field.name.endswith("_source_url"):
                    assert getattr(spec, field.name) not in paths
    assert RATTLE_SPEC.conveyor_source_url not in catalog["rattle_sort"]
    assert RATTLE_SPEC.conveyor_asset in catalog["rattle_sort"]


def test_scene_asset_uris_keep_their_local_catalog_mapping():
    catalog = _catalog()
    for task, manifest in task_manifests().items():
        profile = get_scene_profile(manifest.scene_profile)
        for uri in (profile.room_usd_path, profile.table_visual_usd_path):
            if uri and uri.startswith(("forcebench://assets/", "coribench://assets/")):
                assert uri.split("://assets/", 1)[1] in catalog[task]


def test_rattle_doctor_checks_local_derivative_and_reports_a_real_missing_file(tmp_path):
    paths = _catalog()["rattle_sort"]
    for name in paths:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"direct-existence fixture")
    report = inspect_assets(tmp_path, ["rattle_sort"])
    assert report["direct_files_present"]
    assert report["tasks"]["rattle_sort"] == {"required_direct_files": len(paths), "missing": []}
    (tmp_path / RATTLE_SPEC.conveyor_asset).unlink()
    missing = inspect_assets(tmp_path, ["rattle_sort"])
    assert not missing["direct_files_present"]
    assert missing["tasks"]["rattle_sort"]["missing"] == [RATTLE_SPEC.conveyor_asset]
