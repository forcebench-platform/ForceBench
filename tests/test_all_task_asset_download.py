"""Offline integration coverage for registry-driven asset acquisition."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import types
import urllib.error
import zipfile

import pytest


def _tool():
    spec = importlib.util.spec_from_file_location(
        "download_assets", Path(__file__).resolve().parents[1] / "tools/download_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pin(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _manifest(files, tasks=None):
    return {"files": [{"path": name, "present": True, "dependencies": [], **_pin(data)}
                       for name, data in files.items()],
            "tasks": tasks or {"example": {"files": list(files)}}}


def _registry(files=None, archives=None, builds=None):
    return {"schema_version": 1, "files": files or {}, "archives": archives or {}, "builds": builds or {}}


class _Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


def _zip(files, *, padding=False):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
        if padding:
            archive.writestr("not-selected.bin", bytes(range(256)) * 8192, compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


def _server(module, monkeypatch, urls, ranges=None, *, ignore_ranges=False):
    calls, responses = [], []
    ranges = ranges or {}
    def opener(request, *, timeout):
        url = request if isinstance(request, str) else request.full_url
        requested_range = None if isinstance(request, str) else request.get_header("Range")
        calls.append((url, requested_range))
        if url in ranges:
            assert requested_range, "A range-only archive must never be requested in full"
            data, etag = ranges[url]
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", requested_range).groups())
            assert end - start + 1 <= 8 * 1024 * 1024
            assert request.get_header("If-match") == f'"{etag}"'
            if ignore_ranges:
                response = _Response(data)
            else:
                payload = data[start:end + 1]
                response = _Response(payload, status=206,
                                     headers={"Content-Range": f"bytes {start}-{end}/{len(data)}",
                                              "Content-Length": str(len(payload)), "ETag": f'"{etag}"'})
        else:
            response = _Response(urls[url])
        responses.append(response)
        return response
    monkeypatch.setattr(module.urllib.request, "urlopen", opener)
    return calls, responses


def _build_recipe(module, *, inputs, fixed_output=False):
    builder = ("tools/asset_builders/build_single_stovetop.py" if fixed_output
               else "tools/asset_builders/build_conveyor_a04.py")
    return {"builder": builder, "builder_sha256": module._sha256(module.ROOT / builder), "inputs": inputs,
            "args": ["{input_dir}"] if fixed_output else ["--source", "{input}", "--output", "{output}", "--metadata", "{metadata}"],
            **({"output_name": "Stovetop012_single_center.usd"} if fixed_output else {})}


def test_mixed_url_range_cached_zip_and_build_are_verified_and_idempotent(tmp_path, monkeypatch):
    module = _tool()
    files = {"plain.usd": b"plain", "shipping/object.usd": b"range object",
             "kitchen/model.usd": b"cached object", "derived.usd": b"derived"}
    manifest = _manifest(files)
    manifest["files"].append({"path": "wrapper.usda", "present": True, "source_group": module.AUTHORED,
                              **_pin(b"wrapper")})
    manifest["files"].append({"path": "OmniPBR.mdl", "present": False})
    manifest["tasks"]["example"]["files"] += ["wrapper.usda", "OmniPBR.mdl"]
    ranged = _zip({"upstream/object.usd": files["shipping/object.usd"]}, padding=True)
    cached = _zip({"upstream/kitchen.usd": files["kitchen/model.usd"]})
    source_input = b"build input"
    build = _build_recipe(module, inputs=[{"name": "source.usd", "url": "https://example.test/input", **_pin(source_input)},
                                          {"name": "parts/plain.usd", "asset": "plain.usd"}])
    registry = _registry(
        files={"plain.usd": {"kind": "url", "url": "https://example.test/plain"},
               "shipping/object.usd": {"kind": "zip", "archive": "shipping", "member": "upstream/object.usd"},
               "kitchen/model.usd": {"kind": "zip", "archive": "kitchen", "member": "upstream/kitchen.usd"},
               "derived.usd": {"kind": "build", "recipe": "derived"}},
        archives={"shipping": {"url": "https://example.test/shipping.zip", "bytes": len(ranged), "etag": "range-etag", "access": "range"},
                  "kitchen": {"url": "https://example.test/kitchen.zip", "access": "cached", **_pin(cached)}},
        builds={"derived": build})
    calls, _ = _server(module, monkeypatch, {"https://example.test/plain": b"plain",
                                           "https://example.test/input": source_input,
                                           "https://example.test/kitchen.zip": cached},
                       {"https://example.test/shipping.zip": (ranged, "range-etag")})
    monkeypatch.setattr(module, "_require_usd_core", lambda: None)
    commands = []
    def build_process(command, *, cwd, shell, capture_output, text):
        assert not shell and capture_output and text
        assert command[:2] == [sys.executable, str(module.ROOT / build["builder"])]
        assert (Path(cwd) / "source.usd").read_bytes() == source_input
        assert (Path(cwd) / "parts/plain.usd").read_bytes() == b"plain"
        Path(command[command.index("--output") + 1]).write_bytes(b"derived")
        commands.append(command)
        return types.SimpleNamespace(returncode=0, stdout="built", stderr="")
    monkeypatch.setattr(module.subprocess, "run", build_process)
    destination, cache = tmp_path / "assets", tmp_path / "cache"
    report = module.download_assets(destination, cache_dir=cache, manifest=manifest, registry=registry)
    assert report["checks_passed"], report
    assert report["downloaded_files"] == 3 and report["built_files"] == 1
    assert len(commands) == 1
    for name, data in files.items():
        assert (destination / name).read_bytes() == data
    assert not (destination / "wrapper.usda").exists()
    assert sum(url == "https://example.test/kitchen.zip" for url, byte_range in calls) == 1
    requested = sum(int(re.fullmatch(r"bytes=(\d+)-(\d+)", value).group(2))
                    - int(re.fullmatch(r"bytes=(\d+)-(\d+)", value).group(1)) + 1
                    for url, value in calls if value)
    assert requested < len(ranged) // 10
    assert not list(cache.rglob("*.part")) and not list((cache / "builds").iterdir())
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network on valid resume"))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("builder on valid resume"))
    resumed = module.download_assets(destination, cache_dir=cache, manifest=manifest, registry=registry)
    assert resumed["checks_passed"] and resumed["already_present_files"] == 4


def test_missing_recipe_is_a_preflight_error_without_any_download(tmp_path, monkeypatch):
    module = _tool()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network before recipe closure"))
    report = module.download_assets(tmp_path / "assets", manifest=_manifest({"known.usd": b"a", "missing.usd": b"b"}),
                                    registry=_registry({"known.usd": {"kind": "url", "url": "https://example.test/a"}}))
    assert not report["checks_passed"]
    assert "Missing source recipe" in report["issues"][0]["detail"]
    assert not (tmp_path / "assets").exists()


def test_public_weight_sort_manifest_remains_a_42_payload_baseline(tmp_path):
    module = _tool()
    manifest = json.loads((module.ROOT / "assets/asset_manifest.json").read_text(encoding="utf-8"))
    selected = set(manifest["tasks"]["weight_sort"]["files"])
    registry = _registry({row["path"]: {"kind": "url", "url": "https://example.test/" + row["path"]}
                          for row in manifest["files"] if row["path"] in selected and row.get("present")
                          and row.get("source_group") != module.AUTHORED})
    nodes, builds, inputs = module._plan(manifest, registry, ["weight_sort"], tmp_path / "assets", tmp_path / "cache")
    assert len(nodes) == 42 and "franka_mimic.usd" in nodes
    assert not builds and not inputs


def test_real_registry_covers_all_eight_tasks_and_orders_all_five_builds(tmp_path, monkeypatch):
    module = _tool()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network during registry audit"))
    manifest = json.loads((module.ROOT / "assets/asset_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((module.ROOT / "assets/asset_sources.json").read_text(encoding="utf-8"))
    tasks = sorted(manifest["tasks"])
    assert len(tasks) == 8
    selected = {name for task in tasks for name in manifest["tasks"][task]["files"]}
    required = {row["path"] for row in manifest["files"] if row["path"] in selected and row.get("present")
                and row.get("source_group") != module.AUTHORED}
    assert len(required) == 283 and set(registry["files"]) == required
    nodes, build_order, inputs = module._plan(manifest, registry, tasks, tmp_path / "assets", tmp_path / "cache")
    assert set(nodes) == required
    assert len(build_order) == 5
    assert {nodes[name]["recipe"]["recipe"] for name in build_order} == {
        "conveyor_a04", "warehouse_rack", "restore_kitchen", "restore_stove", "single_stovetop"}
    positions = {name: index for index, name in enumerate(build_order)}
    for name in build_order:
        for source in nodes[name]["build"]["inputs"]:
            if source.get("asset") in positions:
                assert positions[source["asset"]] < positions[name]
    restored_stove = "lightwheel_kitchen/Stovetop012/Stovetop012.usd"
    single_stove = "lightwheel_kitchen/Stovetop012/Stovetop012_single_center.usd"
    assert positions[restored_stove] < positions[single_stove]
    assert any(item["kind"] == "zip" and item["archive"] == "lightwheel_v1" for item in inputs.values())
    assert not (tmp_path / "assets").exists() and not (tmp_path / "cache").exists()


def test_real_registry_pins_archives_builders_http_sources_and_nut_contract():
    module = _tool()
    manifest = json.loads((module.ROOT / "assets/asset_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((module.ROOT / "assets/asset_sources.json").read_text(encoding="utf-8"))
    expected_archives = {
        "shipping01": (21907097879, "range", "etag", "822c58760f11488f10f4c2f0c2bb6bce-2612"),
        "lightwheel_v1": (699682909, "cached", "sha256", "721fb9e47d09b2c43a7db396c1d63c659e987d5925db0a5e78cf4fd40cd0b0b0"),
        "dom_objects_mirror": (192600903, "cached", "sha256", "02aaa3f87d7efd2e06e16d767cdea81e53299d2531505bea22db861c82c9cd63"),
    }
    assert set(registry["archives"]) == set(expected_archives)
    for identifier, (size, access, identity_key, identity_value) in expected_archives.items():
        archive = module._archive_recipe(identifier, registry)
        assert (archive["bytes"], archive["access"], archive[identity_key]) == (size, access, identity_value)
        assert archive["url"].startswith("https://")
    assert "/10b0f7a86135a27e8c2fba9c690f6bf5c4f06bb1/" in registry["archives"]["lightwheel_v1"]["url"]
    assert "/77dda64f9942cb81d97ed00d4d61b8721d749210/" in registry["archives"]["dom_objects_mirror"]["url"]
    rows = {row["path"]: row for row in manifest["files"]}
    for name, recipe in registry["files"].items():
        if recipe["kind"] == "url":
            assert recipe["url"].startswith("https://")
            assert module._download_pin(rows[name], recipe) in module._approved(rows[name])
    for recipe in registry["builds"].values():
        builder = module._builder(recipe)
        assert module._sha256(builder) == recipe["builder_sha256"]
        for source in recipe["inputs"]:
            if "url" in source:
                assert source["url"].startswith("https://")
                module._pin(source, source["name"])
    spec = importlib.util.spec_from_file_location("registry_nut_contract", module.ROOT / "forcebench/nut_assets.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    for name, exports in contract.SOURCE_ALLOWED_EXPORTS.items():
        assert rows[name]["sha256"] == contract.SOURCE_SHA256[name]
        assert {(item["bytes"], item["sha256"]) for item in module._approved(rows[name])} == {
            (item["bytes"], item["sha256"]) for item in exports}
        selected_pin = module._download_pin(rows[name], registry["files"][name])
        assert contract.matching_source_export(name, selected_pin["bytes"], selected_pin["sha256"]) is not None


def test_unknown_existing_payload_is_preserved_before_network(tmp_path, monkeypatch):
    module = _tool()
    (tmp_path / "part.usd").write_bytes(b"changed")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network after unknown destination"))
    report = module.download_assets(tmp_path, manifest=_manifest({"part.usd": b"pinned!"}),
                                    registry=_registry({"part.usd": {"kind": "url", "url": "https://example.test/a"}}))
    assert not report["checks_passed"] and report["files"][0]["status"] == "failed"
    assert (tmp_path / "part.usd").read_bytes() == b"changed"


@pytest.mark.parametrize("name", ["franka_mimic.usd", "nut_m16.usd", "bolt_m16.usd"])
def test_recipe_can_select_only_a_manifest_approved_variant(tmp_path, monkeypatch, name):
    module = _tool()
    manifest = _manifest({name: b"historical"})
    variant = {**_pin(b"official"), "verification_report": "verified.json"}
    manifest["files"][0]["verified_equivalent_exports"] = [variant]
    registry = _registry({name: {"kind": "url", "url": "https://example.test/official", **_pin(b"official")}})
    _server(module, monkeypatch, {"https://example.test/official": b"official"})
    report = module.download_assets(tmp_path, manifest=manifest, registry=registry)
    assert report["checks_passed"] and report["files"][0]["sha256"] == variant["sha256"]
    assert (tmp_path / name).read_bytes() == b"official"


def test_unknown_variant_cannot_be_registered_by_the_source_recipe(tmp_path, monkeypatch):
    module = _tool()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("unapproved export download"))
    report = module.download_assets(tmp_path, manifest=_manifest({"nut_m16.usd": b"historical"}),
                                    registry=_registry({"nut_m16.usd": {"kind": "url", "url": "https://example.test/other",
                                                                       **_pin(b"other export")}}))
    assert not report["checks_passed"] and "unapproved export" in report["issues"][0]["detail"]


def test_existing_historical_export_is_kept_when_recipe_selects_official_variant(tmp_path, monkeypatch):
    module = _tool()
    name = "nut_m16.usd"
    manifest = _manifest({name: b"historical"})
    manifest["files"][0]["verified_equivalent_exports"] = [{**_pin(b"official"), "verification_report": "verified.json"}]
    registry = _registry({name: {"kind": "url", "url": "https://example.test/official", **_pin(b"official")}})
    (tmp_path / name).write_bytes(b"historical")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network for approved existing variant"))
    report = module.download_assets(tmp_path, manifest=manifest, registry=registry)
    assert report["checks_passed"] and report["already_present_files"] == 1
    identity = report["files"][0]
    assert identity["expected_sha256"] == _pin(b"official")["sha256"]
    assert identity["sha256"] == _pin(b"historical")["sha256"]
    assert (tmp_path / name).read_bytes() == b"historical"


@pytest.mark.parametrize("payload", [b"short", b"x" * len(b"expected bytes")])
def test_wrong_url_length_or_hash_leaves_no_payload_or_part(tmp_path, monkeypatch, payload):
    module = _tool()
    _server(module, monkeypatch, {"https://example.test/model": payload})
    report = module.download_assets(tmp_path, manifest=_manifest({"part.usd": b"expected bytes"}),
                                    registry=_registry({"part.usd": {"kind": "url", "url": "https://example.test/model"}}))
    assert not report["checks_passed"] and report["files"][0]["status"] == "failed"
    assert not (tmp_path / "part.usd").exists() and not list(tmp_path.rglob("*.part"))


def test_range_ignored_response_is_rejected_before_reading_body(tmp_path, monkeypatch):
    module = _tool()
    data = _zip({"source.usd": b"model"})
    _, responses = _server(module, monkeypatch, {}, {"https://example.test/a.zip": (data, "etag")}, ignore_ranges=True)
    registry = _registry({"model.usd": {"kind": "zip", "archive": "a", "member": "source.usd"}},
                         {"a": {"url": "https://example.test/a.zip", "bytes": len(data), "etag": "etag", "access": "range"}})
    report = module.download_assets(tmp_path, manifest=_manifest({"model.usd": b"model"}), registry=registry)
    assert not report["checks_passed"] and all(response.read_calls == 0 for response in responses)
    assert not (tmp_path / "model.usd").exists()


def test_shipping_sized_archive_cannot_use_full_cache_mode(tmp_path, monkeypatch):
    module = _tool()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("21 GB full download"))
    registry = _registry({"model.usd": {"kind": "zip", "archive": "shipping", "member": "source.usd"}},
                         {"shipping": {"url": "https://example.test/a.zip", "bytes": 21907097879,
                                       "sha256": "0" * 64, "access": "cached"}})
    report = module.download_assets(tmp_path, manifest=_manifest({"model.usd": b"model"}), registry=registry)
    assert not report["checks_passed"] and "require range access" in report["issues"][0]["detail"]


def test_two_range_archives_do_not_share_seek_or_cached_directory_bytes(tmp_path, monkeypatch):
    module = _tool()
    first, second = _zip({"same.usd": b"first"}), _zip({"same.usd": b"other"})
    registry = _registry(
        {f"{name}.usd": {"kind": "zip", "archive": name, "member": "same.usd"} for name in ("a", "b")},
        {name: {"url": f"https://example.test/{name}.zip", "bytes": len(data), "etag": name, "access": "range"}
         for name, data in (("a", first), ("b", second))})
    _server(module, monkeypatch, {}, {"https://example.test/a.zip": (first, "a"),
                                     "https://example.test/b.zip": (second, "b")})
    report = module.download_assets(tmp_path, manifest=_manifest({"a.usd": b"first", "b.usd": b"other"}), registry=registry)
    assert report["checks_passed"], report
    assert (tmp_path / "a.usd").read_bytes() == b"first" and (tmp_path / "b.usd").read_bytes() == b"other"


def test_fixed_output_builder_uses_isolated_input_dir(tmp_path, monkeypatch):
    module = _tool()
    build = _build_recipe(module, inputs=[{"name": "Stovetop012.usd", "url": "https://example.test/source", **_pin(b"source")}],
                          fixed_output=True)
    registry = _registry({"derived.usd": {"kind": "build", "recipe": "stove"}}, builds={"stove": build})
    _server(module, monkeypatch, {"https://example.test/source": b"source"})
    monkeypatch.setattr(module, "_require_usd_core", lambda: None)
    def run(command, *, cwd, **kwargs):
        assert command[-1] == cwd
        work = Path(cwd)
        assert (work / "Stovetop012.usd").read_bytes() == b"source"
        assert work.is_relative_to(tmp_path / "cache/builds")
        (work / build["output_name"]).write_bytes(b"derived")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    report = module.download_assets(tmp_path / "assets", cache_dir=tmp_path / "cache", manifest=_manifest({"derived.usd": b"derived"}),
                                    registry=registry)
    assert report["checks_passed"] and report["built_files"] == 1
    assert not list((tmp_path / "cache/builds").iterdir())


@pytest.mark.parametrize("basename,asset", [("KitchenRoom.usd", "kitchen"), ("Stovetop012.usd", "stove")])
def test_restore_builder_with_same_input_output_basename_keeps_input_bytes(tmp_path, monkeypatch, basename, asset):
    module = _tool()
    raw, restored = b"public original layer", b"historical restored layer"
    builder = "tools/asset_builders/restore_lightwheel_layers.py"
    recipe = {"builder": builder, "builder_sha256": module._sha256(module.ROOT / builder),
              "inputs": [{"name": basename, "url": "https://example.test/original", **_pin(raw)}],
              "args": ["--asset", asset, "--source", "{input}", "--output", "{output}"]}
    target = "layers/" + basename
    registry = _registry({target: {"kind": "build", "recipe": "restore"}}, builds={"restore": recipe})
    _server(module, monkeypatch, {"https://example.test/original": raw})
    monkeypatch.setattr(module, "_require_usd_core", lambda: None)
    def run(command, *, cwd, **kwargs):
        source = Path(command[command.index("--source") + 1])
        output = Path(command[command.index("--output") + 1])
        assert source.name == output.name == basename
        assert source != output and source.parent == Path(cwd) and output.parent != Path(cwd)
        assert output.parent.is_relative_to(Path(cwd)) and output.parent.is_dir()
        assert source.read_bytes() == raw
        with output.open("xb") as stream:
            stream.write(restored)
        assert source.read_bytes() == raw
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    report = module.download_assets(tmp_path / "assets", manifest=_manifest({target: restored}), registry=registry)
    assert report["checks_passed"], report
    assert (tmp_path / "assets" / target).read_bytes() == restored


@pytest.mark.parametrize("access", ["cached", "range"])
def test_archive_build_input_reuses_selected_archive_and_is_staged_in_isolation(tmp_path, monkeypatch, access):
    module = _tool()
    archive = _zip({"Collected_KitchenRoom/KitchenRoom.usd": b"raw kitchen layer", "other.usd": b"other model"})
    input_spec = {"name": "KitchenRoom.usd", "archive": "lightwheel_v1",
                  "member": "Collected_KitchenRoom/KitchenRoom.usd", **_pin(b"raw kitchen layer")}
    build = _build_recipe(module, inputs=[input_spec])
    archive_recipe = {"url": "https://example.test/lightwheel.zip", "access": access,
                      **_pin(archive), **({"etag": "media-etag"} if access == "range" else {})}
    registry = _registry(
        {"other.usd": {"kind": "zip", "archive": "lightwheel_v1", "member": "other.usd"},
         "restored.usd": {"kind": "build", "recipe": "restore"}},
        {"lightwheel_v1": archive_recipe}, {"restore": build})
    calls, _ = _server(module, monkeypatch, {"https://example.test/lightwheel.zip": archive},
                       {"https://example.test/lightwheel.zip": (archive, "media-etag")} if access == "range" else {})
    monkeypatch.setattr(module, "_require_usd_core", lambda: None)
    def run(command, *, cwd, **kwargs):
        assert Path(cwd).is_relative_to(tmp_path / "cache/builds")
        assert (Path(cwd) / "KitchenRoom.usd").read_bytes() == b"raw kitchen layer"
        Path(command[command.index("--output") + 1]).write_bytes(b"restored layer")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    report = module.download_assets(tmp_path / "assets", cache_dir=tmp_path / "cache", registry=registry,
                                    manifest=_manifest({"other.usd": b"other model", "restored.usd": b"restored layer"}))
    assert report["checks_passed"], report
    assert report["build_inputs"][0]["source"]["archive"] == "lightwheel_v1"
    assert report["build_inputs"][0]["sha256"] == _pin(b"raw kitchen layer")["sha256"]
    assert (tmp_path / "assets/restored.usd").read_bytes() == b"restored layer"
    if access == "cached":
        assert calls == [("https://example.test/lightwheel.zip", None)]
    assert not list((tmp_path / "cache/builds").iterdir())


def test_archive_build_input_with_wrong_pin_cannot_run_builder(tmp_path, monkeypatch):
    module = _tool()
    archive = _zip({"KitchenRoom.usd": b"wrong raw source"})
    build = _build_recipe(module, inputs=[{"name": "KitchenRoom.usd", "archive": "media", "member": "KitchenRoom.usd",
                                          **_pin(b"pinned raw input")}])
    registry = _registry({"restored.usd": {"kind": "build", "recipe": "restore"}},
                         {"media": {"url": "https://example.test/media.zip", "access": "cached", **_pin(archive)}},
                         {"restore": build})
    _server(module, monkeypatch, {"https://example.test/media.zip": archive})
    monkeypatch.setattr(module, "_require_usd_core", lambda: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("build with unverified input"))
    report = module.download_assets(tmp_path / "assets", manifest=_manifest({"restored.usd": b"restored"}), registry=registry)
    assert not report["checks_passed"] and report["build_inputs"][0]["status"] == "failed"
    assert not (tmp_path / "assets/restored.usd").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_local_archive_import_supplies_cached_payload_and_build_input_without_network(tmp_path, monkeypatch):
    module = _tool()
    archive = _zip({"source.usd": b"raw input", "other.usd": b"payload"})
    private_source = tmp_path / "private input=archive.zip"
    private_source.write_bytes(archive)
    recipe = {"url": "https://example.test/media.zip", "access": "cached", **_pin(archive)}
    build = _build_recipe(module, inputs=[{"name": "source.usd", "archive": "media", "member": "source.usd", **_pin(b"raw input")}])
    registry = _registry({"other.usd": {"kind": "zip", "archive": "media", "member": "other.usd"},
                          "derived.usd": {"kind": "build", "recipe": "derived"}},
                         {"media": recipe}, {"derived": build})
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network despite supplied archive"))
    monkeypatch.setattr(module, "_require_usd_core", lambda: None)
    def run(command, *, cwd, **kwargs):
        assert (Path(cwd) / "source.usd").read_bytes() == b"raw input"
        Path(command[command.index("--output") + 1]).write_bytes(b"derived")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    archive_files = module._parse_archive_files([f"media={private_source}"])
    assert archive_files == {"media": str(private_source)}
    destination, cache = tmp_path / "assets", tmp_path / "cache"
    manifest = _manifest({"other.usd": b"payload", "derived.usd": b"derived"})
    report = module.download_assets(destination, cache_dir=cache, manifest=manifest, registry=registry, archive_files=archive_files)
    assert report["checks_passed"], report
    assert module._cached_archive_path(cache, "media", recipe).read_bytes() == archive
    assert report["manual_archives"] == [{"archive_id": "media", "source_url": recipe["url"],
                                            "status": "imported", **_pin(archive)}]
    assert str(private_source) not in json.dumps(report)
    assert private_source.read_bytes() == archive
    resumed = module.download_assets(destination, cache_dir=cache, manifest=manifest, registry=registry, archive_files=archive_files)
    assert resumed["checks_passed"] and resumed["manual_archives"][0]["status"] == "already_present"
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("failure", ["wrong_archive", "unknown_cache"])
def test_local_archive_preflight_rejects_wrong_bytes_or_unknown_cache_without_writing(tmp_path, monkeypatch, failure):
    module = _tool()
    archive = _zip({"source.usd": b"pinned payload"})
    private_source = tmp_path / "private-local.zip"
    private_source.write_bytes(b"x" * len(archive) if failure == "wrong_archive" else archive)
    recipe = {"url": "https://example.test/media.zip", "access": "cached", **_pin(archive)}
    registry = _registry({"model.usd": {"kind": "zip", "archive": "media", "member": "source.usd"}}, {"media": recipe})
    cache = tmp_path / "cache"
    cached_file = module._cached_archive_path(cache, "media", recipe)
    if failure == "unknown_cache":
        cached_file.parent.mkdir(parents=True)
        cached_file.write_bytes(b"existing unknown cache")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network after invalid manual import"))
    destination = tmp_path / "assets"
    report = module.download_assets(destination, cache_dir=cache, manifest=_manifest({"model.usd": b"pinned payload"}),
                                    registry=registry, archive_files={"media": private_source})
    assert not report["checks_passed"] and report["issues"][0]["kind"] == "manual_archive_preflight_failed"
    assert not destination.exists()
    if failure == "unknown_cache":
        assert cached_file.read_bytes() == b"existing unknown cache"
    else:
        assert not cached_file.exists()
    assert str(private_source) not in json.dumps(report)
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("unrelated", [False, True])
def test_local_archive_import_refuses_range_or_unrelated_archive_before_reading_file(tmp_path, monkeypatch, unrelated):
    module = _tool()
    selected = {"url": "https://example.test/selected.zip", "access": "cached", **_pin(b"zip")}
    if unrelated:
        imported = {"url": "https://example.test/unrelated.zip", "access": "cached", **_pin(b"other")}
    else:
        imported = {"url": "https://example.test/shipping.zip", "access": "range", "bytes": 21907097879, "etag": "pinned"}
    registry = _registry({"model.usd": {"kind": "zip", "archive": "selected", "member": "source.usd"}},
                         {"selected": selected, "imported": imported})
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network for refused import"))
    report = module.download_assets(tmp_path / "assets", manifest=_manifest({"model.usd": b"model"}), registry=registry,
                                    archive_files={"imported": tmp_path / "never-read.zip"})
    assert not report["checks_passed"]
    expected = "unrelated" if unrelated else "range archive refused"
    assert expected in report["issues"][0]["detail"]
    assert not (tmp_path / "assets").exists()


@pytest.mark.parametrize("corrupt_output", [False, True])
def test_build_failure_or_wrong_output_hash_never_publishes(tmp_path, monkeypatch, corrupt_output):
    module = _tool()
    build = _build_recipe(module, inputs=[{"name": "input.usd", "url": "https://example.test/input", **_pin(b"input")}])
    registry = _registry({"derived.usd": {"kind": "build", "recipe": "derived"}}, builds={"derived": build})
    _server(module, monkeypatch, {"https://example.test/input": b"input"})
    monkeypatch.setattr(module, "_require_usd_core", lambda: None)
    def run(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_bytes(b"wrong!!")
        return types.SimpleNamespace(returncode=0 if corrupt_output else 1, stdout="", stderr="builder failed")
    monkeypatch.setattr(module.subprocess, "run", run)
    report = module.download_assets(tmp_path / "assets", manifest=_manifest({"derived.usd": b"derived"}), registry=registry)
    assert not report["checks_passed"] and report["files"][0]["status"] == "failed"
    assert not (tmp_path / "assets/derived.usd").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_modified_builder_and_missing_usd_are_clear_preflight_errors(tmp_path, monkeypatch):
    module = _tool()
    build = _build_recipe(module, inputs=[{"name": "input.usd", "url": "https://example.test/input", **_pin(b"input")}])
    registry = _registry({"derived.usd": {"kind": "build", "recipe": "derived"}}, builds={"derived": build})
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network before build preflight"))
    valid_hash = build["builder_sha256"]
    build["builder_sha256"] = "0" * 64
    report = module.download_assets(tmp_path, manifest=_manifest({"derived.usd": b"derived"}), registry=registry)
    assert not report["checks_passed"] and "builder SHA256" in report["issues"][0]["detail"]
    build["builder_sha256"] = valid_hash
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "24.11")
    report = module.download_assets(tmp_path, manifest=_manifest({"derived.usd": b"derived"}), registry=registry)
    assert not report["checks_passed"] and "usd-core==25.11" in report["issues"][0]["detail"]


def test_task_selection_does_not_require_other_task_recipes(tmp_path, monkeypatch):
    module = _tool()
    manifest = _manifest({"selected.usd": b"selected", "unselected.usd": b"not selected"},
                         {"first": {"files": ["selected.usd"]}, "second": {"files": ["unselected.usd"]}})
    registry = _registry({"selected.usd": {"kind": "url", "url": "https://example.test/selected"}})
    _server(module, monkeypatch, {"https://example.test/selected": b"selected"})
    report = module.download_assets(tmp_path, tasks=["first"], manifest=manifest, registry=registry)
    assert report["checks_passed"] and report["required_external_files"] == 1
    assert not (tmp_path / "unselected.usd").exists()


def test_interrupted_url_download_keeps_other_valid_file_and_cleans_partial(tmp_path, monkeypatch):
    module = _tool()
    class Interrupted(_Response):
        def read(self, size=-1):
            if self.tell():
                raise urllib.error.URLError("disconnected")
            return super().read(2)
    def opener(request, **kwargs):
        return _Response(b"good") if request.full_url.endswith("/good") else Interrupted(b"broken")
    monkeypatch.setattr(module.urllib.request, "urlopen", opener)
    registry = _registry({name: {"kind": "url", "url": f"https://example.test/{name.removesuffix('.usd')}"}
                          for name in ("good.usd", "bad.usd")})
    report = module.download_assets(tmp_path, manifest=_manifest({"good.usd": b"good", "bad.usd": b"broken"}), registry=registry)
    assert not report["checks_passed"] and report["downloaded_files"] == 1
    assert (tmp_path / "good.usd").read_bytes() == b"good" and not (tmp_path / "bad.usd").exists()
    assert not list(tmp_path.rglob("*.part"))
