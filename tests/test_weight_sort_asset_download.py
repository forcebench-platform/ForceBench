"""Offline coverage for bounded historical-ZIP extraction and exact asset pins."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import types
import urllib.error
import zipfile

import pytest


def _tool():
    spec = importlib.util.spec_from_file_location(
        "download_weight_sort_assets", Path(__file__).resolve().parents[1] / "tools/download_weight_sort_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(files):
    return {
        "files": [{"path": name, "present": True, "bytes": len(data),
                   "sha256": hashlib.sha256(data).hexdigest(), "dependencies": []}
                  for name, data in files.items()],
        "tasks": {"weight_sort": {"files": list(files)}},
    }


class _Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


def _archive(module, files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in files.items():
            member = module.ARCHIVE_MEMBER_PREFIX + name.removeprefix("simready/")
            archive.writestr(member, data, compress_type=zipfile.ZIP_DEFLATED)
        # An unselected large entry makes accidental full-archive reads visible.
        archive.writestr("unselected/large.bin", bytes(range(256)) * 8192, compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


def _range_server(module, archive, monkeypatch, *, mode=None, table_files=None):
    monkeypatch.setattr(module, "ARCHIVE_BYTES", len(archive))
    monkeypatch.setattr(module, "ARCHIVE_ETAG", "test-etag")
    calls, responses = [], []
    table_files = table_files or {}

    def open_response(request, timeout):
        if isinstance(request, str):
            assert request.startswith(module.TABLE_URL_PREFIX)
            data = table_files[request.removeprefix(module.TABLE_URL_PREFIX)]
            return _Response(data, headers={"Content-Length": str(len(data))})
        assert request.full_url == module.ARCHIVE_URL
        assert request.get_header("If-match") == '"test-etag"'
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        assert match
        start, end = map(int, match.groups())
        assert 0 <= start <= end < len(archive)
        assert end - start + 1 <= module.MAX_RANGE_BYTES
        calls.append((start, end))
        data = archive[start:end + 1]
        headers = {"Content-Range": f"bytes {start}-{end}/{len(archive)}",
                   "Content-Length": str(len(data)), "ETag": '"test-etag"'}
        if mode == "ignored":
            response = _Response(archive, status=200, headers={"Content-Length": str(len(archive))})
        else:
            if mode == "bad_range":
                headers["Content-Range"] = f"bytes {start}-{end}/{len(archive) + 1}"
            elif mode == "bad_etag":
                headers["ETag"] = '"different"'
            elif mode == "truncated":
                data = data[:-1]
            response = _Response(data, status=206, headers=headers)
        responses.append(response)
        return response
    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    return calls, responses


def test_public_manifest_selects_only_34_shipping_and_7_table_files_plus_robot():
    module = _tool()
    manifest = json.loads((module.ROOT / "assets/asset_manifest.json").read_text(encoding="utf-8"))
    selected = module._selected_rows(manifest)
    assert len(selected) == 42
    assert sum(kind == "zip" for row, kind, source in selected) == 34
    assert sum(kind == "table" for row, kind, source in selected) == 7
    assert sum(kind == "robot" for row, kind, source in selected) == 1
    assert module.ARCHIVE_BYTES == 21907097879
    assert module.ARCHIVE_ETAG == "822c58760f11488f10f4c2f0c2bb6bce-2612"
    assert module.ARCHIVE_URL.endswith("SimReady_Containers_Shipping_01_NVD%4010010.zip")
    for row, kind, source in selected:
        if kind == "zip":
            assert source == "Assets/simready_content/common_assets/props/" + row["path"].removeprefix("simready/")


def test_selected_zip_and_table_payloads_are_verified_without_downloading_archive(tmp_path, monkeypatch):
    module = _tool()
    zip_files = {"simready/flatbox_a02/flatbox_a02.usd": b"exact box",
                 "simready/container_f27/materials/Plastic_B.mdl": b"exact material"}
    table = {"residential/east_rural/EastRural_Table.usd": b"exact table"}
    files = {**zip_files, **table}
    archive = _archive(module, zip_files)
    calls, _ = _range_server(module, archive, monkeypatch, table_files={"EastRural_Table.usd": b"exact table"})
    report = module.download_weight_sort_assets(tmp_path, manifest=_manifest(files), workers=4)
    assert report["checks_passed"] and report["downloaded_files"] == 3
    for name, data in files.items():
        assert (tmp_path / name).read_bytes() == data
    assert sum(end - start + 1 for start, end in calls) < len(archive) // 10
    assert not (tmp_path / "unselected").exists()
    assert not list(tmp_path.rglob("*.part"))
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network on valid resume"))
    resumed = module.download_weight_sort_assets(tmp_path, manifest=_manifest(files), workers=4)
    assert resumed["checks_passed"] and resumed["already_present_files"] == 3
    assert resumed["downloaded_files"] == 0


def test_robot_uses_existing_robot_downloader_and_wrappers_are_deferred(tmp_path, monkeypatch):
    module = _tool()
    files = {"franka_mimic.usd": b"robot", "normalized/weight_sort/box.usda": b"authored"}
    manifest = _manifest(files)
    manifest["files"][1]["source_group"] = "forcebench_authored_wrapper_or_primitive"
    calls = []
    def download_robot(source_dir, *, timeout):
        calls.append((source_dir, timeout))
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "franka_mimic.usd").write_bytes(b"robot")
        return {"status": "downloaded", "source_url": "https://example.invalid/robot"}
    monkeypatch.setattr(module, "_robot_tool", lambda: types.SimpleNamespace(download_robot=download_robot))
    report = module.download_weight_sort_assets(tmp_path, manifest=manifest, timeout=12)
    assert report["checks_passed"] and report["required_external_files"] == 1
    assert calls == [(tmp_path, 12)]
    assert report["files"][0]["sha256"] == hashlib.sha256(b"robot").hexdigest()
    assert not (tmp_path / "normalized/weight_sort/box.usda").exists()


def test_unknown_existing_file_is_preserved_and_blocks_network(tmp_path, monkeypatch):
    module = _tool()
    files = {"simready/flatbox_a02/flatbox_a02.usd": b"expected",
             "simready/container_f27/container_f27.usd": b"missing"}
    destination = tmp_path / "simready/flatbox_a02/flatbox_a02.usd"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"x" * len(b"expected"))
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network after invalid preflight"))
    report = module.download_weight_sort_assets(tmp_path, manifest=_manifest(files))
    assert not report["checks_passed"]
    assert destination.read_bytes() == b"x" * len(b"expected")
    assert {item["status"] for item in report["files"]} == {"failed", "missing"}
    assert not (tmp_path / "simready/container_f27/container_f27.usd").exists()


@pytest.mark.parametrize("mode", ["ignored", "bad_range", "bad_etag", "truncated"])
def test_invalid_range_response_never_publishes_asset(tmp_path, monkeypatch, mode):
    module = _tool()
    files = {"simready/flatbox_a02/flatbox_a02.usd": b"expected"}
    calls, responses = _range_server(module, _archive(module, files), monkeypatch, mode=mode)
    report = module.download_weight_sort_assets(tmp_path, manifest=_manifest(files), workers=1)
    assert not report["checks_passed"] and report["files"][0]["status"] == "failed"
    assert calls
    if mode == "ignored":
        assert all(response.read_calls == 0 for response in responses)
        assert "ignored" in report["files"][0]["error"]
    assert not list(tmp_path.rglob("*.usd"))
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("payload", [b"too short", b"x" * len(b"exact pinned bytes")])
def test_wrong_zip_length_or_sha_leaves_no_completed_file(tmp_path, monkeypatch, payload):
    module = _tool()
    name = "simready/flatbox_a02/flatbox_a02.usd"
    _range_server(module, _archive(module, {name: payload}), monkeypatch)
    report = module.download_weight_sort_assets(tmp_path, manifest=_manifest({name: b"exact pinned bytes"}))
    assert not report["checks_passed"]
    assert not (tmp_path / name).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_zip_crc_failure_is_reported_and_cleaned(tmp_path, monkeypatch):
    module = _tool()
    name = "simready/flatbox_a02/flatbox_a02.usd"
    payload = b"exact pinned bytes"
    output = io.BytesIO()
    member = module.ARCHIVE_MEMBER_PREFIX + name.removeprefix("simready/")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, payload)
    corrupted = bytearray(output.getvalue())
    corrupted[30 + len(member)] ^= 1
    _range_server(module, bytes(corrupted), monkeypatch)
    report = module.download_weight_sort_assets(tmp_path, manifest=_manifest({name: payload}))
    assert not report["checks_passed"] and "CRC" in report["files"][0]["error"]
    assert not (tmp_path / name).exists()
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("payload", [b"wrong", b"x" * len(b"pinned table")])
def test_table_length_or_sha_failure_does_not_publish(tmp_path, monkeypatch, payload):
    module = _tool()
    name = "residential/east_rural/EastRural_Table.usd"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs:
                        _Response(payload, headers={"Content-Length": str(len(payload))}))
    report = module.download_weight_sort_assets(tmp_path, manifest=_manifest({name: b"pinned table"}))
    assert not report["checks_passed"]
    assert not (tmp_path / name).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_range_reader_rejects_unbounded_full_archive_read_before_network(monkeypatch):
    module = _tool()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("unbounded network"))
    with pytest.raises(ValueError, match="Range limit"):
        module.HTTPRangeReader().read()


def test_successful_files_survive_a_different_file_failure_and_resume(tmp_path, monkeypatch):
    module = _tool()
    good = "simready/flatbox_a02/flatbox_a02.usd"
    retry = "simready/container_f27/container_f27.usd"
    expected = {good: b"good payload", retry: b"expected retry"}
    _range_server(module, _archive(module, {good: expected[good], retry: b"wrong contents"}), monkeypatch)
    first = module.download_weight_sort_assets(tmp_path, manifest=_manifest(expected), workers=2)
    assert not first["checks_passed"] and first["downloaded_files"] == 1
    assert (tmp_path / good).read_bytes() == expected[good]
    assert not (tmp_path / retry).exists()
    saved_mtime = (tmp_path / good).stat().st_mtime_ns
    _range_server(module, _archive(module, expected), monkeypatch)
    second = module.download_weight_sort_assets(tmp_path, manifest=_manifest(expected), workers=2)
    assert second["checks_passed"] and second["downloaded_files"] == second["already_present_files"] == 1
    assert (tmp_path / good).stat().st_mtime_ns == saved_mtime


def test_network_failure_after_partial_table_data_cleans_temporary_file(tmp_path, monkeypatch):
    module = _tool()
    name = "residential/east_rural/EastRural_Table.usd"
    class Interrupted(_Response):
        def read(self, size=-1):
            if self.tell():
                raise urllib.error.URLError("connection lost")
            return super().read(3)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: Interrupted(b"pinned table"))
    report = module.download_weight_sort_assets(tmp_path, manifest=_manifest({name: b"pinned table"}))
    assert not report["checks_passed"] and "connection lost" in report["files"][0]["error"]
    assert not (tmp_path / name).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_concurrent_user_file_is_preserved_during_publication(tmp_path, monkeypatch):
    module = _tool()
    name = "residential/east_rural/EastRural_Table.usd"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: _Response(b"pinned table"))
    original_link = module.os.link
    def competing_link(source, destination):
        destination.write_bytes(b"concurrent changes")
        original_link(source, destination)
    monkeypatch.setattr(module.os, "link", competing_link)
    report = module.download_weight_sort_assets(tmp_path, manifest=_manifest({name: b"pinned table"}))
    assert not report["checks_passed"] and "preserved" in report["files"][0]["error"]
    assert (tmp_path / name).read_bytes() == b"concurrent changes"
    assert not list(tmp_path.rglob("*.part"))


def test_paths_outside_audited_weight_sources_are_rejected(tmp_path):
    module = _tool()
    with pytest.raises(ValueError, match="No audited Weight Sort source"):
        module.download_weight_sort_assets(tmp_path, manifest=_manifest({"other_task/model.usd": b"other"}))
    with pytest.raises(ValueError, match="escaping asset path"):
        module.download_weight_sort_assets(tmp_path, manifest=_manifest({"simready/flatbox_a02/../../outside.usd": b"other"}))
