"""The robot downloader verifies bytes before publishing and preserves user files."""
from __future__ import annotations

import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import urllib.error

import pytest


def _tool():
    spec = importlib.util.spec_from_file_location(
        "download_robot_asset", Path(__file__).resolve().parents[1] / "tools/download_robot_asset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def robot_tool(monkeypatch):
    module = _tool()
    for prefix, data in (("OFFICIAL", b"verified official export"), ("HISTORICAL", b"historical export")):
        monkeypatch.setattr(module, prefix + "_BYTES", len(data))
        monkeypatch.setattr(module, prefix + "_SHA256", hashlib.sha256(data).hexdigest())
    return module


def _response(data, length=None):
    stream = io.BytesIO(data)
    stream.headers = {} if length is None else {"Content-Length": str(length)}
    return stream


def test_production_endpoint_and_byte_pins_are_fixed():
    module = _tool()
    assert module.SOURCE_URL == (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/IsaacLab/Factory/franka_mimic.usd"
    )
    assert (module.OFFICIAL_BYTES, module.OFFICIAL_SHA256) == (
        7075031, "86d0017a093669b222a8c640bf581cc3e3715035aa7c596600f478da011f6c8e"
    )
    assert (module.HISTORICAL_BYTES, module.HISTORICAL_SHA256) == (
        7073847, "6ca52983cbede63f4caf1557896213e732f14fd8da9234658a6981ec2515b1bf"
    )


@pytest.mark.parametrize("with_length", [False, True])
def test_verified_download_is_published_and_returns_identity(tmp_path, robot_tool, monkeypatch, with_length):
    data = b"verified official export"
    calls = []
    def open_response(url, timeout):
        calls.append((url, timeout))
        return _response(data, len(data) if with_length else None)
    monkeypatch.setattr(robot_tool.urllib.request, "urlopen", open_response)
    report = robot_tool.download_robot(tmp_path / "source", timeout=12)
    assert calls == [(robot_tool.SOURCE_URL, 12)]
    assert report["status"] == "downloaded" and report["export"] == "official"
    assert report["sha256"] == hashlib.sha256(data).hexdigest()
    assert (tmp_path / "source/franka_mimic.usd").read_bytes() == data
    assert list((tmp_path / "source").iterdir()) == [tmp_path / "source/franka_mimic.usd"]


@pytest.mark.parametrize("data,export", [(b"verified official export", "official"), (b"historical export", "historical")])
def test_existing_audited_export_needs_no_network(tmp_path, robot_tool, monkeypatch, data, export):
    destination = tmp_path / "franka_mimic.usd"
    destination.write_bytes(data)
    monkeypatch.setattr(robot_tool.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("unexpected network"))
    report = robot_tool.download_robot(tmp_path)
    assert report["status"] == "already_present" and report["export"] == export
    assert destination.read_bytes() == data


def test_unknown_existing_asset_is_preserved_without_network(tmp_path, robot_tool, monkeypatch):
    destination = tmp_path / "franka_mimic.usd"
    destination.write_bytes(b"x" * robot_tool.OFFICIAL_BYTES)
    monkeypatch.setattr(robot_tool.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="preserved"):
        robot_tool.download_robot(tmp_path)
    assert destination.read_bytes() == b"x" * robot_tool.OFFICIAL_BYTES


@pytest.mark.parametrize("data,length", [
    (b"truncated", None),
    (b"truncated", len(b"verified official export")),
    (b"x" * len(b"verified official export"), None),
    (b"verified official export" + b"extra", None),
    (b"verified official export", 999),
])
def test_invalid_responses_leave_no_asset_or_temporary_file(tmp_path, robot_tool, monkeypatch, data, length):
    monkeypatch.setattr(robot_tool.urllib.request, "urlopen", lambda *args, **kwargs: _response(data, length))
    with pytest.raises(ValueError):
        robot_tool.download_robot(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_network_failure_leaves_no_asset_or_temporary_file(tmp_path, robot_tool, monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(robot_tool.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        robot_tool.download_robot(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_interrupted_response_is_cleaned_and_cli_emits_json(tmp_path, robot_tool, monkeypatch, capsys):
    class Interrupted(io.BytesIO):
        headers = {}
        def read(self, *args):
            raise http.client.IncompleteRead(b"partial", 100)
    monkeypatch.setattr(robot_tool.urllib.request, "urlopen", lambda *args, **kwargs: Interrupted())
    monkeypatch.setattr("sys.argv", ["download_robot_asset.py", "--source-dir", str(tmp_path)])
    assert robot_tool.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed" and "IncompleteRead" in report["error"]
    assert list(tmp_path.iterdir()) == []


def test_file_created_during_download_is_never_replaced(tmp_path, robot_tool, monkeypatch):
    monkeypatch.setattr(robot_tool.urllib.request, "urlopen", lambda *args, **kwargs: _response(b"verified official export"))
    original_link = robot_tool.os.link
    def competing_link(source, destination):
        destination.write_bytes(b"concurrent user file")
        original_link(source, destination)
    monkeypatch.setattr(robot_tool.os, "link", competing_link)
    with pytest.raises(ValueError, match="preserved"):
        robot_tool.download_robot(tmp_path)
    assert (tmp_path / "franka_mimic.usd").read_bytes() == b"concurrent user file"
    assert list(tmp_path.iterdir()) == [tmp_path / "franka_mimic.usd"]


def test_publication_failure_cleans_verified_temporary_file(tmp_path, robot_tool, monkeypatch):
    monkeypatch.setattr(robot_tool.urllib.request, "urlopen", lambda *args, **kwargs: _response(b"verified official export"))
    def fail_link(*args):
        raise OSError("filesystem publication failed")
    monkeypatch.setattr(robot_tool.os, "link", fail_link)
    with pytest.raises(OSError, match="publication failed"):
        robot_tool.download_robot(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_import_has_no_network_side_effect(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network during import"))
    _tool()
