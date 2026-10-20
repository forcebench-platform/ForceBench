"""Command-building helpers shared by expert backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from forcebench.expert_api import ExpertRequest


def isaac_command(request: ExpertRequest, script: str) -> list[str]:
    launcher = request.isaaclab_root / "isaaclab.sh"
    source = request.code_root / script
    if not launcher.is_file():
        raise FileNotFoundError(f"Isaac Lab launcher is missing: {launcher}")
    if not source.is_file():
        raise FileNotFoundError(f"expert generator is missing: {source}")
    return [str(launcher), "-p", str(source)]


def append_options(
    command: list[str],
    options: Mapping[str, Any],
    allowed: Mapping[str, type | tuple[type, ...]],
) -> None:
    unknown = set(options) - set(allowed)
    if unknown:
        raise ValueError(f"unknown expert backend options: {sorted(unknown)}")
    for name in sorted(options):
        value = options[name]
        expected = allowed[name]
        if not isinstance(value, expected) or isinstance(value, bool) != (expected is bool):
            raise TypeError(
                f"expert backend option {name!r} must be {expected}, got {type(value)}"
            )
        flag = "--" + name
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))


def common_camera_args(request: ExpertRequest) -> list[str]:
    return [
        "--main_image_size",
        str(request.main_image_size),
        "--wrist_image_size",
        str(request.wrist_image_size),
        "--headless",
        "--enable_cameras",
        "--device",
        request.device,
    ]


def canonical_physics_args(request: ExpertRequest) -> list[str]:
    return [
        "--physics_rate_hz",
        str(request.physics_rate_hz),
        "--solver_position_iterations",
        str(request.solver_position_iterations),
    ]


def video_path(request: ExpertRequest) -> Path:
    return request.artifact_dir / "episode.mp4"
