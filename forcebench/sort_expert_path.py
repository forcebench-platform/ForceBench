"""Simulator-free path helpers shared by the parallel sorting experts."""

from __future__ import annotations

import numpy as np


def direct_transport_waypoints(
    current_tcp_positions_m,
    destination_positions_m,
    *,
    clearance_z_m: float,
    release_z_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raise, traverse halfway, then descend above the current destination.

    The horizontal coordinates are derived exclusively from the observed TCP
    position and the task specification's destination.  This prevents stale
    scene-layout coordinates from surviving a later fixture move.
    """

    current = np.asarray(current_tcp_positions_m, dtype=np.float32)
    destination = np.asarray(destination_positions_m, dtype=np.float32)
    if current.ndim != 2 or current.shape[1:] != (3,):
        raise ValueError("current TCP positions must have shape (num_envs, 3)")
    if destination.shape != current.shape:
        raise ValueError("destination positions must match current TCP positions")
    if not np.isfinite(current).all() or not np.isfinite(destination).all():
        raise ValueError("transport positions must be finite")
    clearance = float(clearance_z_m)
    release = float(release_z_m)
    if not np.isfinite(clearance) or not np.isfinite(release) or clearance <= release:
        raise ValueError("clearance_z_m must be finite and above release_z_m")

    raised = current.copy()
    raised[:, 2] = clearance
    midpoint = raised.copy()
    midpoint[:, :2] = np.float32(0.5) * (
        current[:, :2] + destination[:, :2]
    )
    release_pose = destination.copy()
    release_pose[:, 2] = release
    return raised, midpoint, release_pose


def bounded_box_feedback_tcp_targets(
    previous_tcp_commands_m,
    actual_box_positions_m,
    target_box_positions_m,
    *,
    gain=0.12,
    max_command_step_m: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate bounded box-pose error into the previous TCP command."""

    previous_command = np.asarray(previous_tcp_commands_m, dtype=np.float32)
    actual_box = np.asarray(actual_box_positions_m, dtype=np.float32)
    target_box = np.asarray(target_box_positions_m, dtype=np.float32)
    if previous_command.ndim != 2 or previous_command.shape[1:] != (3,):
        raise ValueError("previous TCP commands must have shape (num_envs, 3)")
    if actual_box.shape != previous_command.shape or target_box.shape != previous_command.shape:
        raise ValueError("box positions must match previous TCP commands")
    if not all(
        np.isfinite(value).all()
        for value in (previous_command, actual_box, target_box)
    ):
        raise ValueError("feedback positions must be finite")
    gain = np.asarray(gain, dtype=np.float32)
    if gain.ndim == 0:
        gain = np.repeat(gain, 3)
    maximum = float(max_command_step_m)
    if gain.shape != (3,) or not np.isfinite(gain).all() or np.any(gain <= 0.0):
        raise ValueError("gain must be a positive finite scalar or 3-vector")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_command_step_m must be finite and positive")

    correction = gain[None, :] * (target_box - actual_box)
    norm = np.linalg.norm(correction, axis=1, keepdims=True)
    scale = np.minimum(1.0, maximum / np.maximum(norm, 1.0e-9)).astype(np.float32)
    bounded = correction * scale
    return previous_command + bounded, bounded


def limit_tcp_command_step(
    previous_tcp_commands_m,
    proposed_tcp_commands_m,
    *,
    max_command_step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the final all-axis norm limit after every absolute guard."""

    previous = np.asarray(previous_tcp_commands_m, dtype=np.float32)
    proposed = np.asarray(proposed_tcp_commands_m, dtype=np.float32)
    if previous.ndim != 2 or previous.shape[1:] != (3,) or proposed.shape != previous.shape:
        raise ValueError("TCP command arrays must have matching shape (num_envs, 3)")
    if not np.isfinite(previous).all() or not np.isfinite(proposed).all():
        raise ValueError("TCP commands must be finite")
    maximum = float(max_command_step_m)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_command_step_m must be finite and positive")
    step = proposed - previous
    norm = np.linalg.norm(step, axis=1, keepdims=True)
    scale = np.minimum(1.0, maximum / np.maximum(norm, 1.0e-9)).astype(np.float32)
    bounded = step * scale
    return previous + bounded, bounded
