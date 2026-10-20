"""Regress the production collector's post-completion trajectory/quality alignment."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forcebench.recording import EpisodeBuffer
from forcebench.seed_balance import final_move_sign
from scripts.expert.independent_release_state import CompletionSnapshot, trim_episode_buffer


SOURCE = Path(__file__).resolve().parents[1] / "scripts/expert/generate_parallel_stove_press_turn.py"
HISTORY_FIELDS = (
    "minimum_press_m", "maximum_turn_press_m", "maximum_easy_progress_deg",
    "maximum_hard_progress_deg", "maximum_tracking_error_m",
)


def production_adjudicator():
    """Compile the actual final-scoring tail, without importing Isaac's launcher."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    expert = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "ParallelStovePressTurnExpert")
    run = next(node for node in expert.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    start = next(i for i, node in enumerate(run.body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
                 and isinstance(node.targets[0].elts[0], ast.Name)
                 and node.targets[0].elts[0].id == "batch_press")
    function = ast.FunctionDef(
        name="adjudicate",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in (
            "self", "completed_states", "release_states", "before_reports", "knob_positions")],
            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=deepcopy(run.body[start:]), decorator_list=[], type_params=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(np=np, math=math, NUM_ENVS=2, SEEDS=[0, 1],
                     args=SimpleNamespace(probe_min_angle_deg=8.0, probe_max_angle_deg=20.0),
                     as_numpy=np.asarray, final_move_sign=final_move_sign,
                     CompletionSnapshot=CompletionSnapshot, trim_episode_buffer=trim_episode_buffer,
                     QUALITY_HISTORY_FIELDS=HISTORY_FIELDS)
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["adjudicate"]


def buffer_with_two_transitions():
    buffer = EpisodeBuffer()
    for frame in range(2):
        buffer.append(state=np.zeros(7), action=np.zeros(7), wrench_raw=np.zeros(6),
                      wrench_ema=np.zeros(6), timestamp=frame / 15.0,
                      privileged={"mechanism": np.asarray([frame], dtype=np.float32)})
    return buffer


def setup_case(*, completion_point, batch_point, completion_press=0.0,
               completion_hard=0.0, batch_hard=0.0):
    histories = dict(zip(HISTORY_FIELDS, [-0.014, -0.008, 60.0, completion_hard, 0.001]))
    report = {"success": True, "knob_angle_deg": 60.0, "wrong_direction_detent_crossed": False}
    snapshot = CompletionSnapshot.capture(report=report, control_frames=1,
        press_m=completion_press, turn_angle_rad=math.radians(60.0),
        fingertip_xyz_m=completion_point, history_metrics=histories)
    expert = SimpleNamespace(
        buffers=[buffer_with_two_transitions() for _ in range(2)],
        env=SimpleNamespace(fingertip_midpoint_pos=np.asarray([batch_point, batch_point])),
        task=SimpleNamespace(evaluate_final_state_for=lambda env, i: deepcopy(report)),
        mechanism_state=lambda: (np.zeros(2), np.full(2, math.radians(60.0)), None, None),
        closed_finger_positions_m=np.full((2, 2), 0.02),
        open_finger_positions_m=np.full((2, 2), 0.04), modes=["direct_expert"] * 2,
        force_switch_triggered=np.zeros(2, dtype=bool), probe_failed=np.zeros(2, dtype=bool),
        switch_frame=np.full(2, -1), switch_angle_deg=np.full(2, np.nan),
        switch_torque_nm=np.full(2, np.nan), angle_before_open_deg=np.full(2, 60.0),
        angle_after_open_deg=np.full(2, 60.0), turn_complete=np.ones(2, dtype=bool),
        settle_complete=np.ones(2, dtype=bool),
        specs=[SimpleNamespace(target_final_angle_deg=60.0, final_angle_tolerance_deg=5.0,
                               press_engage_position_m=-0.007)] * 2,
    )
    for name, value in histories.items():
        setattr(expert, name, np.full(2, batch_hard if name == "maximum_hard_progress_deg" else value))
    for name in ("phase_traces", "source_type_traces", "angle_traces", "press_traces", "ema_torque_traces"):
        setattr(expert, name, [[0, 1], [0, 1]])
    release_states = [SimpleNamespace(stage="done", release_passed=True, failure_reason=""),
                      SimpleNamespace(stage="failed", release_passed=False, failure_reason="motion_not_stable")]
    before = [deepcopy(report), deepcopy(report)]
    _, reports = production_adjudicator()(expert, [snapshot, None], release_states, before, np.zeros((2, 3)))
    return expert, reports


def good_point():
    return (0.0, final_move_sign(0) * 0.1, 0.2)


def test_peer_wait_cannot_make_trimmed_incomplete_release_successful():
    expert, reports = setup_case(completion_point=(0.0, 0.0, 0.02), batch_point=good_point())
    assert not reports[0]["success"]
    assert not reports[0]["lifted"] and not reports[0]["retreated"]
    assert reports[0]["collection_quality_state_timepoint"] == "per_clone_completion_post_action"
    assert reports[0]["collection_quality_control_frames"] == 1
    assert len(expert.buffers[0]) == len(expert.buffers[0].privileged_fields["mechanism"]) == 1
    assert len(expert.buffers[1]) == 2
    assert reports[1]["collection_quality_state_timepoint"] == "batch_end_post_action"
    assert not reports[1]["success"]


def test_later_peer_physics_cannot_reject_successful_retained_episode():
    expert, reports = setup_case(completion_point=good_point(), batch_point=(0.0, 0.0, 0.0), batch_hard=25.0)
    assert reports[0]["success"] and expert.buffers[0].successes[-1]
    assert reports[0]["maximum_hard_progress_deg"] == 0.0
    assert expert.maximum_hard_progress_deg[0] == 0.0
    assert expert.maximum_hard_progress_deg[1] == 25.0
    assert expert.total_control_frames == 2


@pytest.mark.parametrize("press,hard", [(-0.003, 0.0), (0.0, 3.0)])
def test_all_final_quality_gates_use_completion_snapshot(press, hard):
    _, reports = setup_case(completion_point=good_point(), batch_point=good_point(),
                            completion_press=press, completion_hard=hard, batch_hard=0.0)
    assert not reports[0]["success"]
    assert reports[0]["final_press_m"] == press
    assert reports[0]["maximum_hard_progress_deg"] == hard


def test_snapshot_detaches_mutable_sensor_and_report_inputs():
    point = np.asarray(good_point())
    report = {"success": True, "nested": {"value": 1}}
    metrics = {"maximum_hard_progress_deg": 0.0}
    snapshot = CompletionSnapshot.capture(report=report, control_frames=1, press_m=0.0,
        turn_angle_rad=0.0, fingertip_xyz_m=point, history_metrics=metrics)
    point[:] = 999
    report["nested"]["value"] = 999
    metrics["maximum_hard_progress_deg"] = 999
    assert snapshot.fingertip_xyz_m == good_point()
    assert snapshot.report["nested"]["value"] == 1
    assert snapshot.history_metrics["maximum_hard_progress_deg"] == 0.0
