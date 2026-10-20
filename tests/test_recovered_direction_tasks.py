"""CPU checks for historically recovered tasks; not simulator reproduction."""

import ast
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from forcebench.task_api import TaskManifest
from forcebench.tasks.faucet_force_turn import MANIFEST as FAUCET_MANIFEST
from forcebench.tasks.faucet_force_turn import spec as faucet
from forcebench.tasks.faucet_force_turn.outcome import evaluate_report as faucet_outcome
from forcebench.tasks.stove_press_turn import MANIFEST as STOVE_MANIFEST
from forcebench.tasks.stove_press_turn import spec as stove
from forcebench.tasks.stove_press_turn.outcome import evaluate_report as stove_outcome


def load_drive_hook():
    """Execute the production config hook without importing Isaac Sim."""
    source = Path(__file__).parents[1] / "forcebench" / "env.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    hook = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_configure_task_robot_drive")
    namespace = {"TaskManifest": TaskManifest}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[hook.name]


def drive_config():
    return SimpleNamespace(
        servo_arm_stiffness=81.0,
        servo_arm_damping=13.0,
        robot=SimpleNamespace(actuators={
            "arm_shoulder": SimpleNamespace(stiffness=81.0, damping=13.0),
            "arm_forearm": SimpleNamespace(stiffness=81.0, damping=13.0),
            "hand": SimpleNamespace(stiffness=4000.0, damping=120.0),
        }),
    )


def test_legacy_task_drive_config_is_unchanged():
    cfg = drive_config()
    before = deepcopy(cfg)
    legacy = TaskManifest("legacy", "test", "example:Task", "legacy")
    load_drive_hook()(cfg, legacy)
    assert cfg == before


@pytest.mark.parametrize("manifest", [FAUCET_MANIFEST, STOVE_MANIFEST])
def test_recovered_task_drives_and_horizon(manifest):
    assert manifest.evaluation_max_steps == 1500
    cfg = drive_config()
    load_drive_hook()(cfg, manifest)
    assert (cfg.servo_arm_stiffness, cfg.servo_arm_damping) == (80.0, 12.0)
    for arm in ("arm_shoulder", "arm_forearm"):
        assert vars(cfg.robot.actuators[arm]) == {"stiffness": 80.0, "damping": 12.0}
    assert vars(cfg.robot.actuators["hand"]) == {"stiffness": 20000.0, "damping": 600.0}


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), True, "80"])
def test_invalid_drive_override_fails_before_robot_mutation(value):
    cfg = drive_config()
    before = deepcopy(cfg)
    with pytest.raises(ValueError):
        load_drive_hook()(cfg, replace(FAUCET_MANIFEST, arm_actuator_stiffness=value))
    assert cfg == before


@pytest.mark.parametrize("direction", [-1, 1])
def test_faucet_original_band_is_preserved_and_wrong_direction_fails(direction):
    episode = replace(faucet.SPEC, easy_direction_sign=direction)
    center = episode.detent_center_magnitude_deg
    for offset in (0.6, 1.35, 2.1):
        assert faucet.evaluate_turn_angle(math.radians(direction * (center + offset)), episode)["success"]
    for offset in (0.59, 2.11, 10.0):
        assert not faucet.evaluate_turn_angle(math.radians(direction * (center + offset)), episode)["success"]
    assert not faucet.evaluate_turn_angle(math.radians(-direction * (center + 1.35)), episode)["success"]


@pytest.mark.parametrize("failed_field", [
    "correct_direction_detent_crossed", "within_final_angle_tolerance",
    "water_visible", "gripper_released", "tcp_lifted_clear", "valve_stationary",
])
def test_faucet_each_completion_condition_remains_required(failed_field):
    report = dict.fromkeys([
        "correct_direction_detent_crossed", "within_final_angle_tolerance",
        "water_visible", "gripper_released", "tcp_lifted_clear", "valve_stationary",
    ], True)
    assert faucet_outcome(report)["success"]
    report[failed_field] = False
    judged = faucet_outcome(report)
    assert not judged["success"]
    assert len(judged["failure_reasons"]) == 1


def test_stove_press_gate_and_historical_record_parameters():
    assert not stove.is_press_engaged(-0.006999)
    assert stove.is_press_engaged(-0.007)
    assert stove.is_press_engaged(-0.008)
    # Independent values from the archived ForceVLA episode reports, seeds 300-302.
    recorded = [
        (300, -1, 78.43087356308985, -81.43087356308985),
        (301, -1, 72.46205309615499, -75.46205309615499),
        (302, -1, 89.45611991528203, -92.45611991528203),
    ]
    for seed, direction, threshold, target in recorded:
        episode = stove.spec_from_layout(stove.sample_initial_state(seed))
        assert episode.easy_direction_sign == direction
        assert episode.success_angle_magnitude_deg == pytest.approx(threshold, abs=1e-10)
        assert episode.target_final_angle_deg == pytest.approx(target, abs=1e-10)


@pytest.mark.parametrize("direction", [-1, 1])
def test_stove_directional_crossing_and_latched_outcome(direction):
    episode = replace(stove.SPEC, easy_direction_sign=direction)
    threshold = episode.success_angle_magnitude_deg
    assert not stove.evaluate_turn_angle(math.radians(direction * (threshold - 0.01)), episode)["success"]
    assert stove.evaluate_turn_angle(math.radians(direction * (threshold + 0.01)), episode)["success"]
    assert not stove.evaluate_turn_angle(math.radians(-direction * (threshold + 0.01)), episode)["success"]
    # At final submission the knob can be released or drift; the recorded event
    # is still the criterion, rather than reimposing final-state diagnostics.
    report = {
        "correct_direction_detent_crossed": True,
        "pressed_at_submission": False,
        "flame_visible": False,
        "within_final_angle_tolerance": False,
    }
    assert stove_outcome(report)["success"]
    report["correct_direction_detent_crossed"] = False
    report["flame_visible"] = True
    report["pressed_at_submission"] = True
    assert not stove_outcome(report)["success"]
