from pathlib import Path

import numpy as np

from coribench.registry import get_task_manifest, task_names
from coribench.tasks.gear_mesh.spec import (
    SPEC as GEAR_SPEC,
    evaluate_final_state as evaluate_gear_final_state,
    sample_initial_medium_gear_yaw_jitter_deg,
    sample_initial_state as sample_gear_initial_state,
)
from coribench.tasks.nut_thread.spec import (
    SPEC as NUT_SPEC,
    evaluate_final_state as evaluate_nut_final_state,
    sample_initial_state as sample_nut_initial_state,
)


def test_forge_asset_tasks_are_discovered_without_simulator_imports():
    assert {"gear_mesh", "nut_thread"}.issubset(task_names())
    assert get_task_manifest("gear_mesh").runtime_class.endswith(":GearMeshTask")
    assert get_task_manifest("nut_thread").runtime_class.endswith(":NutThreadTask")


def test_gear_layout_is_deterministic_and_separated():
    first = sample_gear_initial_state(42)
    assert first == sample_gear_initial_state(42)
    assert first != sample_gear_initial_state(43)
    assert first["initial_separation_m"] >= GEAR_SPEC.minimum_initial_separation_m
    target = np.asarray(first["target_medium_shaft_position_m"][:2])
    pickup = np.asarray(first["pickup_gear_center_position_m"][:2])
    np.testing.assert_allclose(np.linalg.norm(target - pickup), first["initial_separation_m"])
    assert first["medium_gear_root_position_m"] == first["pickup_gear_center_position_m"]
    assert first["target_medium_shaft_position_m"][2] == GEAR_SPEC.medium_gear_target_offset_m[2]
    minimum_jitter, maximum_jitter = GEAR_SPEC.initial_medium_gear_yaw_jitter_abs_range_deg
    assert minimum_jitter <= abs(first["medium_gear_yaw_jitter_deg"]) <= maximum_jitter
    assert np.isclose(
        first["medium_gear_yaw_rad"],
        np.deg2rad(
            GEAR_SPEC.initial_medium_gear_yaw_deg + first["medium_gear_yaw_jitter_deg"]
        ),
    )
    assert first["target_base_yaw_rad"] == 0.0
    second = sample_gear_initial_state(43)
    assert second["medium_gear_yaw_jitter_deg"] != first["medium_gear_yaw_jitter_deg"]
    assert second["target_base_yaw_rad"] == 0.0


def test_gear_initial_yaw_jitter_is_seeded_small_and_nonzero():
    values = [sample_initial_medium_gear_yaw_jitter_deg(seed) for seed in range(64)]
    minimum, maximum = GEAR_SPEC.initial_medium_gear_yaw_jitter_abs_range_deg
    assert values == [sample_initial_medium_gear_yaw_jitter_deg(seed) for seed in range(64)]
    assert all(minimum <= abs(value) <= maximum for value in values)
    assert any(value < 0.0 for value in values)
    assert any(value > 0.0 for value in values)


def test_gear_success_requires_seating_release_and_retraction():
    kwargs = dict(
        gear_center_position_m=[0.64, 0.01, 0.0],
        gear_axis_world=[0.0, 0.0, 1.0],
        target_shaft_position_m=[0.64, 0.01, 0.0],
        gripper_open_fraction=1.0,
        eef_position_m=[0.64, 0.01, 0.09],
    )
    assert evaluate_gear_final_state(**kwargs).success
    assert not evaluate_gear_final_state(**{**kwargs, "gripper_open_fraction": 0.0}).success
    assert not evaluate_gear_final_state(**{**kwargs, "gear_center_position_m": [0.65, 0.01, 0.0]}).success
    assert not evaluate_gear_final_state(**{**kwargs, "eef_position_m": [0.64, 0.01, 0.02]}).success


def test_nut_layout_is_deterministic_and_uses_forge_thread_geometry():
    first = sample_nut_initial_state(42)
    assert first == sample_nut_initial_state(42)
    assert first != sample_nut_initial_state(43)
    assert first["initial_separation_m"] >= NUT_SPEC.minimum_initial_separation_m
    assert NUT_SPEC.thread_pitch_m == 0.002
    assert NUT_SPEC.target_nut_base_offset_m == 0.032
    assert first["pickup_nut_root_position_m"][2] == NUT_SPEC.pickup_support_size_m[2]
    assert first["pickup_nut_yaw_rad"] == 0.0
    assert first["target_bolt_yaw_rad"] == 0.0
    assert sample_nut_initial_state(43)["pickup_nut_yaw_rad"] == 0.0
    assert sample_nut_initial_state(43)["target_bolt_yaw_rad"] == 0.0


def test_forge_tasks_use_normalized_object_assets():
    for asset in (
        GEAR_SPEC.gear_base_asset,
        GEAR_SPEC.medium_gear_asset,
        GEAR_SPEC.small_gear_asset,
        GEAR_SPEC.large_gear_asset,
        NUT_SPEC.nut_asset,
        NUT_SPEC.bolt_asset,
    ):
        assert asset.startswith("normalized/")
        assert asset.endswith(".usda")


def test_gear_grasp_and_mesh_phase_match_forge_geometry():
    assert GEAR_SPEC.asset_scale_xyz == (2.0, 2.0, 2.0)
    assert GEAR_SPEC.mass_scale == 8.0
    assert GEAR_SPEC.grasp_gripper_open_fraction == 0.70
    assert GEAR_SPEC.pickup_grasp_tcp_z_m == 0.040
    assert GEAR_SPEC.medium_gear_target_offset_m == (0.04050, 0.0, 0.010)
    assert GEAR_SPEC.initial_medium_gear_yaw_deg == -4.5
    assert GEAR_SPEC.expert_twist_alignment_offset_deg == -4.5


def test_normalized_thread_assets_preserve_source_sdf_collisions():
    asset_root = Path(__file__).resolve().parents[1] / "assets" / "normalized" / "nut_thread"
    for filename, source_asset in (
        ("nut.usda", "nut_m16.usd"),
        ("bolt.usda", "bolt_m16.usd"),
    ):
        wrapper = (asset_root / filename).read_text(encoding="utf-8")
        assert source_asset in wrapper
        assert "convexDecomposition" not in wrapper


def test_nut_success_requires_thread_depth_release_and_retraction():
    target = [0.64, 0.01, NUT_SPEC.target_nut_base_offset_m]
    kwargs = dict(
        nut_base_position_m=target,
        nut_axis_world=[0.0, 0.0, 1.0],
        target_nut_base_position_m=target,
        gripper_open_fraction=1.0,
        eef_position_m=[0.64, 0.01, 0.10],
    )
    assert evaluate_nut_final_state(**kwargs).success
    assert not evaluate_nut_final_state(
        **{**kwargs, "nut_base_position_m": [0.64, 0.01, target[2] + NUT_SPEC.thread_pitch_m]}
    ).success
    assert not evaluate_nut_final_state(**{**kwargs, "nut_base_position_m": [0.65, 0.01, target[2]]}).success
    assert not evaluate_nut_final_state(**{**kwargs, "gripper_open_fraction": 0.0}).success



