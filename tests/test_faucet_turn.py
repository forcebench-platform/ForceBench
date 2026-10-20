import math
from pathlib import Path

import pytest

from forcebench.registry import get_task_manifest, task_names
from forcebench.task_api import CameraProfile
from forcebench.tasks.faucet_turn.spec import (
    SPEC,
    detent_effort_nm,
    detent_resistance_nm,
    evaluate_angle,
    flow_fraction,
    sample_initial_state,
    spec_from_layout,
)


def test_faucet_task_is_discoverable_and_uses_the_default_main_camera():
    assert "faucet_turn" in task_names()
    manifest = get_task_manifest("faucet_turn")
    assert manifest.runtime_class.endswith(":FaucetTurnTask")
    runtime_source = (
        Path(__file__).resolve().parents[1]
        / "forcebench"
        / "tasks"
        / "faucet_turn"
        / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "table_position_m=" not in runtime_source
    assert "table_focal_length_mm=" not in runtime_source
    assert "UsdPhysics.FilteredPairsAPI.Apply(lever_prim)" in runtime_source
    assert "filtered_pairs.CreateFilteredPairsRel().AddTarget(table_path)" in runtime_source
    assert '"panda_hand"' in runtime_source
    assert '"panda_leftfinger"' in runtime_source
    assert '"panda_rightfinger"' in runtime_source
    assert "UsdPhysics.FilteredPairsAPI.Apply(\n                    gripper_prim" in runtime_source
    camera = CameraProfile()
    assert camera.table_position_m == (1.30, 0.00, 0.69)
    assert camera.table_focal_length_mm == 24.0


def test_faucet_valve_reuses_the_source_sink_lever_on_the_detent_joint():
    valve_asset = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "lightwheel_kitchen"
        / "Sink054"
        / "faucet_detent_valve.usda"
    ).read_text(encoding="utf-8")
    assert '@./Sink054.usd@</root/Sink054_handle>' in valve_asset
    assert 'def Xform "lever"' in valve_asset
    assert 'def Mesh "SourceLeverVisual"' in valve_asset
    assert 'double sourceLeverYawDeg = -90' in valve_asset
    assert 'float3 xformOp:rotateXYZ = (0, 0, -90)' in valve_asset
    assert 'delete apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysicsRigidBodyAPI"]' in valve_asset
    assert SPEC.lever_pivot_offset_m == (0.01, 0.0, -0.01)
    assert (
        'double3 xformOp:translate = (0.01, 0, -0.01)'
        in valve_asset
    )
    assert (
        'point3f physics:localPos0 = (0.01, 0, -0.01)'
        in valve_asset
    )
    assert 'def Cylinder "LeverRodCollider"' in valve_asset
    assert 'double radius = 0.0052' in valve_asset
    assert 'double3 xformOp:translate = (-0.00009, -0.01165, 0.059)' in valve_asset
    assert 'def Cube "LeverCollider"' not in valve_asset
    assert 'def Cylinder "ConnectorShaftVisual"' in valve_asset
    assert 'double height = 0.12779866876392887' in valve_asset
    assert (
        'double3 xformOp:translate = (0.0101355, 0.063898, -0.00960991)'
        in valve_asset
    )
    assert (
        'float3 xformOp:rotateXYZ = (0.349779065393, 0, -0.121499365973)'
        in valve_asset
    )
    assert 'def Xform "knob"' not in valve_asset
    assert 'def Cylinder "BaseVisual"' not in valve_asset
    assert 'def Cylinder "MountCollarVisual"' not in valve_asset
    assert 'def PhysicsRevoluteJoint "valve_joint"' in valve_asset
    assert 'uniform token physics:axis = "Y"' in valve_asset
    assert 'quatf physics:localRot0 = (0, 1, 0, 0)' in valve_asset
    assert 'quatf physics:localRot1 = (0, 1, 0, 0)' in valve_asset
    assert SPEC.lever_rotation_sign == -1.0
    assert 'float physics:upperLimit = 110' in valve_asset


def test_faucet_robot_base_support_uses_a_uniform_gray_material():
    support_asset = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "lightwheel_kitchen"
        / "Sink054"
        / "robot_base_counter_extension.usda"
    ).read_text(encoding="utf-8")
    assert 'def Material "GrayCountertop"' in support_asset
    assert 'color3f inputs:diffuseColor = (0.38, 0.40, 0.42)' in support_asset
    assert (
        'rel material:binding = </RobotBaseCounterExtension/GrayCountertop>'
        in support_asset
    )
    assert "UsdUVTexture" not in support_asset
    assert "Kitchen_InsularShelf_BC001.jpg" not in support_asset


def test_faucet_layout_exposes_a_tunable_continuous_load_profile():
    layout = sample_initial_state(3)
    assert layout == sample_initial_state(3)
    assert layout != sample_initial_state(4)
    assert SPEC.countertop_z_m == 0.0
    support_top_z = (
        SPEC.robot_support_position_m[2] + SPEC.robot_support_size_m[2] / 2.0
    )
    support_min_x = (
        SPEC.robot_support_position_m[0] - SPEC.robot_support_size_m[0] / 2.0
    )
    support_max_x = (
        SPEC.robot_support_position_m[0] + SPEC.robot_support_size_m[0] / 2.0
    )
    assert support_top_z == pytest.approx(SPEC.countertop_z_m)
    assert support_min_x < -0.15
    assert support_max_x > 0.18
    assert SPEC.sink_translation_m == (0.45, 0.33, -0.049304)
    assert SPEC.valve_position_m == (0.653, 0.015, -0.037)
    valve_pivot = tuple(
        base + offset
        for base, offset in zip(SPEC.valve_position_m, SPEC.lever_pivot_offset_m)
    )
    assert valve_pivot == pytest.approx((0.663, 0.015, -0.047))
    assert SPEC.expert_grasp_radius_m == pytest.approx(0.090)
    assert SPEC.lever_rod_centerline_y_offset_m == pytest.approx(
        -0.013866420774313262
    )
    valve_pipe_clearance = abs(SPEC.water_outlet_position_m[1] - valve_pivot[1])
    assert valve_pipe_clearance == pytest.approx(0.127796)
    assert valve_pipe_clearance >= SPEC.minimum_valve_pipe_clearance_m
    assert SPEC.sink_yaw_deg == -90.0
    assert layout["phase"] == "interactive_detent"
    assert layout["valve_load_profile"]["damping_nm_s_rad"] == SPEC.damping_nm_s_rad
    assert SPEC.detent_center_range_deg[0] <= layout["valve_load_profile"]["detent_center_deg"] <= SPEC.detent_center_range_deg[1]
    sampled_peak = layout["valve_load_profile"]["detent_peak_torque_nm"]
    assert sampled_peak == pytest.approx(SPEC.detent_peak_torque_nm)
    assert layout["randomization"]["detent_peak_torque_randomized"] is False
    sink_offset = [
        sampled - base
        for sampled, base in zip(layout["sink_translation_m"], SPEC.sink_translation_m)
    ]
    assert sink_offset == pytest.approx([0.0, 0.0, 0.0])
    assert layout["randomization"]["geometry_randomized"] is False
    valve_offset = [
        sampled - base
        for sampled, base in zip(layout["valve_position_m"], SPEC.valve_position_m)
    ]
    assert valve_offset == pytest.approx(sink_offset)
    water_offset = [
        sampled - base
        for sampled, base in zip(
            layout["water_stream_center_m"], SPEC.water_stream_center_m
        )
    ]
    assert water_offset == pytest.approx(sink_offset)
    episode_spec = spec_from_layout(layout)
    assert episode_spec.valve_position_m == pytest.approx(layout["valve_position_m"])
    assert episode_spec.water_outlet_position_m == pytest.approx(
        layout["water_outlet_position_m"]
    )
    assert episode_spec.water_stream_center_m == pytest.approx(
        layout["water_stream_center_m"]
    )
    episode_pivot_y = (
        episode_spec.valve_position_m[1] + episode_spec.lever_pivot_offset_m[1]
    )
    assert abs(
        episode_spec.water_outlet_position_m[1] - episode_pivot_y
    ) == pytest.approx(valve_pipe_clearance)
    sequence = layout["expert_detent_sequence"]
    assert sequence["rise_start_angle_deg"] == pytest.approx(
        episode_spec.detent_center_deg - episode_spec.detent_rise_width_deg
    )
    assert sequence["critical_peak_angle_deg"] == pytest.approx(
        episode_spec.detent_center_deg
    )
    assert sequence["drop_end_angle_deg"] == pytest.approx(
        episode_spec.detent_center_deg + episode_spec.detent_drop_width_deg
    )
    assert sequence["low_load_end_angle_deg"] == pytest.approx(
        sequence["drop_end_angle_deg"] + episode_spec.post_drop_low_width_deg
    )
    assert sequence["post_drop_rise_end_angle_deg"] == pytest.approx(
        sequence["low_load_end_angle_deg"] + episode_spec.post_drop_rise_width_deg
    )
    assert sequence["overshoot_target_deg"] == pytest.approx(
        episode_spec.detent_center_deg + episode_spec.expert_overshoot_after_peak_deg
    )
    assert sequence["backoff_target_deg"] > episode_spec.detent_center_deg
    assert (
        sequence["rise_start_angle_deg"]
        < episode_spec.detent_center_deg
        < sequence["drop_end_angle_deg"]
        < sequence["backoff_target_deg"]
        < sequence["overshoot_target_deg"]
        < sequence["low_load_end_angle_deg"]
        < sequence["post_drop_rise_end_angle_deg"]
    )
    assert sequence["load_drop_threshold_nm"] == pytest.approx(
        episode_spec.expert_load_drop_fraction
        * (
            episode_spec.detent_peak_torque_nm
            - episode_spec.base_friction_torque_nm
        )
    )


def test_detent_resistance_rises_drops_into_a_low_zone_then_rises_again():
    rise_start_deg = SPEC.detent_center_deg - SPEC.detent_rise_width_deg
    drop_end_deg = SPEC.detent_center_deg + SPEC.detent_drop_width_deg
    low_end_deg = drop_end_deg + SPEC.post_drop_low_width_deg
    post_rise_end_deg = low_end_deg + SPEC.post_drop_rise_width_deg
    angles_deg = [
        index / 100.0 for index in range(0, round(100.0 * SPEC.open_limit_deg) + 1)
    ]
    samples = [
        detent_resistance_nm(math.radians(angle_deg)) for angle_deg in angles_deg
    ]
    peak_index = max(range(len(samples)), key=samples.__getitem__)
    assert angles_deg[peak_index] == pytest.approx(SPEC.detent_center_deg)
    assert samples[peak_index] == pytest.approx(SPEC.detent_peak_torque_nm)
    assert detent_resistance_nm(math.radians(0.0)) == pytest.approx(
        SPEC.base_friction_torque_nm
    )
    assert detent_resistance_nm(math.radians(drop_end_deg)) == pytest.approx(
        SPEC.base_friction_torque_nm
    )
    assert detent_resistance_nm(math.radians(low_end_deg)) == pytest.approx(
        SPEC.base_friction_torque_nm
    )
    assert detent_resistance_nm(math.radians(post_rise_end_deg)) == pytest.approx(
        SPEC.post_drop_resistance_nm
    )
    rising = [
        detent_resistance_nm(math.radians(value))
        for value in (rise_start_deg, rise_start_deg + 2.5, rise_start_deg + 5.0,
                      rise_start_deg + 7.5, SPEC.detent_center_deg)
    ]
    falling = [
        detent_resistance_nm(math.radians(value))
        for value in (
            SPEC.detent_center_deg,
            SPEC.detent_center_deg + 0.25 * SPEC.detent_drop_width_deg,
            SPEC.detent_center_deg + 0.50 * SPEC.detent_drop_width_deg,
            SPEC.detent_center_deg + 0.75 * SPEC.detent_drop_width_deg,
            drop_end_deg,
        )
    ]
    assert rising == sorted(rising)
    assert falling == sorted(falling, reverse=True)
    post_drop_rising = [
        detent_resistance_nm(math.radians(value))
        for value in (
            low_end_deg,
            low_end_deg + 1.0,
            low_end_deg + 2.0,
            low_end_deg + 3.0,
            post_rise_end_deg,
        )
    ]
    assert post_drop_rising == sorted(post_drop_rising)
    assert SPEC.detent_drop_width_deg < SPEC.detent_rise_width_deg
    # The 0.6-degree snap-through is intentionally fast, but its C1 easing
    # still bounds every 0.01-degree sample-to-sample change.
    assert max(abs(right - left) for left, right in zip(samples, samples[1:])) < 0.005


def test_effort_opposes_velocity_without_a_static_torque_jump():
    center = math.radians(SPEC.detent_center_deg)
    assert detent_effort_nm(center, 0.0) == pytest.approx(0.0)
    assert detent_effort_nm(center, 1.0) < 0.0
    assert detent_effort_nm(center, -1.0) > 0.0
    assert abs(detent_effort_nm(center, 100.0)) <= SPEC.max_effort_nm


def test_flow_is_off_before_detent_then_decreases_after_maximum():
    assert SPEC.detent_center_range_deg == (20.0, 30.0)
    assert SPEC.detent_center_deg == pytest.approx(25.0)
    assert flow_fraction(math.radians(SPEC.detent_center_deg - 1.0)) == 0.0
    assert flow_fraction(math.radians(SPEC.detent_center_deg)) == pytest.approx(1.0)
    midpoint_deg = 0.5 * (SPEC.detent_center_deg + SPEC.open_limit_deg)
    assert flow_fraction(math.radians(midpoint_deg)) == pytest.approx(
        0.5 * (1.0 + SPEC.minimum_flow_fraction)
    )
    assert flow_fraction(math.radians(SPEC.open_limit_deg)) == pytest.approx(
        SPEC.minimum_flow_fraction
    )
    assert not evaluate_angle(math.radians(SPEC.success_angle_deg - 0.01))["success"]
    assert evaluate_angle(math.radians(SPEC.success_angle_deg))["success"]
