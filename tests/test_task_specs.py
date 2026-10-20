import numpy as np

from forcebench.tasks.peg_insert.spec import SPEC, evaluate_final_state, sample_initial_state


def test_seeded_layout_is_deterministic_and_separated():
    first = sample_initial_state(42)
    second = sample_initial_state(42)
    other = sample_initial_state(43)
    assert first == second
    assert first != other
    assert first["initial_separation_m"] >= SPEC.minimum_initial_separation_m
    hole = np.asarray(first["target_hole_position_m"][:2])
    pickup = np.asarray(first["pickup_hole_position_m"][:2])
    assert np.linalg.norm(hole - pickup) >= SPEC.minimum_initial_separation_m
    assert first["peg_upright_yaw_rad"] == 0.0
    assert first["target_hole_yaw_rad"] == 0.0
    assert other["peg_upright_yaw_rad"] == 0.0
    assert other["target_hole_yaw_rad"] == 0.0


def test_target_never_overlaps_pickup_socket_across_many_seeds():
    pickup = np.asarray([SPEC.pickup_center_x_m, SPEC.pickup_center_y_m])
    realized_targets = set()
    for seed in range(10_000):
        layout = sample_initial_state(seed)
        target = np.asarray(layout["target_hole_position_m"][:2])
        assert layout["pickup_hole_position_m"] == [
            SPEC.pickup_center_x_m,
            SPEC.pickup_center_y_m,
            SPEC.hole_root_z_m,
        ]
        assert layout["peg_position_m"] == [
            SPEC.pickup_center_x_m,
            SPEC.pickup_center_y_m,
            SPEC.upright_peg_root_z_m,
        ]
        realized_targets.add(tuple(layout["target_hole_position_m"][:2]))
        assert np.linalg.norm(target - pickup) >= SPEC.minimum_initial_separation_m
    assert len(realized_targets) > 1


def test_contact_profile_prioritizes_grasp_without_jamming_socket():
    assert SPEC.friction_combine_mode == "min"
    effective_grasp_friction = min(SPEC.gripper_friction, SPEC.peg_friction)
    assert 1.0 <= effective_grasp_friction <= 2.0
    assert min(SPEC.peg_friction, SPEC.hole_friction) <= 0.01
    assert SPEC.upright_peg_root_z_m > SPEC.hole_root_z_m
    assert 0.008 < SPEC.pickup_support_width_m < 0.02
    assert 0.15 <= SPEC.start_tcp_position_m[2] <= 0.22


def test_success_requires_insert_release_and_retract():
    kwargs = dict(
        peg_position_m=[0.62, 0.01, SPEC.hole_mate_height_m],
        peg_axis_world=[0.0, 0.0, 1.0],
        target_hole_position_m=[0.62, 0.01, SPEC.hole_mate_height_m],
        gripper_open_fraction=1.0,
        eef_position_m=[0.62, 0.01, 0.12],
    )
    assert evaluate_final_state(**kwargs).success
    assert not evaluate_final_state(**{**kwargs, "gripper_open_fraction": 0.0}).success
    assert not evaluate_final_state(**{**kwargs, "eef_position_m": [0.62, 0.01, 0.03]}).success
    assert not evaluate_final_state(
        **{**kwargs, "peg_position_m": [0.63, 0.01, SPEC.hole_mate_height_m]}
    ).success
    tilted = np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    assert not evaluate_final_state(**{**kwargs, "peg_axis_world": tilted}).success
