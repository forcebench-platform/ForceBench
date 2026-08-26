import pytest

from coribench.scene_profiles import DEFAULT_SCENE_PROFILE, get_scene_profile, scene_profile_names


def test_contact_lab_is_the_default_and_only_standard_profiles_are_registered():
    assert DEFAULT_SCENE_PROFILE == "contact_lab"
    assert set(scene_profile_names()) == {
        "contact_lab",
        "weight_sort_residential",
        "parcel_sort_station",
        "gear_assembly_cell",
        "kitchen_sink",
        "kitchen_stove",
        "simple_room_drawer",
        "simple_room_sort_table",
    }


def test_contact_lab_uses_an_exact_height_native_table():
    profile = get_scene_profile("contact_lab")
    assert profile.room_usd_path is None
    assert profile.table_visual_usd_path is None
    assert profile.room_orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert profile.table_visual_translation_m == (0.0, 0.0, 0.0)
    assert profile.table_size_m == (1.35, 0.80, 0.05)
    assert profile.table_translation_m == (0.475, 0.0, -0.025)
    table_min_x = profile.table_translation_m[0] - profile.table_size_m[0] / 2.0
    table_top_z = profile.table_translation_m[2] + profile.table_size_m[2] / 2.0
    assert table_min_x == pytest.approx(-0.20)
    assert table_min_x <= -0.154086  # Franka link0 visual x minimum.
    assert table_top_z == pytest.approx(0.0)
    assert {box.name for box in profile.visual_boxes} >= {
        "floor",
        "rear_wall",
        "side_wall_left",
        "side_wall_right",
        "wall_art",
        "cabinet_left",
        "bench_front_apron",
        "bench_leg_front_left",
    }
    assert profile.table_color[0] > profile.table_color[1] > profile.table_color[2]
    assert profile.table_metallic == 0.0
    assert profile.table_visible is True
    assert profile.ground_visible is False


def test_visual_shell_is_collision_independent_by_construction():
    contact = get_scene_profile("contact_lab")
    assert contact.visual_boxes
    assert all(box.size_m for box in contact.visual_boxes)


def test_weight_sort_residential_uses_a_wide_camera_aligned_table():
    profile = get_scene_profile("weight_sort_residential")
    assert profile.table_visual_usd_path.endswith("EastRural_Table_weight_sort.usda")
    assert profile.table_size_m[1] > profile.table_size_m[0]
    assert profile.table_size_m[:2] == (1.10, 1.55)
    assert profile.table_translation_m[2] + profile.table_size_m[2] / 2.0 == pytest.approx(0.0)
    assert profile.table_visible is False
    assert all(not box.name.startswith("bench_") for box in profile.visual_boxes)


def test_parcel_sort_station_keeps_the_robot_support_geometry_invariant():
    profile = get_scene_profile("parcel_sort_station")
    assert profile.table_size_m == (1.35, 0.80, 0.05)
    assert profile.table_translation_m == (0.475, 0.0, -0.025)
    assert profile.table_translation_m[2] + profile.table_size_m[2] / 2.0 == pytest.approx(0.0)
    assert {box.name for box in profile.visual_boxes} >= {
        "parcel_floor",
        "parcel_rear_wall",
        "bollard_left",
    }
    assert profile.room_usd_path.endswith("small_warehouse_digital_twin/rack.usd")
    assert profile.room_scale == (0.65, 0.65, 0.65)
    assert profile.room_translation_m[0] < 0.2
    assert all(not box.name.startswith("conveyor_") for box in profile.visual_boxes)
    assert all(not box.name.startswith("rack_") for box in profile.visual_boxes)


def test_kitchen_sink_uses_lightwheel_asset_at_the_invariant_surface_height():
    profile = get_scene_profile("kitchen_sink")
    assert profile.room_usd_path.endswith("KitchenRoom_background.usda")
    assert profile.room_translation_m == pytest.approx(
        (-1.827888317449588, -1.0030593951112115, -0.89842848812593)
    )
    assert profile.table_visual_usd_path.endswith("Sink054_environment.usda")
    assert profile.table_visual_scale == (1.0, 1.0, 1.0)
    assert profile.table_visual_orientation_xyzw == (
        0.0,
        0.0,
        -0.70710678,
        0.70710678,
    )
    assert profile.table_visual_translation_m[1] == pytest.approx(0.33)
    assert profile.table_visual_translation_m[2] == pytest.approx(-0.049304)
    assert profile.table_translation_m[1] == pytest.approx(0.33)
    assert profile.table_translation_m[2] + profile.table_size_m[2] / 2.0 == pytest.approx(0.0)
    assert profile.table_visible is False
    assert profile.visual_boxes == ()


def test_kitchen_stove_reuses_faucet_room_and_preserves_its_native_table():
    faucet = get_scene_profile("kitchen_sink")
    profile = get_scene_profile("kitchen_stove")
    assert profile.room_usd_path == faucet.room_usd_path
    assert profile.room_translation_m == faucet.room_translation_m
    assert profile.room_orientation_xyzw == faucet.room_orientation_xyzw
    assert profile.room_scale == faucet.room_scale
    assert profile.hidden_room_prims == faucet.hidden_room_prims
    assert profile.visual_boxes == faucet.visual_boxes == ()
    assert profile.table_visual_usd_path is None
    assert profile.table_size_m == (1.10, 1.50, 0.05)
    assert profile.table_translation_m == (0.45, 0.10, -0.025)
    assert profile.table_translation_m[2] + profile.table_size_m[2] / 2.0 == pytest.approx(0.0)
    assert profile.table_visible is True


def test_simple_room_drawer_preserves_the_canonical_support_geometry():
    profile = get_scene_profile("simple_room_drawer")
    assert profile.room_usd_path is None
    assert profile.table_size_m == (1.35, 0.80, 0.05)
    assert profile.table_translation_m == (0.475, 0.0, -0.025)
    assert profile.table_translation_m[2] + profile.table_size_m[2] / 2.0 == pytest.approx(0.0)
    assert {box.name for box in profile.visual_boxes} >= {
        "simple_floor",
        "simple_rear_wall",
        "simple_window",
        "simple_bench_front",
    }
    assert profile.dome_light_intensity > 0.0


def test_simple_room_sort_table_reuses_the_box_sort_table():
    drawer = get_scene_profile("simple_room_sort_table")
    box_sort = get_scene_profile("weight_sort_residential")
    assert drawer.table_visual_usd_path == box_sort.table_visual_usd_path
    assert drawer.table_visual_translation_m == box_sort.table_visual_translation_m
    assert drawer.table_visual_orientation_xyzw == box_sort.table_visual_orientation_xyzw
    assert drawer.table_visual_scale == box_sort.table_visual_scale
    assert drawer.table_visual_collision_prim == box_sort.table_visual_collision_prim
    assert drawer.table_size_m == box_sort.table_size_m
    assert drawer.table_translation_m == box_sort.table_translation_m
    assert drawer.table_visible is False
    assert all(not box.name.startswith("simple_bench_") for box in drawer.visual_boxes)


def test_unknown_scene_profile_has_an_actionable_error():
    with pytest.raises(
        ValueError,
        match=(
                "contact_lab, weight_sort_residential, parcel_sort_station, "
                "gear_assembly_cell, kitchen_sink, kitchen_stove, "
                "simple_room_drawer, simple_room_sort_table"
        ),
    ):
        get_scene_profile("warehouse_everything")
