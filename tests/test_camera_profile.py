from pathlib import Path

from coribench.task_api import CameraProfile


def test_default_main_camera_is_a_centered_workspace_overview():
    profile = CameraProfile()
    assert profile.table_position_m == (1.30, 0.00, 0.69)
    assert profile.table_target_m == (0.55732250, 0.00, 0.02035074)
    assert profile.table_rotation_wxyz == (
        0.64607454,
        0.28738073,
        0.28738073,
        0.64607454,
    )
    assert profile.table_focal_length_mm == 24.0
    assert profile.wrist_image_rotation_deg == 0
    # Keep the camera elevated and beyond the workbench; the wrist camera owns
    # the close-up view.  y=0 makes the overview exactly front-aligned.
    assert profile.table_position_m[2] >= 0.6
    assert profile.table_position_m[0] >= 1.2
    assert profile.table_position_m[1] == 0.0


def test_weight_sort_keeps_the_28cm_main_camera_push_in():
    runtime_source = Path("coribench/tasks/weight_sort/runtime.py").read_text(
        encoding="utf-8"
    )
    assert "camera_profile = CameraProfile(" in runtime_source
    assert "table_position_m=(1.09205030, 0.0, 0.50249821)" in runtime_source


def test_peg_and_nut_use_the_tight_224px_main_camera_view():
    expected = (
        "table_position_m=(1.05856784, 0.0, 0.53963208)",
        "table_target_m=(0.52384004, 0.0, 0.05748461)",
        "table_focal_length_mm=40.0",
    )
    for task_name in ("peg_insert", "nut_thread"):
        runtime_source = (
            Path("coribench/tasks") / task_name / "runtime.py"
        ).read_text(encoding="utf-8")
        assert "camera_profile = CameraProfile(" in runtime_source
        for camera_field in expected:
            assert camera_field in runtime_source
