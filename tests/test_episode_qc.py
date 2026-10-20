import numpy as np
import pytest

from forcebench.episode_qc import (
    reference_change_fraction,
    weight_sort_visual_errors,
    weight_sort_visual_metrics,
)


def valid_main_image(size=100):
    image = np.full((size, size, 3), 55, dtype=np.uint8)
    # The retained 224 px camera's reset probe has ~6% blue container pixels.
    # This synthetic fixture exercises QC only, not simulator appearance.
    image[:20, :30] = (20, 40, 150)
    return image


def test_weight_sort_visual_gate_accepts_expected_reset_probe():
    main = valid_main_image()
    wrist = np.full((50, 50, 3), 60, dtype=np.uint8)
    metrics = weight_sort_visual_metrics(main, wrist)
    assert metrics["main_blue_fraction"] == pytest.approx(0.060)
    assert weight_sort_visual_errors(metrics) == ()


@pytest.mark.parametrize("color", ((0, 0, 0), (10, 50, 220)))
def test_weight_sort_visual_gate_rejects_large_occlusions(color):
    main = np.full((100, 100, 3), color, dtype=np.uint8)
    wrist = np.full((50, 50, 3), 60, dtype=np.uint8)
    assert weight_sort_visual_errors(weight_sort_visual_metrics(main, wrist))


def test_weight_sort_visual_gate_rejects_reference_scene_changes():
    reference = valid_main_image()
    changed = reference.copy()
    changed[30:60, :30] = 255
    wrist = np.full((50, 50, 3), 60, dtype=np.uint8)
    metrics = weight_sort_visual_metrics(
        changed,
        wrist,
        reference_main=reference,
        reference_wrist=wrist,
    )
    assert reference_change_fraction(changed, reference) == pytest.approx(0.09)
    assert any("main reference change" in error for error in weight_sort_visual_errors(metrics))


def test_weight_sort_visual_gate_allows_seeded_box_motion():
    reference = valid_main_image()
    changed = reference.copy()
    changed[30:50, :20] = 255
    wrist = np.full((50, 50, 3), 60, dtype=np.uint8)
    metrics = weight_sort_visual_metrics(
        changed,
        wrist,
        reference_main=reference,
        reference_wrist=wrist,
    )
    assert reference_change_fraction(changed, reference) == pytest.approx(0.04)
    assert weight_sort_visual_errors(metrics) == ()


def test_visual_metrics_require_uint8_hwc_rgb():
    with pytest.raises(TypeError, match="uint8"):
        weight_sort_visual_metrics(
            np.zeros((32, 32, 3), dtype=np.float32),
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
