"""Simulator-independent quality gates for persistent expert episodes."""

from __future__ import annotations

from typing import Any

import numpy as np


WEIGHT_SORT_MAIN_BLUE_FRACTION_RANGE = (0.012, 0.035)
WEIGHT_SORT_MAIN_MIN_MEAN_INTENSITY = 40.0
WEIGHT_SORT_MAIN_MAX_DARK_FRACTION = 0.52
# Seeded pickup positions legitimately move the box across a sizeable portion
# of the wrist view.  Reference differences are therefore only a guard for the
# giant stale-container modes; the tighter mild-blue intrusion check comes from
# the absolute main-camera blue fraction above.
WEIGHT_SORT_MAIN_MAX_REFERENCE_CHANGE_FRACTION = 0.08
WEIGHT_SORT_WRIST_MAX_REFERENCE_CHANGE_FRACTION = 0.25


def _rgb_image(value: Any, *, label: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} must be an HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"{label} must use uint8 pixels, got {image.dtype}")
    if image.shape[0] < 16 or image.shape[1] < 16:
        raise ValueError(f"{label} is too small for visual QC: {image.shape}")
    return image


def reference_change_fraction(image: Any, reference: Any) -> float:
    """Fraction of pixels whose maximum RGB-channel change exceeds 40."""

    current = _rgb_image(image, label="image")
    baseline = _rgb_image(reference, label="reference")
    if current.shape != baseline.shape:
        raise ValueError(
            f"image/reference shape mismatch: {current.shape} != {baseline.shape}"
        )
    difference = np.max(
        np.abs(current.astype(np.int16) - baseline.astype(np.int16)), axis=2
    )
    return float(np.mean(difference > 40))


def weight_sort_visual_metrics(
    main_image: Any,
    wrist_image: Any,
    *,
    reference_main: Any | None = None,
    reference_wrist: Any | None = None,
) -> dict[str, float | None]:
    """Measure the large blue/dark occlusions seen in corrupted weight_sort runs."""

    main = _rgb_image(main_image, label="main_image")
    wrist = _rgb_image(wrist_image, label="wrist_image")
    red = main[..., 0].astype(np.int16)
    green = main[..., 1].astype(np.int16)
    blue = main[..., 2].astype(np.int16)
    blue_mask = (blue > 80) & (blue > 1.25 * red) & (blue > 1.08 * green)
    dark_mask = np.max(main, axis=2) < 45
    return {
        "main_mean_intensity": float(np.mean(main)),
        "main_blue_fraction": float(np.mean(blue_mask)),
        "main_dark_fraction": float(np.mean(dark_mask)),
        "main_reference_change_fraction": (
            None
            if reference_main is None
            else reference_change_fraction(main, reference_main)
        ),
        "wrist_reference_change_fraction": (
            None
            if reference_wrist is None
            else reference_change_fraction(wrist, reference_wrist)
        ),
    }


def weight_sort_visual_errors(metrics: dict[str, float | None]) -> tuple[str, ...]:
    """Return fail-closed reasons for a weight_sort reset probe."""

    errors: list[str] = []
    blue_fraction = float(metrics["main_blue_fraction"])
    blue_min, blue_max = WEIGHT_SORT_MAIN_BLUE_FRACTION_RANGE
    if not blue_min <= blue_fraction <= blue_max:
        errors.append(
            f"main blue fraction {blue_fraction:.4f} outside [{blue_min:.4f}, {blue_max:.4f}]"
        )
    mean_intensity = float(metrics["main_mean_intensity"])
    if mean_intensity < WEIGHT_SORT_MAIN_MIN_MEAN_INTENSITY:
        errors.append(
            f"main mean intensity {mean_intensity:.2f} below "
            f"{WEIGHT_SORT_MAIN_MIN_MEAN_INTENSITY:.2f}"
        )
    dark_fraction = float(metrics["main_dark_fraction"])
    if dark_fraction > WEIGHT_SORT_MAIN_MAX_DARK_FRACTION:
        errors.append(
            f"main dark fraction {dark_fraction:.4f} above "
            f"{WEIGHT_SORT_MAIN_MAX_DARK_FRACTION:.4f}"
        )
    main_change = metrics["main_reference_change_fraction"]
    if (
        main_change is not None
        and float(main_change) > WEIGHT_SORT_MAIN_MAX_REFERENCE_CHANGE_FRACTION
    ):
        errors.append(
            f"main reference change {float(main_change):.4f} above "
            f"{WEIGHT_SORT_MAIN_MAX_REFERENCE_CHANGE_FRACTION:.4f}"
        )
    wrist_change = metrics["wrist_reference_change_fraction"]
    if (
        wrist_change is not None
        and float(wrist_change) > WEIGHT_SORT_WRIST_MAX_REFERENCE_CHANGE_FRACTION
    ):
        errors.append(
            f"wrist reference change {float(wrist_change):.4f} above "
            f"{WEIGHT_SORT_WRIST_MAX_REFERENCE_CHANGE_FRACTION:.4f}"
        )
    return tuple(errors)
