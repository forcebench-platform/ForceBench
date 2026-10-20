import numpy as np
import pytest

from forcebench.tasks.peg_insert.spec import evaluate_release_stability_batch


def test_release_gate_accepts_stable_contact_seats_per_environment():
    result = evaluate_release_stability_batch(
        [0.0004, 0.0015, 0.0008],
        [0.0042, -0.0055, 0.0000],
        [2.0, 8.0, 0.0],
    )
    np.testing.assert_array_equal(result, [True, True, True])


def test_release_gate_rejects_center_height_and_tilt_instability():
    result = evaluate_release_stability_batch(
        [0.0016, 0.0004, 0.0004],
        [0.0000, 0.0056, 0.0000],
        [0.0, 0.0, 8.1],
    )
    np.testing.assert_array_equal(result, [False, False, False])


@pytest.mark.parametrize(
    "center,height,tilt",
    [([0.0], [0.0, 0.0], [0.0]), ([0.0], [0.0], [[0.0]]), ([np.nan], [0.0], [0.0])],
)
def test_release_gate_rejects_malformed_or_nonfinite_inputs(center, height, tilt):
    with pytest.raises(ValueError):
        evaluate_release_stability_batch(center, height, tilt)
