import numpy as np
import pytest

from medcalib_api.data import FEATURES, make_cohort
from medcalib_api.metrics import brier_score, expected_calibration_error


def test_ece_known_value():
    # A model that always says 0.9 but is right half the time: ECE = |0.5 - 0.9| = 0.4
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.9)
    assert expected_calibration_error(y, p) == pytest.approx(0.4)


def test_ece_perfectly_calibrated_sample_is_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p) < 0.02


def test_ece_handles_probability_one():
    assert expected_calibration_error([1, 1], [1.0, 1.0]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "y, p",
    [([1, 0], [0.5]), ([], []), ([1], [1.5]), ([[1]], [[0.5]])],
)
def test_ece_rejects_invalid_input(y, p):
    with pytest.raises(ValueError):
        expected_calibration_error(y, p)


def test_brier_score_known_value():
    assert brier_score([1, 0], [0.8, 0.2]) == pytest.approx(0.04)


def test_cohort_is_reproducible_and_well_formed():
    X1, y1 = make_cohort(500, seed=7)
    X2, y2 = make_cohort(500, seed=7)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert X1.shape == (500, len(FEATURES))
    assert set(np.unique(y1)) <= {0, 1}
    assert X1[:, 1].min() >= 10 and X1[:, 1].max() <= 30  # MMSE range


def test_cohort_rejects_non_positive_size():
    with pytest.raises(ValueError):
        make_cohort(0)
