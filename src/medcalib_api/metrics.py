"""Calibration metrics for binary classifiers."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins on [0, 1].

    ECE = sum_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|, where the
    "accuracy" of a bin is the observed fraction of positives and the
    "confidence" is the mean predicted probability.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_prob must be 1-D arrays of the same length")
    if y_true.size == 0:
        raise ValueError("inputs must not be empty")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    # Bin index; probability 1.0 belongs to the last bin.
    idx = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    return float(np.mean((y_prob - y_true) ** 2))
