"""Synthetic multimodal cohort used to train and test the demo model.

Real patient data (OASIS, ADNI, partner clinic) cannot be shipped inside a
public repository or a Docker image, so the service is trained on a seeded
synthetic cohort whose features follow plausible clinical ranges. The
generator is deterministic for a given seed, which makes every build and
every test run reproducible.
"""

from __future__ import annotations

import numpy as np

FEATURES: tuple[str, ...] = (
    "age",  # years
    "mmse",  # Mini-Mental State Examination, 0-30
    "hippocampal_volume",  # ml, both hemispheres, normalised to head size
    "entorhinal_thickness",  # mm
    "apoe4_carrier",  # 0 / 1
)


def make_cohort(n: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return a feature matrix X (n x 5) and binary labels y (1 = early AD / MCI converter)."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)

    age = rng.normal(72, 7, n).clip(55, 95)
    apoe4 = rng.binomial(1, 0.3, n)

    # Latent disease severity drives every observed marker.
    latent = 0.08 * (age - 72) + 0.9 * apoe4 + rng.normal(0, 1, n)

    mmse = (28.5 - 2.2 * latent + rng.normal(0, 1.2, n)).clip(10, 30)
    hippo = (7.2 - 0.45 * latent + rng.normal(0, 0.35, n)).clip(4.0, 9.5)
    entorhinal = (3.4 - 0.2 * latent + rng.normal(0, 0.18, n)).clip(2.0, 4.5)

    prob = 1.0 / (1.0 + np.exp(-(2.4 * latent - 1.0)))
    y = rng.binomial(1, prob)

    X = np.column_stack([age, mmse, hippo, entorhinal, apoe4]).astype(float)
    return X, y.astype(int)
