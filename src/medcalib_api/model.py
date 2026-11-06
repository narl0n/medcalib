"""Training, calibration and persistence of the risk model."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from .data import FEATURES, make_cohort
from .metrics import brier_score, expected_calibration_error

MODEL_VERSION = "0.1.0"
DEFAULT_MODEL_PATH = Path(os.environ.get("MEDCALIB_MODEL_PATH", "artifacts/model.joblib"))


@dataclass
class TrainReport:
    model_version: str
    seed: int
    n_train: int
    n_test: int
    auc: float
    ece_uncalibrated: float
    ece_calibrated: float
    brier_uncalibrated: float
    brier_calibrated: float


def train(seed: int = 42, n: int = 3000):
    """Train Gaussian naive Bayes, calibrate it with isotonic regression, evaluate both."""
    X, y = make_cohort(n=n, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Naive Bayes ranks well but is over-confident when features are correlated,
    # which is exactly the failure mode calibration has to fix.
    base = GaussianNB()
    base.fit(X_train, y_train)
    p_raw = base.predict_proba(X_test)[:, 1]

    calibrated = CalibratedClassifierCV(
        GaussianNB(),
        method="isotonic",
        cv=5,
    )
    calibrated.fit(X_train, y_train)
    p_cal = calibrated.predict_proba(X_test)[:, 1]

    report = TrainReport(
        model_version=MODEL_VERSION,
        seed=seed,
        n_train=len(y_train),
        n_test=len(y_test),
        auc=round(float(roc_auc_score(y_test, p_cal)), 4),
        ece_uncalibrated=round(expected_calibration_error(y_test, p_raw), 4),
        ece_calibrated=round(expected_calibration_error(y_test, p_cal), 4),
        brier_uncalibrated=round(brier_score(y_test, p_raw), 4),
        brier_calibrated=round(brier_score(y_test, p_cal), 4),
    )
    return calibrated, report


def save(model, report: TrainReport, path: Path = DEFAULT_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "report": asdict(report), "features": FEATURES}, path)


def load(path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Load a saved bundle; train a fresh one if no artefact exists yet."""
    if path.exists():
        return joblib.load(path)
    model, report = train()
    return {"model": model, "report": asdict(report), "features": FEATURES}


def predict_proba(bundle: dict, rows: np.ndarray) -> np.ndarray:
    return bundle["model"].predict_proba(np.atleast_2d(rows))[:, 1]


def main() -> None:
    """Entry point used at image build time: `python -m medcalib_api.model`."""
    model, report = train()
    save(model, report)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
