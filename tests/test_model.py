"""Model quality gates: the build fails if the trained model gets worse."""

import numpy as np
import pytest

from medcalib_api import model as model_lib


@pytest.fixture(scope="module")
def trained():
    return model_lib.train(seed=42)


def test_discrimination_gate(trained):
    _, report = trained
    assert report.auc >= 0.85


def test_calibration_gate(trained):
    _, report = trained
    assert report.ece_calibrated <= 0.07
    assert report.ece_calibrated < report.ece_uncalibrated
    assert report.brier_calibrated < report.brier_uncalibrated


def test_training_is_deterministic():
    _, r1 = model_lib.train(seed=1, n=800)
    _, r2 = model_lib.train(seed=1, n=800)
    assert r1 == r2


def test_risk_increases_with_lower_mmse(trained):
    clf, report = trained
    bundle = {"model": clf, "report": report}
    healthy = np.array([70, 29, 7.4, 3.5, 0])
    impaired = np.array([70, 21, 6.2, 2.9, 0])
    p_healthy, p_impaired = model_lib.predict_proba(bundle, np.vstack([healthy, impaired]))
    assert 0 <= p_healthy < p_impaired <= 1


def test_save_and_load_roundtrip(tmp_path, trained):
    clf, report = trained
    path = tmp_path / "model.joblib"
    model_lib.save(clf, report, path)
    bundle = model_lib.load(path)
    assert bundle["report"]["model_version"] == model_lib.MODEL_VERSION
    assert bundle["features"] == model_lib.FEATURES
