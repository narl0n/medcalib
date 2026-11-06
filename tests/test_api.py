import pytest
from fastapi.testclient import TestClient

from medcalib_api.app import app, risk_band

PATIENT = {
    "age": 74,
    "mmse": 24,
    "hippocampal_volume": 6.5,
    "entorhinal_thickness": 3.1,
    "apoe4_carrier": True,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model").json()
    assert body["features"][1] == "mmse"
    assert "ece_calibrated" in body["report"]


def test_predict_returns_calibrated_probability(client):
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "high"}
    assert "Not for clinical use" in body["disclaimer"]


@pytest.mark.parametrize("field, value", [("mmse", 31), ("age", 10), ("hippocampal_volume", -1)])
def test_predict_rejects_out_of_range_input(client, field, value):
    r = client.post("/predict", json={**PATIENT, field: value})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client):
    data = {k: v for k, v in PATIENT.items() if k != "mmse"}
    assert client.post("/predict", json=data).status_code == 422


@pytest.mark.parametrize("p, band", [(0.05, "low"), (0.3, "moderate"), (0.8, "high")])
def test_risk_band(p, band):
    assert risk_band(p) == band
