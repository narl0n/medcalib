"""FastAPI service exposing the calibrated risk model."""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import model as model_lib

DISCLAIMER = "Research prototype trained on synthetic data. Not for clinical use."
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["bundle"] = model_lib.load()
    yield
    STATE.clear()


app = FastAPI(
    title="MedCalib API",
    version=model_lib.MODEL_VERSION,
    description="Calibrated early-Alzheimer's risk estimation (demo service).",
    lifespan=lifespan,
)


class Patient(BaseModel):
    age: float = Field(..., ge=40, le=110, description="Age in years")
    mmse: float = Field(..., ge=0, le=30, description="MMSE score")
    hippocampal_volume: float = Field(..., gt=0, le=15, description="ml")
    entorhinal_thickness: float = Field(..., gt=0, le=6, description="mm")
    apoe4_carrier: bool

    def as_row(self) -> list[float]:
        return [
            self.age,
            self.mmse,
            self.hippocampal_volume,
            self.entorhinal_thickness,
            float(self.apoe4_carrier),
        ]


class Prediction(BaseModel):
    risk: float
    risk_band: str
    model_version: str
    disclaimer: str


def risk_band(p: float) -> str:
    if p < 0.2:
        return "low"
    if p < 0.5:
        return "moderate"
    return "high"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "bundle" in STATE}


@app.get("/model")
def model_info() -> dict:
    bundle = STATE["bundle"]
    return {"features": list(bundle["features"]), "report": bundle["report"]}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient) -> Prediction:
    p = float(model_lib.predict_proba(STATE["bundle"], np.array(patient.as_row()))[0])
    return Prediction(
        risk=round(p, 4),
        risk_band=risk_band(p),
        model_version=model_lib.MODEL_VERSION,
        disclaimer=DISCLAIMER,
    )
