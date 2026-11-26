"""FastAPI service for MP160 (Tiffin Relay delivery time).

Run locally:  uvicorn app.main:app --reload
"""

import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.logging_config import get_logger

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FEATURES = [
    "distance_km", "tiffins", "handovers", "wait_at_station_min",
    "rider_age", "temp_c", "transport", "route_zone",
]

app = FastAPI(title="MP160 Tiffin Relay Delivery Time API", version="1.0")
logger = get_logger()
_model = None


class PredictRequest(BaseModel):
    distance_km: float = Field(..., ge=0)    # client rule 2: below 0 is a typing mistake
    tiffins: int = Field(..., ge=0)
    handovers: int = Field(..., ge=0)
    wait_at_station_min: float = Field(..., ge=0)
    rider_age: float = Field(..., ge=0)
    temp_c: float = Field(..., le=75)        # client rule 1: above 75 is impossible
    transport: str
    route_zone: str


class PredictResponse(BaseModel):
    prediction: float


def get_model():
    """Load the champion model once (models/model.joblib)."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    value = float(get_model().predict(row)[0])
    # one JSON line per prediction; rider_age (personal detail) is NOT logged
    logger.info("prediction", extra={"fields": {
        "event": "predict",
        "distance_km": req.distance_km,
        "tiffins": req.tiffins,
        "handovers": req.handovers,
        "transport": req.transport,
        "route_zone": req.route_zone,
        "prediction": round(value, 3),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }})
    return PredictResponse(prediction=value)
