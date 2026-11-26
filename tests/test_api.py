"""API tests with FastAPI TestClient. A tiny model is built inside the test,
so no CSV and no saved model file is needed."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from src.tiffin.model import build_pipeline

NUM = ["distance_km", "tiffins", "handovers", "wait_at_station_min", "rider_age", "temp_c"]
CAT = ["transport", "route_zone"]

VALID = {
    "distance_km": 10.0, "tiffins": 30, "handovers": 3, "wait_at_station_min": 8.5,
    "rider_age": 35, "temp_c": 30.0, "transport": "scooter", "route_zone": "Central",
}


@pytest.fixture()
def client(monkeypatch):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "distance_km": rng.uniform(1, 20, n), "tiffins": rng.integers(10, 60, n),
        "handovers": rng.integers(1, 6, n), "wait_at_station_min": rng.uniform(0, 20, n),
        "rider_age": rng.integers(20, 60, n), "temp_c": rng.uniform(10, 40, n),
        "transport": rng.choice(["bicycle", "scooter", "local_train"], n),
        "route_zone": rng.choice(["Central", "Suburb", "Old City"], n),
    })
    y = 3 * df["distance_km"] + df["wait_at_station_min"]
    model = build_pipeline("ridge", {"alpha": 1.0}, NUM, CAT).fit(df, y)
    monkeypatch.setattr(main, "get_model", lambda: model)
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_ok(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


def test_temp_above_75_rejected(client):
    assert client.post("/predict", json={**VALID, "temp_c": 76}).status_code == 422


def test_negative_distance_rejected(client):
    assert client.post("/predict", json={**VALID, "distance_km": -1}).status_code == 422


def test_missing_field_rejected(client):
    bad = {k: v for k, v in VALID.items() if k != "transport"}
    assert client.post("/predict", json=bad).status_code == 422


def test_prediction_logged_as_json_without_personal_data(client, capsys):
    client.post("/predict", json=VALID)
    line = [l for l in capsys.readouterr().out.splitlines() if l.strip()][-1]
    entry = json.loads(line)
    assert entry["event"] == "predict" and "prediction" in entry
    assert "rider_age" not in entry
