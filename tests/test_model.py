"""Tests for the model pipeline on tiny synthetic data."""

import numpy as np
import pandas as pd

from src.tiffin.model import build_pipeline, mae

NUM = ["distance_km", "wait"]
CAT = ["transport"]


def _df(n=40):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"distance_km": rng.uniform(1, 20, n),
                      "wait": rng.uniform(0, 15, n),
                      "transport": rng.choice(["bicycle", "scooter"], n)})
    d["y"] = 3 * d["distance_km"] + d["wait"]
    return d


def test_pipeline_fits_and_predicts_with_missing_values():
    d = _df()
    d.loc[0, "wait"] = np.nan  # imputer must cope
    pipe = build_pipeline("random_forest", {"n_estimators": 10, "random_state": 0}, NUM, CAT)
    pipe.fit(d[NUM + CAT], d["y"])
    assert pipe.predict(d[NUM + CAT]).shape == (len(d),)


def test_pipeline_ignores_unseen_category():
    d = _df()
    pipe = build_pipeline("ridge", {"alpha": 1.0}, NUM, CAT)
    pipe.fit(d[NUM + CAT], d["y"])
    new = d.iloc[:2].copy()
    new["transport"] = "hoverboard"
    assert len(pipe.predict(new[NUM + CAT])) == 2


def test_mae_value():
    assert mae(pd.Series([1.0, 2.0]), [2.0, 4.0]) == 1.5
