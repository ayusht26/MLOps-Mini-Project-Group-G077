"""Tests for the cleaning code. Synthetic data only: CSVs are not in git."""

import numpy as np
import pandas as pd

from src.tiffin.clean import blank_stuck_sensor, clean_train, drop_leak, fix_unit_mixup

PARAMS = {
    "audit": {"date_column": "date"},
    "cleaning": {
        "leak_column": "leak",
        "unit_fix": {"column": "wait", "group_column": "zone",
                     "group_value": "Central", "factor": 60},
        "stuck_sensor": {"column": "age", "value": 33,
                         "start_date": "2026-03-08", "end_date": "2026-03-29"},
    },
}


def _df():
    return pd.DataFrame({
        "date": ["2026-03-10", "2026-03-11", "2026-04-01", "2026-03-12"],
        "zone": ["Central", "Suburb", "Central", "Suburb"],
        "wait": [600.0, 10.0, 1200.0, 12.0],
        "age": [33, 33, 33, 40],
        "leak": [1.0, 2.0, 3.0, 4.0],
    })


def test_drop_leak_removes_column():
    assert "leak" not in drop_leak(_df(), "leak").columns


def test_unit_mixup_only_changes_central():
    out = fix_unit_mixup(_df(), PARAMS["cleaning"]["unit_fix"])
    assert out.loc[0, "wait"] == 10.0
    assert out.loc[1, "wait"] == 10.0  # Suburb untouched


def test_stuck_sensor_blanks_only_run():
    out = blank_stuck_sensor(_df(), PARAMS["cleaning"]["stuck_sensor"])
    assert out["age"].isna().tolist() == [True, True, False, False]


def test_clean_train_full_and_removes_duplicates():
    df = pd.concat([_df(), _df().iloc[[0]]], ignore_index=True)
    out = clean_train(df, PARAMS)
    assert len(out) == 4 and "leak" not in out.columns
    assert out["date"].is_monotonic_increasing
