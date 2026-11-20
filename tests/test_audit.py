"""Tests for data audit checks using synthetic data (no external CSVs needed)."""

import pandas as pd
import pytest

from src.tiffin.data_audit import (
    check_leaking_column,
    check_unit_mixup,
    check_sentinel_value,
    check_duplicate_rows,
    check_stuck_sensor,
    check_target_typos,
    check_spelling_variants,
)


def test_check_duplicate_rows():
    # Dataset without duplicates
    df_clean = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    res_clean = check_duplicate_rows(df_clean)
    assert res_clean["present"] is False
    assert res_clean["count"] is None

    # Dataset with duplicates
    df_dups = pd.DataFrame({"a": [1, 1, 2], "b": [4, 4, 5]})
    res_dups = check_duplicate_rows(df_dups)
    assert res_dups["present"] is True
    assert res_dups["count"] == 1


def test_check_leaking_column():
    train_df = pd.DataFrame({
        "feature_1": [1, 2, 3, 4],
        "receipt_col": [10.1, 20.2, 30.1, 40.0],
        "delivery_min": [10.0, 20.0, 30.0, 40.0]
    })
    holdout_df = pd.DataFrame({
        "feature_1": [5, 6]
    })
    
    res = check_leaking_column(train_df, holdout_df, target_col="delivery_min")
    assert res["present"] is True
    assert res["column"] == "receipt_col"


def test_check_unit_mixup():
    df = pd.DataFrame({
        "route_zone": ["Suburb"] * 5 + ["Central"] * 5,
        "wait_min": [10.0, 11.0, 10.5, 9.5, 10.0] + [600.0, 660.0, 630.0, 570.0, 600.0]
    })
    res = check_unit_mixup(df, group_col="route_zone", numeric_cols=["wait_min"])
    assert res["present"] is True
    assert res["column"] == "wait_min"
    assert res["group_value"] == "Central"
    assert res["factor"] == 60
