"""Cleaning code: turns the messy train.csv into a model-ready table."""

from typing import Any, Dict

import numpy as np
import pandas as pd


def drop_leak(df: pd.DataFrame, leak_column: str) -> pd.DataFrame:
    """Remove the column that is only known after delivery."""
    return df.drop(columns=[leak_column], errors="ignore")


def fix_unit_mixup(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Convert seconds back to minutes for the affected group."""
    out = df.copy()
    mask = out[cfg["group_column"]] == cfg["group_value"]
    out.loc[mask, cfg["column"]] = out.loc[mask, cfg["column"]] / cfg["factor"]
    return out


def blank_stuck_sensor(df: pd.DataFrame, cfg: Dict[str, Any], date_column: str = "date") -> pd.DataFrame:
    """Set the frozen sensor readings to NaN (the model pipeline imputes them)."""
    out = df.copy()
    in_run = (
        (out[date_column] >= cfg["start_date"])
        & (out[date_column] <= cfg["end_date"])
        & (out[cfg["column"]] == cfg["value"])
    )
    out.loc[in_run, cfg["column"]] = np.nan
    return out


def clean_train(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Apply every fix that the audit found, in a fixed order."""
    c = params["cleaning"]
    out = df.drop_duplicates().copy()
    out = drop_leak(out, c["leak_column"])
    out = fix_unit_mixup(out, c["unit_fix"])
    out = blank_stuck_sensor(out, c["stuck_sensor"], params["audit"]["date_column"])
    return out.sort_values(params["audit"]["date_column"]).reset_index(drop=True)
