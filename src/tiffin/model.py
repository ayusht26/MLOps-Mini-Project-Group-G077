"""Model pipeline builder and metric helpers."""

from typing import Any, Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def build_pipeline(model_name: str, model_params: Dict[str, Any],
                   numeric: List[str], categorical: List[str]) -> Pipeline:
    """Impute + scale numbers, one-hot the text columns, then fit a regressor."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre), ("model", MODELS[model_name](**model_params))])


def mae(y_true: pd.Series, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))
