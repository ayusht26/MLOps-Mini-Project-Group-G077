"""General utility functions for hashing, data loading, and validation."""

import hashlib
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def get_file_sha256(filepath: str) -> str:
    """Compute the SHA-256 fingerprint of a file.
    
    Used to verify dataset integrity and logged to MLflow runs.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hashing: {filepath}")
    
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_data(train_path: str = "data/raw/train.csv",
                  later_path: str = "data/raw/later.csv",
                  holdout_path: str = "data/raw/holdout_features.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, later, and holdout datasets from CSV files."""
    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path) if Path(later_path).exists() else pd.DataFrame()
    holdout_df = pd.read_csv(holdout_path) if Path(holdout_path).exists() else pd.DataFrame()
    return train_df, later_df, holdout_df
