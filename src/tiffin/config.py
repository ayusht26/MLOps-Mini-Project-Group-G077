"""Configuration loader for the MLOps pipeline."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_params(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Read params.yaml using yaml.safe_load as required by the student handbook."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params
