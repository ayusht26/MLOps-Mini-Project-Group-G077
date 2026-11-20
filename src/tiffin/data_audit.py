"""Data audit module to hunt and verify the seven data problems.

Required by SCSE3040 MLOps Mini Project - Stage 1.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from src.tiffin.config import load_params
from src.tiffin.utils import get_file_sha256, load_raw_data


def check_leaking_column(train_df: pd.DataFrame, 
                         holdout_df: pd.DataFrame, 
                         target_col: str = "delivery_min") -> Dict[str, Any]:
    """1. Hunt for leaking columns.
    
    A column that is only filled in after the event:
    - Missing from holdout_features.csv
    - Agrees almost exactly with the target
    """
    holdout_cols = set(holdout_df.columns)
    candidates = [c for c in train_df.columns if c not in holdout_cols and c != target_col]
    
    for col in candidates:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            corr = float(train_df[col].corr(train_df[target_col]))
            mae_diff = float((train_df[col] - train_df[target_col]).abs().mean())
            if corr > 0.95 or mae_diff < 5.0:
                return {
                    "present": True,
                    "column": col,
                    "correlation": corr,
                    "mae_diff": mae_diff
                }
                
    return {"present": False, "column": None}


def check_unit_mixup(train_df: pd.DataFrame,
                     group_col: str = "route_zone",
                     numeric_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """2. Hunt for unit mixups across categories of group_col.
    
    Compare column's minimum, median, mean, and maximum across groups.
    One group is ~10, 12, 60, 100, or 1000 times bigger or smaller.
    factor is the exact unit conversion factor (wrong = right * factor).
    """
    if numeric_cols is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    candidate_factors = [10, 12, 60, 100, 1000]
    
    for col in numeric_cols:
        if col in ["record_id", "delivery_min"]:
            continue
        group_means = train_df.groupby(group_col)[col].mean()
        if len(group_means) < 2:
            continue
            
        for group_val, mean_val in group_means.items():
            other_means = group_means.drop(index=group_val)
            ref_mean = other_means.median()
            if ref_mean == 0:
                continue
            ratio = mean_val / ref_mean
            
            for f in candidate_factors:
                if 0.7 * f <= ratio <= 1.3 * f:
                    return {
                        "present": True,
                        "column": col,
                        "group_value": group_val,
                        "factor": f,
                        "group_mean": float(mean_val),
                        "reference_mean": float(ref_mean)
                    }
                elif 0.7 * (1.0 / f) <= ratio <= 1.3 * (1.0 / f):
                    return {
                        "present": True,
                        "column": col,
                        "group_value": group_val,
                        "factor": 1 / f,
                        "group_mean": float(mean_val),
                        "reference_mean": float(ref_mean)
                    }
                    
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(train_df: pd.DataFrame,
                         numeric_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """3. Hunt for sentinel values.
    
    Someone typed a special number like -1, -999, 999, 9999 instead of leaving blank.
    value_counts() shows one impossible number repeated many times.
    """
    if numeric_cols is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        
    known_sentinels = {-1, -99, -999, 999, 9999, 99999}
    
    for col in numeric_cols:
        vc = train_df[col].value_counts()
        for sent in known_sentinels:
            if sent in vc and vc[sent] > 3:
                return {
                    "present": True,
                    "column": col,
                    "value": sent,
                    "count": int(vc[sent])
                }
    
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train_df: pd.DataFrame) -> Dict[str, Any]:
    """4. Hunt for duplicate rows.
    
    df.duplicated().sum() gives the number of extra copies.
    """
    extra_copies = int(train_df.duplicated().sum())
    return {
        "present": extra_copies > 0,
        "count": extra_copies if extra_copies > 0 else None
    }


def check_stuck_sensor(train_df: pd.DataFrame,
                       date_col: str = "date",
                       numeric_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """5. Hunt for stuck sensors.
    
    Sort by date. One column shows the same value on every row for a long run of days.
    """
    if numeric_cols is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        
    df_sorted = train_df.sort_values(by=date_col).reset_index(drop=True)
    
    for col in numeric_cols:
        if col in ["record_id", "delivery_min"]:
            continue
        series = df_sorted[col]
        # Identify consecutive blocks with identical values
        blocks = (series != series.shift()).cumsum()
        block_sizes = df_sorted.groupby(blocks).size()
        
        max_block_id = block_sizes.idxmax()
        max_len = block_sizes.max()
        
        # A stuck sensor runs over many days (e.g. >= 20 rows and spanning weeks)
        if max_len >= 15:
            block_df = df_sorted[blocks == max_block_id]
            start_date = str(block_df[date_col].iloc[0])
            end_date = str(block_df[date_col].iloc[-1])
            stuck_val = block_df[col].iloc[0]
            if start_date != end_date:
                return {
                    "present": True,
                    "column": col,
                    "value": stuck_val,
                    "start_date": start_date,
                    "end_date": end_date,
                    "run_length": int(max_len)
                }
                
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train_df: pd.DataFrame, 
                       target_col: str = "delivery_min") -> Dict[str, Any]:
    """6. Hunt for target typos (e.g. typed with an extra zero: 10 times too big).
    
    Sort the target. A small group of values sits far above all others with a big gap.
    """
    sorted_targets = train_df[target_col].sort_values().values
    q75 = np.percentile(sorted_targets, 75)
    iqr = q75 - np.percentile(sorted_targets, 25)
    
    # Check if there is an extreme gap where values are ~10x median
    median_val = np.median(sorted_targets)
    extreme_typos = train_df[train_df[target_col] >= 8 * median_val]
    
    if len(extreme_typos) > 0:
        return {
            "present": True,
            "count": int(len(extreme_typos))
        }
        
    return {"present": False, "count": None}


def check_spelling_variants(train_df: pd.DataFrame,
                            later_df: pd.DataFrame,
                            cat_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """7. Hunt for spelling variants in categorical columns.
    
    One text column has the same category written in different ways (e.g. 'Rain', 'rain ', 'RAI.').
    later.csv has clean spellings. true_levels is the number of real categories.
    """
    if cat_cols is None:
        cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in ["record_id", "date"]]
        
    for col in cat_cols:
        train_vals = set(train_df[col].dropna().unique())
        later_vals = set(later_df[col].dropna().unique()) if not later_df.empty else set()
        
        # If train has more unique categories than clean later, or whitespace variants
        cleaned_train = set(train_df[col].dropna().astype(str).str.strip().str.lower().unique())
        if len(train_vals) > len(cleaned_train) or (later_vals and len(train_vals) > len(later_vals)):
            return {
                "present": True,
                "column": col,
                "true_levels": len(later_vals) if later_vals else len(cleaned_train)
            }
            
    return {"present": False, "column": None, "true_levels": None}


def run_full_audit(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Run full audit of all seven issues based on params.yaml."""
    params = load_params(params_path)
    train_path = params["data"]["raw_train_path"]
    later_path = params["data"]["raw_later_path"]
    holdout_path = params["data"]["raw_holdout_path"]
    
    train_df, later_df, holdout_df = load_raw_data(train_path, later_path, holdout_path)
    sha256 = get_file_sha256(train_path)
    
    print("=" * 65)
    print(f" AUDIT REPORT: {train_path}")
    print(f" SHA-256: {sha256}")
    print(f" Train Rows: {len(train_df)}, Columns: {len(train_df.columns)}")
    print("=" * 65)
    
    res_leaking = check_leaking_column(train_df, holdout_df, params["audit"]["target_column"])
    res_unit = check_unit_mixup(train_df, params["audit"]["group_column"], params["audit"]["numeric_features"])
    res_sentinel = check_sentinel_value(train_df, params["audit"]["numeric_features"])
    res_duplicates = check_duplicate_rows(train_df)
    res_stuck = check_stuck_sensor(train_df, params["audit"]["date_column"], params["audit"]["numeric_features"])
    res_typos = check_target_typos(train_df, params["audit"]["target_column"])
    res_spelling = check_spelling_variants(train_df, later_df, params["audit"]["categorical_features"])
    
    findings = {
        "leaking_column": res_leaking,
        "unit_mixup": res_unit,
        "sentinel_value": res_sentinel,
        "duplicate_rows": res_duplicates,
        "stuck_sensor": res_stuck,
        "target_typos": res_typos,
        "spelling_variants": res_spelling,
    }
    
    print("\n--- Summary of Findings ---")
    for key, val in findings.items():
        pres = val.get("present")
        status_str = "[PRESENT]" if pres else "[ABSENT] "
        print(f" {status_str} {key}: {val}")
    print("=" * 65)
    
    return findings


if __name__ == "__main__":
    run_full_audit()
