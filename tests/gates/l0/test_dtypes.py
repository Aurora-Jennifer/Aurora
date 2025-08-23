"""
L0 Dtype & Numeric Stability Contract
- Enforce a single float dtype policy (default: float32)
- All numeric columns finite (no NaN/Inf)
- No silent up/down-casts across pipeline outputs
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

FLOAT_POLICY = os.getenv("DTYPE_POLICY", "float32")  # "float32" or "float64"
_ALLOWED_FLOATS = {"float32": "float32", "float64": "float64"}
assert FLOAT_POLICY in _ALLOWED_FLOATS, f"Invalid DTYPE_POLICY={FLOAT_POLICY}"

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def test_numeric_columns_match_policy(features_df: pd.DataFrame):
    want = _ALLOWED_FLOATS[FLOAT_POLICY]
    for c in _numeric_cols(features_df):
        got = str(features_df[c].dtype)
        assert got == want, f"Column {c} has dtype {got}, expected {want}"

def test_no_nans_or_infs(features_df: pd.DataFrame):
    num_cols = _numeric_cols(features_df)
    assert num_cols, "No numeric columns detected in features"
    arr = features_df[num_cols].to_numpy()
    assert np.isfinite(arr).all(), "Found NaN/Inf in numeric features"

def test_no_implicit_casts_across_outputs(features_df: pd.DataFrame, e2d_out_dir):
    """
    Guard against mixed dtypes produced in different artifacts.
    If additional outputs are present (e.g., predictions.parquet),
    they must obey the same dtype policy.
    """
    want = _ALLOWED_FLOATS[FLOAT_POLICY]
    for extra in ("predictions.parquet", "scores.parquet"):
        p = e2d_out_dir / extra
        if p.exists():
            df = pd.read_parquet(p)
            for c in _numeric_cols(df):
                got = str(df[c].dtype)
                assert got == want, f"{extra}:{c} dtype {got} != policy {want}"
