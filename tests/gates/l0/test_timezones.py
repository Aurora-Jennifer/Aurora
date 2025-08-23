"""
L0 Timezone/Ordering Contract
- Index must be tz-aware UTC
- Index must be strictly monotonic within each symbol
- No duplicate timestamps per symbol
Optionally enforces session hours if your calendar util is available.
"""

import importlib
import numpy as np
import pandas as pd

def _is_utc_index(idx: pd.Index) -> bool:
    try:
        return getattr(idx, "tz", None) is not None and str(idx.tz) == "UTC"
    except Exception:
        return False

def _has_dupes_per_symbol(df: pd.DataFrame, symbol_col: str) -> bool:
    g = df.groupby(symbol_col, dropna=False)
    return any(g.apply(lambda x: x.index.duplicated().any()))

def _is_monotonic_per_symbol(df: pd.DataFrame, symbol_col: str) -> bool:
    g = df.groupby(symbol_col, dropna=False)
    return all(g.apply(lambda x: x.index.is_monotonic_increasing))

def test_index_is_tz_aware_utc(features_df: pd.DataFrame):
    idx = features_df.index
    assert hasattr(idx, "tz"), "Index must be timezone-aware"
    assert _is_utc_index(idx), f"Index tz must be UTC; got tz={getattr(idx, 'tz', None)}"

def test_no_duplicate_timestamps_per_symbol(features_df: pd.DataFrame):
    symbol_col = "symbol" if "symbol" in features_df.columns else None
    if symbol_col:
        assert not _has_dupes_per_symbol(features_df, symbol_col), "Duplicate timestamps within a symbol"
    else:
        # If no symbol col, ensure global uniqueness
        assert not features_df.index.duplicated().any(), "Duplicate timestamps in index"

def test_monotonic_time_order(features_df: pd.DataFrame):
    symbol_col = "symbol" if "symbol" in features_df.columns else None
    if symbol_col:
        assert _is_monotonic_per_symbol(features_df, symbol_col), "Index must be monotonic per symbol"
    else:
        assert features_df.index.is_monotonic_increasing, "Index must be monotonic increasing"

def test_optional_market_hours_if_calendar_available(features_df: pd.DataFrame):
    """
    Optional hardening: if your repo exposes core.market_calendar.is_session(ts, sym),
    we enforce it. Otherwise, we skip (no flaky heuristics).
    """
    try:
        cal_mod = importlib.import_module("core.market_calendar")
    except Exception:
        return  # calendar not present; optional gate
    if not hasattr(cal_mod, "is_session"):
        return
    symbol_col = "symbol" if "symbol" in features_df.columns else None
    if symbol_col:
        bad = []
        for sym, chunk in features_df.groupby(symbol_col, dropna=False):
            mask = chunk.index.to_series().apply(lambda t: bool(cal_mod.is_session(t, sym)))
            if not mask.all():
                bad.append((sym, (~mask).sum()))
        assert not bad, f"Off-session rows found: {bad}"
    else:
        mask = features_df.index.to_series().apply(lambda t: bool(cal_mod.is_session(t, None)))
        assert mask.all(), f"Off-session rows: {int((~mask).sum())}"
