"""
Data corruption utilities for metamorphic testing of DataSanity validation.

These functions inject specific data quality issues to test that DataSanity
correctly detects and reports violations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union


def inject_lookahead(df: pd.DataFrame, col: str = "Close", shift: int = -1) -> pd.DataFrame:
    """
    Inject lookahead contamination by shifting a column.
    
    Args:
        df: Input DataFrame
        col: Column to corrupt (default: "Close")
        shift: Shift amount (negative = future leak, positive = past leak)
    
    Returns:
        DataFrame with lookahead contamination
    """
    df = df.copy()
    if col in df.columns:
        shifted = df[col].shift(shift)
        # Fill NaN values created by shift to avoid triggering non-finite checks
        if shift < 0:  # Future leak - fill end with last value
            shifted = shifted.fillna(method='ffill')
        else:  # Past leak - fill beginning with first value
            shifted = shifted.fillna(method='bfill')
        df[col] = shifted
    return df


def inject_nans(df: pd.DataFrame, frac: float = 0.02, 
                cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Inject NaN values into specified columns.
    
    Args:
        df: Input DataFrame
        frac: Fraction of rows to corrupt (default: 0.02)
        cols: Columns to corrupt (default: OHLC columns)
    
    Returns:
        DataFrame with NaN contamination
    """
    df = df.copy()
    if cols is None:
        cols = ["Open", "High", "Low", "Close"]
    
    n = max(1, int(len(df) * frac))
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    
    for col in cols:
        if col in df.columns:
            df.loc[df.index[idx], col] = np.nan
    
    return df


def inject_infs(df: pd.DataFrame, frac: float = 0.02,
                cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Inject infinite values into specified columns.
    
    Args:
        df: Input DataFrame
        frac: Fraction of rows to corrupt (default: 0.02)
        cols: Columns to corrupt (default: OHLC columns)
    
    Returns:
        DataFrame with infinite value contamination
    """
    df = df.copy()
    if cols is None:
        cols = ["Open", "High", "Low", "Close"]
    
    n = max(1, int(len(df) * frac))
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    
    for col in cols:
        if col in df.columns:
            df.loc[df.index[idx], col] = np.inf
    
    return df


def inject_duplicates(df: pd.DataFrame, frac: float = 0.01) -> pd.DataFrame:
    """
    Inject duplicate timestamps by duplicating rows.
    
    Args:
        df: Input DataFrame
        frac: Fraction of rows to duplicate (default: 0.01)
    
    Returns:
        DataFrame with duplicate timestamps
    """
    n = max(1, int(len(df) * frac))
    pick = df.sample(n, replace=False).index
    return pd.concat([df, df.loc[pick]]).sort_index()


def inject_non_monotonic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject non-monotonic timestamps by swapping adjacent rows.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with non-monotonic timestamps
    """
    if len(df) < 3:
        return df
    
    df = df.copy()
    # Swap first two rows to create non-monotonic index
    idx1, idx2 = df.index[0], df.index[1]
    df.loc[idx1], df.loc[idx2] = df.loc[idx2].copy(), df.loc[idx1].copy()
    
    return df


def inject_tz_mess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove timezone information to simulate naive timestamps.
    
    Args:
        df: Input DataFrame with timezone-aware index
    
    Returns:
        DataFrame with naive timestamps
    """
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def inject_extreme_prices(df: pd.DataFrame, multiplier: float = 1000000.0,
                         cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Inject extreme price values.
    
    Args:
        df: Input DataFrame
        multiplier: Multiplier for extreme values (default: 1M)
        cols: Columns to corrupt (default: OHLC columns)
    
    Returns:
        DataFrame with extreme price values
    """
    df = df.copy()
    if cols is None:
        cols = ["Open", "High", "Low", "Close"]
    
    for col in cols:
        if col in df.columns:
            df.loc[df.index[0], col] = df[col].mean() * multiplier
    
    return df


def inject_negative_prices(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Inject negative price values.
    
    Args:
        df: Input DataFrame
        cols: Columns to corrupt (default: OHLC columns)
    
    Returns:
        DataFrame with negative price values
    """
    df = df.copy()
    if cols is None:
        cols = ["Open", "High", "Low", "Close"]
    
    for col in cols:
        if col in df.columns:
            df.loc[df.index[0], col] = -abs(df[col].mean())
    
    return df


def inject_string_dtype(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """
    Convert numeric column to string dtype.
    
    Args:
        df: Input DataFrame
        col: Column to corrupt (default: "Close")
    
    Returns:
        DataFrame with string dtype in specified column
    """
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].astype(str)
    return df


def inject_zero_volume(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    """
    Inject zero volume values.
    
    Args:
        df: Input DataFrame
        frac: Fraction of rows to corrupt (default: 0.1)
    
    Returns:
        DataFrame with zero volume values
    """
    df = df.copy()
    if "Volume" in df.columns:
        n = max(1, int(len(df) * frac))
        idx = np.linspace(0, len(df) - 1, n, dtype=int)
        df.loc[df.index[idx], "Volume"] = 0.0
    
    return df
