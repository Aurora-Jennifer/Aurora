"""
Newey-West Sharpe calculation for overlapping returns
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

def newey_west_sharpe(returns: Union[np.ndarray, pd.Series], 
                     lag: int = 4,
                     annualization_factor: float = 252.0) -> float:
    """
    Calculate Newey-West adjusted Sharpe ratio for overlapping returns
    
    Args:
        returns: Array or Series of returns
        lag: Number of lags for autocorrelation adjustment (typically horizon-1)
        annualization_factor: Days per year for annualization (252 for daily)
    
    Returns:
        Newey-West adjusted Sharpe ratio
    """
    returns = np.asarray(returns)
    
    # Remove NaN values
    valid_mask = np.isfinite(returns)
    if not valid_mask.any():
        return np.nan
    
    r = returns[valid_mask]
    T = len(r)
    
    if T < 2:
        return np.nan
    
    # Calculate mean return
    mean_ret = np.mean(r)
    
    # Calculate variance with Newey-West adjustment
    demeaned = r - mean_ret
    var = np.var(demeaned, ddof=1)
    
    # Add autocorrelation terms
    for l in range(1, min(lag + 1, T)):
        if T > l:
            gamma = np.mean(demeaned[l:] * demeaned[:-l])
            weight = 1 - l / (lag + 1)
            var += 2 * weight * gamma
    
    # Annualize
    ann_mean = mean_ret * annualization_factor
    ann_std = np.sqrt(var) * np.sqrt(annualization_factor)
    
    # Return Sharpe ratio
    return ann_mean / (ann_std + 1e-12)

def classic_sharpe(returns: Union[np.ndarray, pd.Series],
                  annualization_factor: float = 252.0) -> float:
    """
    Calculate classic Sharpe ratio (no autocorrelation adjustment)
    """
    returns = np.asarray(returns)
    valid_mask = np.isfinite(returns)
    
    if not valid_mask.any():
        return np.nan
    
    r = returns[valid_mask]
    if len(r) < 2:
        return np.nan
    
    ann_mean = np.mean(r) * annualization_factor
    ann_std = np.std(r, ddof=1) * np.sqrt(annualization_factor)
    
    return ann_mean / (ann_std + 1e-12)

def calculate_ic_metrics(panel_df: pd.DataFrame,
                        pred_col: str = 'prediction',
                        ret_col: str = 'excess_ret_fwd_5') -> dict:
    """
    Calculate Information Coefficient (IC) and Rank-IC metrics
    Fixed version that avoids oracle-like IC values.
    
    Args:
        panel_df: DataFrame with columns ['date', 'symbol', pred_col, ret_col]
        pred_col: Name of prediction column
        ret_col: Name of forward return column
    
    Returns:
        Dictionary with IC metrics
    """
    # Only what we need; avoids include_groups deprecation warnings
    df = panel_df[["date", pred_col, ret_col]].dropna().copy()
    df.rename(columns={ret_col: "fwd"}, inplace=True)

    # Sanity: per-date predictions must vary
    by = df.groupby("date")
    pred_std = by[pred_col].std()
    if (pred_std == 0).any():
        print(f"⚠️  Warning: Predictions constant within {pred_std.eq(0).sum()} dates")
    
    def _ic(d):
        # Spearman ranks both sides internally; no need to pre-rank
        if len(d) < 2 or d[pred_col].std() == 0 or d["fwd"].std() == 0:
            return pd.Series({"ic": np.nan, "rank_ic": np.nan})
        
        ic = d[pred_col].corr(d["fwd"], method='pearson')
        rank_ic = d[pred_col].corr(d["fwd"], method='spearman')
        return pd.Series({"ic": ic, "rank_ic": rank_ic})

    # Use include_groups=False for newer pandas versions, fallback for older
    try:
        daily = by.apply(_ic, include_groups=False)
    except TypeError:
        # Fallback for older pandas versions
        try:
            daily = by.apply(_ic, group_keys=False)
        except TypeError:
            daily = by.apply(_ic)
    
    # Calculate summary statistics (convert to Python types for JSON serialization)
    ic_stats = {
        'median_ic': float(daily['ic'].median()),
        'mean_ic': float(daily['ic'].mean()),
        'std_ic': float(daily['ic'].std()),
        'ic_ir': float(daily['ic'].mean() / (daily['ic'].std() + 1e-12)),
        'median_rank_ic': float(daily['rank_ic'].median()),
        'mean_rank_ic': float(daily['rank_ic'].mean()),
        'std_rank_ic': float(daily['rank_ic'].std()),
        'rank_ic_ir': float(daily['rank_ic'].mean() / (daily['rank_ic'].std() + 1e-12)),
        'positive_ic_days': int((daily['ic'] > 0).sum()),
        'total_days': int(len(daily)),
        'ic_hit_rate': float((daily['ic'] > 0).mean())
    }
    
    return ic_stats


def compute_daily_ic(panel: pd.DataFrame, pred_col="pred", target_col="fwd_ret_5"):
    """
    Alternative IC calculation function with sanity checks.
    Use this for debugging IC inflation issues.
    """
    # Only what we need; avoids include_groups deprecation warnings
    df = panel[["date", pred_col, target_col]].dropna().copy()
    df.rename(columns={target_col: "fwd"}, inplace=True)

    # Sanity: per-date predictions must vary
    by = df.groupby("date")
    assert (by[pred_col].std() > 0).all(), "Predictions constant within some dates"

    def _ic(d):
        # Spearman ranks both sides internally; no need to pre-rank
        ic = d[pred_col].corr(d["fwd"])
        ric = d[pred_col].corr(d["fwd"], method="spearman")
        return pd.Series({"ic": ic, "rank_ic": ric})

    try:
        daily = by.apply(_ic, group_keys=False)
    except TypeError:
        # Fallback for older pandas versions
        daily = by.apply(_ic)

    metrics = {
        "ic_mean": float(daily["ic"].mean()),
        "rank_ic_mean": float(daily["rank_ic"].mean()),
        "hit_rate": float((daily["rank_ic"] > 0).mean()),
        "n_days": int(len(daily)),
    }
    return daily, metrics
