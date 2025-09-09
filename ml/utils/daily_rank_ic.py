# ml/utils/daily_rank_ic.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def daily_rank_ic(panel: pd.DataFrame, pred_col: str = "prediction", 
                  target_col: str = "cs_target") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute daily IC and Rank-IC with proper pandas compatibility.
    
    Args:
        panel: DataFrame with columns ['date', pred_col, target_col]
        pred_col: Name of prediction column
        target_col: Name of target column
        
    Returns:
        Tuple of (daily_ic_df, summary_metrics)
    """
    # Select only needed columns and drop NaN
    df = panel[["date", pred_col, target_col]].dropna().copy()
    
    if len(df) == 0:
        return pd.DataFrame(), {}
    
    # Sanity: per-date predictions must vary
    by = df.groupby("date")
    pred_std = by[pred_col].std()
    if (pred_std == 0).any():
        print(f"⚠️  Warning: Predictions constant within {pred_std.eq(0).sum()} dates")
    
    def _ic(d):
        """Compute IC and Rank-IC for a single date."""
        if len(d) < 2 or d[pred_col].std() == 0 or d[target_col].std() == 0:
            return pd.Series({"ic": np.nan, "rank_ic": np.nan})
        
        ic = d[pred_col].corr(d[target_col], method='pearson')
        rank_ic = d[pred_col].corr(d[target_col], method='spearman')
        
        return pd.Series({"ic": ic, "rank_ic": rank_ic})
    
    # Use include_groups=False for newer pandas versions, fallback for older
    try:
        daily = by.apply(_ic, include_groups=False)
    except TypeError:
        # Fallback for older pandas versions
        daily = by.apply(_ic)
    
    # Compute summary metrics
    metrics = {
        "ic_mean": float(daily["ic"].mean()),
        "ic_std": float(daily["ic"].std()),
        "ic_median": float(daily["ic"].median()),
        "rank_ic_mean": float(daily["rank_ic"].mean()),
        "rank_ic_std": float(daily["rank_ic"].std()),
        "rank_ic_median": float(daily["rank_ic"].median()),
        "hit_rate": float((daily["rank_ic"] > 0).mean()),
        "n_days": int(len(daily)),
        "positive_ic_days": int((daily["ic"] > 0).sum()),
        "positive_rank_ic_days": int((daily["rank_ic"] > 0).sum())
    }
    
    return daily, metrics
