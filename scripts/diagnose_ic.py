#!/usr/bin/env python3
"""
Diagnostic script to test IC calculation and identify oracle-like values
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from ml.utils.newey_west import compute_daily_ic


def test_ic_calculation(panel_file: str):
    """Test IC calculation with various scenarios"""
    
    print(f"🔍 Loading panel data from: {panel_file}")
    panel = pd.read_parquet(panel_file)
    
    print(f"Panel shape: {panel.shape}")
    print(f"Columns: {list(panel.columns)}")
    
    # Test 1: Original IC calculation
    print("\n📊 Test 1: Original IC calculation")
    daily, metrics = compute_daily_ic(panel, pred_col='prediction', target_col='excess_ret_fwd_5')
    print(f"IC: {metrics['ic_mean']:.4f}")
    print(f"Rank-IC: {metrics['rank_ic_mean']:.4f}")
    print(f"Hit Rate: {metrics['hit_rate']:.1%}")
    
    # Test 2: Shuffle predictions within each date (should give IC ~ 0)
    print("\n🎲 Test 2: Shuffled predictions (should give IC ~ 0)")
    test_panel = panel.copy()
    test_panel['prediction'] = test_panel.groupby('date')['prediction'].transform(
        lambda x: np.random.permutation(x.values)
    )
    daily_shuffled, metrics_shuffled = compute_daily_ic(test_panel, pred_col='prediction', target_col='excess_ret_fwd_5')
    print(f"IC: {metrics_shuffled['ic_mean']:.4f}")
    print(f"Rank-IC: {metrics_shuffled['rank_ic_mean']:.4f}")
    print(f"Hit Rate: {metrics_shuffled['hit_rate']:.1%}")
    
    # Test 3: Lag predictions by one day (should reduce IC)
    print("\n⏰ Test 3: Lagged predictions (should reduce IC)")
    lag_panel = panel.copy()
    lag_panel['prediction'] = lag_panel.groupby('symbol')['prediction'].shift(1)
    lag_panel = lag_panel.dropna()
    daily_lagged, metrics_lagged = compute_daily_ic(lag_panel, pred_col='prediction', target_col='excess_ret_fwd_5')
    print(f"IC: {metrics_lagged['ic_mean']:.4f}")
    print(f"Rank-IC: {metrics_lagged['rank_ic_mean']:.4f}")
    print(f"Hit Rate: {metrics_lagged['hit_rate']:.1%}")
    
    # Test 4: Check if predictions are derived from returns
    print("\n🔍 Test 4: Checking for potential leakage")
    
    # Check correlation between predictions and returns
    pred_ret_corr = panel['prediction'].corr(panel['excess_ret_fwd_5'])
    print(f"Direct correlation (pred vs ret): {pred_ret_corr:.4f}")
    
    # Check if predictions are ranks of returns
    ret_ranks = panel.groupby('date')['excess_ret_fwd_5'].rank(pct=True)
    pred_rank_corr = panel['prediction'].corr(ret_ranks)
    print(f"Correlation (pred vs ret_ranks): {pred_rank_corr:.4f}")
    
    # Check if predictions are just returns
    pred_is_ret = np.allclose(panel['prediction'], panel['excess_ret_fwd_5'], rtol=1e-10)
    print(f"Predictions are returns: {pred_is_ret}")
    
    # Test 5: Check cross-sectional ranking
    print("\n📈 Test 5: Cross-sectional ranking analysis")
    sample_date = panel['date'].iloc[0]
    sample_data = panel[panel['date'] == sample_date]
    print(f"Sample date: {sample_date}")
    print(f"Predictions range: {sample_data['prediction'].min():.4f} to {sample_data['prediction'].max():.4f}")
    print(f"Returns range: {sample_data['excess_ret_fwd_5'].min():.4f} to {sample_data['excess_ret_fwd_5'].max():.4f}")
    print(f"Prediction std: {sample_data['prediction'].std():.4f}")
    print(f"Return std: {sample_data['excess_ret_fwd_5'].std():.4f}")
    
    # Check if predictions are just ranks
    pred_ranks = sample_data['prediction'].rank(pct=True)
    ret_ranks_sample = sample_data['excess_ret_fwd_5'].rank(pct=True)
    rank_corr = pred_ranks.corr(ret_ranks_sample)
    print(f"Rank correlation (sample): {rank_corr:.4f}")
    
    return metrics


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/diagnose_ic.py <panel_predictions.parquet>")
        sys.exit(1)
    
    panel_file = sys.argv[1]
    test_ic_calculation(panel_file)
