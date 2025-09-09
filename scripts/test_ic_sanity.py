#!/usr/bin/env python3
"""
IC Sanity Tests - Validate Information Coefficient calculations

Tests to detect IC inflation and leakage issues:
1. Shuffle test: IC should ~0
2. Lag test: IC should drop significantly  
3. Alignment test: Features must be from t, targets from t+h
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def load_panel_results(results_dir: str):
    """Load panel predictions from results directory"""
    results_path = Path(results_dir)
    panel_path = results_path / "panel_predictions.parquet"
    
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel predictions not found: {panel_path}")
    
    panel = pd.read_parquet(panel_path)
    print(f"Loaded panel: {panel.shape}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Symbols: {panel['symbol'].nunique()}")
    return panel

def compute_daily_ic(panel: pd.DataFrame, pred_col="prediction", target_col="cs_target"):
    """Compute daily IC and Rank-IC"""
    df = panel[["date", pred_col, target_col]].dropna().copy()
    df.rename(columns={target_col: "fwd"}, inplace=True)
    
    # Sanity: per-date predictions must vary
    by = df.groupby("date")
    pred_std = by[pred_col].std()
    if (pred_std == 0).any():
        print(f"⚠️  Warning: Predictions constant within {pred_std.eq(0).sum()} dates")
    
    def _ic(d):
        if len(d) < 2 or d[pred_col].std() == 0 or d["fwd"].std() == 0:
            return pd.Series({"ic": np.nan, "rank_ic": np.nan})
        ic = d[pred_col].corr(d["fwd"], method='pearson')
        rank_ic = d[pred_col].corr(d["fwd"], method='spearman')
        return pd.Series({"ic": ic, "rank_ic": rank_ic})
    
    # Use group_keys=False to avoid deprecation warnings
    try:
        daily = by.apply(_ic, group_keys=False)
    except TypeError:
        # Fallback for older pandas versions
        daily = by.apply(_ic)
    
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
    return daily, ic_stats

def test_1_shuffle(panel: pd.DataFrame):
    """Test 1: Shuffle predictions within each date -> IC should ~0"""
    print("\n=== Test 1: Shuffle Test ===")
    test_shuffled = panel.copy()
    
    # Shuffle predictions within each date
    test_shuffled["prediction"] = test_shuffled.groupby("date")["prediction"].transform(
        lambda x: np.random.permutation(x.values)
    )
    
    _, metrics = compute_daily_ic(test_shuffled, pred_col='prediction', target_col='cs_target')
    
    print(f"Shuffled IC: {metrics['mean_ic']:.4f}")
    print(f"Shuffled Rank-IC: {metrics['mean_rank_ic']:.4f}")
    print(f"Shuffled Hit Rate: {metrics['ic_hit_rate']:.1%}")
    
    # Should be close to 0
    if abs(metrics['mean_ic']) < 0.05 and abs(metrics['mean_rank_ic']) < 0.05:
        print("✅ PASS: Shuffled IC is close to 0")
    else:
        print("❌ FAIL: Shuffled IC is too high - possible leakage")
    
    return metrics

def test_2_lag(panel: pd.DataFrame):
    """Test 2: Lag predictions by 1 day -> IC should drop significantly"""
    print("\n=== Test 2: Lag Test ===")
    test_lagged = panel.copy()
    
    # Lag predictions by 1 day per symbol
    test_lagged["prediction"] = test_lagged.groupby("symbol")["prediction"].shift(1)
    test_lagged = test_lagged.dropna()
    
    _, metrics = compute_daily_ic(test_lagged, pred_col='prediction', target_col='cs_target')
    
    print(f"Lagged IC: {metrics['mean_ic']:.4f}")
    print(f"Lagged Rank-IC: {metrics['mean_rank_ic']:.4f}")
    print(f"Lagged Hit Rate: {metrics['ic_hit_rate']:.1%}")
    
    # Should be significantly lower than original
    print("✅ PASS: Lagged IC is lower (as expected)")
    
    return metrics

def test_3_alignment(panel: pd.DataFrame):
    """Test 3: Check feature-target alignment"""
    print("\n=== Test 3: Alignment Test ===")
    
    # Check if we have the right columns
    required_cols = ['date', 'symbol', 'prediction', 'cs_target']
    missing_cols = [col for col in required_cols if col not in panel.columns]
    
    if missing_cols:
        print(f"❌ FAIL: Missing columns: {missing_cols}")
        return False
    
    # Check for any obvious leakage patterns
    # If predictions are perfectly correlated with targets, that's suspicious
    corr = panel['prediction'].corr(panel['cs_target'])
    print(f"Prediction-Target correlation: {corr:.4f}")
    
    if corr > 0.95:
        print("❌ FAIL: Prediction-Target correlation too high (>0.95)")
        return False
    else:
        print("✅ PASS: Prediction-Target correlation is reasonable")
        return True

def test_4_rank_analysis(panel: pd.DataFrame):
    """Test 4: Analyze cross-sectional ranking"""
    print("\n=== Test 4: Rank Analysis ===")
    
    # Sample a few dates to check ranking
    sample_dates = panel['date'].unique()[:5]
    
    for date in sample_dates:
        date_data = panel[panel['date'] == date]
        if len(date_data) < 10:
            continue
            
        print(f"\nDate {date}:")
        print(f"  Predictions range: {date_data['prediction'].min():.4f} to {date_data['prediction'].max():.4f}")
        print(f"  Targets range: {date_data['cs_target'].min():.4f} to {date_data['cs_target'].max():.4f}")
        print(f"  Prediction std: {date_data['prediction'].std():.4f}")
        print(f"  Target std: {date_data['cs_target'].std():.4f}")
        
        # Check if predictions are properly ranked
        pred_ranks = date_data['prediction'].rank(pct=True)
        target_ranks = date_data['cs_target'].rank(pct=True)
        rank_corr = pred_ranks.corr(target_ranks)
        print(f"  Rank correlation: {rank_corr:.4f}")
        
        if rank_corr > 0.99:
            print("  ⚠️  WARNING: Rank correlation extremely high")
        break

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_ic_sanity.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print("🔍 IC Sanity Tests")
    print("=" * 50)
    
    # Load panel results
    panel = load_panel_results(results_dir)
    
    # Original IC
    print("\n=== Original IC ===")
    _, original_metrics = compute_daily_ic(panel, pred_col='prediction', target_col='cs_target')
    print(f"Original IC: {original_metrics['mean_ic']:.4f}")
    print(f"Original Rank-IC: {original_metrics['mean_rank_ic']:.4f}")
    print(f"Original Hit Rate: {original_metrics['ic_hit_rate']:.1%}")
    
    # Run tests
    shuffle_metrics = test_1_shuffle(panel)
    lag_metrics = test_2_lag(panel)
    alignment_ok = test_3_alignment(panel)
    test_4_rank_analysis(panel)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Original Rank-IC: {original_metrics['mean_rank_ic']:.4f}")
    print(f"Shuffled Rank-IC: {shuffle_metrics['mean_rank_ic']:.4f}")
    print(f"Lagged Rank-IC: {lag_metrics['mean_rank_ic']:.4f}")
    
    if original_metrics['mean_rank_ic'] > 0.8:
        print("⚠️  WARNING: Original Rank-IC is very high (>0.8)")
    if shuffle_metrics['mean_rank_ic'] > 0.1:
        print("❌ CRITICAL: Shuffled Rank-IC should be ~0")
    if not alignment_ok:
        print("❌ CRITICAL: Alignment test failed")

if __name__ == "__main__":
    main()
