#!/usr/bin/env python3
"""
Simple IC validation script to detect leakage and overfitting.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

def lag_more_test(panel: pd.DataFrame, pred_col="prediction", target_col="cs_target", 
                  max_lags=5) -> dict:
    """Test IC degradation with increasing lag to detect subtle leakage."""
    results = []
    
    for lag in range(max_lags + 1):
        if lag == 0:
            test_panel = panel.copy()
        else:
            test_panel = panel.copy()
            test_panel[pred_col] = test_panel.groupby('symbol')[pred_col].shift(lag)
            test_panel = test_panel.dropna()
        
        if len(test_panel) < 100:
            continue
            
        # Compute daily IC
        daily_ic = []
        for date in test_panel['date'].unique():
            date_data = test_panel[test_panel['date'] == date]
            if len(date_data) > 1:
                ic = date_data[pred_col].corr(date_data[target_col])
                if not np.isnan(ic):
                    daily_ic.append(ic)
        
        if daily_ic:
            mean_ic = np.mean(daily_ic)
            rank_ic = pd.Series(test_panel[pred_col]).corr(pd.Series(test_panel[target_col]), method='spearman')
            
            results.append({
                'lag': lag,
                'n_obs': len(test_panel),
                'mean_ic': mean_ic,
                'rank_ic': rank_ic,
                'hit_rate': (rank_ic > 0) if not np.isnan(rank_ic) else np.nan
            })
    
    return {
        'lag_results': results,
        'ic_degradation': results[0]['mean_ic'] - results[-1]['mean_ic'] if len(results) > 1 else 0
    }

def feature_leakage_audit(panel: pd.DataFrame, feat_cols, target_col="cs_target") -> dict:
    """Audit features for potential leakage by checking correlations with future returns."""
    results = {}
    
    for feat in feat_cols:
        # Check correlation with target
        corr = panel[feat].corr(panel[target_col])
        
        # Check if feature has suspiciously high correlation
        results[feat] = {
            'correlation_with_target': corr,
            'suspicious': abs(corr) > 0.8,  # Flag high correlations
            'std': panel[feat].std(),
            'null_pct': panel[feat].isnull().mean()
        }
    
    suspicious_features = [f for f, r in results.items() if r['suspicious']]
    
    return {
        'suspicious_features': suspicious_features,
        'feature_details': results,
        'n_suspicious': len(suspicious_features)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/simple_ic_validation.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load panel predictions
    panel_path = results_path / "panel_predictions.parquet"
    if not panel_path.exists():
        print(f"❌ Panel predictions not found: {panel_path}")
        sys.exit(1)
    
    print("🔍 Simple IC Validation")
    print("=" * 50)
    
    panel = pd.read_parquet(panel_path)
    print(f"Loaded panel: {panel.shape}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Symbols: {panel['symbol'].nunique()}")
    
    # Identify feature columns (exclude metadata columns)
    exclude_cols = {'date', 'symbol', 'prediction', 'cs_target', 'excess_ret_fwd_5', 'excess_ret_fwd_3', 'excess_ret_fwd_10'}
    feat_cols = [col for col in panel.columns if col not in exclude_cols]
    print(f"Feature columns: {len(feat_cols)}")
    
    # Run validation tests
    print("\n🔍 Running validation tests...")
    
    # 1. Feature leakage audit
    print("  📊 Feature leakage audit...")
    leak_results = feature_leakage_audit(panel, feat_cols, target_col='cs_target')
    
    # 2. Lag-more test
    print("  ⏰ Lag-more test...")
    lag_results = lag_more_test(panel)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    # Leakage audit
    print(f"Leakage Audit:")
    print(f"  Suspicious Features: {leak_results.get('n_suspicious', 0)}")
    if leak_results.get('suspicious_features'):
        print(f"  Suspicious: {leak_results['suspicious_features'][:5]}...")  # Show first 5
    
    # Lag test
    print(f"Lag Test:")
    print(f"  IC Degradation: {lag_results.get('ic_degradation', 'N/A'):.4f}")
    if lag_results.get('lag_results'):
        lag0 = lag_results['lag_results'][0]
        lag5 = lag_results['lag_results'][-1] if len(lag_results['lag_results']) > 1 else lag0
        print(f"  Lag 0 Rank-IC: {lag0.get('rank_ic', 'N/A'):.4f}")
        print(f"  Lag 5 Rank-IC: {lag5.get('rank_ic', 'N/A'):.4f}")
    
    # Save detailed results
    validation_results = {
        'leakage_audit': leak_results,
        'lag_test': lag_results
    }
    
    output_path = results_path / "simple_ic_validation.json"
    with open(output_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    
    # Overall assessment
    print("\n" + "=" * 50)
    print("🎯 OVERALL ASSESSMENT")
    print("=" * 50)
    
    # Check for red flags
    red_flags = []
    
    # Suspicious features
    if leak_results.get('n_suspicious', 0) > 0:
        red_flags.append(f"Suspicious features detected ({leak_results['n_suspicious']})")
    
    # Low IC degradation with lag
    if lag_results.get('ic_degradation', 0) < 0.1:
        red_flags.append("Low IC degradation with lag (possible leakage)")
    
    if red_flags:
        print("🚨 RED FLAGS DETECTED:")
        for flag in red_flags:
            print(f"  ❌ {flag}")
        print("\n⚠️  These results suggest possible overfitting or data leakage.")
    else:
        print("✅ No major red flags detected.")
        print("📈 Results appear to be legitimate.")

if __name__ == "__main__":
    main()
