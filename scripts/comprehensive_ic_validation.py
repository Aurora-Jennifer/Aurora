#!/usr/bin/env python3
"""
Comprehensive IC validation script to detect leakage and overfitting.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import validation functions directly
sys.path.insert(0, str(project_root))
from ml.validation import comprehensive_ic_validation

def make_xgb_model():
    """Create a fresh XGBoost model for validation."""
    return xgb.XGBRegressor(
        tree_method='hist',
        device='cuda',
        n_estimators=100,  # Smaller for faster validation
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/comprehensive_ic_validation.py <results_directory>")
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
    
    print("🔍 Comprehensive IC Validation")
    print("=" * 50)
    
    panel = pd.read_parquet(panel_path)
    print(f"Loaded panel: {panel.shape}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Symbols: {panel['symbol'].nunique()}")
    
    # Identify feature columns (exclude metadata columns)
    exclude_cols = {'date', 'symbol', 'prediction', 'cs_target', 'excess_ret_fwd_5', 'excess_ret_fwd_3', 'excess_ret_fwd_10'}
    feat_cols = [col for col in panel.columns if col not in exclude_cols]
    print(f"Feature columns: {len(feat_cols)}")
    
    # Run comprehensive validation
    validation_results = comprehensive_ic_validation(
        panel, feat_cols, target_col='cs_target', make_model=make_xgb_model
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    # Cross-validation results
    cv = validation_results.get('cross_validation', {})
    if 'error' not in cv:
        print(f"Cross-Validation:")
        print(f"  Mean Rank-IC: {cv.get('mean_rank_ic', 'N/A'):.4f} ± {cv.get('std_rank_ic', 'N/A'):.4f}")
        print(f"  Hit Rate: {cv.get('mean_hit_rate', 'N/A'):.1%}")
        print(f"  Folds: {cv.get('n_folds', 'N/A')}")
    else:
        print(f"Cross-Validation: ERROR - {cv['error']}")
    
    # Walk-forward results
    wf = validation_results.get('walk_forward', {})
    if 'error' not in wf:
        print(f"Walk-Forward:")
        print(f"  Mean Rank-IC: {wf.get('mean_rank_ic', 'N/A'):.4f} ± {wf.get('std_rank_ic', 'N/A'):.4f}")
        print(f"  Hit Rate: {wf.get('mean_hit_rate', 'N/A'):.1%}")
        print(f"  Periods: {wf.get('n_periods', 'N/A')}")
    else:
        print(f"Walk-Forward: ERROR - {wf['error']}")
    
    # Leakage audit
    leak = validation_results.get('leakage_audit', {})
    if 'error' not in leak:
        print(f"Leakage Audit:")
        print(f"  Suspicious Features: {leak.get('n_suspicious', 0)}")
        if leak.get('suspicious_features'):
            print(f"  Suspicious: {leak['suspicious_features'][:5]}...")  # Show first 5
    else:
        print(f"Leakage Audit: ERROR - {leak['error']}")
    
    # Lag test
    lag = validation_results.get('lag_test', {})
    if 'error' not in lag:
        print(f"Lag Test:")
        print(f"  IC Degradation: {lag.get('ic_degradation', 'N/A'):.4f}")
        if lag.get('lag_results'):
            lag0 = lag['lag_results'][0]
            lag5 = lag['lag_results'][-1] if len(lag['lag_results']) > 1 else lag0
            print(f"  Lag 0 Rank-IC: {lag0.get('rank_ic', 'N/A'):.4f}")
            print(f"  Lag 5 Rank-IC: {lag5.get('rank_ic', 'N/A'):.4f}")
    else:
        print(f"Lag Test: ERROR - {lag['error']}")
    
    # Save detailed results
    output_path = results_path / "comprehensive_ic_validation.json"
    with open(output_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    
    # Overall assessment
    print("\n" + "=" * 50)
    print("🎯 OVERALL ASSESSMENT")
    print("=" * 50)
    
    # Check for red flags
    red_flags = []
    
    # High IC values
    if cv.get('mean_rank_ic', 0) > 0.8:
        red_flags.append("Very high cross-validation Rank-IC (>0.8)")
    
    # Perfect hit rates
    if cv.get('mean_hit_rate', 0) == 1.0:
        red_flags.append("Perfect hit rate in cross-validation")
    
    # Suspicious features
    if leak.get('n_suspicious', 0) > 0:
        red_flags.append(f"Suspicious features detected ({leak['n_suspicious']})")
    
    # Low IC degradation with lag
    if lag.get('ic_degradation', 0) < 0.1:
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
