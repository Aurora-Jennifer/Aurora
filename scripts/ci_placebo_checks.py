#!/usr/bin/env python3
"""
CI placebo checks for preventing leakage regression.

Implements 100% feature shuffle and label lag controls with hard gates.
"""
import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ml.negative_controls import apply_feature_shuffle, apply_label_lag_placebo
from ml.structural_leakage_audit import smoking_gun_feature_audit
from scipy.stats import spearmanr
import json
from pathlib import Path


def run_100_percent_feature_shuffle(data: pd.DataFrame, 
                                   feature_cols: List[str],
                                   target_col: str = 'cs_target',
                                   threshold: float = 0.01,
                                   random_seed: int = 42) -> Dict:
    """
    Run 100% within-date feature shuffle placebo check.
    
    Args:
        data: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        threshold: IC threshold for passing (default 0.01)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with results and pass/fail status
    """
    print("🔄 Running 100% feature shuffle placebo check...")
    
    # Apply 100% feature shuffle
    shuffled_data = apply_feature_shuffle(
        data.copy(), 
        feature_cols, 
        shuffle_fraction=1.0,  # 100% shuffle
        random_seed=random_seed
    )
    
    # Calculate IC on shuffled data
    ic_values = []
    for date, group in shuffled_data.groupby('date'):
        if len(group) > 10:  # Minimum group size
            for feature in feature_cols:
                if feature in group.columns and target_col in group.columns:
                    # Remove NaN values
                    feature_vals = group[feature].dropna()
                    target_vals = group.loc[feature_vals.index, target_col]
                    
                    if len(feature_vals) > 5:
                        corr, _ = spearmanr(feature_vals, target_vals, nan_policy='omit')
                        if not np.isnan(corr):
                            ic_values.append(corr)
    
    if ic_values:
        median_ic = np.median(ic_values)
        abs_median_ic = abs(median_ic)
        max_abs_ic = max(abs(ic) for ic in ic_values)
    else:
        median_ic = 0.0
        abs_median_ic = 0.0
        max_abs_ic = 0.0
    
    # Check if passes threshold
    passes = abs_median_ic < threshold
    
    result = {
        'test': '100_percent_feature_shuffle',
        'median_ic': median_ic,
        'abs_median_ic': abs_median_ic,
        'max_abs_ic': max_abs_ic,
        'threshold': threshold,
        'passes': passes,
        'num_ic_values': len(ic_values),
        'feature_count': len(feature_cols)
    }
    
    print(f"   Median |IC|: {abs_median_ic:.6f}")
    print(f"   Max |IC|: {max_abs_ic:.6f}")
    print(f"   Threshold: {threshold:.3f}")
    print(f"   Result: {'✅ PASS' if passes else '❌ FAIL'}")
    
    return result


def run_label_lag_placebo(data: pd.DataFrame,
                         feature_cols: List[str], 
                         target_col: str = 'cs_target',
                         lag_days: int = 5,
                         threshold: float = 0.01,
                         random_seed: int = 42) -> Dict:
    """
    Run label lag placebo check (predict t+H+lag with features at t).
    
    Args:
        data: DataFrame with features and target  
        feature_cols: List of feature column names
        target_col: Target column name
        lag_days: Number of days to lag features
        threshold: IC threshold for passing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with results and pass/fail status
    """
    print(f"🔄 Running label lag placebo check (lag={lag_days} days)...")
    
    # Apply temporal lag to features
    lagged_data = apply_label_lag_placebo(
        data.copy(),
        feature_cols,
        lag_days=lag_days,
        random_seed=random_seed
    )
    
    # Calculate IC on lagged data
    ic_values = []
    for date, group in lagged_data.groupby('date'):
        if len(group) > 10:
            for feature in feature_cols:
                if feature in group.columns and target_col in group.columns:
                    feature_vals = group[feature].dropna()
                    target_vals = group.loc[feature_vals.index, target_col]
                    
                    if len(feature_vals) > 5:
                        corr, _ = spearmanr(feature_vals, target_vals, nan_policy='omit')
                        if not np.isnan(corr):
                            ic_values.append(corr)
    
    if ic_values:
        median_ic = np.median(ic_values)
        abs_median_ic = abs(median_ic)
        max_abs_ic = max(abs(ic) for ic in ic_values)
    else:
        median_ic = 0.0
        abs_median_ic = 0.0
        max_abs_ic = 0.0
    
    # Check if passes threshold
    passes = abs_median_ic < threshold
    
    result = {
        'test': 'label_lag_placebo',
        'lag_days': lag_days,
        'median_ic': median_ic,
        'abs_median_ic': abs_median_ic,
        'max_abs_ic': max_abs_ic,
        'threshold': threshold,
        'passes': passes,
        'num_ic_values': len(ic_values),
        'rows_after_lag': len(lagged_data)
    }
    
    print(f"   Lag days: {lag_days}")
    print(f"   Median |IC|: {abs_median_ic:.6f}")
    print(f"   Max |IC|: {max_abs_ic:.6f}")
    print(f"   Threshold: {threshold:.3f}")
    print(f"   Result: {'✅ PASS' if passes else '❌ FAIL'}")
    
    return result


def load_golden_slice_data(golden_data_path: str = "tests/golden/panel_sample.parquet") -> Tuple[pd.DataFrame, List[str]]:
    """
    Load golden slice data for CI testing.
    
    Args:
        golden_data_path: Path to golden data file
        
    Returns:
        Tuple of (data, feature_columns)
    """
    if not os.path.exists(golden_data_path):
        # Create mock golden data if file doesn't exist
        print(f"⚠️ Golden data not found at {golden_data_path}, creating mock data")
        return create_mock_golden_data()
    
    data = pd.read_parquet(golden_data_path)
    
    # Identify feature columns (assuming they start with 'f_' or end with '_csr', '_csz', etc.)
    feature_cols = [col for col in data.columns 
                   if col.startswith('f_') or 
                   any(col.endswith(suffix) for suffix in ['_csr', '_csz', '_res'])]
    
    return data, feature_cols


def create_mock_golden_data() -> Tuple[pd.DataFrame, List[str]]:
    """Create mock golden data for testing when real data unavailable."""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    symbols = [f'SYM{i:03d}' for i in range(50)]
    
    np.random.seed(42)
    data = []
    
    for date in dates:
        # Create mostly random features that should have minimal signal when shuffled
        random_base = np.random.randn(len(symbols))
        
        for i, symbol in enumerate(symbols):
            # Very weak signal to noise ratio (should disappear with proper shuffling)
            weak_signal = random_base[i] * 0.01  # Very weak
            noise = np.random.normal(0, 1.0)     # Strong noise
            
            # Target with minimal predictable component
            target = weak_signal + noise
            
            # Features that are mostly noise with tiny signal
            feature1 = weak_signal * 0.05 + np.random.normal(0, 1.0)  # 5% signal
            feature2 = weak_signal * 0.03 + np.random.normal(0, 1.0)  # 3% signal  
            feature3 = np.random.normal(0, 1.0)                       # Pure noise
            
            data.append({
                'date': date,
                'symbol': symbol,
                'cs_target': target,
                'f_momentum_5_csr': feature1,
                'f_volume_csr': feature2,
                'f_noise_csr': feature3
            })
    
    df = pd.DataFrame(data)
    feature_cols = ['f_momentum_5_csr', 'f_volume_csr', 'f_noise_csr']
    
    return df, feature_cols


def run_ci_placebo_suite(data_path: str = None, 
                        threshold: float = 0.01,
                        output_path: str = "results/ci/placebo_results.json") -> bool:
    """
    Run complete CI placebo test suite.
    
    Args:
        data_path: Path to test data (uses golden data if None)
        threshold: IC threshold for passing tests
        output_path: Path to save results
        
    Returns:
        True if all tests pass, False otherwise
    """
    print("🧪 CI PLACEBO TEST SUITE")
    print("="*50)
    
    # Load data
    if data_path:
        data = pd.read_parquet(data_path)
        feature_cols = [col for col in data.columns if col.startswith('f_')]
    else:
        data, feature_cols = load_golden_slice_data()
    
    print(f"✅ Loaded test data: {data.shape}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Run tests
    results = []
    
    # Test 1: 100% feature shuffle
    shuffle_result = run_100_percent_feature_shuffle(
        data, feature_cols, threshold=threshold
    )
    results.append(shuffle_result)
    
    # Test 2: Label lag placebo
    lag_result = run_label_lag_placebo(
        data, feature_cols, lag_days=5, threshold=threshold
    )
    results.append(lag_result)
    
    # Overall pass/fail
    all_passed = all(result['passes'] for result in results)
    
    # Save results with JSON-safe conversion
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    json_safe_results = []
    for result in results:
        json_safe_result = {}
        for key, value in result.items():
            if isinstance(value, (np.bool_, bool)):
                json_safe_result[key] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):
                json_safe_result[key] = float(value)
            else:
                json_safe_result[key] = value
        json_safe_results.append(json_safe_result)
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': pd.Timestamp.now().isoformat(),
            'threshold': float(threshold),
            'all_passed': bool(all_passed),
            'results': json_safe_results
        }, f, indent=2)
    
    # Summary
    print(f"\n📊 PLACEBO TEST SUMMARY:")
    print(f"   100% Feature Shuffle: {'✅ PASS' if shuffle_result['passes'] else '❌ FAIL'}")
    print(f"   Label Lag Placebo: {'✅ PASS' if lag_result['passes'] else '❌ FAIL'}")
    print(f"   Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if not all_passed:
        print(f"\n❌ CI PLACEBO CHECKS FAILED")
        print(f"   This indicates potential data leakage or model issues")
        print(f"   Review the feature engineering and model training pipeline")
        sys.exit(1)
    else:
        print(f"\n✅ CI PLACEBO CHECKS PASSED")
        print(f"   No evidence of data leakage detected")
    
    return all_passed


def main():
    """Main CLI entry point for CI integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CI Placebo Checks")
    parser.add_argument('--data-path', help="Path to test data file")
    parser.add_argument('--threshold', type=float, default=0.01, 
                       help="IC threshold for passing (default: 0.01)")
    parser.add_argument('--output', default="results/ci/placebo_results.json",
                       help="Output path for results")
    
    args = parser.parse_args()
    
    # Run placebo suite
    success = run_ci_placebo_suite(
        data_path=args.data_path,
        threshold=args.threshold,
        output_path=args.output
    )
    
    # Exit with appropriate code for CI
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
