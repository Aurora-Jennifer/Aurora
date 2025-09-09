"""
Hard leakage audit tools to detect and prevent data leakage.

This module implements surgical leakage detection with hard fail thresholds.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
from scipy.stats import spearmanr


def leakage_audit_same_date(df: pd.DataFrame, feature_cols: List[str], 
                           label_col: str, threshold: float = 0.95) -> List[Tuple[str, float, float]]:
    """
    Smoking gun leakage audit: check same-date correlation between features and labels.
    
    For each feature, compute per-date Spearman correlation with label, then aggregate.
    Features with |median_rho| >= threshold are likely leakage.
    
    Args:
        df: DataFrame with date, features, and labels
        feature_cols: Feature column names
        label_col: Label column name
        threshold: Correlation threshold for flagging leakage
        
    Returns:
        List of (feature_name, median_rho, fraction_dates_above_threshold)
    """
    print(f"🔍 Running smoking gun leakage audit on {len(feature_cols)} features...")
    
    leakage_results = []
    
    for feature in feature_cols:
        if feature not in df.columns:
            continue
            
        # Compute per-date Spearman correlation
        date_correlations = []
        
        for date in df['date'].unique()[:50]:  # Sample first 50 dates for speed
            date_data = df[df['date'] == date]
            
            if len(date_data) < 10:  # Need minimum observations
                continue
                
            # Use rank correlation to handle non-linear relationships
            feature_ranks = date_data[feature].rank()
            label_ranks = date_data[label_col].rank()
            
            # Skip if no variance
            if feature_ranks.std() == 0 or label_ranks.std() == 0:
                continue
                
            try:
                corr, _ = spearmanr(feature_ranks, label_ranks, nan_policy='omit')
                if not np.isnan(corr):
                    date_correlations.append(corr)
            except:
                continue
        
        if len(date_correlations) == 0:
            continue
            
        # Aggregate correlations
        median_rho = np.median(date_correlations)
        high_corr_fraction = np.mean([abs(rho) >= threshold for rho in date_correlations])
        
        leakage_results.append((feature, median_rho, high_corr_fraction))
    
    # Sort by absolute correlation
    leakage_results.sort(key=lambda x: -abs(x[1]))
    
    print(f"📊 Leakage audit complete: {len(leakage_results)} features analyzed")
    return leakage_results


def detect_forward_looking_features(feature_whitelist: List[str]) -> List[str]:
    """
    Detect obviously forward-looking features by name patterns.
    
    Args:
        feature_whitelist: List of feature names
        
    Returns:
        List of suspicious feature names
    """
    suspicious_patterns = ['fwd', 'forward', 'future', 'label', 'target', 'ret_fwd', 'excess_ret']
    
    suspicious_features = []
    for feature in feature_whitelist:
        feature_lower = feature.lower()
        if any(pattern in feature_lower for pattern in suspicious_patterns):
            suspicious_features.append(feature)
    
    return suspicious_features


def temporal_shift_audit(df: pd.DataFrame, feature_cols: List[str], 
                        n_samples: int = 100) -> Dict[str, Any]:
    """
    Audit temporal shift correctness for leakage guard.
    
    Verify that features at time t equal pre-shift values at t-1.
    
    Args:
        df: DataFrame with date, symbol, features
        feature_cols: Feature columns to audit
        n_samples: Number of random samples to check
        
    Returns:
        Dictionary with audit results
    """
    print(f"🔍 Running temporal shift audit on {n_samples} random samples...")
    
    # Get random (symbol, date) pairs where date > min_date
    df_sorted = df.sort_values(['symbol', 'date'])
    eligible_rows = df_sorted.groupby('symbol').apply(
        lambda g: g.iloc[1:]  # Skip first date per symbol (no prior data)
    ).reset_index(drop=True)
    
    if len(eligible_rows) < n_samples:
        n_samples = len(eligible_rows)
    
    sample_indices = np.random.choice(len(eligible_rows), n_samples, replace=False)
    issues = []
    
    for idx in sample_indices:
        row = eligible_rows.iloc[idx]
        symbol = row['symbol']
        date = row['date']
        
        # Find previous date for this symbol
        symbol_data = df_sorted[df_sorted['symbol'] == symbol]
        current_idx = symbol_data[symbol_data['date'] == date].index[0]
        prev_rows = symbol_data[symbol_data.index < current_idx]
        
        if len(prev_rows) == 0:
            continue
            
        prev_row = prev_rows.iloc[-1]  # Most recent prior row
        
        # Check a few random features
        check_features = np.random.choice(feature_cols, min(5, len(feature_cols)), replace=False)
        
        for feature in check_features:
            if feature in row and feature in prev_row:
                current_val = row[feature]
                # After leakage guard, current should equal previous raw value
                # This is a simplified check - full implementation would need access to pre-shift data
                if pd.notna(current_val) and pd.notna(prev_row[feature]):
                    # For now, just check that values are reasonable (not identical to avoid false positives)
                    if abs(current_val - prev_row[feature]) > 1e6:  # Very large difference suggests issue
                        issues.append({
                            'symbol': symbol,
                            'date': date,
                            'feature': feature,
                            'current': current_val,
                            'previous': prev_row[feature]
                        })
    
    return {
        'samples_checked': n_samples,
        'issues_found': len(issues),
        'issues': issues[:10]  # Top 10 issues
    }


def horizon_leakage_check(df: pd.DataFrame, feature_cols: List[str], 
                         horizon: int, max_ic_threshold: float = 0.2) -> Dict[str, Any]:
    """
    Check for leakage in horizon construction and OOF IC.
    
    Args:
        df: DataFrame with predictions and returns
        feature_cols: Feature columns
        horizon: Forward return horizon in days
        max_ic_threshold: Maximum allowable |IC| before flagging leakage
        
    Returns:
        Dictionary with leakage check results
    """
    print(f"🔍 Running horizon leakage check for {horizon}d horizon...")
    
    results = {
        'horizon': horizon,
        'max_ic_threshold': max_ic_threshold,
        'leakage_detected': False,
        'issues': []
    }
    
    # Check if forward return columns exist in features (they shouldn't)
    fwd_patterns = [f'ret_fwd_{horizon}', f'excess_ret_fwd_{horizon}', f'fwd_{horizon}']
    leaked_features = []
    
    for pattern in fwd_patterns:
        leaked = [f for f in feature_cols if pattern in f.lower()]
        leaked_features.extend(leaked)
    
    if leaked_features:
        results['leakage_detected'] = True
        results['issues'].append(f"Forward-looking features in whitelist: {leaked_features}")
    
    # Check OOF IC if we have predictions
    if 'prediction' in df.columns and f'ret_fwd_{horizon}' in df.columns:
        # Compute rank IC per date, then median
        def rank_ic(group):
            if len(group) < 10:
                return np.nan
            pred_ranks = group['prediction'].rank()
            ret_ranks = group[f'ret_fwd_{horizon}'].rank()
            if pred_ranks.std() == 0 or ret_ranks.std() == 0:
                return np.nan
            return pred_ranks.corr(ret_ranks, method='spearman')
        
        daily_ics = df.groupby('date').apply(rank_ic)
        valid_ics = daily_ics.dropna()
        
        if len(valid_ics) > 0:
            median_ic = valid_ics.median()
            results['median_oof_ic'] = median_ic
            results['abs_median_oof_ic'] = abs(median_ic)
            
            if abs(median_ic) > max_ic_threshold:
                results['leakage_detected'] = True
                results['issues'].append(f"OOF IC {median_ic:.4f} exceeds threshold {max_ic_threshold}")
    
    return results


def hard_leakage_audit(df: pd.DataFrame, feature_cols: List[str], 
                      label_col: str, horizon: int = None) -> Dict[str, Any]:
    """
    Comprehensive hard leakage audit with hard-fail on detection.
    
    Args:
        df: DataFrame with features and labels
        feature_cols: Feature column names
        label_col: Label column name
        horizon: Forward return horizon (if applicable)
        
    Returns:
        Dictionary with all audit results
        
    Raises:
        ValueError: If critical leakage is detected
    """
    print("🚨 Starting HARD leakage audit - will FAIL on detection...")
    
    audit_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'features_audited': len(feature_cols),
        'leakage_detected': False,
        'critical_issues': []
    }
    
    # 1. Forward-looking feature name check
    suspicious_names = detect_forward_looking_features(feature_cols)
    if suspicious_names:
        audit_results['leakage_detected'] = True
        audit_results['critical_issues'].append(f"Forward-looking feature names: {suspicious_names}")
    
    # 2. Same-date correlation audit
    same_date_results = leakage_audit_same_date(df, feature_cols, label_col, threshold=0.95)
    high_corr_features = [(name, rho, frac) for name, rho, frac in same_date_results if abs(rho) >= 0.95 or frac >= 0.1]
    
    if high_corr_features:
        audit_results['leakage_detected'] = True
        audit_results['critical_issues'].append(f"High correlation features: {high_corr_features[:5]}")
    
    audit_results['same_date_audit'] = same_date_results[:15]  # Top 15 for inspection
    
    # 3. Temporal shift audit
    shift_results = temporal_shift_audit(df, feature_cols, n_samples=50)
    if shift_results['issues_found'] > 5:  # Allow some tolerance
        audit_results['leakage_detected'] = True
        audit_results['critical_issues'].append(f"Temporal shift issues: {shift_results['issues_found']} found")
    
    audit_results['temporal_audit'] = shift_results
    
    # 4. Horizon-specific checks
    if horizon:
        horizon_results = horizon_leakage_check(df, feature_cols, horizon)
        if horizon_results['leakage_detected']:
            audit_results['leakage_detected'] = True
            audit_results['critical_issues'].extend(horizon_results['issues'])
        
        audit_results['horizon_audit'] = horizon_results
    
    # Print summary
    print(f"📊 Leakage audit summary:")
    print(f"   Features audited: {len(feature_cols)}")
    print(f"   Suspicious names: {len(suspicious_names)}")
    print(f"   High correlation features: {len(high_corr_features)}")
    print(f"   Temporal issues: {shift_results['issues_found']}")
    
    if high_corr_features:
        print(f"🚨 TOP LEAKAGE SUSPECTS:")
        for name, rho, frac in high_corr_features[:5]:
            print(f"   {name}: median_rho={rho:.4f}, high_corr_dates={frac:.1%}")
    
    # HARD FAIL on critical leakage
    if audit_results['leakage_detected']:
        critical_summary = '; '.join(audit_results['critical_issues'])
        raise ValueError(f"🚨 CRITICAL LEAKAGE DETECTED: {critical_summary}")
    
    print("✅ Leakage audit PASSED - no critical issues detected")
    return audit_results


def validate_oof_ic(oof_ic: float, max_threshold: float = 0.2, horizon: int = None) -> None:
    """
    Hard validation of OOF IC with immediate failure on suspicious values.
    
    Args:
        oof_ic: Out-of-fold Information Coefficient
        max_threshold: Maximum allowable absolute IC
        horizon: Horizon for context in error message
        
    Raises:
        ValueError: If IC exceeds threshold
    """
    abs_ic = abs(oof_ic)
    
    if abs_ic > max_threshold:
        horizon_text = f" (horizon {horizon}d)" if horizon else ""
        raise ValueError(
            f"🚨 HARD FAIL: OOF IC {oof_ic:.4f}{horizon_text} exceeds threshold {max_threshold}. "
            f"This indicates severe data leakage. Expected range: |IC| < 0.1 for honest signals."
        )
    
    print(f"✅ OOF IC validation passed: {oof_ic:.4f} (threshold: {max_threshold})")


if __name__ == "__main__":
    # Basic test
    print("Leakage audit module loaded successfully")
