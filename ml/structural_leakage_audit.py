"""
Structural leakage audit to identify features with forward-looking contamination.

This module implements comprehensive audits to detect features that are monotonic
transformations of labels or contain future information.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
from scipy.stats import spearmanr


def smoking_gun_feature_audit(df: pd.DataFrame, feature_cols: List[str], 
                             label_col: str, contamination_threshold: float = 0.2) -> List[Tuple[str, float, float]]:
    """
    Smoking gun audit: detect features with high same-date correlation to labels.
    
    This catches features that are monotonic transformations of forward returns
    or otherwise contaminated with future information.
    
    Args:
        df: DataFrame with features and labels (post-leakage-guard)
        feature_cols: Feature column names to audit
        label_col: Label column name
        contamination_threshold: |correlation| threshold for flagging contamination
        
    Returns:
        List of (feature_name, median_rank_correlation, fraction_high_correlation)
        
    Raises:
        RuntimeError: If contaminated features detected above threshold
    """
    print(f"🔍 SMOKING GUN AUDIT: Testing {len(feature_cols)} features vs {label_col}")
    print(f"   Contamination threshold: |ρ| ≥ {contamination_threshold}")
    
    audit_results = []
    by_date = df.groupby('date')
    
    for feature in feature_cols:
        if feature not in df.columns:
            continue
            
        # Compute same-date rank correlation per date
        date_correlations = []
        
        for date_name, group in by_date:
            if len(group) < 10:  # Need minimum observations
                continue
                
            # Use rank correlation to detect monotonic relationships
            feature_ranks = group[feature].rank()
            label_ranks = group[label_col].rank()
            
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
            
        # Aggregate statistics
        median_rho = np.median(date_correlations)
        high_corr_fraction = np.mean([abs(rho) >= 0.95 for rho in date_correlations])
        
        audit_results.append((feature, median_rho, high_corr_fraction))
    
    # Sort by absolute correlation (highest first)
    audit_results.sort(key=lambda x: -abs(x[1]))
    
    # Check for contamination
    contaminated_features = [
        result for result in audit_results 
        if abs(result[1]) >= contamination_threshold or result[2] > 0.05
    ]
    
    print(f"📊 AUDIT SUMMARY:")
    print(f"   Features tested: {len(audit_results)}")
    print(f"   Contaminated features: {len(contaminated_features)}")
    
    if contaminated_features:
        print(f"🚨 CONTAMINATED FEATURES (top 10):")
        for i, (feature, rho, frac) in enumerate(contaminated_features[:10]):
            print(f"  {i+1:2d}. {feature}: median_ρ={rho:.4f}, high_corr_dates={frac:.1%}")
        
        # HARD FAIL on contamination
        top_offenders = contaminated_features[:5]
        raise RuntimeError(
            f"🚨 STRUCTURAL LEAKAGE DETECTED: {len(contaminated_features)} contaminated features. "
            f"Top offenders: {[(f, round(r, 4)) for f, r, _ in top_offenders]}"
        )
    
    print(f"✅ No structural leakage detected (all |ρ| < {contamination_threshold})")
    
    # Show top correlations for monitoring
    print(f"📋 TOP CORRELATIONS (monitoring):")
    for i, (feature, rho, frac) in enumerate(audit_results[:15]):
        status = "🚨" if abs(rho) >= 0.1 else "✅"
        print(f"  {i+1:2d}. {status} {feature}: ρ={rho:.4f}, high_dates={frac:.1%}")
    
    return audit_results


def detect_forward_looking_names(feature_cols: List[str]) -> List[str]:
    """
    Detect obviously forward-looking features by name patterns.
    
    Args:
        feature_cols: Feature column names
        
    Returns:
        List of suspicious feature names
    """
    BANNED_TOKENS = [
        'fwd', 'forward', 'future', 'label', 'target', 'ret_fwd', 'excess_ret_fwd',
        'oof', 'rank_ret_fwd', '_fwd_', 'ahead', 'lookahead', 'next_'
    ]
    
    suspicious = []
    for feature in feature_cols:
        feature_lower = feature.lower()
        for token in BANNED_TOKENS:
            if token in feature_lower:
                suspicious.append(feature)
                break
    
    return suspicious


def build_positive_allowlist(df: pd.DataFrame) -> List[str]:
    """
    Build a positive allowlist of safe features based on known-safe sources.
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        List of safe feature column names
    """
    # Known-safe base patterns (past-only data)
    SAFE_BASE_PATTERNS = [
        'close', 'volume', 'ret_1', 'ret_5', 'vol_5', 'vol_20', 'vol_',
        'momentum', 'ma_', 'bb_', 'breakout', 'liquidity', 'sharpe',
        'mean_reversion', 'ema_', 'sma_', 'rsi_', 'atr_', 'stoch_',
        'bollinger', 'macd_', 'adx_', 'williams', 'cci_', 'roc_',
        'trix_', 'dpo_', 'ppo_', 'ultimate_osc'
    ]
    
    # Known-safe transformation suffixes
    SAFE_SUFFIXES = ['_csr', '_csz', '_sec_res', '_sector_res']
    
    # Banned patterns (forward-looking)
    BANNED_PATTERNS = ['fwd', 'forward', 'future', 'target', 'label', 'next', 'ahead']
    
    # Build allowlist
    safe_features = []
    
    for col in df.columns:
        # Skip non-feature columns
        if col in ['date', 'symbol', 'close', 'market_ret1']:
            continue
            
        # Skip forward-looking columns
        col_lower = col.lower()
        if any(banned in col_lower for banned in BANNED_PATTERNS):
            continue
            
        # Check if it has a safe transformation suffix (CS transformed features)
        has_safe_suffix = any(col.endswith(suffix) for suffix in SAFE_SUFFIXES)
        
        if has_safe_suffix:
            # Extract base name by removing suffix
            base_name = col
            for suffix in SAFE_SUFFIXES:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            
            # Remove f_ prefix if present
            if base_name.startswith('f_'):
                base_name = base_name[2:]
            
            # If base contains any safe pattern, it's allowed
            is_safe_base = any(pattern in base_name.lower() for pattern in SAFE_BASE_PATTERNS)
            
            if is_safe_base:
                safe_features.append(col)
        else:
            # Also check columns that might be features but don't have expected suffixes
            # Skip if obviously not a feature (raw data columns)
            if col in df.columns and col not in ['close', 'volume', 'high', 'low', 'open']:
                # If it contains safe patterns and no banned patterns, tentatively allow
                is_safe_base = any(pattern in col_lower for pattern in SAFE_BASE_PATTERNS)
                if is_safe_base:
                    # Only add if it looks like a derived feature (has underscores typically)
                    if '_' in col:
                        safe_features.append(col)
    
    print(f"📋 POSITIVE ALLOWLIST: {len(safe_features)} safe features identified")
    
    # Show feature breakdown
    suffix_counts = {}
    for feature in safe_features:
        for suffix in SAFE_SUFFIXES:
            if feature.endswith(suffix):
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
                break
    
    print(f"   Breakdown: {dict(suffix_counts)}")
    
    # Show sample features for transparency
    if safe_features:
        print(f"   Sample features: {safe_features[:10]}")
    
    return safe_features


def causality_check_streaming(df: pd.DataFrame, symbol: str, feature_base: str, window: int = 5) -> bool:
    """
    Check if a feature could be computed with streaming (past-only) data.
    
    Args:
        df: DataFrame with symbol data
        symbol: Symbol to test
        feature_base: Base feature name (e.g., 'vol_5')
        window: Rolling window size
        
    Returns:
        True if feature passes causality check
    """
    print(f"🔍 CAUSALITY CHECK: {feature_base} for {symbol}")
    
    # Get symbol data
    symbol_data = df[df['symbol'] == symbol].sort_values('date').copy()
    
    if len(symbol_data) < window * 2:
        print(f"   ⚠️ Insufficient data for {symbol}")
        return True  # Can't test, assume safe
    
    # Compute streaming version (past-only)
    def stream_feature(series, window_size):
        """Compute feature using only past data."""
        out = []
        for i in range(len(series)):
            if i < window_size:
                out.append(np.nan)  # Not enough history
            else:
                # Use only past data (excluding current)
                past_data = series.iloc[i-window_size:i]
                if feature_base.startswith('vol'):
                    out.append(past_data.std(ddof=0))
                elif feature_base.startswith('mom'):
                    out.append(past_data.iloc[-1] - past_data.iloc[0])
                else:
                    out.append(past_data.mean())  # Default
        return pd.Series(out, index=series.index)
    
    # Compute streaming feature
    if 'ret_1' in symbol_data.columns:
        streaming_result = stream_feature(symbol_data['ret_1'], window)
        
        # Compare with existing feature if available
        if feature_base in symbol_data.columns:
            existing_result = symbol_data[feature_base]
            
            # Check alignment (ignoring NaNs)
            valid_mask = ~(streaming_result.isna() | existing_result.isna())
            if valid_mask.sum() > 0:
                max_diff = np.abs(streaming_result[valid_mask] - existing_result[valid_mask]).max()
                
                if max_diff > 1e-6:
                    print(f"   🚨 CAUSALITY VIOLATION: max_diff={max_diff:.2e}")
                    return False
                else:
                    print(f"   ✅ CAUSALITY CHECK PASSED: max_diff={max_diff:.2e}")
                    return True
    
    print(f"   ⚠️ Could not perform causality check")
    return True


def comprehensive_structural_audit(df: pd.DataFrame, feature_cols: List[str], 
                                 label_col: str) -> Dict[str, Any]:
    """
    Run comprehensive structural leakage audit.
    
    Args:
        df: DataFrame with features and labels (post-leakage-guard)
        feature_cols: Feature columns to audit
        label_col: Label column name
        
    Returns:
        Dictionary with audit results
        
    Raises:
        RuntimeError: If any structural leakage detected
    """
    print("🔍 COMPREHENSIVE STRUCTURAL LEAKAGE AUDIT")
    print("=" * 60)
    
    audit_summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'features_tested': len(feature_cols),
        'label_column': label_col,
        'leakage_detected': False,
        'issues': []
    }
    
    # 1. Name-based detection
    print("\n1️⃣ NAME-BASED DETECTION")
    suspicious_names = detect_forward_looking_names(feature_cols)
    if suspicious_names:
        audit_summary['leakage_detected'] = True
        audit_summary['issues'].append(f"Forward-looking names: {suspicious_names}")
        print(f"🚨 SUSPICIOUS NAMES: {suspicious_names}")
    else:
        print("✅ No suspicious feature names detected")
    
    # 2. Same-date correlation audit
    print("\n2️⃣ SAME-DATE CORRELATION AUDIT")
    try:
        correlation_results = smoking_gun_feature_audit(df, feature_cols, label_col, threshold=0.2)
        audit_summary['correlation_audit'] = correlation_results[:10]
    except RuntimeError as e:
        audit_summary['leakage_detected'] = True
        audit_summary['issues'].append(str(e))
        raise e
    
    # 3. Positive allowlist check
    print("\n3️⃣ POSITIVE ALLOWLIST VALIDATION")
    safe_features = build_positive_allowlist(df)
    unsafe_features = [f for f in feature_cols if f not in safe_features]
    
    if unsafe_features:
        print(f"⚠️ FEATURES NOT IN ALLOWLIST: {len(unsafe_features)}")
        for feature in unsafe_features[:10]:
            print(f"   - {feature}")
        if len(unsafe_features) > 10:
            print(f"   ... and {len(unsafe_features) - 10} more")
    else:
        print("✅ All features in positive allowlist")
    
    audit_summary['safe_features'] = len(safe_features)
    audit_summary['unsafe_features'] = len(unsafe_features)
    
    # 4. Sample causality check
    print("\n4️⃣ CAUSALITY SPOT CHECK")
    if 'symbol' in df.columns:
        sample_symbols = df['symbol'].unique()[:3]
        for symbol in sample_symbols:
            causality_check_streaming(df, symbol, 'vol_5', window=5)
    
    print("\n" + "=" * 60)
    if audit_summary['leakage_detected']:
        print("🚨 STRUCTURAL LEAKAGE AUDIT FAILED")
        print(f"Issues: {audit_summary['issues']}")
    else:
        print("✅ STRUCTURAL LEAKAGE AUDIT PASSED")
    
    return audit_summary


if __name__ == "__main__":
    print("Structural leakage audit module loaded successfully")
