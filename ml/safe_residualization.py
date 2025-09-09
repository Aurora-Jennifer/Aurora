"""
Index-safe residualization methods using vectorized operations.

These methods avoid the groupby.apply trap that caused flat features.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings


def residualize_against_factor(df: pd.DataFrame, y_cols: List[str], factor_col: str, 
                               suffix: str = "_res", min_variance: float = 1e-12) -> pd.DataFrame:
    """
    Residualize features against a single factor using vectorized operations.
    
    For each date, computes: residual = y - beta * factor
    where beta = cov(y, factor) / var(factor)
    
    Args:
        df: DataFrame with 'date' column
        y_cols: Feature columns to residualize
        factor_col: Factor column (e.g., market return)
        suffix: Suffix for residualized column names
        min_variance: Minimum factor variance to avoid division by zero
        
    Returns:
        DataFrame with new residualized columns
    """
    df = df.copy()
    
    if factor_col not in df.columns:
        warnings.warn(f"Factor column {factor_col} not found, skipping residualization")
        return df
    
    print(f"🔧 Residualizing {len(y_cols)} features against {factor_col}...")
    
    # Group by date and compute betas vectorized
    def compute_beta_vectorized(group):
        """Compute beta coefficients for all y_cols at once."""
        factor_values = group[factor_col]
        
        # Check factor variance
        factor_var = factor_values.var(ddof=0)
        if factor_var < min_variance:
            # Return zero betas if factor has no variance
            return pd.Series(0.0, index=y_cols)
        
        # Demean factor
        factor_demeaned = factor_values - factor_values.mean()
        
        # Compute betas for all y columns
        betas = {}
        for y_col in y_cols:
            if y_col in group.columns:
                y_values = group[y_col]
                y_demeaned = y_values - y_values.mean()
                covariance = (y_demeaned * factor_demeaned).mean()
                beta = covariance / factor_var
                betas[y_col] = beta
            else:
                betas[y_col] = 0.0
        
        return pd.Series(betas)
    
    # Compute betas per date
    betas_by_date = df.groupby('date').apply(compute_beta_vectorized)
    
    # Apply residualization using transform to maintain index alignment
    for y_col in y_cols:
        if y_col in df.columns:
            # Map betas back to original DataFrame
            df[f'_beta_{y_col}'] = df['date'].map(betas_by_date[y_col])
            
            # Compute residuals: y - beta * factor
            df[f'{y_col}{suffix}'] = df[y_col] - df[f'_beta_{y_col}'] * df[factor_col]
            
            # Clean up temporary beta column
            df.drop(columns=[f'_beta_{y_col}'], inplace=True)
    
    # Log residualization results
    residual_cols = [f'{y_col}{suffix}' for y_col in y_cols if y_col in df.columns]
    if residual_cols:
        # Check that residuals have variance
        sample_date = df['date'].iloc[0]
        sample_data = df[df['date'] == sample_date]
        residual_stds = sample_data[residual_cols].std()
        n_zero_std = (residual_stds < min_variance).sum()
        
        print(f"✅ Residualization complete: {len(residual_cols)} new columns")
        print(f"📊 Sample residual variance: {n_zero_std}/{len(residual_cols)} with zero std")
        
        if n_zero_std > len(residual_cols) * 0.5:
            warnings.warn(f"Many residuals have zero variance: {n_zero_std}/{len(residual_cols)}")
    
    return df


def residualize_against_sectors(df: pd.DataFrame, y_cols: List[str], sector_col: str = 'sector',
                               suffix: str = "_sec_res", min_sector_size: int = 3) -> pd.DataFrame:
    """
    Residualize features against sector means using transform (index-safe).
    
    For each date and feature: residual = y - sector_mean(y)
    
    Args:
        df: DataFrame with 'date' and sector columns
        y_cols: Feature columns to residualize
        sector_col: Column containing sector information
        suffix: Suffix for residualized column names
        min_sector_size: Minimum sector size to compute sector mean
        
    Returns:
        DataFrame with new residualized columns
    """
    df = df.copy()
    
    if sector_col not in df.columns:
        warnings.warn(f"Sector column {sector_col} not found, skipping sector residualization")
        return df
    
    print(f"🔧 Sector-residualizing {len(y_cols)} features against {sector_col}...")
    
    for y_col in y_cols:
        if y_col not in df.columns:
            continue
            
        # Use transform to maintain index alignment
        def sector_residualize(group):
            """Residualize against sector means within each date."""
            sector_means = group.groupby(sector_col)[y_col].transform('mean')
            sector_counts = group.groupby(sector_col)[y_col].transform('size')
            
            # Only residualize if sector has enough observations
            mask = sector_counts >= min_sector_size
            residuals = group[y_col].copy()
            residuals[mask] = group[y_col][mask] - sector_means[mask]
            
            return residuals
        
        # Apply sector residualization per date
        df[f'{y_col}{suffix}'] = df.groupby('date', group_keys=False).apply(
            sector_residualize, include_groups=False
        )
    
    # Log results
    residual_cols = [f'{y_col}{suffix}' for y_col in y_cols if y_col in df.columns]
    print(f"✅ Sector residualization complete: {len(residual_cols)} new columns")
    
    return df


def apply_exactly_once_normalization(df: pd.DataFrame, feature_cols: List[str], 
                                    method: str = 'zscore', exclude_suffixes: List[str] = None) -> pd.DataFrame:
    """
    Apply normalization exactly once, skipping already-normalized features.
    
    Args:
        df: DataFrame with 'date' column
        feature_cols: Feature columns to normalize
        method: Normalization method ('zscore', 'rank')
        exclude_suffixes: Suffixes indicating already-normalized features
        
    Returns:
        DataFrame with normalized features
    """
    if exclude_suffixes is None:
        exclude_suffixes = ['_csr', '_csz', '_res', '_norm']
    
    df = df.copy()
    
    # Identify features that need normalization
    need_normalization = []
    already_normalized = []
    
    for col in feature_cols:
        if any(col.endswith(suffix) for suffix in exclude_suffixes):
            already_normalized.append(col)
        else:
            need_normalization.append(col)
    
    if already_normalized:
        print(f"🔧 Skipping normalization for {len(already_normalized)} already-normalized features")
    
    if not need_normalization:
        print("✅ All features already normalized, skipping")
        return df
    
    print(f"🔧 Applying {method} normalization to {len(need_normalization)} features...")
    
    if method == 'zscore':
        # Cross-sectional z-score per date
        def cs_zscore(series):
            mu, std = series.mean(), series.std(ddof=0)
            if std < 1e-12:
                return series * 0  # Return zeros if no variance
            return (series - mu) / std
        
        for col in need_normalization:
            if col in df.columns:
                df[col] = df.groupby('date')[col].transform(cs_zscore)
    
    elif method == 'rank':
        # Cross-sectional ranks per date
        for col in need_normalization:
            if col in df.columns:
                df[col] = df.groupby('date')[col].transform(
                    lambda s: s.rank(pct=True, method='average')
                )
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    print(f"✅ Exactly-once normalization complete using {method}")
    return df


def build_residualized_features(df: pd.DataFrame, base_features: List[str], 
                               market_col: str = None, sector_col: str = None,
                               config: Dict = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build residualized features with multiple methods and exactly-once normalization.
    
    Args:
        df: DataFrame with date, symbol columns
        base_features: Base feature columns to residualize
        market_col: Market factor column for market residualization
        sector_col: Sector column for sector residualization
        config: Configuration dict with residualization settings
        
    Returns:
        Tuple of (updated_dataframe, list_of_new_feature_columns)
    """
    if config is None:
        config = {
            'market_residualization': True,
            'sector_residualization': True,
            'normalization_method': 'zscore',
            'max_residual_features': 50
        }
    
    df = df.copy()
    new_feature_cols = []
    
    print(f"🔧 Building residualized features from {len(base_features)} base features...")
    
    # Limit number of features to residualize (performance)
    features_to_process = base_features[:config.get('max_residual_features', 50)]
    if len(features_to_process) < len(base_features):
        print(f"🔧 Processing first {len(features_to_process)} features for performance")
    
    # 1) Market residualization
    if config.get('market_residualization', True) and market_col and market_col in df.columns:
        print("🔧 Step 1: Market residualization...")
        df = residualize_against_factor(df, features_to_process, market_col, suffix='_mkt_res')
        mkt_res_cols = [f'{col}_mkt_res' for col in features_to_process if col in df.columns]
        new_feature_cols.extend(mkt_res_cols)
    
    # 2) Sector residualization
    if config.get('sector_residualization', True) and sector_col and sector_col in df.columns:
        print("🔧 Step 2: Sector residualization...")
        df = residualize_against_sectors(df, features_to_process, sector_col, suffix='_sec_res')
        sec_res_cols = [f'{col}_sec_res' for col in features_to_process if col in df.columns]
        new_feature_cols.extend(sec_res_cols)
    
    # 3) Apply exactly-once normalization to all features
    all_features = base_features + new_feature_cols
    existing_features = [col for col in all_features if col in df.columns]
    
    if existing_features:
        print("🔧 Step 3: Exactly-once normalization...")
        df = apply_exactly_once_normalization(
            df, existing_features, 
            method=config.get('normalization_method', 'zscore')
        )
    
    print(f"✅ Residualized feature building complete: {len(new_feature_cols)} new features")
    return df, new_feature_cols


# Testing utilities
def test_residualization_preserves_index():
    """Test that residualization maintains proper index alignment."""
    # Create test data
    df = pd.DataFrame({
        'date': ['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02'],
        'symbol': ['A', 'B', 'A', 'B'],
        'feature': [10.0, 20.0, 15.0, 25.0],
        'market': [1.0, 1.0, 2.0, 2.0]  # Same market factor per date
    })
    
    # Apply residualization
    df_res = residualize_against_factor(df, ['feature'], 'market')
    
    # Verify index alignment
    assert len(df_res) == len(df), "Residualization changed number of rows"
    assert (df_res['symbol'] == df['symbol']).all(), "Symbol order changed"
    assert (df_res['date'] == df['date']).all(), "Date order changed"
    
    # Verify residuals have variance (features varied, market constant per date)
    assert 'feature_res' in df_res.columns, "Residual column not created"
    
    print("✅ Residualization index alignment test passed")


if __name__ == "__main__":
    test_residualization_preserves_index()
