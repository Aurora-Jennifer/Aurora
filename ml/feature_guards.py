"""
Feature engineering guards and utilities to prevent regression.

These guards prevent the flat-feature bug and ensure data quality.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings


def assert_cs_dispersion(df: pd.DataFrame, cols: List[str], name: str = "dataset"):
    """
    Hard assertion that features have cross-sectional variance.
    
    This is the core guard against the flat-feature bug that was caused by
    groupby.apply index alignment issues.
    
    Args:
        df: DataFrame with 'date' column
        cols: Feature column names to check
        name: Dataset name for error messages
        
    Raises:
        ValueError: If any feature is flat on all dates
        
    Returns:
        Dict with dispersion metrics for monitoring
    """
    if not cols:
        return {"status": "skipped", "reason": "no_features"}
        
    g = df.groupby("date", sort=False)[cols]
    flat_by_feat = (g.nunique() <= 1).sum()  # how many dates each feature is flat
    total_dates = g.ngroups
    
    # Critical: fail if ANY feature is flat on ALL dates
    completely_flat = flat_by_feat[flat_by_feat == total_dates]
    if len(completely_flat) > 0:
        bad_features = list(completely_flat.index)
        raise ValueError(
            f"❌ FATAL: {len(bad_features)} features are flat on ALL dates in {name}: {bad_features[:10]}... "
            "This indicates the groupby.apply index alignment bug has returned."
        )
    
    # Check a sample date for spot verification
    dispersion_metrics = {"status": "passed", "total_dates": total_dates}
    
    if total_dates > 0:
        sample_date = df["date"].iloc[0]
        snap = df.loc[df["date"] == sample_date, cols].std(ddof=0).sort_values()
        n_zero_std = (snap == 0).sum()
        
        dispersion_metrics.update({
            "sample_date": str(sample_date),
            "flat_features_sample": n_zero_std,
            "total_features": len(cols),
            "flat_ratio_sample": n_zero_std / len(cols),
            "std_min": snap.min(),
            "std_median": snap.median(),
            "std_max": snap.max()
        })
        
        print(f"📊 {name} dispersion check: {n_zero_std}/{len(cols)} features have zero std on sample date {sample_date}")
        print(f"📊 Sample feature std range: min={snap.min():.6f}, median={snap.median():.6f}, max={snap.max():.6f}")
        
        # Warn if too many flat features on sample date
        if n_zero_std > len(cols) * 0.3:
            warnings.warn(
                f"High flat feature ratio on sample date: {n_zero_std}/{len(cols)} features flat",
                UserWarning
            )
            dispersion_metrics["warning"] = "high_flat_ratio_sample"
    
    return dispersion_metrics


def assert_no_leakage(df: pd.DataFrame, feature_cols: List[str], target_col: str = None):
    """
    Basic leakage detection - ensure features don't use future information.
    
    Args:
        df: DataFrame with 'date' column
        feature_cols: Feature columns to check
        target_col: Target column (allowed to use future data)
    """
    # Basic check: ensure all feature values are properly lagged/historical
    # This is a placeholder for more sophisticated leakage detection
    
    if target_col and target_col in df.columns:
        # Verify target uses future data (as expected)
        pass  # Target leakage is intentional
    
    # Check for obvious issues like features that are too correlated with future returns
    if 'future_ret' in df.columns:
        for col in feature_cols[:5]:  # Sample check to avoid computation
            if col in df.columns:
                corr = df[col].corr(df['future_ret'])
                if abs(corr) > 0.95:  # Suspiciously high correlation
                    warnings.warn(f"Feature {col} highly correlated with future returns: {corr:.3f}")


def forbid_double_normalization(df: pd.DataFrame, cols: List[str], normalized_suffix: str = "_norm"):
    """
    Prevent double normalization by tracking which features are already normalized.
    
    Args:
        df: DataFrame
        cols: Columns to normalize
        normalized_suffix: Suffix that indicates already normalized
        
    Returns:
        List of columns safe to normalize
    """
    safe_cols = []
    already_normalized = []
    
    for col in cols:
        if col.endswith(normalized_suffix) or col.endswith("_csr") or col.endswith("_csz"):
            already_normalized.append(col)
        else:
            safe_cols.append(col)
    
    if already_normalized:
        print(f"🔧 Skipping normalization for {len(already_normalized)} already-normalized features: {already_normalized[:3]}...")
    
    return safe_cols


def validate_device_consistency(model_device: str, data_device: str = "cpu"):
    """
    Ensure XGBoost model and data are on the same device.
    
    Args:
        model_device: Device where model is configured ('cpu', 'cuda', etc.)
        data_device: Device where data tensors are located
        
    Returns:
        Tuple of (recommended_device, warning_message)
    """
    if model_device != data_device:
        warning = f"Device mismatch: model on {model_device}, data on {data_device}"
        
        # Recommend GPU if available
        if model_device == "cuda":
            return "cuda", f"{warning} - convert data to GPU for optimal performance"
        else:
            return "cpu", f"{warning} - using CPU for both"
    
    return model_device, None


def enforce_float32(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Convert feature columns to float32 for XGBoost efficiency.
    
    Args:
        df: DataFrame
        feature_cols: Feature columns to convert
        
    Returns:
        DataFrame with converted columns
    """
    df = df.copy()
    
    for col in feature_cols:
        if col in df.columns and df[col].dtype != 'float32':
            df[col] = df[col].astype('float32')
    
    print(f"✅ Converted {len(feature_cols)} features to float32")
    return df


def drop_near_constant_features(df: pd.DataFrame, feature_cols: List[str], 
                               threshold: float = 1e-6, min_dates_ratio: float = 0.95) -> List[str]:
    """
    Drop features that are near-constant across most dates.
    
    Args:
        df: DataFrame with 'date' column
        feature_cols: Feature columns to check
        threshold: Minimum standard deviation threshold
        min_dates_ratio: Minimum fraction of dates that must have low variance
        
    Returns:
        List of feature columns to keep
    """
    if not feature_cols:
        return []
    
    # Check per-date standard deviation
    by_date_std = df.groupby('date')[feature_cols].std(ddof=0)
    
    # Count dates where each feature has low variance
    low_var_ratio = (by_date_std <= threshold).mean()
    
    # Keep features that have sufficient variance on enough dates
    keep_features = low_var_ratio[low_var_ratio < min_dates_ratio].index.tolist()
    drop_features = low_var_ratio[low_var_ratio >= min_dates_ratio].index.tolist()
    
    if drop_features:
        print(f"🗑️  Dropping {len(drop_features)} near-constant features: {drop_features[:3]}...")
        print(f"📊 Kept {len(keep_features)}/{len(feature_cols)} features with sufficient variance")
    
    return keep_features


class FeaturePipelineGuard:
    """Context manager for feature pipeline with automatic validation."""
    
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], name: str = "pipeline"):
        self.df = df
        self.feature_cols = feature_cols
        self.name = name
        self.initial_metrics = None
        
    def __enter__(self):
        """Pre-pipeline validation."""
        print(f"🛡️  Starting feature pipeline guard for {self.name}")
        
        # Baseline dispersion check
        try:
            self.initial_metrics = assert_cs_dispersion(self.df, self.feature_cols, f"{self.name}_input")
        except ValueError as e:
            print(f"⚠️  Input data failed dispersion check: {e}")
            # Don't fail here - input might legitimately have issues we're fixing
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Post-pipeline validation."""
        if exc_type is not None:
            print(f"❌ Feature pipeline failed with {exc_type.__name__}: {exc_val}")
            return False
        
        # Final dispersion check
        final_metrics = assert_cs_dispersion(self.df, self.feature_cols, f"{self.name}_output")
        
        # Compare metrics
        if self.initial_metrics and self.initial_metrics.get("status") == "passed":
            initial_flat = self.initial_metrics.get("flat_features_sample", 0)
            final_flat = final_metrics.get("flat_features_sample", 0)
            
            if final_flat > initial_flat * 2:  # Significant increase in flat features
                print(f"⚠️  WARNING: Flat features increased from {initial_flat} to {final_flat}")
        
        print(f"✅ Feature pipeline guard completed for {self.name}")
        return True
