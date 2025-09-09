"""
Negative controls for proving non-spurious results.

These controls shuffle data to destroy any genuine signal,
proving that observed IC comes from real predictive content.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings


def apply_date_wise_label_shuffle(df: pd.DataFrame, label_col: str = 'cs_target', 
                                 random_seed: int = 42) -> pd.DataFrame:
    """
    Apply date-wise label shuffle control.
    
    Within each date, randomly permute the labels across symbols.
    This destroys any genuine cross-sectional signal while preserving
    the distribution of labels per date.
    
    Args:
        df: DataFrame with date, symbol, and label columns
        label_col: Name of the label column to shuffle
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with shuffled labels
    """
    print(f"🔄 Applying date-wise label shuffle control...")
    
    df_control = df.copy()
    np.random.seed(random_seed)
    
    # Shuffle labels within each date
    for date, group in df_control.groupby('date'):
        group_indices = group.index
        group_labels = group[label_col].values
        shuffled_group_labels = np.random.permutation(group_labels)
        df_control.loc[group_indices, label_col] = shuffled_group_labels
    
    print(f"✅ Label shuffle complete: {len(df_control)} rows processed")
    return df_control


def apply_feature_shuffle(df: pd.DataFrame, feature_cols: List[str], 
                         shuffle_fraction: float = 1.0, random_seed: int = 42) -> pd.DataFrame:
    """
    Apply within-date feature shuffle control.
    
    For a fraction of features, randomly permute values within each date.
    This partially destroys predictive signal while maintaining feature distributions.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        shuffle_fraction: Fraction of features to shuffle (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with shuffled features
    """
    print(f"🔄 Applying feature shuffle control (fraction: {shuffle_fraction})...")
    
    df_control = df.copy()
    np.random.seed(random_seed)
    
    # Select features to shuffle
    n_features_to_shuffle = int(len(feature_cols) * shuffle_fraction)
    features_to_shuffle = np.random.choice(feature_cols, size=n_features_to_shuffle, replace=False)
    
    print(f"📋 Shuffling {len(features_to_shuffle)} features: {list(features_to_shuffle)[:5]}...")
    
    # Shuffle selected features within each date
    for feature in features_to_shuffle:
        if feature in df_control.columns:
            shuffled_values = []
            
            for date, group in df_control.groupby('date'):
                group_values = group[feature].values
                shuffled_group_values = np.random.permutation(group_values)
                shuffled_values.extend(shuffled_group_values)
            
            df_control[feature] = shuffled_values
    
    print(f"✅ Feature shuffle complete: {len(features_to_shuffle)} features shuffled")
    return df_control


def apply_label_lag_placebo(df: pd.DataFrame, feature_cols: List[str], 
                           lag_days: int = 5, random_seed: int = 42) -> pd.DataFrame:
    """
    Apply label lag placebo control.
    
    Use features from t-lag_days to predict labels at t+H.
    This creates a temporal mismatch that should destroy signal.
    
    Args:
        df: DataFrame with date-sorted data
        feature_cols: Feature columns to lag
        lag_days: Number of days to lag features
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with lagged features
    """
    print(f"🔄 Applying label lag placebo control (lag: {lag_days} days)...")
    
    df_control = df.copy()
    df_control = df_control.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Lag features by specified days
    for feature in feature_cols:
        if feature in df_control.columns:
            df_control[feature] = df_control.groupby('symbol')[feature].shift(lag_days)
    
    # Remove rows where lagged features are NaN
    initial_rows = len(df_control)
    df_control = df_control.dropna(subset=feature_cols, how='any')
    final_rows = len(df_control)
    
    print(f"✅ Label lag placebo complete: {initial_rows} → {final_rows} rows after lag removal")
    return df_control


def run_negative_control_test(control_type: str, df: pd.DataFrame, 
                             feature_cols: List[str], label_col: str = 'cs_target',
                             shuffle_fraction: float = 0.2, random_seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    """
    Run negative control test and return controlled dataset with metadata.
    
    Args:
        control_type: Type of control ('label_shuffle' or 'feature_shuffle')
        df: Original dataset
        feature_cols: Feature column names
        label_col: Label column name
        shuffle_fraction: Fraction for feature shuffle
        random_seed: Random seed
        
    Returns:
        Tuple of (controlled_dataframe, control_metadata)
    """
    metadata = {
        'control_type': control_type,
        'original_shape': df.shape,
        'random_seed': random_seed,
        'features_available': len(feature_cols)
    }
    
    if control_type == 'label_shuffle':
        df_control = apply_date_wise_label_shuffle(df, label_col, random_seed)
        metadata['method'] = 'date_wise_label_permutation'
        
    elif control_type == 'feature_shuffle':
        df_control = apply_feature_shuffle(df, feature_cols, shuffle_fraction, random_seed)
        metadata['method'] = 'within_date_feature_permutation'
        metadata['shuffle_fraction'] = shuffle_fraction
        metadata['features_shuffled'] = int(len(feature_cols) * shuffle_fraction)
        
    elif control_type == 'label_lag_placebo':
        lag_days = 5  # Default lag
        df_control = apply_label_lag_placebo(df, feature_cols, lag_days, random_seed)
        metadata['method'] = 'temporal_feature_lag'
        metadata['lag_days'] = lag_days
        
    else:
        raise ValueError(f"Unknown control type: {control_type}")
    
    metadata['controlled_shape'] = df_control.shape
    
    return df_control, metadata


def validate_control_ic(ic_value: float, control_type: str, threshold: float = 0.01) -> bool:
    """
    Validate that control IC is near zero as expected.
    
    Args:
        ic_value: Information Coefficient from control run
        control_type: Type of control applied
        threshold: Maximum acceptable |IC| for control
        
    Returns:
        True if control passes validation
    """
    abs_ic = abs(ic_value)
    
    print(f"🔍 CONTROL VALIDATION ({control_type}):")
    print(f"   IC: {ic_value:.6f}")
    print(f"   |IC|: {abs_ic:.6f}")
    print(f"   Threshold: {threshold:.6f}")
    
    if abs_ic < threshold:
        print(f"✅ CONTROL PASSED: |IC| < {threshold}")
        return True
    else:
        print(f"🚨 CONTROL FAILED: |IC| ≥ {threshold}")
        print(f"   This suggests potential issues with the control implementation or data.")
        return False


if __name__ == "__main__":
    print("Negative controls module loaded successfully")
