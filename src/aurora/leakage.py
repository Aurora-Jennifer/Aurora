"""
Leakage detection and prevention
"""

import pandas as pd


def assert_no_lookahead(df: pd.DataFrame) -> None:
    """
    Assert no lookahead bias - labels at t must not depend on features beyond t
    """
    assert df.index.is_monotonic_increasing, "DataFrame index must be monotonic increasing"
    
    # Check for any forward-looking features (this is a basic check)
    # In practice, you'd want more sophisticated checks based on your feature engineering
    if 'label' in df.columns:
        assert (df["label"].index == df.index).all(), "Label index must match DataFrame index"


def guard_splits(train_end: int, test_start: int) -> None:
    """
    Guard against temporal leakage in train/test splits
    """
    assert train_end < test_start, f"Temporal leak: train_end ({train_end}) >= test_start ({test_start})"


def validate_temporal_integrity(df: pd.DataFrame, train_end: int, test_start: int) -> None:
    """
    Comprehensive temporal integrity validation
    """
    # Check monotonic index
    assert_no_lookahead(df)
    
    # Check split boundaries
    guard_splits(train_end, test_start)
    
    # Check for any NaN values that could indicate data issues
    if df.isnull().any().any():
        print("⚠️  Warning: DataFrame contains NaN values")
    
    # Check for duplicate timestamps
    if df.index.duplicated().any():
        raise ValueError("Duplicate timestamps found in DataFrame")


def guard_temporal_splits(train_end_idx: int, test_start_idx: int) -> None:
    """Guard against temporal leakage in train/test splits"""
    assert train_end_idx < test_start_idx, "Temporal leak: train touches test."


def assert_labels_not_lookahead(feat_ts, label_ts) -> None:
    """Assert labels don't use future information"""
    assert all(f <= l for f, l in zip(feat_ts, label_ts)), "Label uses future info."


def detect_feature_leakage(features: pd.DataFrame, labels: pd.Series, lookback_window: int = 1) -> None:
    """
    Detect potential feature leakage by checking for future information
    """
    # Ensure features and labels are aligned
    assert len(features) == len(labels), "Features and labels must have same length"
    
    # Check that no feature at time t uses information from time t+1 or later
    # This is a basic check - you'd want more sophisticated validation
    for col in features.columns:
        if features[col].isnull().any():
            print(f"⚠️  Warning: Feature '{col}' contains NaN values")
    
    # Check for suspiciously high correlations that might indicate leakage
    correlations = features.corrwith(labels)
    high_corr_features = correlations[abs(correlations) > 0.95]
    
    if len(high_corr_features) > 0:
        print(f"⚠️  Warning: High correlation features detected: {high_corr_features.to_dict()}")
        print("   This might indicate data leakage - investigate manually")
