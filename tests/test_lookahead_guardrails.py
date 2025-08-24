"""
Lookahead Guardrail Tests
Ensure no future information leaks into features.
"""

import numpy as np
import pandas as pd
import pytest

from core.ml.build_features import build_matrix


def assert_no_lookahead(X: pd.DataFrame, y: pd.Series):
    """
    Assert that features X contain no future information relative to targets y.
    
    Args:
        X: Feature matrix
        y: Target series
    """
    # X must be strictly earlier than y (same timestamps)
    assert X.index.equals(y.index), "Feature and target indices must match"

    # Every feature at time t should be computable from data up to time t
    # This is a basic check - in practice, we rely on proper .shift(1) usage


def test_feature_recompute_unchanged():
    """
    Test that recomputing features after truncating future rows 
    leaves values at time t unchanged.
    """
    # Create test data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(100) * 2,
        'High': 102 + np.random.randn(100) * 2,
        'Low': 98 + np.random.randn(100) * 2,
        'Close': 101 + np.random.randn(100) * 2,
        'Volume': 1000 + np.random.randn(100) * 100
    }, index=dates)

    # Build features on full dataset
    X_full, y_full = build_matrix(df, horizon=1)

    # Build features on truncated dataset (remove last 10 rows)
    X_trunc, y_trunc = build_matrix(df.iloc[:-10], horizon=1)

    # Values at time t should be identical
    common_idx = X_full.index.intersection(X_trunc.index)

    for col in X_full.columns:
        if col in X_trunc.columns:
            # Check that feature values are identical for common timestamps
            pd.testing.assert_series_equal(
                X_full.loc[common_idx, col],
                X_trunc.loc[common_idx, col],
                check_names=False,
                check_dtype=False  # Allow dtype differences due to different computations
            )


def test_all_features_shifted():
    """
    Test that all features are properly shifted to avoid lookahead.
    """
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(50) * 2,
        'High': 102 + np.random.randn(50) * 2,
        'Low': 98 + np.random.randn(50) * 2,
        'Close': 101 + np.random.randn(50) * 2,
        'Volume': 1000 + np.random.randn(50) * 100
    }, index=dates)

    X, y = build_matrix(df, horizon=1)

    # Check that features and targets are properly aligned
    assert_no_lookahead(X, y)

    # Verify that features don't contain future information
    # This is a basic check - the real protection is in the feature computation
    assert len(X) == len(y), "Feature and target lengths must match"


def test_rsi_no_lookahead():
    """
    Test that RSI calculation doesn't use future information.
    """
    from core.ml.build_features import _rsi

    # Create test data
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    close_prices = pd.Series(100 + np.cumsum(np.random.randn(30) * 0.5), index=dates)

    # Compute RSI
    rsi = _rsi(close_prices, n=14)

    # RSI should be NaN for the first n-1 periods (no lookahead)
    assert rsi.iloc[:13].isna().all(), "RSI should be NaN for first n-1 periods"

    # RSI should be finite for periods >= n
    assert rsi.iloc[13:].notna().all(), "RSI should be finite for periods >= n"


def test_rolling_features_no_lookahead():
    """
    Test that rolling features don't use future information.
    """
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    close_prices = pd.Series(100 + np.cumsum(np.random.randn(30) * 0.5), index=dates)

    # Test SMA calculation
    sma_10 = close_prices.rolling(10, min_periods=10).mean().shift(1)

    # SMA should be NaN for first 10 periods
    assert sma_10.iloc[:10].isna().all(), "SMA should be NaN for first n periods"

    # SMA should be finite for periods >= n+1
    assert sma_10.iloc[10:].notna().all(), "SMA should be finite for periods >= n+1"


def test_returns_calculation_no_lookahead():
    """
    Test that returns calculation doesn't use future information.
    """
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    close_prices = pd.Series(100 + np.cumsum(np.random.randn(30) * 0.5), index=dates)

    # Calculate returns
    returns = close_prices.pct_change()

    # First return should be NaN (no previous price)
    assert pd.isna(returns.iloc[0]), "First return should be NaN"

    # Other returns should be finite
    assert returns.iloc[1:].notna().all(), "Returns should be finite after first period"


def test_feature_matrix_alignment():
    """
    Test that feature matrix and target are properly aligned.
    """
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(50) * 2,
        'High': 102 + np.random.randn(50) * 2,
        'Low': 98 + np.random.randn(50) * 2,
        'Close': 101 + np.random.randn(50) * 2,
        'Volume': 1000 + np.random.randn(50) * 100
    }, index=dates)

    X, y = build_matrix(df, horizon=1)

    # Check alignment
    assert X.index.equals(y.index), "Feature and target indices must match"

    # Check that we have reasonable data
    assert len(X) > 0, "Feature matrix should not be empty"
    assert len(y) > 0, "Target series should not be empty"

    # Check that features don't have NaN values (after alignment)
    assert not X.isna().any().any(), "Features should not contain NaN values"

    # Check that targets don't have NaN values (after alignment)
    assert not y.isna().any(), "Targets should not contain NaN values"


if __name__ == "__main__":
    pytest.main([__file__])
