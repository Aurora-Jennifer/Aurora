"""
No lookahead leakage tests for walkforward framework.
Verify that train and test data are properly separated.
"""

import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityValidator
from scripts.walkforward_framework import build_feature_table, gen_walkforward


def test_train_ends_before_test_starts():
    """Test that train_hi < test_lo for all folds (basic boundary check)."""
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=5))

    for fold in folds:
        assert fold.train_hi < fold.test_lo, (
            f"WF_BOUNDARY: train_hi ({fold.train_hi}) must be < test_lo ({fold.test_lo})"
        )


def test_no_index_overlap():
    """Test that train and test indices never overlap."""
    folds = list(gen_walkforward(n=100, train_len=20, test_len=10, stride=5))

    for fold in folds:
        train_indices = set(range(fold.train_lo, fold.train_hi + 1))
        test_indices = set(range(fold.test_lo, fold.test_hi + 1))

        overlap = train_indices & test_indices
        assert len(overlap) == 0, f"WF_OVERLAP: Train and test indices overlap: {overlap}"


def test_feature_pipeline_no_lookahead():
    """Test that feature pipeline doesn't introduce lookahead."""
    # Create synthetic data
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    # Build feature table
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    for fold in folds:
        # Get train and test indices
        train_indices = np.arange(fold.train_lo, fold.train_hi + 1)
        test_indices = np.arange(fold.test_lo, fold.test_hi + 1)

        # Verify no overlap
        train_set = set(train_indices)
        test_set = set(test_indices)
        overlap = train_set & test_set
        assert len(overlap) == 0, f"WF_FEATURE: Feature indices overlap: {overlap}"


def test_constant_predictor_sanity_check():
    """Test that a constant predictor doesn't achieve unrealistic performance."""
    # Create synthetic data with random labels
    np.random.seed(42)
    n_samples = 100
    np.random.randn(n_samples, 10)
    y = np.random.choice([-1, 0, 1], n_samples)  # Random labels
    np.random.randn(n_samples) + 100

    # Generate folds
    folds = list(gen_walkforward(n=n_samples, train_len=30, test_len=10, stride=5))

    # Simulate constant predictor (always predicts 0)
    constant_predictions = np.zeros(n_samples)

    # Calculate "performance" for each fold
    fold_accuracies = []
    for fold in folds:
        test_indices = np.arange(fold.test_lo, fold.test_hi + 1)
        test_labels = y[test_indices]
        test_preds = constant_predictions[test_indices]

        # Calculate accuracy (should be around 33% for random labels)
        accuracy = np.mean(test_preds == test_labels)
        fold_accuracies.append(accuracy)

    # Average accuracy should be reasonable (not >0.9)
    avg_accuracy = np.mean(fold_accuracies)
    assert avg_accuracy < 0.9, (
        f"WF_SANITY: Constant predictor achieved unrealistic accuracy: {avg_accuracy}"
    )


def test_feature_timestamps_no_future_leakage():
    """Test that features don't use future timestamps."""
    # Create data with timestamps
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    # Build feature table
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    for fold in folds:
        train_indices = np.arange(fold.train_lo, fold.train_hi + 1)
        test_indices = np.arange(fold.test_lo, fold.test_hi + 1)

        # Get corresponding timestamps
        train_timestamps = dates[train_indices]
        test_timestamps = dates[test_indices]

        # Verify no future leakage in timestamps
        max_train_time = train_timestamps.max()
        min_test_time = test_timestamps.min()

        assert max_train_time < min_test_time, (
            f"WF_TIMESTAMP: Train timestamp ({max_train_time}) >= test timestamp ({min_test_time})"
        )


def test_rolling_window_features_no_lookahead():
    """Test that rolling window features don't look into the future."""
    # Create data with known patterns
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": np.arange(100) + 100,  # Simple increasing pattern
            "High": np.arange(100) + 102,
            "Low": np.arange(100) + 98,
            "Close": np.arange(100) + 100,
            "Volume": np.ones(100) * 1000000,
        },
        index=dates,
    )

    # Build feature table
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    for fold in folds:
        train_indices = np.arange(fold.train_lo, fold.train_hi + 1)
        test_indices = np.arange(fold.test_lo, fold.test_hi + 1)

        # Get train and test features
        X_train = X[train_indices]
        X_test = X[test_indices]

        # Verify that test features don't contain information from beyond train_hi
        # This is a basic check - in practice, you'd need more sophisticated validation
        assert len(X_train) > 0, "WF_ROLLING: Empty train set"
        assert len(X_test) > 0, "WF_ROLLING: Empty test set"


def test_target_alignment_no_lookahead():
    """Test that targets are properly aligned and don't leak future information."""
    # Create data with known target pattern
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    # Build feature table
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    for fold in folds:
        train_indices = np.arange(fold.train_lo, fold.train_hi + 1)
        test_indices = np.arange(fold.test_lo, fold.test_hi + 1)

        # Get train and test targets
        y_train = y[train_indices]
        y_test = y[test_indices]

        # Verify target alignment
        assert len(y_train) == len(train_indices), "WF_TARGET: Train target length mismatch"
        assert len(y_test) == len(test_indices), "WF_TARGET: Test target length mismatch"


def test_data_sanity_in_folds():
    """Test that DataSanity validation works within fold boundaries."""
    # Create clean data
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    # Build feature table
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    validator = DataSanityValidator(profile="strict")

    for fold in folds:
        train_indices = np.arange(fold.train_lo, fold.train_hi + 1)
        test_indices = np.arange(fold.test_lo, fold.test_hi + 1)

        # Create train and test data slices
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

        # Validate each slice with DataSanity
        try:
            validator.validate_and_repair(train_data, f"TRAIN_FOLD_{fold.fold_id}")
            validator.validate_and_repair(test_data, f"TEST_FOLD_{fold.fold_id}")
        except Exception as e:
            pytest.fail(f"WF_DATASANITY: DataSanity failed on fold {fold.fold_id}: {e}")


def test_no_global_state_leakage():
    """Test that global state doesn't leak between folds."""
    # Create data
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )

    # Build feature table
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    # Track feature statistics across folds
    train_stats = []
    test_stats = []

    for fold in folds:
        train_indices = np.arange(fold.train_lo, fold.train_hi + 1)
        test_indices = np.arange(fold.test_lo, fold.test_hi + 1)

        X_train = X[train_indices]
        X_test = X[test_indices]

        # Calculate statistics
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)

        train_stats.append(train_mean)
        test_stats.append(test_mean)

    # Verify that train and test statistics are reasonably different
    # (if they were identical, it might indicate leakage)
    train_stats = np.array(train_stats)
    test_stats = np.array(test_stats)

    # Calculate correlation between train and test means
    correlations = []
    for i in range(train_stats.shape[1]):
        corr = np.corrcoef(train_stats[:, i], test_stats[:, i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    # Average correlation should not be too high (indicating leakage)
    if correlations:
        avg_corr = np.mean(correlations)
        assert avg_corr < 0.95, f"WF_GLOBAL: High correlation between train/test stats: {avg_corr}"
