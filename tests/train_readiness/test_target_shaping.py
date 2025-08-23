import numpy as np
import pandas as pd
import pytest

from core.ml.shape_labels import shape_labels, validate_no_leakage


@pytest.mark.sanity
def test_rank_labels_no_leakage():
    """Test that rank labels don't introduce leakage"""
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    y_raw = pd.Series(np.random.randn(500), index=dates)

    config = {
        "method": "rank",
        "rank_window": 252,
        "options": {
            "rank_type": "time_series",
            "rank_method": "percentile"
        }
    }

    y_raw, y_ranked = shape_labels(y_raw, config)

    # Check no leakage
    assert validate_no_leakage(y_raw, y_ranked, config)

    # Check that ranked values are between 0 and 1
    assert y_ranked.min() >= 0
    assert y_ranked.max() <= 1

    # Check monotonicity (correlation should be positive)
    corr = y_raw.corr(y_ranked)
    assert corr > 0.5


@pytest.mark.sanity
def test_winsorize_labels():
    """Test winsorization of labels"""
    # Create synthetic data with outliers
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    y_raw = pd.Series(np.random.randn(500), index=dates)
    # Add some outliers
    y_raw.iloc[0] = 10.0
    y_raw.iloc[1] = -10.0

    config = {
        "method": "winsorize",
        "winsor_limits": [0.01, 0.99],
        "options": {
            "winsorize_symmetric": True
        }
    }

    y_raw, y_winsorized = shape_labels(y_raw, config)

    # Check that outliers are clipped
    assert y_winsorized.max() < 10.0
    assert y_winsorized.min() > -10.0

    # Check that most values are unchanged
    unchanged_ratio = (y_raw == y_winsorized).mean()
    assert unchanged_ratio > 0.95  # 95% should be unchanged


@pytest.mark.sanity
def test_zscore_labels():
    """Test z-score normalization of labels"""
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    y_raw = pd.Series(np.random.randn(500), index=dates)

    config = {
        "method": "zscore",
        "zscore_window": 252,
        "options": {
            "zscore_robust": False
        }
    }

    y_raw, y_zscore = shape_labels(y_raw, config)

    # Check that z-scores have reasonable range
    assert y_zscore.abs().max() < 5.0  # Shouldn't have extreme z-scores

    # Check that transformation preserves sign
    sign_preserved = ((y_raw > 0) == (y_zscore > 0)).mean()
    assert sign_preserved > 0.8  # 80% should preserve sign


@pytest.mark.sanity
def test_leakage_detection():
    """Test that leakage detection works"""
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    y_raw = pd.Series(np.random.randn(500), index=dates)

    # Create a "leaky" transformation (using future values)
    y_leaky = y_raw.shift(-1)  # This would be leakage

    config = {
        "method": "rank",
        "rank_window": 252
    }

    # This should detect leakage
    assert not validate_no_leakage(y_raw, y_leaky, config)
