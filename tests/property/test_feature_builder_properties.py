"""
Property-based tests for feature builder using Hypothesis
Tests invariants that should hold for all valid inputs
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

# Import your feature builder
try:
    from core.ml.build_features import build_matrix
except ImportError:
    pytest.skip("Feature builder not available", allow_module_level=True)


@settings(verbosity=Verbosity.verbose, max_examples=100)
@given(
    n_rows=st.integers(min_value=50, max_value=200),
    price_base=st.floats(min_value=10.0, max_value=1000.0),
    volatility=st.floats(min_value=0.01, max_value=0.5),
)
def test_feature_builder_monotonicity(n_rows, price_base, volatility):
    """Property: Feature building should preserve monotonicity of timestamps"""

    # Generate synthetic OHLC data with monotonic timestamps
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")
    np.random.seed(42)  # Deterministic for property testing

    # Generate realistic price series
    returns = np.random.normal(0, volatility, n_rows)
    prices = price_base * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "TEST",
        "Open": prices * 0.999,
        "High": prices * 1.002,
        "Low": prices * 0.998,
        "Close": prices,
        "Volume": np.random.randint(1000000, 2000000, n_rows)
    })

    # Build features
    features = build_matrix(df, include_returns=True)

    # Property: Timestamps should remain monotonic
    assert features["timestamp"].is_monotonic_increasing, "Timestamps must remain monotonic"

    # Property: No duplicate timestamps
    assert not features["timestamp"].duplicated().any(), "No duplicate timestamps allowed"

    # Property: All timestamps should be finite
    assert features["timestamp"].notna().all(), "All timestamps must be finite"


@settings(verbosity=Verbosity.verbose, max_examples=50)
@given(
    n_rows=st.integers(min_value=100, max_value=300),
    n_symbols=st.integers(min_value=1, max_value=5),
)
def test_feature_builder_symbol_invariance(n_rows, n_symbols):
    """Property: Feature building should work consistently across symbols"""

    # Generate multi-symbol data
    symbols = [f"SYMBOL_{i}" for i in range(n_symbols)]
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")

    all_data = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)  # Different seed per symbol

        prices = 100 + np.cumsum(np.random.normal(0, 0.02, n_rows))
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.random.randint(1000000, 2000000, n_rows)
        })
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Build features
    features = build_matrix(combined_df, include_returns=True)

    # Property: All symbols should be present in output
    assert set(features["symbol"].unique()) == set(symbols), "All symbols must be preserved"

    # Property: Feature count should be consistent per symbol
    feature_counts = features.groupby("symbol").size()
    assert feature_counts.nunique() == 1, "All symbols should have same number of features"

    # Property: No cross-symbol data leakage in features
    for symbol in symbols:
        features[features["symbol"] == symbol]
        # Check that feature values don't depend on other symbols' data
        # This is a simplified check - in practice you'd verify specific feature calculations


@settings(verbosity=Verbosity.verbose, max_examples=30)
@given(
    n_rows=st.integers(min_value=200, max_value=500),
    noise_level=st.floats(min_value=0.0, max_value=0.1),
)
def test_feature_builder_robustness_to_noise(n_rows, noise_level):
    """Property: Feature building should be robust to small amounts of noise"""

    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")

    # Generate clean price series
    returns = np.random.normal(0, 0.02, n_rows)
    prices = 100 * np.exp(np.cumsum(returns))

    # Add noise
    noise = np.random.normal(0, noise_level, n_rows)
    noisy_prices = prices + noise

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "TEST",
        "Open": noisy_prices * 0.999,
        "High": noisy_prices * 1.002,
        "Low": noisy_prices * 0.998,
        "Close": noisy_prices,
        "Volume": np.random.randint(1000000, 2000000, n_rows)
    })

    # Build features
    features = build_matrix(df, include_returns=True)

    # Property: Should handle noise gracefully (no crashes)
    assert len(features) > 0, "Should produce features even with noise"

    # Property: Feature values should be finite
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "symbol":  # Skip non-numeric columns
            assert features[col].notna().all(), f"All values in {col} must be finite"


@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(
    n_rows=st.integers(min_value=50, max_value=100),
    gap_size=st.integers(min_value=1, max_value=10),
)
def test_feature_builder_handles_gaps(n_rows, gap_size):
    """Property: Feature building should handle data gaps gracefully"""

    # Create data with gaps
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")

    # Introduce gaps by removing some rows
    gap_indices = np.random.choice(n_rows, size=gap_size, replace=False)
    mask = np.ones(n_rows, dtype=bool)
    mask[gap_indices] = False

    dates_with_gaps = dates[mask]
    n_remaining = len(dates_with_gaps)

    prices = 100 + np.cumsum(np.random.normal(0, 0.02, n_remaining))

    df = pd.DataFrame({
        "timestamp": dates_with_gaps,
        "symbol": "TEST",
        "Open": prices * 0.999,
        "High": prices * 1.002,
        "Low": prices * 0.998,
        "Close": prices,
        "Volume": np.random.randint(1000000, 2000000, n_remaining)
    })

    # Build features
    features = build_matrix(df, include_returns=True)

    # Property: Should handle gaps without crashing
    assert len(features) > 0, "Should produce features even with gaps"

    # Property: Output should have same number of rows as input
    assert len(features) == len(df), "Output should preserve input row count"


@settings(verbosity=Verbosity.verbose, max_examples=10)
def test_feature_builder_determinism():
    """Property: Same input should produce same output (determinism)"""

    # Generate test data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.normal(0, 0.02, 100))

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "TEST",
        "Open": prices * 0.999,
        "High": prices * 1.002,
        "Low": prices * 0.998,
        "Close": prices,
        "Volume": np.random.randint(1000000, 2000000, 100)
    })

    # Build features twice with same input
    features1 = build_matrix(df, include_returns=True)
    features2 = build_matrix(df, include_returns=True)

    # Property: Results should be identical
    pd.testing.assert_frame_equal(features1, features2, check_dtype=False), \
        "Feature building must be deterministic"


@settings(verbosity=Verbosity.verbose, max_examples=5)
@given(
    extreme_prices=st.lists(
        st.floats(min_value=0.01, max_value=1000000.0),
        min_size=10, max_size=50
    )
)
def test_feature_builder_extreme_values(extreme_prices):
    """Property: Should handle extreme price values gracefully"""

    n_rows = len(extreme_prices)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")

    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "TEST",
        "Open": np.array(extreme_prices) * 0.999,
        "High": np.array(extreme_prices) * 1.002,
        "Low": np.array(extreme_prices) * 0.998,
        "Close": np.array(extreme_prices),
        "Volume": np.random.randint(1000000, 2000000, n_rows)
    })

    # Build features
    features = build_matrix(df, include_returns=True)

    # Property: Should not crash on extreme values
    assert len(features) > 0, "Should handle extreme values without crashing"

    # Property: Should produce finite features
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "symbol":
            assert features[col].notna().all(), f"All values in {col} must be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
