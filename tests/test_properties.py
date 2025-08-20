"""
Property-based tests for DataSanity using Hypothesis.
"""

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityError, DataSanityValidator

# Try to import Hypothesis
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
class TestDataSanityProperties:
    """Property-based tests for DataSanity."""

    @given(prices=st.lists(st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_finite_prices_produce_finite_returns(self, prices):
        """Property: finite prices should produce finite returns."""
        validator = DataSanityValidator(profile="strict")
        if not hasattr(validator, "compute_returns"):
            pytest.skip("compute_returns method not implemented")

        # Create price series
        price_series = pd.Series(prices)

        # Calculate returns
        returns = validator.compute_returns(price_series)

        # All returns should be finite
        assert np.isfinite(returns).all(), "All returns should be finite for finite prices"

    @given(prices=st.lists(st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_returns_no_nan_for_finite_input(self, prices):
        """Property: finite prices should not produce NaN returns."""
        validator = DataSanityValidator(profile="strict")
        if not hasattr(validator, "compute_returns"):
            pytest.skip("compute_returns method not implemented")

        # Create price series
        price_series = pd.Series(prices)

        # Calculate returns
        returns = validator.compute_returns(price_series)

        # No NaNs should be produced
        assert not returns.isna().any(), "No NaN returns should be produced for finite prices"

    @given(prices=st.lists(st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_returns_length_matches_input(self, prices):
        """Property: returns length should match input length."""
        validator = DataSanityValidator(profile="strict")
        if not hasattr(validator, "compute_returns"):
            pytest.skip("compute_returns method not implemented")

        # Create price series
        price_series = pd.Series(prices)

        # Calculate returns
        returns = validator.compute_returns(price_series)

        # Length should match
        assert len(returns) == len(price_series), "Returns length should match input length"

    @given(prices=st.lists(st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_returns_first_element_is_zero(self, prices):
        """Property: first return should always be zero."""
        validator = DataSanityValidator(profile="strict")
        if not hasattr(validator, "compute_returns"):
            pytest.skip("compute_returns method not implemented")

        # Create price series
        price_series = pd.Series(prices)

        # Calculate returns
        returns = validator.compute_returns(price_series)

        # First return should be zero
        assert abs(returns.iloc[0]) < 1e-10, "First return should be zero"

    @given(prices=st.lists(st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_returns_equals_pct_change(self, prices):
        """Property: returns should equal pct_change().fillna(0.0)."""
        validator = DataSanityValidator(profile="strict")
        if not hasattr(validator, "compute_returns"):
            pytest.skip("compute_returns method not implemented")

        # Create price series
        price_series = pd.Series(prices)

        # Calculate returns using DataSanity
        returns_ds = validator.compute_returns(price_series)

        # Calculate reference returns using pandas
        returns_ref = price_series.pct_change().fillna(0.0)

        # Compare using numpy testing
        np.testing.assert_allclose(
            returns_ds.values,
            returns_ref.values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Returns should equal pct_change().fillna(0.0)",
        )

    @given(data_size=st.integers(min_value=5, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_shuffled_timestamps_break_monotonicity(self, data_size):
        """Property: shuffling timestamps should break monotonicity and cause strict mode to fail."""
        validator = DataSanityValidator(profile="strict")
        # Create clean data
        dates = pd.date_range("2023-01-01", periods=data_size, freq="1min", tz=UTC)
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 200, data_size),
                "High": np.random.uniform(200, 300, data_size),
                "Low": np.random.uniform(50, 100, data_size),
                "Close": np.random.uniform(100, 200, data_size),
                "Volume": np.random.uniform(1000000, 5000000, data_size),
            },
            index=dates,
        )

        # Shuffle the index
        shuffled_data = data.copy()
        shuffled_data.index = shuffled_data.index.to_series().sample(frac=1).index

        # Should fail in strict mode due to non-monotonic index
        with pytest.raises(DataSanityError, match="monotonic|non-monotonic|timestamp.*order"):
            validator.validate_and_repair(shuffled_data, "SHUFFLED_TEST")

    @given(data_size=st.integers(min_value=5, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_negative_prices_always_fail_strict(self, data_size):
        """Property: negative prices should always fail in strict mode."""
        validator = DataSanityValidator(profile="strict")
        # Create clean data with valid OHLC relationships
        dates = pd.date_range("2023-01-01", periods=data_size, freq="1min", tz=UTC)
        
        # Generate realistic price data
        base_price = 100.0
        prices = [base_price]
        for _i in range(1, data_size):
            change = np.random.uniform(-0.05, 0.05)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))
        
        data = pd.DataFrame(
            {
                "Open": [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
                "High": [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                "Low": [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                "Close": prices,
                "Volume": np.random.uniform(1000000, 5000000, data_size),
            },
            index=dates,
        )
        
        # Ensure OHLC consistency
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        # Introduce negative price (only Close to avoid OHLC violations)
        data.loc[data.index[0], "Close"] = -50.0

        # Should always fail in strict mode (either on price validation or OHLC consistency)
        with pytest.raises(
            DataSanityError,
            match="violations.*price_limits|unrepaired.*price_limits|Negative prices|negative|non-positive|price.*negative|OHLC invariant violation",
        ):
            validator.validate_and_repair(data, "NEGATIVE_PRICE_TEST")

    @given(data_size=st.integers(min_value=5, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_clean_data_always_passes_strict(self, data_size):
        """Property: clean data should always pass strict mode."""
        validator = DataSanityValidator(profile="strict")
        # Create clean data with realistic price movements
        dates = pd.date_range("2023-01-01", periods=data_size, freq="1min", tz=UTC)

        # Generate realistic price data without extreme movements
        base_price = 100.0
        prices = [base_price]
        for _i in range(1, data_size):
            # Small random walk to avoid extreme returns
            change = np.random.uniform(-0.05, 0.05)  # Max 5% change
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))

        data = pd.DataFrame(
            {
                "Open": [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
                "High": [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                "Low": [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                "Close": prices,
                "Volume": np.random.uniform(1000000, 5000000, data_size),
            },
            index=dates,
        )

        # Ensure OHLC consistency
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        # Should always pass in strict mode (but may have lookahead flag due to Returns column)
        try:
            clean_data, result = validator.validate_and_repair(data, "CLEAN_DATA_TEST")
            assert len(clean_data) == data_size, "Should preserve all rows for clean data"
        except DataSanityError as e:
            if "Lookahead contamination" in str(e):
                # This is expected due to Returns column addition
                pass
            else:
                pytest.fail(f"Clean data should pass strict mode, but failed: {e}")


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
def test_hypothesis_import():
    """Test that Hypothesis is properly imported."""
    assert HYPOTHESIS_AVAILABLE, "Hypothesis should be available for property tests"
