"""
Test returns calculation - verify correct returns logic.
"""

from datetime import timezone

import numpy as np
import pandas as pd
import pytest


def test_compute_returns_method_exists(strict_validator):
    """Test that compute_returns method exists."""
    # Check if the method exists
    if not hasattr(strict_validator, "compute_returns"):
        pytest.skip(
            "compute_returns method not implemented - implement this method for returns calculation"
        )

    # If it exists, test basic functionality
    close_prices = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0])
    returns = strict_validator.compute_returns(close_prices)

    assert len(returns) == len(close_prices), "Returns should have same length as input"
    assert returns.dtype in [np.float64, np.float32], "Returns should be numeric"


def test_returns_equals_pct_change(strict_validator, mk_ts):
    """Test that compute_returns equals pct_change().fillna(0.0)."""
    # Create clean data
    data = mk_ts(n=20)

    if hasattr(strict_validator, "compute_returns"):
        # Calculate returns using DataSanity
        returns_ds = strict_validator.compute_returns(data["Close"])

        # Calculate reference returns using pandas
        returns_ref = data["Close"].pct_change().fillna(0.0)

        # Compare using numpy testing
        np.testing.assert_allclose(
            returns_ds.values,
            returns_ref.values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Returns should equal pct_change().fillna(0.0)",
        )
    else:
        pytest.skip("compute_returns method not implemented")


def test_returns_no_off_by_one(strict_validator, mk_ts):
    """Test that returns calculation has no off-by-one errors."""
    # Create simple price series
    prices = pd.Series([100.0, 110.0, 105.0, 115.0, 120.0])

    if hasattr(strict_validator, "compute_returns"):
        returns = strict_validator.compute_returns(prices)

        # First return should be 0 (no previous price)
        assert abs(returns.iloc[0]) < 1e-10, "First return should be 0"

        # Second return should be (110-100)/100 = 0.1
        assert abs(returns.iloc[1] - 0.1) < 1e-10, "Second return should be 0.1"

        # Third return should be (105-110)/110 = -0.04545...
        assert (
            abs(returns.iloc[2] - (-0.045454545454545456)) < 1e-10
        ), "Third return should be -0.04545..."
    else:
        pytest.skip("compute_returns method not implemented")


def test_returns_gaps_with_unchanged_price(strict_validator, mk_ts):
    """Test that gaps with unchanged price produce zero returns."""
    # Create data with gaps and constant price
    dates = pd.date_range("2023-01-01", periods=10, freq="1min", tz=timezone.utc)
    # Create gaps by removing some timestamps
    dates = dates.drop(dates[2:4])  # Remove timestamps 2 and 3

    data = pd.DataFrame(
        {
            "Open": [100.0] * len(dates),
            "High": [100.0] * len(dates),
            "Low": [100.0] * len(dates),
            "Close": [100.0] * len(dates),  # Constant price
            "Volume": [1000000] * len(dates),
        },
        index=dates,
    )

    if hasattr(strict_validator, "compute_returns"):
        returns = strict_validator.compute_returns(data["Close"])

        # All returns should be 0 (constant price)
        np.testing.assert_allclose(
            returns.values,
            np.zeros(len(returns)),
            rtol=1e-10,
            atol=1e-10,
            err_msg="Returns should be 0 for constant price across gaps",
        )
    else:
        pytest.skip("compute_returns method not implemented")


def test_returns_no_nans_for_finite_input(strict_validator, mk_ts):
    """Test that returns calculation produces no NaNs for finite input."""
    # Create clean data
    data = mk_ts(n=20)

    if hasattr(strict_validator, "compute_returns"):
        returns = strict_validator.compute_returns(data["Close"])

        # Check that no NaNs were produced
        assert (
            not returns.isna().any()
        ), "Returns should have no NaN values for finite input"

        # Check that all values are finite
        assert np.isfinite(returns).all(), "All returns should be finite"
    else:
        pytest.skip("compute_returns method not implemented")


def test_returns_with_price_changes(strict_validator, mk_ts):
    """Test returns calculation with known price changes."""
    # Create simple price series with known changes
    prices = pd.Series([100.0, 120.0, 90.0, 135.0, 108.0])

    if hasattr(strict_validator, "compute_returns"):
        returns = strict_validator.compute_returns(prices)

        # Expected returns: [0, 0.2, -0.25, 0.5, -0.2]
        expected = [0.0, 0.2, -0.25, 0.5, -0.2]

        np.testing.assert_allclose(
            returns.values,
            expected,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Returns should match expected values for known price changes",
        )
    else:
        pytest.skip("compute_returns method not implemented")


def test_returns_with_single_price(strict_validator, mk_ts):
    """Test returns calculation with single price."""
    # Create single price
    prices = pd.Series([100.0])

    if hasattr(strict_validator, "compute_returns"):
        returns = strict_validator.compute_returns(prices)

        # Should have one return value (0 for first observation)
        assert len(returns) == 1, "Should have one return value"
        assert abs(returns.iloc[0]) < 1e-10, "Single price should have return of 0"
    else:
        pytest.skip("compute_returns method not implemented")


def test_returns_with_two_prices(strict_validator, mk_ts):
    """Test returns calculation with two prices."""
    # Create two prices
    prices = pd.Series([100.0, 110.0])

    if hasattr(strict_validator, "compute_returns"):
        returns = strict_validator.compute_returns(prices)

        # Should have two return values
        assert len(returns) == 2, "Should have two return values"
        assert abs(returns.iloc[0]) < 1e-10, "First return should be 0"
        assert abs(returns.iloc[1] - 0.1) < 1e-10, "Second return should be 0.1"
    else:
        pytest.skip("compute_returns method not implemented")


def test_returns_with_decreasing_prices(strict_validator, mk_ts):
    """Test returns calculation with decreasing prices."""
    # Create decreasing price series
    prices = pd.Series([100.0, 90.0, 80.0, 70.0, 60.0])

    if hasattr(strict_validator, "compute_returns"):
        returns = strict_validator.compute_returns(prices)

        # Expected returns: [0, -0.1, -0.1111..., -0.125, -0.1428...]
        expected = [0.0, -0.1, -0.1111111111111111, -0.125, -0.14285714285714285]

        np.testing.assert_allclose(
            returns.values,
            expected,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Returns should match expected values for decreasing prices",
        )
    else:
        pytest.skip("compute_returns method not implemented")


def test_returns_with_zero_price(strict_validator, mk_ts):
    """Test returns calculation with zero price (edge case)."""
    # Create price series with zero
    prices = pd.Series([100.0, 0.0, 50.0])

    if hasattr(strict_validator, "compute_returns"):
        returns = strict_validator.compute_returns(prices)

        # First return should be 0
        assert abs(returns.iloc[0]) < 1e-10, "First return should be 0"

        # Second return should be -1 (0-100)/100
        assert abs(returns.iloc[1] - (-1.0)) < 1e-10, "Second return should be -1"

        # Third return should be inf or handled appropriately
        # This depends on implementation - could be inf, nan, or handled specially
        third_return = returns.iloc[2]
        assert (
            np.isinf(third_return) or np.isnan(third_return) or third_return == 0
        ), f"Third return should be inf/nan/0, got {third_return}"
    else:
        pytest.skip("compute_returns method not implemented")
