"""
Fixtures for DataSanity test suite.
"""

import os
from datetime import timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def seed_rng():
    """Seed RNG for deterministic tests."""
    np.random.seed(42)
    return 42


@pytest.fixture(scope="session", autouse=True)
def force_strict_mode():
    """Force strict mode for all tests."""
    # Set environment variable to force strict mode
    os.environ["DATASANITY_PROFILE"] = "strict"
    yield
    # Clean up
    if "DATASANITY_PROFILE" in os.environ:
        del os.environ["DATASANITY_PROFILE"]


@pytest.fixture
def tz_utc():
    """UTC timezone for consistent testing."""
    return timezone.utc


@pytest.fixture
def mk_ts():
    """Create clean time series DataFrame with OHLCV data."""

    def _mk_ts(n=100, start="2023-01-01", freq="1min", tz=timezone.utc):
        """Create a clean OHLCV DataFrame with realistic values.

        Args:
            n: Number of rows
            start: Start date string
            freq: Frequency string
            tz: Timezone

        Returns:
            pd.DataFrame with OHLCV columns and DatetimeIndex
        """
        # Create datetime index
        dates = pd.date_range(start=start, periods=n, freq=freq, tz=tz)

        # Generate realistic price data without lookahead
        np.random.seed(42)  # Ensure deterministic
        base_price = 100.0

        # Generate prices without using future information
        prices = [base_price]
        for i in range(1, n):
            # Simple random walk without lookahead
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Ensure positive prices

        prices = np.array(prices)

        # Create OHLCV data without lookahead
        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.001, n)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
                "Close": prices,
                "Volume": np.random.lognormal(10, 0.5, n),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        # Ensure all values are finite and positive
        data = data.clip(lower=0.01)  # Minimum price of 1 cent

        return data

    return _mk_ts


@pytest.fixture
def strict_validator():
    """Create a DataSanityValidator in strict mode."""
    from core.data_sanity import DataSanityValidator

    # Create validator with strict profile
    validator = DataSanityValidator(profile="strict")
    return validator


@pytest.fixture
def default_validator():
    """Create a DataSanityValidator in default mode."""
    from core.data_sanity import DataSanityValidator

    # Create validator with default profile
    validator = DataSanityValidator(profile="default")
    return validator
