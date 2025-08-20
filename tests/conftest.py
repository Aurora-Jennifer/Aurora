"""
Fixtures for DataSanity test suite.
"""

import os
import pathlib
import socket
from datetime import UTC

import numpy as np
import pandas as pd
import pytest


def pytest_collection_modifyitems(config, items):
    """Skip attic/quarantine tests by default."""
    attic = pathlib.Path("attic").resolve()
    skip_attic = pytest.mark.skip(reason="Skipped attic/quarantine tests")
    for it in items:
        p = pathlib.Path(str(getattr(it, "fspath", ""))).resolve()
        if attic in p.parents:
            it.add_marker(skip_attic)


@pytest.fixture(scope="session", autouse=True)
def seed_rng():
    """Seed RNG for deterministic tests."""
    np.random.seed(42)
    return 42


@pytest.fixture(scope="session", autouse=True)
def no_network():
    """Block network access during tests."""
    # Only enforce when tests explicitly require offline or for whole suite
    if os.getenv("AURORA_TEST_OFFLINE", "1") != "1":
        return
    
    def guard(*a, **k):
        raise RuntimeError("Network access is forbidden in tests")
    
    # Store original functions to restore later
    original_socket = socket.socket
    original_urlopen = None
    
    try:
        import urllib.request as _u
        original_urlopen = _u.urlopen
        _u.urlopen = guard
    except Exception:
        pass
    
    socket.socket = guard
    
    yield
    
    # Restore original functions
    socket.socket = original_socket
    if original_urlopen:
        try:
            import urllib.request as _u
            _u.urlopen = original_urlopen
        except Exception:
            pass


@pytest.fixture(scope="session", autouse=True)
def force_strict_mode():
    """Force strict mode for all tests."""
    # Set environment variable to force strict mode
    os.environ["DATASANITY_PROFILE"] = "strict"
    os.environ["AURORA_SANITY_MODE"] = "raise"
    yield
    # Clean up
    if "DATASANITY_PROFILE" in os.environ:
        del os.environ["DATASANITY_PROFILE"]
    if "AURORA_SANITY_MODE" in os.environ:
        del os.environ["AURORA_SANITY_MODE"]


@pytest.fixture
def tz_utc():
    """UTC timezone for consistent testing."""
    return UTC


@pytest.fixture
def mk_ts():
    """Create clean time series DataFrame with OHLCV data."""

    def _mk_ts(n=100, start="2023-01-01", freq="1min", tz=UTC):
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
        for _i in range(1, n):
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
