"""
L0 Gate: Timezone Contract
Ensures all timestamps are UTC and bars are within market hours.
"""

from pathlib import Path

import pandas as pd
import pytest


def test_timestamps_are_utc(e2d_output):
    """Test that all timestamps in E2D output are UTC."""
    # Load the E2D output data
    if hasattr(e2d_output, 'df'):
        df = e2d_output.df
    else:
        # Assume e2d_output is a path to the output directory
        output_path = Path(e2d_output)
        if (output_path / "data.parquet").exists():
            df = pd.read_parquet(output_path / "data.parquet")
        else:
            pytest.skip("No data.parquet found in E2D output")

    # Check that index has timezone info
    assert df.index.tz is not None, "DataFrame index must have timezone info"
    assert str(df.index.tz) == "UTC", f"Expected UTC timezone, got {df.index.tz}"


def test_bars_within_market_hours(e2d_output, market_calendar):
    """Test that all bars are within market hours."""
    # Load the E2D output data
    if hasattr(e2d_output, 'df'):
        df = e2d_output.df
    else:
        output_path = Path(e2d_output)
        if (output_path / "data.parquet").exists():
            df = pd.read_parquet(output_path / "data.parquet")
        else:
            pytest.skip("No data.parquet found in E2D output")

    # Simple market hours check (9:30 AM - 4:00 PM ET, Monday-Friday)
    # Convert to UTC (ET is UTC-5 or UTC-4 depending on DST)
    # For simplicity, check 14:30-21:00 UTC (covers both EST/EDT)
    bad_bars = []
    for timestamp in df.index:
        # Check if it's a weekday
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            bad_bars.append(timestamp)
            continue

        # Check if it's within market hours (14:30-21:00 UTC)
        hour = timestamp.hour
        minute = timestamp.minute
        time_minutes = hour * 60 + minute

        if time_minutes < 14 * 60 + 30 or time_minutes > 21 * 60:  # Before 14:30 or after 21:00
            bad_bars.append(timestamp)

    assert len(bad_bars) == 0, f"Found {len(bad_bars)} out-of-hours bars: {bad_bars[:5]}"


def test_timestamp_monotonicity(e2d_output):
    """Test that timestamps are monotonically increasing."""
    if hasattr(e2d_output, 'df'):
        df = e2d_output.df
    else:
        output_path = Path(e2d_output)
        if (output_path / "data.parquet").exists():
            df = pd.read_parquet(output_path / "data.parquet")
        else:
            pytest.skip("No data.parquet found in E2D output")

    # Check that index is monotonically increasing
    assert df.index.is_monotonic_increasing, "Timestamps must be monotonically increasing"


@pytest.fixture
def e2d_output():
    """Fixture providing E2D output path."""
    return "artifacts/e2d/last"


@pytest.fixture
def market_calendar():
    """Fixture providing market calendar for validation."""
    # Simple market calendar implementation
    class MarketCalendar:
        def open_bars(self):
            """Return pandas DatetimeIndex of market open bars."""
            # This would be implemented with actual market calendar logic
            # For now, return empty index as placeholder
            return pd.DatetimeIndex([])

    return MarketCalendar()
