"""
Test edge cases - verify graceful handling of missing columns and mixed timezones.
"""

from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityError


def test_missing_required_columns(strict_validator, mk_ts):
    """Test that missing required columns raise clear error."""
    # Create clean data
    data = mk_ts(n=10)

    # Drop required column
    data = data.drop("Close", axis=1)

    # Should fail with clear error message
    with pytest.raises(
        DataSanityError, match="missing.*Close|required.*Close|Close.*missing"
    ):
        strict_validator.validate_and_repair(data, "MISSING_CLOSE_TEST")


def test_multiple_missing_columns(strict_validator, mk_ts):
    """Test that multiple missing columns are all listed in error."""
    # Create clean data
    data = mk_ts(n=10)

    # Drop multiple required columns
    data = data.drop(["Close", "Volume"], axis=1)

    # Should fail with clear error message listing all missing columns
    with pytest.raises(
        DataSanityError,
        match="missing.*Close.*Volume|missing.*Volume.*Close|required.*Close.*Volume",
    ):
        strict_validator.validate_and_repair(data, "MISSING_MULTIPLE_TEST")


def test_missing_ohlc_columns(strict_validator, mk_ts):
    """Test that missing OHLC columns are detected."""
    # Create clean data
    data = mk_ts(n=10)

    # Drop OHLC columns
    data = data.drop(["Open", "High", "Low"], axis=1)

    # Should fail with clear error message
    with pytest.raises(
        DataSanityError, match="missing.*Open.*High.*Low|required.*Open.*High.*Low"
    ):
        strict_validator.validate_and_repair(data, "MISSING_OHLC_TEST")


def test_mixed_timezone_handling(strict_validator, mk_ts):
    """Test handling of mixed timezone data."""
    # Create data with mixed timezones
    dates_utc = pd.date_range("2023-01-01", periods=5, freq="1min", tz=timezone.utc)
    dates_eastern = pd.date_range("2023-01-01", periods=5, freq="1min", tz="US/Eastern")

    # Create mixed timezone data
    data_utc = mk_ts(n=5)
    data_utc.index = dates_utc

    data_eastern = mk_ts(n=5)
    data_eastern.index = dates_eastern

    # Combine with mixed timezones
    mixed_data = pd.concat([data_utc, data_eastern])

    # Should fail with clear error message about mixed timezones
    with pytest.raises(
        DataSanityError,
        match="No valid datetime index found|mixed.*timezone|timezone.*mixed|timezone.*UTC|Lookahead contamination",
    ):
        strict_validator.validate_and_repair(mixed_data, "MIXED_TIMEZONE_TEST")


def test_naive_timezone_handling(strict_validator, mk_ts):
    """Test handling of naive (no timezone) datetime index."""
    # Create data with naive timezone
    data = mk_ts(n=10)
    data.index = data.index.tz_localize(None)  # Remove timezone

    # Should fail with clear error message about missing timezone
    with pytest.raises(
        DataSanityError, match="timezone|UTC|timezone.*UTC|Lookahead contamination"
    ):
        strict_validator.validate_and_repair(data, "NAIVE_TIMEZONE_TEST")


def test_empty_dataframe_handling(strict_validator):
    """Test handling of empty DataFrame."""
    # Create empty DataFrame
    empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Should fail with clear error message about insufficient data
    with pytest.raises(
        DataSanityError, match="Empty data not allowed|insufficient|empty|no.*data"
    ):
        strict_validator.validate_and_repair(empty_df, "EMPTY_DF_TEST")


def test_single_row_handling(strict_validator, mk_ts):
    """Test handling of single row DataFrame."""
    # Create single row
    data = mk_ts(n=1)

    # Should pass (single row is valid)
    try:
        clean_data, result = strict_validator.validate_and_repair(
            data, "SINGLE_ROW_TEST"
        )
        assert len(clean_data) == 1, "Should preserve single row"
    except DataSanityError as e:
        # If single row is not allowed, error should be clear
        assert (
            "insufficient" in str(e).lower() or "minimum" in str(e).lower()
        ), f"Unexpected error for single row: {e}"


def test_duplicate_timestamps_handling(strict_validator, mk_ts):
    """Test handling of duplicate timestamps."""
    # Create clean data
    data = mk_ts(n=10)

    # Create duplicate timestamps
    duplicate_index = data.index.tolist()
    duplicate_index[2] = duplicate_index[1]  # Make index[2] same as index[1]
    data.index = pd.DatetimeIndex(duplicate_index)

    # Should fail with clear error message
    with pytest.raises(
        DataSanityError,
        match="Index has duplicates|duplicate.*timestamp|timestamp.*duplicate",
    ):
        strict_validator.validate_and_repair(data, "DUPLICATE_TIMESTAMP_TEST")


def test_non_monotonic_index_handling(strict_validator, mk_ts):
    """Test handling of non-monotonic index."""
    # Create clean data
    data = mk_ts(n=10)

    # Shuffle index to make it non-monotonic
    data.index = data.index[::-1]

    # Should fail with clear error message
    with pytest.raises(
        DataSanityError, match="monotonic|non-monotonic|timestamp.*order"
    ):
        strict_validator.validate_and_repair(data, "NON_MONOTONIC_TEST")


def test_missing_values_handling(strict_validator, mk_ts):
    """Test handling of missing values."""
    # Create clean data
    data = mk_ts(n=10)

    # Introduce missing values
    data.loc[data.index[5], "Close"] = np.nan
    data.loc[data.index[6], "Volume"] = np.nan

    # Should fail with clear error message about missing values
    with pytest.raises(
        DataSanityError, match="Non-finite values|NaN|missing.*values|non-finite"
    ):
        strict_validator.validate_and_repair(data, "MISSING_VALUES_TEST")


def test_extreme_values_handling(strict_validator, mk_ts):
    """Test handling of extreme values."""
    # Create clean data
    data = mk_ts(n=10)

    # Introduce extreme values
    data.loc[data.index[5], "Close"] = 1e12  # Very large price
    data.loc[data.index[6], "Volume"] = 1e15  # Very large volume

    # Should fail with clear error message about extreme values
    with pytest.raises(
        DataSanityError,
        match="Prices >|Excessive volume|extreme|price.*max|volume.*max",
    ):
        strict_validator.validate_and_repair(data, "EXTREME_VALUES_TEST")


def test_negative_values_handling(strict_validator, mk_ts):
    """Test handling of negative values."""
    # Create clean data
    data = mk_ts(n=10)

    # Introduce negative values
    data.loc[data.index[5], "Close"] = -50.0
    data.loc[data.index[6], "Volume"] = -1000.0

    # Should fail with clear error message about negative values
    with pytest.raises(DataSanityError, match="Negative prices|Negative volume"):
        strict_validator.validate_and_repair(data, "NEGATIVE_VALUES_TEST")
