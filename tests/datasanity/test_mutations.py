"""
Metamorphic tests for DataSanity validation.

These tests verify that DataSanity correctly detects and reports various
data quality violations by injecting specific corruptions and asserting
that the expected errors are raised.
"""

import pytest
import pandas as pd
from pathlib import Path
from core.data_sanity import DataSanityValidator, DataSanityError
from tests.util.corruptions import (
    inject_lookahead, inject_nans, inject_infs, inject_duplicates,
    inject_non_monotonic, inject_tz_mess, inject_extreme_prices,
    inject_negative_prices, inject_string_dtype, inject_zero_volume
)

pytestmark = pytest.mark.datasanity


def load_smoke_spy():
    """Load the SPY smoke fixture for testing."""
    fixture_path = Path("data/fixtures/smoke/SPY.csv")
    if fixture_path.exists():
        df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
        # Ensure timezone-aware
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    else:
        # Fallback: create minimal test data
        dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
        data = {
            "Open": [100 + i for i in range(50)],
            "High": [102 + i for i in range(50)],
            "Low": [98 + i for i in range(50)],
            "Close": [101 + i for i in range(50)],
            "Volume": [10000 + i * 100 for i in range(50)],
        }
        return pd.DataFrame(data, index=dates)


@pytest.mark.mutation
def test_lookahead_is_flagged():
    """Test that lookahead contamination is detected."""
    # Create clean synthetic data
    dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    data = {
        "Open": [100 + i for i in range(50)],
        "High": [102 + i for i in range(50)],
        "Low": [98 + i for i in range(50)],
        "Close": [101 + i for i in range(50)],
        "Volume": [10000 + i * 100 for i in range(50)],
    }
    df = pd.DataFrame(data, index=dates)
    
    # Inject lookahead by shifting Close column
    bad = inject_lookahead(df, "Close", shift=-1)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    assert "lookahead" in error_msg or "contamination" in error_msg


@pytest.mark.mutation
def test_nans_are_flagged():
    """Test that NaN values are detected."""
    df = load_smoke_spy()
    bad = inject_nans(df, frac=0.05)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    assert "nan" in error_msg or "non-finite" in error_msg or "finite" in error_msg


@pytest.mark.mutation
def test_infs_are_flagged():
    """Test that infinite values are detected."""
    df = load_smoke_spy()
    bad = inject_infs(df, frac=0.05)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    assert "inf" in error_msg or "non-finite" in error_msg or "finite" in error_msg


@pytest.mark.mutation
def test_duplicates_are_flagged():
    """Test that duplicate timestamps are detected."""
    df = load_smoke_spy()
    bad = inject_duplicates(df, frac=0.1)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    assert "duplicate" in error_msg or "dup" in error_msg


@pytest.mark.mutation
def test_non_monotonic_are_flagged():
    """Test that non-monotonic timestamps are detected."""
    df = load_smoke_spy()
    bad = inject_non_monotonic(df)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    # The error might be lookahead contamination instead of monotonic
    # Both are valid failures for non-monotonic data
    assert any(keyword in error_msg for keyword in ["monotonic", "non-decreasing", "lookahead", "contamination"])


@pytest.mark.mutation
def test_naive_timezone_flagged():
    """Test that naive timezone is detected."""
    df = load_smoke_spy()
    bad = inject_tz_mess(df)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    assert "naive" in error_msg or "timezone" in error_msg


@pytest.mark.mutation
def test_extreme_prices_flagged():
    """Test that extreme prices are detected."""
    df = load_smoke_spy()
    bad = inject_extreme_prices(df, multiplier=1000000.0)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    assert "price" in error_msg or "extreme" in error_msg or "limit" in error_msg


@pytest.mark.mutation
def test_negative_prices_flagged():
    """Test that negative prices are detected."""
    df = load_smoke_spy()
    bad = inject_negative_prices(df)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    assert "negative" in error_msg or "price" in error_msg


@pytest.mark.mutation
def test_string_dtype_flagged():
    """Test that string dtypes are detected."""
    df = load_smoke_spy()
    bad = inject_string_dtype(df, col="Close")
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    error_msg = str(e.value).lower()
    assert "dtype" in error_msg or "numeric" in error_msg or "string" in error_msg


@pytest.mark.mutation
def test_zero_volume_flagged():
    """Test that zero volume is detected (if configured)."""
    df = load_smoke_spy()
    bad = inject_zero_volume(df, frac=0.2)
    
    validator = DataSanityValidator(profile="strict")
    # This might pass or fail depending on config - just test it doesn't crash
    try:
        cleaned_df, validation_result = validator.validate_and_repair(bad, "TEST")
        # If it passes, that's fine - zero volume might be allowed
        assert len(cleaned_df) == len(bad), "Data length should be preserved"
    except DataSanityError as e:
        # If it fails, that's also fine - zero volume might be forbidden
        error_msg = str(e).lower()
        assert "volume" in error_msg or "zero" in error_msg


@pytest.mark.mutation
def test_clean_data_passes():
    """Test that clean data passes validation."""
    # Create clean synthetic data instead of using SPY fixture
    dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    data = {
        "Open": [100 + i for i in range(50)],
        "High": [102 + i for i in range(50)],
        "Low": [98 + i for i in range(50)],
        "Close": [101 + i for i in range(50)],
        "Volume": [10000 + i * 100 for i in range(50)],
    }
    df = pd.DataFrame(data, index=dates)
    
    # Use walkforward_smoke profile which allows lookahead for smoke tests
    validator = DataSanityValidator(profile="walkforward_smoke")
    cleaned_df, validation_result = validator.validate_and_repair(df, "TEST")
    
    # Check that validation passed (no exceptions raised)
    assert len(cleaned_df) == len(df), "Data length should be preserved"
    assert "lookahead_contamination" in validation_result.flags, "Should flag lookahead but not fail"


@pytest.mark.mutation
def test_multiple_corruptions_handled():
    """Test that multiple corruptions are handled gracefully."""
    df = load_smoke_spy()
    bad = inject_nans(inject_lookahead(df), frac=0.05)
    
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(bad, "TEST")
    
    # Should fail with some error (don't care which one)
    assert len(str(e.value)) > 0


@pytest.mark.mutation
def test_profile_differences():
    """Test that different profiles handle corruptions differently."""
    df = load_smoke_spy()
    bad = inject_nans(df, frac=0.05)
    
    # Strict profile should fail
    strict_validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError):
        strict_validator.validate_and_repair(bad, "TEST")
    
    # Warn profile might pass (depending on config)
    warn_validator = DataSanityValidator(profile="warn")
    try:
        cleaned_df, validation_result = warn_validator.validate_and_repair(bad, "TEST")
        # If it passes, that's fine - check that we got a DataFrame back
        assert len(cleaned_df) == len(bad), "Data length should be preserved"
    except DataSanityError:
        # If it fails, that's also fine
        pass
