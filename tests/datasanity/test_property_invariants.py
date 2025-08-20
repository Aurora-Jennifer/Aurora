import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, Verbosity
from hypothesis.extra.pandas import data_frames, column, indexes
from tests.datasanity._golden import EXPECTED
from tests.datasanity._mutate import inject_duplicates, inject_non_monotonic, inject_nans, inject_infs, inject_lookahead, inject_string_dtype
from core.data_sanity import DataSanityValidator, DataSanityError

# Property test strategies
@st.composite
def clean_ohlcv_dataframes(draw):
    """Generate clean OHLCV DataFrames for property testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    # Generate realistic price movements
    base_price = draw(st.floats(min_value=50, max_value=500))
    price_changes = draw(st.lists(
        st.floats(min_value=-0.05, max_value=0.05),  # ±5% daily changes
        min_size=100, max_size=100
    ))
    
    # Build price series with realistic movements
    prices = [base_price]
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(1.0, new_price))  # Ensure positive prices
    
    # Generate OHLC from close prices with realistic spreads
    data = {
        "Close": prices,
        "Volume": [draw(st.floats(min_value=1000, max_value=1000000)) for _ in range(100)],
    }
    
    # Generate realistic OHLC from close prices
    ohlc_data = []
    for i, close_price in enumerate(prices):
        # Small random spread around close price
        spread = close_price * 0.01  # 1% spread
        open_price = close_price + draw(st.floats(min_value=-spread, max_value=spread))
        high_price = max(open_price, close_price) + draw(st.floats(min_value=0, max_value=spread))
        low_price = min(open_price, close_price) - draw(st.floats(min_value=0, max_value=spread))
        
        ohlc_data.append({
            "Open": max(0.1, open_price),
            "High": max(0.1, high_price),
            "Low": max(0.1, low_price),
            "Close": max(0.1, close_price),
            "Volume": data["Volume"][i]
        })
    
    df = pd.DataFrame(ohlc_data, index=dates)
    
    # Ensure OHLC invariants
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    
    return df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})

@st.composite
def tz_aware_dataframes(draw):
    """Generate timezone-aware DataFrames."""
    df = draw(clean_ohlcv_dataframes())
    df.index = df.index.tz_localize("UTC")
    return df

# Property Tests
@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_clean_data_passes_validation(df):
    """Property: Clean data always passes DataSanity validation."""
    validator = DataSanityValidator(profile="strict")
    result = validator.validate_and_repair(df, "TEST")
    assert result[0] is True, f"Clean data failed validation: {result[1]}"

@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_duplicates_always_detected(df):
    """Property: Duplicate timestamps are always detected in strict mode."""
    df_with_dups = inject_duplicates(df, frac=0.1)
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(df_with_dups, "TEST")
    assert EXPECTED["DUP_TS"] in str(e.value)

@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_nans_always_detected(df):
    """Property: NaN values are always detected in strict mode."""
    df_with_nans = inject_nans(df, cols=["Open", "Close"], frac=0.1)
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(df_with_nans, "TEST")
    assert EXPECTED["NONFINITE"] in str(e.value)

@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_infs_always_detected(df):
    """Property: Infinite values are always detected in strict mode."""
    df_with_infs = inject_infs(df, cols=["High", "Low"], frac=0.1)
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(df_with_infs, "TEST")
    assert EXPECTED["NONFINITE"] in str(e.value)

@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_lookahead_always_detected(df):
    """Property: Lookahead contamination is always detected in strict mode."""
    df_with_lookahead = inject_lookahead(df, shift=1)
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(df_with_lookahead, "TEST")
    assert EXPECTED["LOOKAHEAD"] in str(e.value)

@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_string_dtype_always_detected(df):
    """Property: String dtypes are always detected in strict mode."""
    df_with_strings = inject_string_dtype(df, col="Close")
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(df_with_strings, "TEST")
    assert EXPECTED["INVALID_DTYPE"] in str(e.value)

@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_naive_timezone_always_detected(df):
    """Property: Naive timezones are always detected in strict mode."""
    df_naive = df.copy()
    df_naive.index = df_naive.index.tz_localize(None)
    validator = DataSanityValidator(profile="strict")
    with pytest.raises(DataSanityError) as e:
        validator.validate_and_repair(df_naive, "TEST")
    assert EXPECTED["NAIVE_TZ"] in str(e.value)

@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_validation_idempotent(df):
    """Property: Validation is idempotent - running twice gives same result."""
    validator = DataSanityValidator(profile="strict")
    result1 = validator.validate_and_repair(df, "TEST")
    result2 = validator.validate_and_repair(df, "TEST")
    assert result1[0] == result2[0]
    if result1[0]:  # If validation passed
        assert result1[1].equals(result2[1])

@pytest.mark.sanity
@pytest.mark.property
@settings(verbosity=Verbosity.verbose, max_examples=20)
@given(df=tz_aware_dataframes())
def test_property_profile_consistency(df):
    """Property: Same profile always gives consistent results."""
    validator1 = DataSanityValidator(profile="strict")
    validator2 = DataSanityValidator(profile="strict")
    result1 = validator1.validate_and_repair(df, "TEST")
    result2 = validator2.validate_and_repair(df, "TEST")
    assert result1[0] == result2[0]
