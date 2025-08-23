from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityError, DataSanityValidator
from tests.datasanity._golden import EXPECTED


class TestGoldenRegression:
    """Golden fixture regression tests for DataSanity validation."""
    
    @pytest.fixture(scope="class")
    def validator(self):
        return DataSanityValidator(profile="strict")
    
    @pytest.fixture(scope="class")
    def clean_df(self):
        """Load clean golden fixture."""
        fixture_path = Path("tests/golden/clean/ohlc_clean.csv")
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
            df.index = df.index.tz_localize("UTC")
            return df
        else:
            # Fallback: create minimal clean data
            dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
            data = {
                "Open": [100 + i for i in range(50)],
                "High": [102 + i for i in range(50)],
                "Low": [98 + i for i in range(50)],
                "Close": [101 + i for i in range(50)],
                "Volume": [10000 + i * 100 for i in range(50)],
            }
            return pd.DataFrame(data, index=dates)
    
    @pytest.mark.golden
    @pytest.mark.regression
    def test_clean_golden_passes_validation(self, validator, clean_df):
        """Golden test: Clean data should always pass validation."""
        result = validator.validate_and_repair(clean_df, "GOLDEN_CLEAN")
        assert result[0] is True, f"Clean golden data failed: {result[1]}"
        assert len(result[1]) == len(clean_df), "Clean data should not be modified"
    
    @pytest.mark.golden
    @pytest.mark.regression
    def test_duplicate_timestamps_golden(self, validator):
        """Golden test: Duplicate timestamps should always fail."""
        fixture_path = Path("tests/golden/violations/dup_timestamps.csv")
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
            df.index = df.index.tz_localize("UTC")
            with pytest.raises(DataSanityError) as e:
                validator.validate_and_repair(df, "GOLDEN_DUP_TS")
            assert EXPECTED["DUP_TS"] in str(e.value)
            assert "code=DUP_TS" in str(e.value)
    
    @pytest.mark.golden
    @pytest.mark.regression
    def test_extreme_prices_golden(self, validator):
        """Golden test: Extreme prices should always fail."""
        fixture_path = Path("tests/golden/violations/extreme_prices.csv")
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
            df.index = df.index.tz_localize("UTC")
            with pytest.raises(DataSanityError) as e:
                validator.validate_and_repair(df, "GOLDEN_EXTREME_PRICES")
            assert EXPECTED["PRICES_GT"] in str(e.value)
    
    @pytest.mark.golden
    @pytest.mark.regression
    def test_negative_prices_golden(self, validator):
        """Golden test: Negative prices should always fail."""
        fixture_path = Path("tests/golden/violations/negative_prices.csv")
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
            df.index = df.index.tz_localize("UTC")
            with pytest.raises(DataSanityError) as e:
                validator.validate_and_repair(df, "GOLDEN_NEGATIVE_PRICES")
            assert "negative" in str(e.value).lower() or "price" in str(e.value).lower()
    
    @pytest.mark.golden
    @pytest.mark.regression
    def test_golden_validation_consistency(self, validator, clean_df):
        """Golden test: Validation results should be consistent across runs."""
        result1 = validator.validate_and_repair(clean_df, "GOLDEN_CONSISTENCY_1")
        result2 = validator.validate_and_repair(clean_df, "GOLDEN_CONSISTENCY_2")
        assert result1[0] == result2[0], "Validation results should be consistent"
        if result1[0]:  # If validation passed
            assert result1[1].equals(result2[1]), "Validated data should be identical"
    
    @pytest.mark.golden
    @pytest.mark.regression
    def test_golden_profile_behavior(self, clean_df):
        """Golden test: Different profiles should behave as expected."""
        strict_validator = DataSanityValidator(profile="strict")
        warn_validator = DataSanityValidator(profile="warn")
        
        # Clean data should pass both profiles
        strict_result = strict_validator.validate_and_repair(clean_df, "GOLDEN_STRICT")
        warn_result = warn_validator.validate_and_repair(clean_df, "GOLDEN_WARN")
        
        assert strict_result[0] == warn_result[0], "Clean data should pass both profiles"
    
    @pytest.mark.golden
    @pytest.mark.regression
    def test_golden_error_messages_consistent(self, validator):
        """Golden test: Error messages should be consistent and informative."""
        # Test with known violation
        dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
        data = {
            "Open": [100] * 10,
            "High": [102] * 10,
            "Low": [98] * 10,
            "Close": [101] * 10,
            "Volume": [10000] * 10,
        }
        df = pd.DataFrame(data, index=dates)
        
        # Inject a known violation
        df.loc[df.index[0], "Close"] = np.nan
        
        with pytest.raises(DataSanityError) as e:
            validator.validate_and_repair(df, "GOLDEN_ERROR_MSG")
        
        error_msg = str(e.value)
        assert "TEST" in error_msg, "Error should include symbol name"
        assert EXPECTED["NONFINITE"] in error_msg, "Error should mention non-finite values"
        assert "code=" in error_msg, "Error should include error code"
    
    @pytest.mark.golden
    @pytest.mark.regression
    def test_golden_data_integrity_preserved(self, validator, clean_df):
        """Golden test: Data integrity should be preserved during validation."""
        original_shape = clean_df.shape
        original_dtypes = clean_df.dtypes.copy()
        original_index = clean_df.index.copy()
        
        result = validator.validate_and_repair(clean_df, "GOLDEN_INTEGRITY")
        
        assert result[0] is True, "Clean data should pass validation"
        validated_df = result[1]
        
        # Shape should be preserved
        assert validated_df.shape == original_shape, "Data shape should be preserved"
        
        # Dtypes should be preserved (numeric)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert np.issubdtype(validated_df[col].dtype, np.number), f"{col} should remain numeric"
        
        # Index should be preserved
        assert validated_df.index.equals(original_index), "Index should be preserved"
        
        # Values should be preserved (no repairs on clean data)
        pd.testing.assert_frame_equal(clean_df, validated_df, check_dtype=False)
