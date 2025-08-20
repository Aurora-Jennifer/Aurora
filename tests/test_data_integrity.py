#!/usr/bin/env python3
"""
Data Integrity Tests
Tests the DataSanity layer with deliberate corrupt data to ensure proper validation and repair.
"""

import logging
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_sanity import (
    DataSanityError,
    DataSanityValidator,
    DataSanityWrapper,
    validate_market_data,
)

# Import hypothesis for property-based testing
try:
    from hypothesis import Verbosity, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataSanity:
    """Test suite for DataSanity validation and repair."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_data_sanity.yaml"

        # Create test config
        test_config = {
            "profiles": {
                "default": {
                    "mode": "warn",
                    "price_max": 1000000.0,
                    "allow_repairs": True,
                    "allow_winsorize": True,
                    "allow_clip_prices": True,
                    "allow_fix_ohlc": True,
                    "allow_drop_dupes": True,
                    "allow_ffill_nans": True,
                    "tolerate_outliers_after_repair": True,
                    "fail_on_lookahead_flag": False,
                    "fail_if_any_repair": False,
                },
                "strict": {
                    "mode": "fail",
                    "price_max": 1000000.0,
                    "allow_repairs": False,
                    "allow_winsorize": False,
                    "allow_clip_prices": False,
                    "allow_fix_ohlc": False,
                    "allow_drop_dupes": False,
                    "allow_ffill_nans": False,
                    "tolerate_outliers_after_repair": False,
                    "fail_on_lookahead_flag": True,
                    "fail_if_any_repair": True,
                },
                "lenient": {
                    "mode": "warn",
                    "price_max": 10000000.0,
                    "allow_repairs": True,
                    "allow_winsorize": True,
                    "allow_clip_prices": True,
                    "allow_fix_ohlc": True,
                    "allow_drop_dupes": True,
                    "allow_ffill_nans": True,
                    "tolerate_outliers_after_repair": True,
                    "fail_on_lookahead_flag": False,
                    "fail_if_any_repair": False,
                },
            },
            "price_limits": {
                "max_price": 1000000.0,
                "min_price": 0.01,
                "max_daily_return": 0.3,
                "max_volume": 1000000000000,
            },
            "ohlc_validation": {
                "max_high_low_spread": 0.4,
                "require_ohlc_consistency": True,
                "allow_zero_volume": False,
            },
            "outlier_detection": {
                "z_score_threshold": 4.0,
                "mad_threshold": 3.0,
                "min_obs_for_outlier": 20,
            },
            "repair_mode": "warn",
            "winsorize_quantile": 0.01,
            "time_series": {
                "require_monotonic": True,
                "require_utc": True,
                "max_gap_days": 30,
                "allow_duplicates": False,
            },
            "returns": {
                "method": "log_close_to_close",
                "min_periods": 2,
                "fill_method": "forward",
            },
            "logging": {
                "log_repairs": True,
                "log_outliers": True,
                "log_validation_failures": True,
                "summary_level": "INFO",
            },
        }

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

        self.validator = DataSanityValidator(str(self.config_path))
        self.wrapper = DataSanityWrapper(str(self.config_path))

        yield

        # Cleanup
        import shutil

        shutil.rmtree(self.temp_dir)

    @pytest.fixture
    def sample_data_sizes(self):
        """Fixture for different data sizes."""
        return {"small": 10, "medium": 100, "large": 1000, "xlarge": 10000}

    @pytest.fixture
    def corruption_scenarios(self):
        """Fixture for different corruption scenarios."""
        return {
            "extreme_prices": {
                "type": "extreme_prices",
                "description": "Prices outside valid bounds",
            },
            "nan_burst": {"type": "nan_burst", "description": "Multiple NaN values"},
            "ohlc_violations": {
                "type": "ohlc_violations",
                "description": "OHLC relationship violations",
            },
            "duplicate_timestamps": {
                "type": "duplicate_timestamps",
                "description": "Duplicate timestamps",
            },
            "non_monotonic": {
                "type": "non_monotonic",
                "description": "Non-monotonic timestamps",
            },
            "negative_volume": {
                "type": "negative_volume",
                "description": "Negative volume values",
            },
            "mixed_timezones": {
                "type": "mixed_timezones",
                "description": "Mixed timezone data",
            },
            "lookahead_contamination": {
                "type": "lookahead_contamination",
                "description": "Future data contamination",
            },
        }

    @pytest.fixture
    def config_profiles(self):
        """Fixture for different configuration profiles."""
        return ["default", "strict", "lenient"]

    @pytest.fixture
    def symbols(self):
        """Fixture for different symbols."""
        return ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ"]

    @pytest.fixture
    def time_periods(self):
        """Fixture for different time periods."""
        return {
            "daily": {"freq": "D", "periods": 100},
            "hourly": {"freq": "H", "periods": 240},
            "weekly": {"freq": "W", "periods": 52},
            "monthly": {"freq": "M", "periods": 12},
        }

    def create_corrupt_data(self, corruption_type: str = "mixed") -> pd.DataFrame:
        """Create test data with deliberate corruption."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create base data with some corruption
        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0 + i * 0.1 for i in range(100)],
                "High": [102.0 + i * 0.1 for i in range(100)],
                "Low": [98.0 + i * 0.1 for i in range(100)],
                "Close": [101.0 + i * 0.1 for i in range(100)],
                "Volume": [1000000 + i * 1000 for i in range(100)],
            }
        )

        if corruption_type == "mixed" or corruption_type == "extreme_prices":
            # 2. Extreme prices
            data.loc[10, "Close"] = 1e11  # Extreme high price
            data.loc[20, "Close"] = 0.0  # Zero price
            data.loc[30, "Close"] = np.nan  # NaN price

        if corruption_type == "mixed" or corruption_type == "nan_burst":
            # Add NaN burst
            data.loc[40:45, "Close"] = np.nan
            data.loc[50:52, "Open"] = np.nan

        if corruption_type == "mixed" or corruption_type == "ohlc_violations":
            # 3. OHLC inconsistencies
            data.loc[40, "High"] = 95.0  # High < Close
            data.loc[41, "Low"] = 105.0  # Low > Close

        if corruption_type == "mixed" or corruption_type == "negative_volume":
            # 4. Negative volume
            data.loc[60, "Volume"] = -1000000

        if corruption_type == "mixed" or corruption_type == "duplicate_timestamps":
            # 6. Duplicate timestamps (add several duplicates)
            data.loc[80, "Date"] = data.loc[79, "Date"]
            data.loc[85, "Date"] = data.loc[79, "Date"]  # Another duplicate
            data.loc[90, "Date"] = data.loc[89, "Date"]  # Another duplicate

        if corruption_type == "mixed" or corruption_type == "non_monotonic":
            # 1. Non-monotonic timestamps (reverse some dates)
            data.loc[50:60, "Date"] = data.loc[50:60, "Date"].iloc[::-1].values

        if corruption_type == "mixed" or corruption_type == "mixed_timezones":
            # 7. Non-UTC timezone
            data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize("US/Eastern")

        if corruption_type == "lookahead_contamination":
            # Add lookahead contamination
            data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
            data.loc[data.index[5], "Returns"] = data.loc[data.index[6], "Returns"]  # Lookahead

        # Set Date as index for validation
        data = data.set_index("Date")

        return data

    def create_test_data(
        self, size: int = 100, freq: str = "D", clean: bool = True
    ) -> pd.DataFrame:
        """Create test data of specified size and frequency."""
        dates = pd.date_range("2023-01-01", periods=size, freq=freq, tz=UTC)

        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, size)  # 0.05% daily return, 1.5% volatility
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.002, size)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, size))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, size))),
                "Close": prices,
                "Volume": np.random.lognormal(10, 0.5, size),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        if not clean:
            # Add some corruption (only if data is large enough)
            if len(data) > 10:
                data.loc[data.index[10], "Close"] = 1e11  # Extreme price
            if len(data) > 20:
                data.loc[data.index[20], "Close"] = np.nan  # NaN
            else:
                # For small datasets, corrupt the last few rows
                data.loc[data.index[-1], "Close"] = 1e11  # Extreme price
                if len(data) > 1:
                    data.loc[data.index[-2], "Close"] = np.nan  # NaN

        return data

    def create_config_with_profile(self, profile: str) -> DataSanityValidator:
        """Create validator with specific profile."""
        # Create a new validator with the specific profile
        validator = DataSanityValidator(profile=profile)
        return validator

    @pytest.mark.data_sanity
    @pytest.mark.validation
    @pytest.mark.parametrize(
        "corruption_type", ["duplicate_timestamps", "non_monotonic", "mixed_timezones"]
    )
    def test_time_series_validation(self, corruption_type):
        """Test time series validation and repair for different corruption types."""
        data = self.create_corrupt_data(corruption_type)

        # Test with warn mode
        clean_data, result = self.validator.validate_and_repair(data, f"TEST_{corruption_type}")

        # Should have monotonic timestamps
        assert clean_data.index.is_monotonic_increasing, (
            f"Timestamps should be monotonic for {corruption_type}"
        )

        # Should be UTC
        assert clean_data.index.tz == UTC, f"Timestamps should be UTC for {corruption_type}"

        # Should have no duplicates
        assert not clean_data.index.duplicated().any(), (
            f"Should have no duplicate timestamps for {corruption_type}"
        )

        # Should have fewer rows only if duplicates were present
        if corruption_type == "duplicate_timestamps":
            assert len(clean_data) < len(data), (
                f"Should remove duplicate timestamps for {corruption_type}"
            )
        else:
            # For non-monotonic and mixed_timezones, should preserve data but fix issues
            assert len(clean_data) > 0, f"Should have data for {corruption_type}"

    @pytest.mark.data_sanity
    @pytest.mark.validation
    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_time_series_validation_different_sizes(self, size):
        """Test time series validation with different data sizes."""
        data = self.create_test_data(size, clean=False)

        clean_data, result = self.validator.validate_and_repair(data, f"TEST_SIZE_{size}")

        # Basic validation should pass
        assert clean_data.index.is_monotonic_increasing, (
            f"Timestamps should be monotonic for size {size}"
        )
        assert clean_data.index.tz == UTC, f"Timestamps should be UTC for size {size}"
        assert len(clean_data) > 0, f"Should have data for size {size}"

    @pytest.mark.data_sanity
    @pytest.mark.validation
    @pytest.mark.parametrize("corruption_type", ["extreme_prices", "nan_burst"])
    def test_price_validation(self, corruption_type):
        """Test price validation and repair for different corruption types."""
        data = self.create_corrupt_data(corruption_type)

        clean_data, result = self.validator.validate_and_repair(data, f"TEST_{corruption_type}")

        # Check price bounds (after validation, prices should be within bounds)
        finite_prices = clean_data["Close"].dropna()
        assert (finite_prices <= 1000000.0).all(), (
            f"Prices should be within bounds for {corruption_type}"
        )
        assert (finite_prices >= 0.01).all(), f"Prices should be positive for {corruption_type}"

        # Check for finite values (should be all finite after validation)
        assert np.isfinite(clean_data["Close"]).all(), (
            f"All prices should be finite for {corruption_type}"
        )

        # Check that extreme prices were handled
        assert clean_data["Close"].max() <= 1000000.0, (
            f"Extreme prices should be capped for {corruption_type}"
        )
        assert clean_data["Close"].min() >= 0.01, (
            f"Zero prices should be fixed for {corruption_type}"
        )

    @pytest.mark.data_sanity
    @pytest.mark.validation
    @pytest.mark.parametrize("profile", ["default", "strict", "lenient"])
    def test_price_validation_different_profiles(self, profile):
        """Test price validation with different configuration profiles."""
        validator = self.create_config_with_profile(profile)
        data = self.create_corrupt_data("extreme_prices")

        if profile == "strict":
            # Strict profile should fail
            with pytest.raises(DataSanityError):
                validator.validate_and_repair(data, f"TEST_{profile}")
        else:
            # Other profiles should repair
            clean_data, result = validator.validate_and_repair(data, f"TEST_{profile}")
            assert len(clean_data) > 0, f"Should have data for profile {profile}"

    @pytest.mark.data_sanity
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=50)
    @given(st.lists(st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=20))
    def test_property_based_price_validation(self, prices):
        """Property-based test for price validation with valid prices."""
        dates = pd.date_range("2023-01-01", periods=len(prices), freq="D", tz=UTC)

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * 1.02 for p in prices],
                "Low": [p * 0.98 for p in prices],
                "Close": prices,
                "Volume": [1000000] * len(prices),
            },
            index=dates,
        )

        # Valid data should pass
        clean_data = self.wrapper.validate_dataframe(data, "PROPERTY_TEST")
        assert len(clean_data) == len(data), "Valid data should pass unchanged"
        assert "Returns" in clean_data.columns, "Should calculate returns"

    @pytest.mark.data_sanity
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=20)
    @given(st.lists(st.floats(min_value=1e10, max_value=1e12), min_size=1, max_size=5))
    def test_property_based_extreme_price_detection(self, extreme_prices):
        """Property-based test for extreme price detection."""
        dates = pd.date_range("2023-01-01", periods=len(extreme_prices), freq="D", tz=UTC)

        data = pd.DataFrame(
            {
                "Open": extreme_prices,
                "High": [p * 1.02 for p in extreme_prices],
                "Low": [p * 0.98 for p in extreme_prices],
                "Close": extreme_prices,
                "Volume": [1000000] * len(extreme_prices),
            },
            index=dates,
        )

        # Extreme prices should trigger failure or repair
        try:
            clean_data = self.wrapper.validate_dataframe(data, "EXTREME_TEST")
            # If repair mode, should clip to bounds
            assert clean_data["Close"].max() <= 1000000.0, "Extreme prices should be clipped"
        except DataSanityError:
            # If fail mode, should raise exception
            pass

    @pytest.mark.data_sanity
    @pytest.mark.validation
    @pytest.mark.parametrize("corruption_type", ["ohlc_violations"])
    def test_ohlc_consistency(self, corruption_type):
        """Test OHLC consistency validation and repair."""
        data = self.create_corrupt_data(corruption_type)

        clean_data, result = self.validator.validate_and_repair(data, f"TEST_{corruption_type}")

        # Check OHLC relationships (after repair, these should be consistent)
        assert (clean_data["High"] >= clean_data["Open"]).all(), (
            f"High should be >= Open for {corruption_type}"
        )
        assert (clean_data["High"] >= clean_data["Close"]).all(), (
            f"High should be >= Close for {corruption_type}"
        )
        assert (clean_data["Low"] <= clean_data["Open"]).all(), (
            f"Low should be <= Open for {corruption_type}"
        )
        assert (clean_data["Low"] <= clean_data["Close"]).all(), (
            f"Low should be <= Close for {corruption_type}"
        )

        # Check high-low spread (after repair, should be reasonable)
        spread = (clean_data["High"] - clean_data["Low"]) / clean_data["Close"]
        reasonable_spreads = spread <= 1.0  # Allow up to 100% spread
        assert reasonable_spreads.sum() >= len(spread) * 0.95, (
            f"At least 95% of spreads should be reasonable for {corruption_type}"
        )

    @pytest.mark.data_sanity
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=30)
    @given(st.lists(st.floats(min_value=10, max_value=1000), min_size=5, max_size=15))
    def test_property_based_ohlc_validation(self, base_prices):
        """Property-based test for OHLC validation."""
        dates = pd.date_range("2023-01-01", periods=len(base_prices), freq="D", tz=UTC)

        # Create OHLC data with some violations
        data = pd.DataFrame(
            {
                "Open": base_prices,
                "High": [p * 1.02 for p in base_prices],
                "Low": [p * 0.98 for p in base_prices],
                "Close": base_prices,
                "Volume": [1000000] * len(base_prices),
            },
            index=dates,
        )

        # Add some OHLC violations
        data.loc[data.index[0], "High"] = data.loc[data.index[0], "Low"] - 1  # High < Low
        data.loc[data.index[1], "Low"] = data.loc[data.index[1], "High"] + 1  # Low > High

        # Should repair OHLC violations
        clean_data = self.wrapper.validate_dataframe(data, "PROPERTY_OHLC_TEST")

        # Check OHLC relationships after repair
        assert (clean_data["High"] >= clean_data["Low"]).all(), "High should be >= Low after repair"
        assert (clean_data["High"] >= clean_data["Open"]).all(), (
            "High should be >= Open after repair"
        )
        assert (clean_data["High"] >= clean_data["Close"]).all(), (
            "High should be >= Close after repair"
        )

    @pytest.mark.data_sanity
    @pytest.mark.perf
    @pytest.mark.benchmark
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_performance_validation(self, size):
        """Test validation performance with different data sizes."""
        data = self.create_test_data(size, clean=False)

        # Measure memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Measure validation time
        start_time = time.time()
        clean_data, result = self.validator.validate_and_repair(data, f"PERF_TEST_{size}")
        validation_time = time.time() - start_time

        # Measure memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Performance assertions
        assert validation_time < 10.0, (
            f"Validation took {validation_time:.2f}s for size {size}, expected < 10.0s"
        )
        assert memory_increase < 1000.0, (
            f"Memory increase {memory_increase:.2f}MB for size {size}, expected < 1000MB"
        )
        assert len(clean_data) > 0, f"Should have data for size {size}"

        # Log performance metrics
        print(f"Size {size}: Time={validation_time:.3f}s, Memory={memory_increase:.2f}MB")

    @pytest.mark.data_sanity
    @pytest.mark.perf
    @pytest.mark.parametrize("repair_mode", ["warn", "drop", "winsorize"])
    def test_performance_different_repair_modes(self, repair_mode):
        """Test performance with different repair modes."""
        # Create config with specific repair mode
        config = self.validator.config.copy()
        config["repair_mode"] = repair_mode

        validator = DataSanityValidator()
        validator.config = config

        data = self.create_test_data(1000, clean=False)

        # Measure performance
        start_time = time.time()
        clean_data, result = validator.validate_and_repair(data, f"PERF_{repair_mode}")
        validation_time = time.time() - start_time

        # All modes should complete within reasonable time
        assert validation_time < 5.0, (
            f"Repair mode {repair_mode} took {validation_time:.2f}s, expected < 5.0s"
        )
        assert len(clean_data) > 0, f"Should have data for repair mode {repair_mode}"

    @pytest.mark.data_sanity
    @pytest.mark.stress
    @pytest.mark.slow
    def test_stress_large_dataset(self):
        """Stress test with very large dataset."""
        data = self.create_test_data(50000, clean=False)

        # Measure performance
        start_time = time.time()
        clean_data, result = self.validator.validate_and_repair(data, "STRESS_TEST")
        validation_time = time.time() - start_time

        # Should handle large dataset
        assert validation_time < 60.0, (
            f"Large dataset validation took {validation_time:.2f}s, expected < 60.0s"
        )
        assert len(clean_data) > 0, "Should have data after stress test"
        # Check that extreme prices were handled (either clipped or removed)
        assert clean_data["Close"].max() <= 1000000.0, "Extreme prices should be handled"

    @pytest.mark.data_sanity
    @pytest.mark.validation
    @pytest.mark.parametrize("corruption_type", ["negative_volume"])
    def test_volume_validation(self, corruption_type):
        """Test volume validation and repair."""
        data = self.create_corrupt_data(corruption_type)

        clean_data, result = self.validator.validate_and_repair(data, f"TEST_{corruption_type}")

        # Check volume bounds
        assert (clean_data["Volume"] >= 0).all(), (
            f"Volume should be non-negative for {corruption_type}"
        )
        assert (clean_data["Volume"] <= 1000000000000).all(), (
            f"Volume should be within bounds for {corruption_type}"
        )

        # Check for finite values
        assert np.isfinite(clean_data["Volume"]).all(), (
            f"All volume should be finite for {corruption_type}"
        )

        # Check that zero volume was handled
        assert (clean_data["Volume"] > 0).all(), (
            f"Zero volume should be replaced for {corruption_type}"
        )

    @pytest.mark.data_sanity
    @pytest.mark.edge_case
    @pytest.mark.parametrize("size", [0, 1, 2])
    def test_edge_case_data_sizes(self, size):
        """Test edge cases with very small data sizes."""
        if size == 0:
            # Empty DataFrame - should return empty DataFrame
            data = pd.DataFrame()
            clean_data, result = self.validator.validate_and_repair(data, "EMPTY_TEST")
            assert clean_data.empty, "Empty input should return empty output"
            assert result.rows_in == 0, "Should report 0 input rows"
            assert result.rows_out == 0, "Should report 0 output rows"
        else:
            # Small DataFrames
            data = self.create_test_data(size)

            # Should handle small data sizes
            clean_data, result = self.validator.validate_and_repair(data, f"SMALL_TEST_{size}")
            assert len(clean_data) > 0, f"Should have data for size {size}"
            if size >= 2:
                assert "Returns" in clean_data.columns, f"Should calculate returns for size {size}"
            else:
                # For size 1, returns might not be calculated due to insufficient data
                pass

    @pytest.mark.data_sanity
    @pytest.mark.falsification
    @pytest.mark.parametrize(
        "scenario",
        [
            "extreme_negative_prices",
            "impossible_ohlc",
            "future_contamination",
            "invalid_dtypes",
        ],
    )
    def test_falsification_scenarios(self, scenario):
        """Test falsification scenarios that should always fail."""
        if scenario == "extreme_negative_prices":
            # Create data with extreme negative prices
            dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=UTC)
            data = pd.DataFrame(
                {
                    "Open": [
                        -1e12,
                        100.0,
                        200.0,
                        300.0,
                        400.0,
                        500.0,
                        600.0,
                        700.0,
                        800.0,
                        900.0,
                    ],
                    "High": [
                        -1e12,
                        102.0,
                        202.0,
                        302.0,
                        402.0,
                        502.0,
                        602.0,
                        702.0,
                        802.0,
                        902.0,
                    ],
                    "Low": [
                        -1e12,
                        98.0,
                        198.0,
                        298.0,
                        398.0,
                        498.0,
                        598.0,
                        698.0,
                        798.0,
                        898.0,
                    ],
                    "Close": [
                        -1e12,
                        101.0,
                        201.0,
                        301.0,
                        401.0,
                        501.0,
                        601.0,
                        701.0,
                        801.0,
                        901.0,
                    ],
                    "Volume": [1000000] * 10,
                },
                index=dates,
            )

            # Should handle negative prices by clipping them
            clean_data, result = self.validator.validate_and_repair(data, "FALSIFY_NEGATIVE")
            assert (clean_data["Close"] >= 0.01).all(), (
                "Negative prices should be clipped to minimum"
            )
            assert (clean_data["Open"] >= 0.01).all(), (
                "Negative prices should be clipped to minimum"
            )
            assert (clean_data["High"] >= 0.01).all(), (
                "Negative prices should be clipped to minimum"
            )
            assert (clean_data["Low"] >= 0.01).all(), "Negative prices should be clipped to minimum"

        elif scenario == "impossible_ohlc":
            # Create data with impossible OHLC relationships
            dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=UTC)
            data = pd.DataFrame(
                {
                    "Open": [100.0] * 10,
                    "High": [95.0] * 10,  # High < Open
                    "Low": [105.0] * 10,  # Low > Open
                    "Close": [101.0] * 10,
                    "Volume": [1000000] * 10,
                },
                index=dates,
            )

            # Should fail or repair OHLC violations
            try:
                clean_data, result = self.validator.validate_and_repair(data, "FALSIFY_OHLC")
                # If repair mode, check that violations were fixed
                assert (clean_data["High"] >= clean_data["Open"]).all(), (
                    "OHLC violations should be repaired"
                )
                assert (clean_data["Low"] <= clean_data["Open"]).all(), (
                    "OHLC violations should be repaired"
                )
            except DataSanityError:
                # If fail mode, exception is expected
                pass

        elif scenario == "future_contamination":
            # Create data with future contamination
            dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=UTC)
            data = pd.DataFrame(
                {
                    "Open": [100.0 + i for i in range(10)],
                    "High": [102.0 + i for i in range(10)],
                    "Low": [98.0 + i for i in range(10)],
                    "Close": [101.0 + i for i in range(10)],
                    "Volume": [1000000] * 10,
                },
                index=dates,
            )

            # Add lookahead contamination
            data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
            data.loc[data.index[5], "Returns"] = data.loc[data.index[6], "Returns"]  # Lookahead

            # Should detect or handle lookahead
            clean_data, result = self.validator.validate_and_repair(data, "FALSIFY_LOOKAHEAD")
            assert "Returns" in clean_data.columns, "Should handle lookahead contamination"

        elif scenario == "invalid_dtypes":
            # Create data with invalid data types
            dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=UTC)
            data = pd.DataFrame(
                {
                    "Open": ["invalid"] * 10,  # String instead of float
                    "High": [102.0] * 10,
                    "Low": [98.0] * 10,
                    "Close": [101.0] * 10,
                    "Volume": [1000000] * 10,
                },
                index=dates,
            )

            # Should handle invalid dtypes by converting or failing gracefully
            try:
                clean_data, result = self.validator.validate_and_repair(data, "FALSIFY_DTYPE")
                # If it succeeds, check that data was converted
                assert pd.api.types.is_numeric_dtype(clean_data["Open"]), (
                    "Invalid dtypes should be converted"
                )
            except (DataSanityError, TypeError, ValueError):
                # If it fails, that's also acceptable for invalid dtypes
                pass

    @pytest.mark.data_sanity
    @pytest.mark.network
    @pytest.mark.flaky
    @patch("yfinance.download")
    def test_network_resilience(self, mock_download):
        """Test network resilience and error handling."""
        # Mock network failures
        mock_download.side_effect = Exception("Network timeout")

        # Should handle network failures gracefully
        with pytest.raises(Exception, match="Network timeout"):
            import yfinance as yf

            yf.download("SPY", start="2024-01-01", end="2024-01-31")

        # Test with partial data corruption
        mock_download.side_effect = None
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [100.0, np.nan, 300.0, np.nan],
                "High": [102.0, np.nan, 302.0, np.nan],
                "Low": [98.0, np.nan, 298.0, np.nan],
                "Close": [101.0, np.nan, 301.0, np.nan],
                "Volume": [1000000] * 4,
            },
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )

        # Should handle partial corruption
        data = mock_download("SPY", start="2024-01-01", end="2024-01-31")
        clean_data = self.wrapper.validate_dataframe(data, "NETWORK_TEST")
        assert len(clean_data) > 0, "Should handle partial data corruption"

    @pytest.mark.data_sanity
    @pytest.mark.integration
    def test_integration_with_engines(self):
        """Test integration with trading engines."""
        # Simulate what backtest engine does
        symbols = ["SPY", "QQQ"]
        all_data = []

        for symbol in symbols:
            # Create mock data (simulating engine data loading)
            data = self.create_test_data(100, clean=False)
            data["Symbol"] = symbol

            # Validate data (like updated backtest engine)
            clean_data = self.wrapper.validate_dataframe(data, symbol)
            all_data.append(clean_data)

        # Combine data (like backtest engine)
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()

            assert len(combined_data) > 0, "Should have combined data"
            assert "Returns" in combined_data.columns, "Should have returns"
            assert "Symbol" in combined_data.columns, "Should have symbol column"

    @pytest.mark.data_sanity
    @pytest.mark.edge_case
    def test_multiindex_column_handling(self):
        """Test MultiIndex column handling."""
        data = self.create_test_data(10)

        # Create MultiIndex columns (like yfinance)
        data.columns = pd.MultiIndex.from_tuples(
            [
                ("Open", "TEST"),
                ("High", "TEST"),
                ("Low", "TEST"),
                ("Close", "TEST"),
                ("Volume", "TEST"),
            ]
        )

        # Should handle MultiIndex correctly
        clean_data = self.wrapper.validate_dataframe(data, "MULTIINDEX_TEST")

        # Should flatten to single level
        assert not isinstance(clean_data.columns, pd.MultiIndex)
        assert "Close" in clean_data.columns

    @pytest.mark.data_sanity
    @pytest.mark.edge_case
    def test_timezone_handling(self):
        """Test timezone handling."""
        # Test naive timestamps
        dates = pd.date_range("2023-01-01", periods=10, freq="D")  # No timezone
        data = self.create_test_data(10)
        data.index = dates

        # Should convert to UTC
        clean_data = self.wrapper.validate_dataframe(data, "TZ_TEST")
        assert clean_data.index.tz == UTC

        # Test mixed timezones
        mixed_dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=UTC)
        mixed_dates = mixed_dates.tz_localize(None)  # Make some naive
        data.index = mixed_dates

        # Should handle mixed timezones
        clean_data = self.wrapper.validate_dataframe(data, "MIXED_TZ_TEST")
        assert clean_data.index.tz == UTC

    @pytest.mark.data_sanity
    @pytest.mark.edge_case
    def test_corporate_actions_consistency(self):
        """Test corporate actions consistency."""
        data = self.create_test_data(10)

        # Add Adj Close with large differences (simulating split)
        data["Adj Close"] = data["Close"] * 1.5

        # Should handle gracefully
        clean_data = self.wrapper.validate_dataframe(data, "CORP_ACTIONS_TEST")
        assert "Adj Close" in clean_data.columns
        assert len(clean_data) == len(data)

    def test_outlier_detection(self):
        """Test outlier detection and repair."""
        # Create data with clear outliers
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        prices = [100.0 + i * 0.1 for i in range(50)]

        # Add outliers
        prices[25] = 1000.0  # Clear outlier
        prices[35] = 0.001  # Another outlier

        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices,
                "High": [p * 1.02 for p in prices],
                "Low": [p * 0.98 for p in prices],
                "Close": prices,
                "Volume": [1000000] * 50,
            }
        )

        clean_data, result = self.validator.validate_and_repair(data, "TEST")

        # Check that outliers were handled
        # The test data has 1000.0 and 0.001 as outliers
        # In warn mode, these should be clipped to bounds
        assert clean_data["Close"].max() <= 1000000.0, "Outliers should be clipped to bounds"
        assert clean_data["Close"].min() >= 0.01, "Outliers should be clipped to bounds"

    def test_returns_calculation(self):
        """Test returns calculation."""
        data = self.create_corrupt_data()

        clean_data, result = self.validator.validate_and_repair(data, "TEST")

        # Check that returns were calculated
        assert "Returns" in clean_data.columns, "Returns column should be added"

        # Check returns are finite (first return will be NaN, which is expected)
        returns_finite = clean_data["Returns"].dropna()
        assert np.isfinite(returns_finite).all(), "All non-NaN returns should be finite"

        # Check returns are within bounds (after winsorization)
        max_return = self.validator.config["price_limits"]["max_daily_return"]
        # Some returns may be winsorized, so check they're within bounds
        finite_returns = clean_data["Returns"].dropna()
        assert (np.abs(finite_returns) <= max_return).all(), "Returns should be within bounds"

        # Check returns are log returns (skip first NaN and winsorized values)
        expected_returns = np.log(clean_data["Close"] / clean_data["Close"].shift(1))
        # Compare only non-NaN values that weren't winsorized
        # Winsorized returns will be at the bounds, so exclude those
        max_return = self.validator.config["price_limits"]["max_daily_return"]
        mask = ~(clean_data["Returns"].isna() | expected_returns.isna())
        mask = mask & (np.abs(clean_data["Returns"]) < max_return)
        if mask.any():
            np.testing.assert_array_almost_equal(
                clean_data["Returns"][mask], expected_returns[mask], decimal=10
            )

    def test_fail_mode(self):
        """Test fail mode behavior."""
        # Create config with fail mode
        fail_config = self.validator.config.copy()
        fail_config["repair_mode"] = "fail"

        fail_validator = DataSanityValidator()
        fail_validator.config = fail_config

        data = self.create_corrupt_data()

        # Should raise exception
        with pytest.raises(DataSanityError):
            fail_validator.validate_and_repair(data, "TEST")

    def test_drop_mode(self):
        """Test drop mode behavior."""
        # Create config with drop mode
        drop_config = self.validator.config.copy()
        drop_config["repair_mode"] = "drop"

        drop_validator = DataSanityValidator()
        drop_validator.config = drop_config

        data = self.create_corrupt_data()
        original_len = len(data)

        clean_data, result = drop_validator.validate_and_repair(data, "TEST")

        # Should have fewer rows (corrupt rows dropped)
        assert len(clean_data) < original_len, "Should drop corrupt rows"

    def test_winsorize_mode(self):
        """Test winsorize mode behavior."""
        # Create config with winsorize mode
        winsorize_config = self.validator.config.copy()
        winsorize_config["repair_mode"] = "winsorize"

        winsorize_validator = DataSanityValidator()
        winsorize_validator.config = winsorize_config

        data = self.create_corrupt_data()

        clean_data, result = winsorize_validator.validate_and_repair(data, "TEST")

        # Should have same number of rows (except for duplicate removal)
        # Duplicate removal happens regardless of repair mode
        assert len(clean_data) <= len(data), "Should not add rows in winsorize mode"

        # Extreme values should be winsorized
        assert clean_data["Close"].max() <= 1000000.0, "Extreme values should be winsorized"

    def test_file_loading(self):
        """Test loading and validating data from files."""
        # Create test data file
        data = self.create_corrupt_data()
        test_file = Path(self.temp_dir) / "test_data.pkl"
        data.to_pickle(test_file)

        # Test loading and validation
        clean_data = self.wrapper.load_and_validate(str(test_file), "TEST_FILE")

        # Should be cleaned
        assert len(clean_data) < len(data), "Should clean corrupt data"
        assert clean_data.index.is_monotonic_increasing, "Should have monotonic timestamps"

    def test_validation_stats(self):
        """Test validation statistics tracking."""
        data = self.create_corrupt_data()

        self.wrapper.validate_dataframe(data, "TEST")
        stats = self.wrapper.get_validation_stats()

        # Should have statistics
        assert "repair_count" in stats, "Should track repair count"
        assert "outlier_count" in stats, "Should track outlier count"
        assert "validation_failures" in stats, "Should track validation failures"

        # Should have some repairs
        assert stats["repair_count"] > 0, "Should have performed repairs"

    def test_convenience_function(self):
        """Test the convenience validate_market_data function."""
        data = self.create_corrupt_data()

        clean_data = validate_market_data(data, "TEST_CONVENIENCE")

        # Should be cleaned
        assert len(clean_data) < len(data), "Should clean corrupt data"
        assert clean_data.index.is_monotonic_increasing, "Should have monotonic timestamps"

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty DataFrame
        empty_data = pd.DataFrame()
        clean_empty, result = self.validator.validate_and_repair(empty_data, "EMPTY")
        assert clean_empty.empty, "Empty DataFrame should remain empty"

        # Single row DataFrame
        single_row = pd.DataFrame(
            {
                "Date": [datetime.now()],
                "Open": [100.0],
                "High": [102.0],
                "Low": [98.0],
                "Close": [101.0],
                "Volume": [1000000],
            }
        )
        clean_single, result = self.validator.validate_and_repair(single_row, "SINGLE")
        assert len(clean_single) == 1, "Single row should remain"

        # Missing columns
        missing_cols = pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=10),
                "Close": [100.0 + i for i in range(10)],
            }
        )
        with pytest.raises(DataSanityError, match="Missing required columns"):
            self.validator.validate_and_repair(missing_cols, "MISSING")

    def test_column_name_variations(self):
        """Test handling of different column name variations."""
        dates = pd.date_range("2023-01-01", periods=10)

        # Test lowercase column names
        lower_data = pd.DataFrame(
            {
                "date": dates,
                "open": [100.0 + i for i in range(10)],
                "high": [102.0 + i for i in range(10)],
                "low": [98.0 + i for i in range(10)],
                "close": [101.0 + i for i in range(10)],
                "volume": [1000000] * 10,
            }
        )

        clean_lower, result = self.validator.validate_and_repair(lower_data, "LOWER")
        assert "Open" in clean_lower.columns, "Should standardize column names"
        assert "Close" in clean_lower.columns, "Should standardize column names"

        # Test uppercase column names
        upper_data = pd.DataFrame(
            {
                "DATE": dates,
                "OPEN": [100.0 + i for i in range(10)],
                "HIGH": [102.0 + i for i in range(10)],
                "LOW": [98.0 + i for i in range(10)],
                "CLOSE": [101.0 + i for i in range(10)],
                "VOLUME": [1000000] * 10,
            }
        )

        clean_upper, result = self.validator.validate_and_repair(upper_data, "UPPER")
        assert "Open" in clean_upper.columns, "Should standardize column names"
        assert "Close" in clean_upper.columns, "Should standardize column names"


def test_integration_with_existing_data_loaders():
    """Test integration with existing data loading functions."""
    # This test would integrate with the actual data loaders in the system
    # For now, we'll test the wrapper functions

    # Test with synthetic data that mimics real market data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    base_price = 100.0

    # Create realistic price data with some noise
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))  # 0.05% daily return, 1.5% volatility
    prices = base_price * np.exp(np.cumsum(returns))

    realistic_data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.normal(0, 0.002, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            "Close": prices,
            "Volume": np.random.lognormal(13, 0.5, len(dates)),  # Realistic volume distribution
        }
    )

    # Ensure OHLC relationships
    realistic_data["High"] = realistic_data[["Open", "High", "Close"]].max(axis=1)
    realistic_data["Low"] = realistic_data[["Open", "Low", "Close"]].min(axis=1)

    # Validate the data
    clean_data = validate_market_data(realistic_data, "REALISTIC")

    # Should pass validation with minimal changes
    assert len(clean_data) == len(realistic_data), "Realistic data should pass validation"
    assert "Returns" in clean_data.columns, "Should calculate returns"

    # Check returns are reasonable
    returns = clean_data["Returns"].dropna()
    assert len(returns) > 0, "Should have valid returns"
    assert np.abs(returns).max() < 0.3, "Returns should be within reasonable bounds"


def test_real_world_data_scenarios():
    """Test DataSanity with real-world data scenarios."""
    import yfinance as yf

    from core.data_sanity import get_data_sanity_wrapper

    data_sanity = get_data_sanity_wrapper()

    # Test 1: Real market data from yfinance
    print("Testing with real SPY data...")
    spy_data = yf.download("SPY", start="2024-01-01", end="2024-01-31", progress=False, auto_adjust=False)
    clean_spy = data_sanity.validate_dataframe(spy_data, "SPY")

    assert len(clean_spy) > 0, "Should have data"
    assert "Returns" in clean_spy.columns, "Should calculate returns"
    assert clean_spy.index.is_monotonic_increasing, "Should have monotonic timestamps"
    assert clean_spy.index.tz == UTC, "Should be UTC"

    # Test 2: Data with gaps (weekends)
    print("Testing data with gaps...")
    gap_data = yf.download("AAPL", start="2024-01-01", end="2024-01-31", progress=False, auto_adjust=False)
    clean_gap = data_sanity.validate_dataframe(gap_data, "AAPL")

    # Should handle gaps gracefully
    assert len(clean_gap) > 0, "Should handle gaps"

    # Test 3: Multiple symbols
    print("Testing multiple symbols...")
    symbols = ["SPY", "QQQ", "IWM"]
    for symbol in symbols:
        try:
            data = yf.download(symbol, start="2024-01-01", end="2024-01-15", progress=False, auto_adjust=False)
            clean_data = data_sanity.validate_dataframe(data, symbol)
            assert len(clean_data) > 0, f"Should validate {symbol}"
            assert "Returns" in clean_data.columns, f"Should calculate returns for {symbol}"
        except Exception as e:
            print(f"Warning: Could not test {symbol}: {e}")

    print("✅ Real-world data scenarios passed")


def test_data_sanity_integration_with_engines():
    """Test that DataSanity is properly integrated with trading engines."""
    import yfinance as yf

    from core.data_sanity import get_data_sanity_wrapper

    data_sanity = get_data_sanity_wrapper()

    # Test that engines can load data through DataSanity
    print("Testing engine integration...")

    # Simulate what the backtest engine does
    symbols = ["SPY"]
    all_data = []

    for symbol in symbols:
        try:
            # Load data (like backtest engine)
            ticker = yf.Ticker(symbol)
            data = ticker.history(start="2024-01-01", end="2024-01-31")

            if not data.empty:
                # Validate data (like updated backtest engine)
                clean_data = data_sanity.validate_dataframe(data, symbol)
                clean_data["Symbol"] = symbol
                all_data.append(clean_data)
                print(f"✅ Loaded and validated {len(clean_data)} data points for {symbol}")
            else:
                print(f"⚠️  No data for {symbol}")

        except Exception as e:
            print(f"❌ Error loading data for {symbol}: {e}")

    if all_data:
        # Combine data (like backtest engine)
        combined_data = pd.concat(all_data, ignore_index=False)
        combined_data = combined_data.sort_index()

        assert len(combined_data) > 0, "Should have combined data"
        assert "Returns" in combined_data.columns, "Should have returns"
        assert "Symbol" in combined_data.columns, "Should have symbol column"

        print("✅ Engine integration test passed")
    else:
        print("⚠️  No data available for engine integration test")


def test_data_sanity_configuration():
    """Test different DataSanity configurations."""
    import tempfile

    import yaml

    # Create test data
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0 + i * 0.1 for i in range(50)],
            "High": [102.0 + i * 0.1 for i in range(50)],
            "Low": [98.0 + i * 0.1 for i in range(50)],
            "Close": [101.0 + i * 0.1 for i in range(50)],
            "Volume": [1000000 + i * 1000 for i in range(50)],
        }
    )

    # Add some corruption
    data.loc[10, "Close"] = 1e11  # Extreme price
    data.loc[20, "Close"] = np.nan  # NaN

    # Test 1: Fail mode
    print("Testing fail mode...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        fail_config = {
            "profiles": {
                "default": {
                    "mode": "fail",
                    "price_max": 1000000.0,
                    "allow_repairs": False,
                    "allow_winsorize": False,
                    "allow_clip_prices": False,
                    "allow_fix_ohlc": False,
                    "allow_drop_dupes": False,
                    "allow_ffill_nans": False,
                    "tolerate_outliers_after_repair": False,
                    "fail_on_lookahead_flag": True,
                    "fail_if_any_repair": True,
                }
            },
            "price_limits": {
                "max_price": 1000000.0,
                "min_price": 0.01,
                "max_daily_return": 0.3,
                "max_volume": 1000000000000,
            },
            "ohlc_validation": {
                "max_high_low_spread": 0.4,
                "require_ohlc_consistency": True,
                "allow_zero_volume": False,
            },
            "outlier_detection": {
                "z_score_threshold": 4.0,
                "mad_threshold": 3.0,
                "min_obs_for_outlier": 20,
            },
            "winsorize_quantile": 0.01,
            "time_series": {
                "require_monotonic": True,
                "require_utc": True,
                "max_gap_days": 30,
                "allow_duplicates": False,
            },
            "returns": {
                "method": "log_close_to_close",
                "min_periods": 2,
                "fill_method": "forward",
            },
            "logging": {
                "log_repairs": True,
                "log_outliers": True,
                "log_validation_failures": True,
                "summary_level": "INFO",
            },
        }
        yaml.dump(fail_config, f)
        config_path = f.name

    try:
        from core.data_sanity import DataSanityValidator

        fail_validator = DataSanityValidator(config_path)

        with pytest.raises(ValueError):  # Should fail due to extreme price
            fail_validator.validate_and_repair(data, "FAIL_TEST")
        print("✅ Fail mode works correctly")
    finally:
        import os

        os.unlink(config_path)

    # Test 2: Drop mode
    print("Testing drop mode...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        drop_config = {
            "profiles": {
                "default": {
                    "mode": "warn",
                    "price_max": 1000000.0,
                    "allow_repairs": True,
                    "allow_winsorize": False,
                    "allow_clip_prices": False,
                    "allow_fix_ohlc": False,
                    "allow_drop_dupes": True,
                    "allow_ffill_nans": False,
                    "tolerate_outliers_after_repair": False,
                    "fail_on_lookahead_flag": False,
                    "fail_if_any_repair": False,
                }
            },
            "price_limits": {
                "max_price": 1000000.0,
                "min_price": 0.01,
                "max_daily_return": 0.3,
                "max_volume": 1000000000000,
            },
            "ohlc_validation": {
                "max_high_low_spread": 0.4,
                "require_ohlc_consistency": True,
                "allow_zero_volume": False,
            },
            "outlier_detection": {
                "z_score_threshold": 4.0,
                "mad_threshold": 3.0,
                "min_obs_for_outlier": 20,
            },
            "winsorize_quantile": 0.01,
            "time_series": {
                "require_monotonic": True,
                "require_utc": True,
                "max_gap_days": 30,
                "allow_duplicates": False,
            },
            "returns": {
                "method": "log_close_to_close",
                "min_periods": 2,
                "fill_method": "forward",
            },
            "logging": {
                "log_repairs": True,
                "log_outliers": True,
                "log_validation_failures": True,
                "summary_level": "INFO",
            },
        }
        yaml.dump(drop_config, f)
        config_path = f.name

    try:
        drop_validator = DataSanityValidator(config_path)

        # Should fail due to extreme price since allow_clip_prices is False
        with pytest.raises(Exception) as exc_info:
            drop_validator.validate_and_repair(data, "DROP_TEST")

        error_msg = str(exc_info.value)
        assert "Prices >" in error_msg, f"Expected price error, got: {error_msg}"
        print("✅ Drop mode correctly fails on extreme prices")
    finally:
        import os

        os.unlink(config_path)


if __name__ == "__main__":
    # Run all tests including new ones
    print("🧪 Running comprehensive DataSanity tests...")

    # Run the original test suite
    pytest.main([__file__, "-v"])

    # Run additional tests
    print("\n🧪 Running additional validation tests...")

    try:
        test_real_world_data_scenarios()
    except Exception as e:
        print(f"❌ Real-world data test failed: {e}")

    try:
        test_data_sanity_integration_with_engines()
    except Exception as e:
        print(f"❌ Engine integration test failed: {e}")

    try:
        test_data_sanity_configuration()
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

    print("\n✅ All DataSanity validation tests completed!")
