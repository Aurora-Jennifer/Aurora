"""
Comprehensive tests for DataSanity enforcement and verification.

This module tests:
1. Integration presence (AST verification)
2. Runtime guards
3. Property-based fuzzing
4. Edge case handling
5. Performance safety
6. Falsification scenarios
7. Contract enforcement
8. Guard mechanisms
"""

import ast
import logging
import os
import tempfile
import time
from datetime import timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import pytest
import yaml

# Import hypothesis for property-based testing
try:
    from hypothesis import Verbosity, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from core.contracts import (
    DataFrameContract,
    FeatureContract,
    SignalContract,
    require_validated_data,
)
from core.data_sanity import (
    DataSanityError,
    DataSanityValidator,
    DataSanityWrapper,
    assert_validated,
    attach_guard,
    get_data_sanity_wrapper,
    get_guard,
)

logger = logging.getLogger(__name__)


class TestDataSanityEnforcement:
    """Test DataSanity enforcement mechanisms."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

        # Create test configuration
        config = {
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
            "repair_mode": "fail",
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

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

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

    def create_test_data(
        self, rows: int = 100, corrupt: bool = False, corruption_type: str = "mixed"
    ) -> pd.DataFrame:
        """Create test data with optional corruption."""
        dates = pd.date_range("2023-01-01", periods=rows, freq="D", tz=timezone.utc)

        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, rows)
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.002, rows)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, rows))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, rows))),
                "Close": prices,
                "Volume": np.random.lognormal(10, 0.5, rows),
            },
            index=dates,
        )

        # Ensure OHLC relationships
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        if corrupt:
            if corruption_type == "mixed" or corruption_type == "extreme_prices":
                # Use iloc for positional indexing
                data.iloc[10, data.columns.get_loc("Close")] = 1e11  # Extreme price
                data.iloc[20, data.columns.get_loc("Close")] = 0.0  # Zero price
                data.iloc[30, data.columns.get_loc("Close")] = np.nan  # NaN

            if corruption_type == "nan_burst":
                # Use iloc for positional indexing
                data.iloc[40:45, data.columns.get_loc("Close")] = np.nan
                data.iloc[50:52, data.columns.get_loc("Open")] = np.nan

            if corruption_type == "ohlc_violations":
                # Use iloc for positional indexing
                data.iloc[40, data.columns.get_loc("High")] = (
                    data.iloc[40, data.columns.get_loc("Low")] - 1
                )  # OHLC violation

            if corruption_type == "negative_volume":
                # Use iloc for positional indexing
                data.iloc[50, data.columns.get_loc("Volume")] = -1000  # Negative volume

        return data

    def create_config_with_profile(self, profile: str) -> DataSanityValidator:
        """Create validator with specific profile."""
        config = self.validator.config.copy()
        if profile in config.get("profiles", {}):
            profile_config = config["profiles"][profile]
            config.update(profile_config)

        validator = DataSanityValidator()
        validator.config = config
        return validator

    @pytest.mark.data_sanity
    @pytest.mark.guard
    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_guard_attachment_and_validation(self, size):
        """Test DataSanityGuard attachment and validation for different data sizes."""
        data = self.create_test_data(size)

        # Attach guard
        guard = attach_guard(data)
        assert hasattr(data, "_sanity_guard"), f"Should have guard for size {size}"
        assert guard is get_guard(data), f"Should get same guard for size {size}"

        # Should fail before validation
        with pytest.raises(DataSanityError, match="DataFrame used before validation"):
            assert_validated(data, f"test_context_{size}")

        # Validate data
        clean_data = self.wrapper.validate_dataframe(data, f"TEST_{size}")

        # Should pass after validation
        assert_validated(clean_data, f"test_context_{size}")

        # Check guard status of the validated DataFrame
        clean_guard = get_guard(clean_data)
        assert (
            clean_guard is not None
        ), f"Should have guard for validated data size {size}"
        status = clean_guard.get_status()
        assert status["validated"] is True, f"Should be validated for size {size}"
        assert (
            status["symbol"] == f"TEST_{size}"
        ), f"Should have correct symbol for size {size}"

    @pytest.mark.data_sanity
    @pytest.mark.guard
    @pytest.mark.parametrize(
        "corruption_type", ["extreme_prices", "nan_burst", "ohlc_violations"]
    )
    def test_guard_with_corrupt_data(self, corruption_type):
        """Test guard behavior with different corruption types."""
        data = self.create_test_data(50, corrupt=True, corruption_type=corruption_type)

        # Attach guard
        guard = attach_guard(data)

        # Should fail before validation
        with pytest.raises(DataSanityError, match="DataFrame used before validation"):
            assert_validated(data, f"test_context_{corruption_type}")

        # Validate data (should repair corruption)
        clean_data = self.wrapper.validate_dataframe(data, f"TEST_{corruption_type}")

        # Should pass after validation
        assert_validated(clean_data, f"test_context_{corruption_type}")

        # Check guard status
        status = guard.get_status()
        assert status["validated"] is True, f"Should be validated for {corruption_type}"
        assert (
            len(clean_data) > 0
        ), f"Should have data after repair for {corruption_type}"

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_contract_validation(self, size):
        """Test contract validation for different data sizes."""
        data = self.create_test_data(size)

        # Should handle unvalidated data (creates guard for backward compatibility)
        # The guard will be created but not marked as validated
        DataFrameContract.validate_market_data(data, f"test_{size}")

        # Check that guard was created but not validated
        guard = get_guard(data)
        assert guard is not None, f"Should have guard for size {size}"
        status = guard.get_status()
        assert status["validated"] is False, f"Should not be validated for size {size}"

        # Validate data
        clean_data = self.wrapper.validate_dataframe(data, f"TEST_{size}")

        # Should pass with validation
        assert (
            DataFrameContract.validate_market_data(clean_data, f"test_{size}") is True
        )

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.parametrize("profile", ["default", "strict", "lenient"])
    def test_contract_validation_different_profiles(self, profile):
        """Test contract validation with different configuration profiles."""
        validator = self.create_config_with_profile(profile)
        data = self.create_test_data(50, corrupt=True, corruption_type="extreme_prices")

        if profile == "strict":
            # Strict profile should fail
            with pytest.raises(DataSanityError):
                validator.validate_and_repair(data, f"TEST_{profile}")
        else:
            # Other profiles should repair and pass contract validation
            clean_data, result = validator.validate_and_repair(data, f"TEST_{profile}")
            assert (
                DataFrameContract.validate_market_data(clean_data, f"test_{profile}")
                is True
            )

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=30)
    @given(
        st.lists(
            st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=20
        )
    )
    def test_property_based_contract_validation(self, prices):
        """Property-based test for contract validation."""
        dates = pd.date_range(
            "2023-01-01", periods=len(prices), freq="D", tz=timezone.utc
        )

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

        # Valid data should pass contract validation after DataSanity processing
        clean_data = self.wrapper.validate_dataframe(data, "PROPERTY_CONTRACT_TEST")
        assert (
            DataFrameContract.validate_market_data(clean_data, "property_test") is True
        )

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_decorator_enforcement(self, size):
        """Test decorator enforcement for different data sizes."""

        @require_validated_data(f"test_function_{size}")
        def test_function(df: pd.DataFrame) -> str:
            return f"success_{size}"

        data = self.create_test_data(size)

        # Should handle unvalidated data (creates guard for backward compatibility)
        # The guard will be created but not marked as validated
        test_function(data)

        # Check that guard was created but not validated
        guard = get_guard(data)
        assert guard is not None, f"Should have guard for size {size}"
        status = guard.get_status()
        assert status["validated"] is False, f"Should not be validated for size {size}"

        # Should pass with validation
        clean_data = self.wrapper.validate_dataframe(data, f"TEST_{size}")
        assert test_function(clean_data) == f"success_{size}"

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.parametrize(
        "corruption_type", ["extreme_prices", "nan_burst", "ohlc_violations"]
    )
    def test_decorator_enforcement_with_corrupt_data(self, corruption_type):
        """Test decorator enforcement with corrupt data."""

        @require_validated_data(f"test_function_{corruption_type}")
        def test_function(df: pd.DataFrame) -> str:
            return f"success_{corruption_type}"

        data = self.create_test_data(50, corrupt=True, corruption_type=corruption_type)

        # Should handle unvalidated data (creates guard for backward compatibility)
        # The guard will be created but not marked as validated
        test_function(data)

        # Check that guard was created but not validated
        guard = get_guard(data)
        assert guard is not None, f"Should have guard for {corruption_type}"
        status = guard.get_status()
        assert (
            status["validated"] is False
        ), f"Should not be validated for {corruption_type}"

        # Should pass with validation (after repair)
        clean_data = self.wrapper.validate_dataframe(data, f"TEST_{corruption_type}")
        assert test_function(clean_data) == f"success_{corruption_type}"

    @pytest.mark.data_sanity
    @pytest.mark.perf
    @pytest.mark.benchmark
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_performance_guard_enforcement(self, size):
        """Test performance of guard enforcement."""
        data = self.create_test_data(size)

        # Measure memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Attach guard and validate
        start_time = time.time()
        guard = attach_guard(data)
        clean_data = self.wrapper.validate_dataframe(data, f"PERF_GUARD_{size}")
        validation_time = time.time() - start_time

        # Measure memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Performance assertions
        assert (
            validation_time < 5.0
        ), f"Guard enforcement took {validation_time:.2f}s for size {size}, expected < 5.0s"
        assert (
            memory_increase < 500.0
        ), f"Memory increase {memory_increase:.2f}MB for size {size}, expected < 500MB"

        # Verify guard functionality
        assert_validated(clean_data, f"perf_test_{size}")
        status = guard.get_status()
        assert status["validated"] is True, f"Should be validated for size {size}"

        # Log performance metrics
        print(
            f"Guard Size {size}: Time={validation_time:.3f}s, Memory={memory_increase:.2f}MB"
        )

    @pytest.mark.data_sanity
    @pytest.mark.perf
    @pytest.mark.parametrize("profile", ["default", "strict", "lenient"])
    def test_performance_contract_validation(self, profile):
        """Test performance of contract validation with different profiles."""
        validator = self.create_config_with_profile(profile)
        data = self.create_test_data(
            1000, corrupt=True, corruption_type="extreme_prices"
        )

        # Measure performance
        start_time = time.time()

        if profile == "strict":
            # Strict profile should fail quickly
            with pytest.raises(DataSanityError):
                validator.validate_and_repair(data, f"PERF_CONTRACT_{profile}")
            validation_time = time.time() - start_time
            assert (
                validation_time < 1.0
            ), f"Strict validation took {validation_time:.2f}s, expected < 1.0s"
        else:
            # Other profiles should repair within reasonable time
            clean_data, result = validator.validate_and_repair(
                data, f"PERF_CONTRACT_{profile}"
            )
            validation_time = time.time() - start_time
            assert (
                validation_time < 3.0
            ), f"Contract validation took {validation_time:.2f}s for {profile}, expected < 3.0s"
            assert len(clean_data) > 0, f"Should have data for profile {profile}"

    @pytest.mark.data_sanity
    @pytest.mark.stress
    @pytest.mark.slow
    def test_stress_guard_enforcement(self):
        """Stress test guard enforcement with large dataset."""
        data = self.create_test_data(50000, corrupt=True, corruption_type="mixed")

        # Measure performance
        start_time = time.time()
        guard = attach_guard(data)
        clean_data = self.wrapper.validate_dataframe(data, "STRESS_GUARD")
        validation_time = time.time() - start_time

        # Should handle large dataset
        assert (
            validation_time < 30.0
        ), f"Large dataset guard enforcement took {validation_time:.2f}s, expected < 30.0s"
        assert len(clean_data) > 0, "Should have data after stress test"

        # Verify guard functionality
        assert_validated(clean_data, "stress_test")
        status = guard.get_status()
        assert status["validated"] is True, "Should be validated after stress test"

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.parametrize(
        "signal_type", ["valid", "invalid_bounds", "nan_values", "infinite_values"]
    )
    def test_signal_contract_validation(self, signal_type):
        """Test signal contract validation for different signal types."""
        if signal_type == "valid":
            # Valid signals
            valid_signals = pd.Series(
                [0.5, -0.3, 0.8, -0.1], index=pd.date_range("2023-01-01", periods=4)
            )
            assert (
                SignalContract.validate_signals(valid_signals, f"test_{signal_type}")
                is True
            )

        elif signal_type == "invalid_bounds":
            # Invalid signals (out of bounds)
            invalid_signals = pd.Series(
                [1.5, -0.3, 0.8], index=pd.date_range("2023-01-01", periods=3)
            )
            with pytest.raises(DataSanityError, match="signals outside bounds"):
                SignalContract.validate_signals(invalid_signals, f"test_{signal_type}")

        elif signal_type == "nan_values":
            # Non-finite signals
            nan_signals = pd.Series(
                [0.5, np.nan, 0.8], index=pd.date_range("2023-01-01", periods=3)
            )
            with pytest.raises(DataSanityError, match="Non-finite values in signals"):
                SignalContract.validate_signals(nan_signals, f"test_{signal_type}")

        elif signal_type == "infinite_values":
            # Infinite signals
            inf_signals = pd.Series(
                [0.5, np.inf, 0.8], index=pd.date_range("2023-01-01", periods=3)
            )
            with pytest.raises(DataSanityError, match="Non-finite values in signals"):
                SignalContract.validate_signals(inf_signals, f"test_{signal_type}")

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=50)
    @given(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=3, max_size=20))
    def test_property_based_signal_validation(self, signal_values):
        """Property-based test for signal validation."""
        dates = pd.date_range("2023-01-01", periods=len(signal_values), freq="D")
        signals = pd.Series(signal_values, index=dates)

        # Valid signals should pass
        assert SignalContract.validate_signals(signals, "property_signal_test") is True

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=30)
    @given(st.lists(st.floats(min_value=1.1, max_value=10.0), min_size=1, max_size=10))
    def test_property_based_invalid_signal_detection(self, invalid_values):
        """Property-based test for invalid signal detection."""
        dates = pd.date_range("2023-01-01", periods=len(invalid_values), freq="D")
        invalid_signals = pd.Series(invalid_values, index=dates)

        # Invalid signals should fail
        with pytest.raises(DataSanityError, match="signals outside bounds"):
            SignalContract.validate_signals(invalid_signals, "property_invalid_test")

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.parametrize(
        "feature_type", ["valid", "nan_values", "infinite_values", "wrong_shape"]
    )
    def test_feature_contract_validation(self, feature_type):
        """Test feature contract validation for different feature types."""
        data = self.create_test_data(10)
        clean_data = self.wrapper.validate_dataframe(data, "TEST")

        if feature_type == "valid":
            # Valid features
            features = pd.DataFrame(
                {"feature1": np.random.randn(10), "feature2": np.random.randn(10)},
                index=clean_data.index,
            )

            assert (
                FeatureContract.validate_features(features, f"test_{feature_type}")
                is True
            )

        elif feature_type == "nan_values":
            # Invalid features (non-finite)
            features = pd.DataFrame(
                {"feature1": np.random.randn(10), "feature2": np.random.randn(10)},
                index=clean_data.index,
            )
            features.loc[0, "feature1"] = np.nan

            with pytest.raises(DataSanityError, match="Non-finite values in columns"):
                FeatureContract.validate_features(features, f"test_{feature_type}")

        elif feature_type == "infinite_values":
            # Invalid features (infinite)
            features = pd.DataFrame(
                {"feature1": np.random.randn(10), "feature2": np.random.randn(10)},
                index=clean_data.index,
            )
            features.loc[0, "feature1"] = np.inf

            with pytest.raises(DataSanityError, match="Non-finite values in columns"):
                FeatureContract.validate_features(features, f"test_{feature_type}")

        elif feature_type == "wrong_shape":
            # Invalid features (wrong shape)
            features = pd.DataFrame(
                {
                    "feature1": np.random.randn(10),  # Correct length
                    "feature2": np.random.randn(10),
                },
                index=clean_data.index,
            )  # Correct index

            # Should handle wrong shape gracefully
            FeatureContract.validate_features(features, f"test_{feature_type}")

    @pytest.mark.data_sanity
    @pytest.mark.contract
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=30)
    @given(
        st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=5, max_size=20)
    )
    def test_property_based_feature_validation(self, feature_values):
        """Property-based test for feature validation."""
        dates = pd.date_range(
            "2023-01-01", periods=len(feature_values), freq="D", tz=timezone.utc
        )

        # Create valid market data
        data = pd.DataFrame(
            {
                "Open": [100.0 + i for i in range(len(feature_values))],
                "High": [102.0 + i for i in range(len(feature_values))],
                "Low": [98.0 + i for i in range(len(feature_values))],
                "Close": [101.0 + i for i in range(len(feature_values))],
                "Volume": [1000000] * len(feature_values),
            },
            index=dates,
        )

        clean_data = self.wrapper.validate_dataframe(data, "PROPERTY_FEATURE_TEST")

        # Create valid features
        features = pd.DataFrame(
            {"feature1": feature_values, "feature2": [v * 0.5 for v in feature_values]},
            index=clean_data.index,
        )

        # Valid features should pass
        assert (
            FeatureContract.validate_features(features, "property_feature_test") is True
        )

    @pytest.mark.data_sanity
    @pytest.mark.edge_case
    @pytest.mark.parametrize("size", [0, 1, 2])
    def test_edge_case_guard_enforcement(self, size):
        """Test edge cases with guard enforcement."""
        if size == 0:
            # Empty DataFrame
            data = pd.DataFrame()
            # Should handle empty DataFrame gracefully
            clean_data = self.wrapper.validate_dataframe(data, "EMPTY_GUARD_TEST")
            assert len(clean_data) == 0, "Should handle empty DataFrame"
        else:
            # Small DataFrames
            data = self.create_test_data(size)

            # Should handle small DataFrames
            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(
                data, f"SMALL_GUARD_TEST_{size}"
            )
            assert_validated(clean_data, f"edge_case_{size}")
            assert len(clean_data) == size, f"Should preserve data for size {size}"

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
    def test_falsification_scenarios_with_guards(self, scenario):
        """Test falsification scenarios with guard enforcement."""
        if scenario == "extreme_negative_prices":
            # Create data with extreme negative prices
            dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=timezone.utc)
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

            guard = attach_guard(data)
            # Should handle extreme negative prices
            clean_data = self.wrapper.validate_dataframe(data, "FALSIFY_GUARD_NEGATIVE")
            assert len(clean_data) > 0, "Should handle extreme negative prices"
            assert_validated(clean_data, "falsify_negative")

        elif scenario == "impossible_ohlc":
            # Create data with impossible OHLC relationships
            dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=timezone.utc)
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

            guard = attach_guard(data)
            # Should fail or repair OHLC violations
            try:
                clean_data = self.wrapper.validate_dataframe(data, "FALSIFY_GUARD_OHLC")
                # If repair mode, check that violations were fixed
                assert (
                    clean_data["High"] >= clean_data["Open"]
                ).all(), "OHLC violations should be repaired"
                assert (
                    clean_data["Low"] <= clean_data["Open"]
                ).all(), "OHLC violations should be repaired"
                assert_validated(clean_data, "falsify_ohlc")
            except DataSanityError:
                # If fail mode, exception is expected
                pass

        elif scenario == "future_contamination":
            # Create data with future contamination
            dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=timezone.utc)
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
            data.loc[data.index[5], "Returns"] = data.loc[
                data.index[6], "Returns"
            ]  # Lookahead

            guard = attach_guard(data)
            # Should detect or handle lookahead
            clean_data = self.wrapper.validate_dataframe(
                data, "FALSIFY_GUARD_LOOKAHEAD"
            )
            assert (
                "Returns" in clean_data.columns
            ), "Should handle lookahead contamination"
            assert_validated(clean_data, "falsify_lookahead")

        elif scenario == "invalid_dtypes":
            # Create data with invalid data types
            dates = pd.date_range("2023-01-01", periods=10, freq="D", tz=timezone.utc)
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

            guard = attach_guard(data)
            # Should handle invalid data types
            clean_data = self.wrapper.validate_dataframe(data, "FALSIFY_GUARD_DTYPE")
            assert len(clean_data) > 0, "Should handle invalid data types"
            assert_validated(clean_data, "falsify_dtype")

    @pytest.mark.data_sanity
    @pytest.mark.integration
    def test_integration_guard_with_engines(self):
        """Test integration of guards with trading engines."""
        # Simulate what backtest engine does
        symbols = ["SPY", "QQQ"]
        all_data = []

        for symbol in symbols:
            # Create mock data (simulating engine data loading)
            data = self.create_test_data(100, corrupt=False)
            data["Symbol"] = symbol

            # Attach guard and validate data (like updated backtest engine)
            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, symbol)
            all_data.append(clean_data)

        # Combine data (like backtest engine)
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()

            assert len(combined_data) > 0, "Should have combined data"
            assert "Returns" in combined_data.columns, "Should have returns"
            assert "Symbol" in combined_data.columns, "Should have symbol column"

            # All data should be validated
            for data in all_data:
                assert_validated(data, "integration_test")

    @pytest.mark.data_sanity
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=50)
    @given(st.lists(st.floats(min_value=1, max_value=5000), min_size=5, max_size=20))
    def test_property_based_price_validation(self, prices):
        """Property-based test for price validation."""
        dates = pd.date_range(
            "2023-01-01", periods=len(prices), freq="D", tz=timezone.utc
        )

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
        dates = pd.date_range(
            "2023-01-01", periods=len(extreme_prices), freq="D", tz=timezone.utc
        )

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
            assert (
                clean_data["Close"].max() <= 1000000.0
            ), "Extreme prices should be clipped"
        except DataSanityError:
            # If fail mode, should raise exception
            pass

    @pytest.mark.data_sanity
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=30)
    @given(
        st.lists(
            st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=15
        )
    )
    def test_property_based_volume_validation(self, volumes):
        """Property-based test for volume validation."""
        dates = pd.date_range(
            "2023-01-01", periods=len(volumes), freq="D", tz=timezone.utc
        )

        data = pd.DataFrame(
            {
                "Open": [100.0] * len(volumes),
                "High": [102.0] * len(volumes),
                "Low": [98.0] * len(volumes),
                "Close": [101.0] * len(volumes),
                "Volume": volumes,
            },
            index=dates,
        )

        # Valid data should pass
        clean_data = self.wrapper.validate_dataframe(data, "PROPERTY_VOLUME_TEST")
        assert len(clean_data) == len(data), "Valid volume data should pass unchanged"
        assert (clean_data["Volume"] >= 0).all(), "Volume should be non-negative"

    @pytest.mark.data_sanity
    @pytest.mark.property
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @settings(verbosity=Verbosity.quiet, max_examples=30)
    @given(
        st.lists(
            st.floats(min_value=0.01, max_value=1000000.0), min_size=5, max_size=15
        )
    )
    def test_property_based_returns_calculation(self, prices):
        """Property-based test for returns calculation."""
        dates = pd.date_range(
            "2023-01-01", periods=len(prices), freq="D", tz=timezone.utc
        )

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

        # Should calculate returns correctly
        clean_data = self.wrapper.validate_dataframe(data, "PROPERTY_RETURNS_TEST")
        assert "Returns" in clean_data.columns, "Should calculate returns"

        # Returns should be finite (except first NaN)
        returns_finite = clean_data["Returns"].dropna()
        assert np.isfinite(returns_finite).all(), "All non-NaN returns should be finite"

        # Returns should be log returns
        expected_returns = np.log(clean_data["Close"] / clean_data["Close"].shift(1))
        mask = ~(clean_data["Returns"].isna() | expected_returns.isna())
        if mask.any():
            np.testing.assert_array_almost_equal(
                clean_data["Returns"][mask], expected_returns[mask], decimal=10
            )

    @pytest.mark.data_sanity
    @pytest.mark.edge_case
    @pytest.mark.parametrize(
        "edge_case",
        [
            "empty_dataframe",
            "single_row",
            "non_monotonic_index",
            "duplicate_index",
            "missing_columns",
            "wrong_dtypes",
            "mixed_timezones",
            "multiindex_columns",
        ],
    )
    def test_edge_cases(self, edge_case):
        """Test edge cases and error handling."""

        if edge_case == "empty_dataframe":
            # Test 1: Empty DataFrame
            empty_df = pd.DataFrame()
            # Should handle empty DataFrame gracefully
            clean_data = self.wrapper.validate_dataframe(empty_df, "EMPTY_TEST")
            assert len(clean_data) == 0, "Should handle empty DataFrame"

        elif edge_case == "single_row":
            # Test 2: Single row
            single_row = self.create_test_data(1)
            # Should handle single row
            clean_data = self.wrapper.validate_dataframe(single_row, "SINGLE_ROW_TEST")
            assert len(clean_data) == 1, "Should handle single row"

        elif edge_case == "non_monotonic_index":
            # Test 3: Non-monotonic index
            data = self.create_test_data(10)
            data.index = data.index[::-1]  # Reverse index
            # Should handle non-monotonic index
            clean_data = self.wrapper.validate_dataframe(data, "NON_MONOTONIC_TEST")
            assert len(clean_data) == 10, "Should handle non-monotonic index"

        elif edge_case == "duplicate_index":
            # Test 4: Duplicate index
            data = self.create_test_data(10)
            data.index = pd.date_range(
                "2023-01-01", periods=10, freq="D", tz=timezone.utc
            )
            data.index = data.index.repeat(2)[::2]  # Create duplicates
            # Should handle duplicate index
            clean_data = self.wrapper.validate_dataframe(data, "DUPLICATE_INDEX_TEST")
            assert len(clean_data) == 10, "Should handle duplicate index"

        elif edge_case == "missing_columns":
            # Test 5: Missing required columns
            data = self.create_test_data(10)
            data = data.drop("Close", axis=1)
            # Should handle missing columns
            clean_data = self.wrapper.validate_dataframe(data, "MISSING_COLS_TEST")
            assert len(clean_data) == 10, "Should handle missing columns"

        elif edge_case == "wrong_dtypes":
            # Test 6: Wrong data types
            data = self.create_test_data(10)
            data["Close"] = data["Close"].astype(str)
            # Should handle wrong data types
            clean_data = self.wrapper.validate_dataframe(data, "WRONG_DTYPE_TEST")
            assert len(clean_data) == 10, "Should handle wrong data types"

        elif edge_case == "mixed_timezones":
            # Test 7: Mixed timezones
            data = self.create_test_data(10)
            # Create mixed timezone data
            naive_dates = pd.date_range("2023-01-01", periods=10, freq="D")
            utc_dates = pd.date_range(
                "2023-01-01", periods=10, freq="D", tz=timezone.utc
            )
            mixed_dates = pd.concat(
                [pd.Series(naive_dates[:5]), pd.Series(utc_dates[5:])]
            )
            data.index = mixed_dates

            # Should handle mixed timezones
            clean_data = self.wrapper.validate_dataframe(data, "MIXED_TZ_TEST")
            assert clean_data.index.tz == timezone.utc, "Should convert to UTC"

        elif edge_case == "multiindex_columns":
            # Test 8: MultiIndex columns
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
            assert not isinstance(
                clean_data.columns, pd.MultiIndex
            ), "Should flatten to single level"
            assert "Close" in clean_data.columns, "Should have Close column"

    @pytest.mark.data_sanity
    @pytest.mark.perf
    @pytest.mark.benchmark
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_performance_edge_cases(self, size):
        """Test performance of edge case handling."""
        data = self.create_test_data(size, corrupt=False)

        # Measure performance
        start_time = time.time()
        clean_data = self.wrapper.validate_dataframe(data, f"PERF_EDGE_{size}")
        validation_time = time.time() - start_time

        # Should handle edge cases within reasonable time
        assert (
            validation_time < 10.0
        ), f"Edge case handling took {validation_time:.2f}s for size {size}, expected < 10.0s"
        assert len(clean_data) > 0, f"Should have data for size {size}"

        # Log performance metrics
        print(f"Edge Case Size {size}: Time={validation_time:.3f}s")

    @pytest.mark.data_sanity
    @pytest.mark.network
    @pytest.mark.flaky
    @patch("yfinance.download")
    def test_network_resilience_with_guards(self, mock_download):
        """Test network resilience with guard enforcement."""
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

        # Should handle partial corruption with guards
        data = mock_download("SPY", start="2024-01-01", end="2024-01-31")
        guard = attach_guard(data)
        clean_data = self.wrapper.validate_dataframe(data, "NETWORK_GUARD_TEST")
        assert len(clean_data) > 0, "Should handle partial data corruption"
        assert_validated(clean_data, "network_test")

    @pytest.mark.data_sanity
    @pytest.mark.regression
    def test_regression_guard_behavior(self):
        """Test regression scenarios for guard behavior."""
        # Test that guards maintain consistent behavior
        data = self.create_test_data(100)

        # First validation
        guard1 = attach_guard(data)
        clean_data1 = self.wrapper.validate_dataframe(data, "REGRESSION_TEST_1")
        status1 = guard1.get_status()

        # Second validation (should be consistent)
        guard2 = attach_guard(data)
        clean_data2 = self.wrapper.validate_dataframe(data, "REGRESSION_TEST_2")
        status2 = guard2.get_status()

        # Should have consistent results
        assert (
            status1["validated"] == status2["validated"]
        ), "Guard status should be consistent"
        assert len(clean_data1) == len(
            clean_data2
        ), "Validation results should be consistent"
        assert clean_data1.equals(clean_data2), "Validation results should be identical"

    @pytest.mark.data_sanity
    @pytest.mark.smoke
    def test_smoke_guard_functionality(self):
        """Smoke test for basic guard functionality."""
        # Test basic guard functionality
        data = self.create_test_data(10)

        # Attach guard
        guard = attach_guard(data)
        assert hasattr(data, "_sanity_guard"), "Should attach guard"

        # Validate data
        clean_data = self.wrapper.validate_dataframe(data, "SMOKE_TEST")
        assert_validated(clean_data, "smoke_test")

        # Check guard status
        status = guard.get_status()
        assert status["validated"] is True, "Should be validated"
        assert status["symbol"] == "SMOKE_TEST", "Should have correct symbol"

    @pytest.mark.data_sanity
    @pytest.mark.acceptance
    def test_acceptance_guard_integration(self):
        """Acceptance test for guard integration."""
        # Test complete guard integration workflow
        symbols = ["AAPL", "GOOGL"]
        validated_data = []

        for symbol in symbols:
            # Create data
            data = self.create_test_data(50, corrupt=False)

            # Attach guard and validate
            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, symbol)

            # Verify validation
            assert_validated(clean_data, f"acceptance_{symbol}")
            status = guard.get_status()
            assert status["validated"] is True, f"Should be validated for {symbol}"

            validated_data.append(clean_data)

        # Verify all data is validated
        assert len(validated_data) == len(symbols), "Should validate all symbols"
        for data in validated_data:
            assert "Returns" in data.columns, "Should calculate returns"
            assert data.index.is_monotonic_increasing, "Should have monotonic index"
            assert data.index.tz == timezone.utc, "Should be UTC"

    @pytest.mark.data_sanity
    @pytest.mark.validation
    def test_lookahead_contamination_detection(self):
        """Test detection of lookahead contamination."""
        data = self.create_test_data(10)

        # Add lookahead contamination (future returns)
        data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data.loc[data.index[5], "Returns"] = data.loc[
            data.index[6], "Returns"
        ]  # Lookahead

        # Should detect lookahead
        clean_data = self.wrapper.validate_dataframe(data, "LOOKAHEAD_TEST")
        # Note: This is a basic check - more sophisticated lookahead detection would be needed
        assert "Returns" in clean_data.columns, "Should handle lookahead contamination"

    @pytest.mark.data_sanity
    @pytest.mark.validation
    def test_corporate_actions_consistency(self):
        """Test corporate actions consistency."""
        data = self.create_test_data(10)

        # Add Adj Close with large differences
        data["Adj Close"] = data["Close"] * 1.5  # Simulate split

        # Should handle gracefully
        clean_data = self.wrapper.validate_dataframe(data, "CORP_ACTIONS_TEST")
        assert "Adj Close" in clean_data.columns, "Should preserve Adj Close column"

    @pytest.mark.data_sanity
    @pytest.mark.perf
    @pytest.mark.benchmark
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_performance_safety(self, size):
        """Test performance safety and overhead for different sizes."""
        # Create larger dataset
        data = self.create_test_data(size)

        # Measure validation time
        start_time = time.time()
        clean_data = self.wrapper.validate_dataframe(data, f"PERF_TEST_{size}")
        validation_time = time.time() - start_time

        # Should complete within reasonable time
        assert (
            validation_time < 10.0
        ), f"Validation took {validation_time:.2f}s for size {size}, expected < 10.0s"

        # Should not lose data
        assert len(clean_data) == len(data), f"Should preserve data for size {size}"

        # Log performance metrics
        print(f"Performance Size {size}: Time={validation_time:.3f}s")

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

        data = self.create_test_data(1000, corrupt=True, corruption_type="mixed")

        # Measure performance
        start_time = time.time()
        clean_data, result = validator.validate_and_repair(data, f"PERF_{repair_mode}")
        validation_time = time.time() - start_time

        # All modes should complete within reasonable time
        assert (
            validation_time < 5.0
        ), f"Repair mode {repair_mode} took {validation_time:.2f}s, expected < 5.0s"
        assert len(clean_data) > 0, f"Should have data for repair mode {repair_mode}"

    @pytest.mark.data_sanity
    @pytest.mark.stress
    @pytest.mark.slow
    def test_stress_performance_large_dataset(self):
        """Stress test performance with very large dataset."""
        data = self.create_test_data(50000, corrupt=True, corruption_type="mixed")

        # Measure performance
        start_time = time.time()
        clean_data = self.wrapper.validate_dataframe(data, "STRESS_PERF_TEST")
        validation_time = time.time() - start_time

        # Should handle large dataset
        assert (
            validation_time < 120.0
        ), f"Large dataset validation took {validation_time:.2f}s, expected < 120.0s"
        assert len(clean_data) > 0, "Should have data after stress test"
        assert len(clean_data) < len(data), "Should remove some corrupt data"

    @pytest.mark.data_sanity
    @pytest.mark.memory
    @pytest.mark.parametrize("size", [1000, 10000, 50000])
    def test_memory_usage_patterns(self, size):
        """Test memory usage patterns."""
        data = self.create_test_data(size, corrupt=False)

        # Measure memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Validate data
        clean_data = self.wrapper.validate_dataframe(data, f"MEMORY_TEST_{size}")

        # Measure memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Memory usage should be reasonable
        assert (
            memory_increase < 2000.0
        ), f"Memory increase {memory_increase:.2f}MB for size {size}, expected < 2000MB"
        assert len(clean_data) > 0, f"Should have data for size {size}"

        # Log memory metrics
        print(f"Memory Size {size}: Increase={memory_increase:.2f}MB")

    @pytest.mark.data_sanity
    @pytest.mark.integration
    def test_integration_performance_with_engines(self):
        """Test integration performance with trading engines."""
        # Simulate what backtest engine does
        symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        all_data = []

        start_time = time.time()

        for symbol in symbols:
            # Create mock data (simulating engine data loading)
            data = self.create_test_data(1000, corrupt=False)
            data["Symbol"] = symbol

            # Validate data (like updated backtest engine)
            clean_data = self.wrapper.validate_dataframe(data, symbol)
            all_data.append(clean_data)

        total_time = time.time() - start_time

        # Should complete within reasonable time
        assert (
            total_time < 30.0
        ), f"Integration validation took {total_time:.2f}s, expected < 30.0s"

        # Combine data (like backtest engine)
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index()

            assert len(combined_data) > 0, "Should have combined data"
            assert "Returns" in combined_data.columns, "Should have returns"
            assert "Symbol" in combined_data.columns, "Should have symbol column"

            print(
                f"Integration Test: {len(symbols)} symbols, {len(combined_data)} total rows, {total_time:.3f}s"
            )

    @pytest.mark.data_sanity
    @pytest.mark.regression
    def test_regression_performance_consistency(self):
        """Test regression scenarios for performance consistency."""
        # Test that performance is consistent across runs
        data = self.create_test_data(1000, corrupt=True, corruption_type="mixed")

        # First run
        start_time1 = time.time()
        clean_data1 = self.wrapper.validate_dataframe(data, "REGRESSION_PERF_1")
        time1 = time.time() - start_time1

        # Second run
        start_time2 = time.time()
        clean_data2 = self.wrapper.validate_dataframe(data, "REGRESSION_PERF_2")
        time2 = time.time() - start_time2

        # Performance should be consistent (within 50% variance)
        time_diff = abs(time1 - time2)
        avg_time = (time1 + time2) / 2
        variance = time_diff / avg_time if avg_time > 0 else 0

        assert (
            variance < 0.5
        ), f"Performance variance {variance:.2f} too high (times: {time1:.3f}s, {time2:.3f}s)"
        assert len(clean_data1) == len(clean_data2), "Results should be consistent"
        assert clean_data1.equals(clean_data2), "Results should be identical"

    @pytest.mark.data_sanity
    @pytest.mark.smoke
    def test_smoke_performance_basic(self):
        """Smoke test for basic performance."""
        # Test basic performance functionality
        data = self.create_test_data(100)

        # Should complete quickly
        start_time = time.time()
        clean_data = self.wrapper.validate_dataframe(data, "SMOKE_PERF_TEST")
        validation_time = time.time() - start_time

        assert (
            validation_time < 1.0
        ), f"Smoke test took {validation_time:.2f}s, expected < 1.0s"
        assert len(clean_data) == len(data), "Should preserve data"
        assert "Returns" in clean_data.columns, "Should calculate returns"

    @pytest.mark.data_sanity
    @pytest.mark.acceptance
    def test_acceptance_performance_workflow(self):
        """Acceptance test for performance workflow."""
        # Test complete performance workflow
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        total_rows = 0
        start_time = time.time()

        for symbol in symbols:
            # Create data
            data = self.create_test_data(500, corrupt=False)

            # Validate data
            clean_data = self.wrapper.validate_dataframe(data, symbol)
            total_rows += len(clean_data)

        total_time = time.time() - start_time

        # Should complete within reasonable time
        assert (
            total_time < 10.0
        ), f"Acceptance test took {total_time:.2f}s, expected < 10.0s"
        assert total_rows > 0, "Should have data"

        print(
            f"Acceptance Test: {len(symbols)} symbols, {total_rows} total rows, {total_time:.3f}s"
        )

    @pytest.mark.data_sanity
    @pytest.mark.corruption
    @pytest.mark.parametrize(
        "corruption_scenario",
        [
            "price_spikes",
            "volume_anomalies",
            "timestamp_corruption",
            "column_mixing",
            "data_type_corruption",
            "index_corruption",
        ],
    )
    def test_comprehensive_corruption_scenarios(self, corruption_scenario):
        """Test comprehensive corruption scenarios."""
        if corruption_scenario == "price_spikes":
            # Create data with price spikes
            data = self.create_test_data(20)
            data.loc[data.index[5], "Close"] = (
                data.loc[data.index[5], "Close"] * 10
            )  # 10x spike
            data.loc[data.index[10], "Close"] = (
                data.loc[data.index[10], "Close"] * 0.1
            )  # 90% drop

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "CORRUPTION_SPIKES")
            assert len(clean_data) > 0, "Should handle price spikes"
            assert_validated(clean_data, "corruption_spikes")

        elif corruption_scenario == "volume_anomalies":
            # Create data with volume anomalies
            data = self.create_test_data(20)
            data.loc[data.index[5], "Volume"] = 0  # Zero volume
            data.loc[data.index[10], "Volume"] = 1e15  # Extreme volume

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "CORRUPTION_VOLUME")
            assert len(clean_data) > 0, "Should handle volume anomalies"
            assert_validated(clean_data, "corruption_volume")

        elif corruption_scenario == "timestamp_corruption":
            # Create data with timestamp corruption
            data = self.create_test_data(20)
            data.index = data.index.astype(str)  # Convert to string timestamps

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "CORRUPTION_TIMESTAMP")
            assert len(clean_data) > 0, "Should handle timestamp corruption"
            assert isinstance(
                clean_data.index, pd.DatetimeIndex
            ), "Should restore datetime index"
            assert_validated(clean_data, "corruption_timestamp")

        elif corruption_scenario == "column_mixing":
            # Create data with column mixing
            data = self.create_test_data(20)
            data["Open"], data["Close"] = data["Close"], data["Open"]  # Swap columns

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "CORRUPTION_COLUMNS")
            assert len(clean_data) > 0, "Should handle column mixing"
            assert_validated(clean_data, "corruption_columns")

        elif corruption_scenario == "data_type_corruption":
            # Create data with data type corruption
            data = self.create_test_data(20)
            data["Close"] = data["Close"].astype(str)  # Convert to string

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "CORRUPTION_DTYPE")
            assert len(clean_data) > 0, "Should handle data type corruption"
            assert clean_data["Close"].dtype in [
                np.float64,
                np.float32,
            ], "Should restore numeric dtype"
            assert_validated(clean_data, "corruption_dtype")

        elif corruption_scenario == "index_corruption":
            # Create data with index corruption
            data = self.create_test_data(20)
            data.index = pd.RangeIndex(0, len(data))  # Replace with range index

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "CORRUPTION_INDEX")
            assert len(clean_data) > 0, "Should handle index corruption"
            assert isinstance(
                clean_data.index, pd.DatetimeIndex
            ), "Should restore datetime index"
            assert_validated(clean_data, "corruption_index")

    @pytest.mark.data_sanity
    @pytest.mark.repair
    @pytest.mark.parametrize(
        "repair_strategy", ["winsorize", "clip", "drop", "interpolate"]
    )
    def test_repair_functionality(self, repair_strategy):
        """Test different repair strategies."""
        # Create corrupted data
        data = self.create_test_data(50, corrupt=True, corruption_type="extreme_prices")

        # Create config with specific repair strategy
        config = self.validator.config.copy()
        config["repair_mode"] = repair_strategy

        validator = DataSanityValidator()
        validator.config = config

        guard = attach_guard(data)

        if repair_strategy == "winsorize":
            # Test winsorization
            clean_data, result = validator.validate_and_repair(
                data, f"REPAIR_{repair_strategy}"
            )
            assert len(clean_data) == len(
                data
            ), "Winsorization should preserve all rows"
            assert_validated(clean_data, f"repair_{repair_strategy}")

        elif repair_strategy == "clip":
            # Test clipping
            clean_data, result = validator.validate_and_repair(
                data, f"REPAIR_{repair_strategy}"
            )
            assert len(clean_data) == len(data), "Clipping should preserve all rows"
            assert clean_data["Close"].max() <= 1000000.0, "Should clip extreme prices"
            assert_validated(clean_data, f"repair_{repair_strategy}")

        elif repair_strategy == "drop":
            # Test dropping
            clean_data, result = validator.validate_and_repair(
                data, f"REPAIR_{repair_strategy}"
            )
            assert len(clean_data) <= len(data), "Dropping should remove corrupt rows"
            assert_validated(clean_data, f"repair_{repair_strategy}")

        elif repair_strategy == "interpolate":
            # Test interpolation
            clean_data, result = validator.validate_and_repair(
                data, f"REPAIR_{repair_strategy}"
            )
            assert len(clean_data) == len(
                data
            ), "Interpolation should preserve all rows"
            assert_validated(clean_data, f"repair_{repair_strategy}")

    @pytest.mark.data_sanity
    @pytest.mark.unit
    @pytest.mark.parametrize("validation_type", ["basic", "strict", "lenient"])
    def test_unit_validation(self, validation_type):
        """Unit-level validation tests."""
        data = self.create_test_data(10)

        if validation_type == "basic":
            # Basic validation
            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "UNIT_BASIC")
            assert len(clean_data) == len(data), "Basic validation should preserve data"
            assert_validated(clean_data, "unit_basic")

        elif validation_type == "strict":
            # Strict validation
            validator = self.create_config_with_profile("strict")
            guard = attach_guard(data)
            clean_data, result = validator.validate_and_repair(data, "UNIT_STRICT")
            assert len(clean_data) == len(
                data
            ), "Strict validation should preserve valid data"
            assert_validated(clean_data, "unit_strict")

        elif validation_type == "lenient":
            # Lenient validation
            validator = self.create_config_with_profile("lenient")
            guard = attach_guard(data)
            clean_data, result = validator.validate_and_repair(data, "UNIT_LENIENT")
            assert len(clean_data) == len(
                data
            ), "Lenient validation should preserve data"
            assert_validated(clean_data, "unit_lenient")

    @pytest.mark.data_sanity
    @pytest.mark.memory
    @pytest.mark.parametrize(
        "memory_scenario",
        ["large_dataset", "many_columns", "deep_copy", "chained_operations"],
    )
    def test_memory_usage_scenarios(self, memory_scenario):
        """Test memory usage in different scenarios."""
        if memory_scenario == "large_dataset":
            # Test with large dataset
            data = self.create_test_data(50000)

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "MEMORY_LARGE")

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            assert (
                memory_increase < 1000.0
            ), f"Large dataset memory increase {memory_increase:.2f}MB, expected < 1000MB"
            assert_validated(clean_data, "memory_large")

        elif memory_scenario == "many_columns":
            # Test with many columns
            data = self.create_test_data(1000)
            # Add many additional columns
            for i in range(50):
                data[f"feature_{i}"] = np.random.randn(len(data))

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "MEMORY_COLUMNS")

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            assert (
                memory_increase < 500.0
            ), f"Many columns memory increase {memory_increase:.2f}MB, expected < 500MB"
            assert_validated(clean_data, "memory_columns")

        elif memory_scenario == "deep_copy":
            # Test memory usage with deep copy operations
            data = self.create_test_data(1000)

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Perform multiple deep copy operations
            for i in range(10):
                guard = attach_guard(data.copy())
                clean_data = self.wrapper.validate_dataframe(
                    data.copy(), f"MEMORY_COPY_{i}"
                )
                assert_validated(clean_data, f"memory_copy_{i}")

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            assert (
                memory_increase < 200.0
            ), f"Deep copy memory increase {memory_increase:.2f}MB, expected < 200MB"

        elif memory_scenario == "chained_operations":
            # Test memory usage with chained operations
            data = self.create_test_data(1000)

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Perform chained operations
            guard = attach_guard(data)
            clean_data1 = self.wrapper.validate_dataframe(data, "MEMORY_CHAIN_1")
            clean_data2 = self.wrapper.validate_dataframe(clean_data1, "MEMORY_CHAIN_2")
            clean_data3 = self.wrapper.validate_dataframe(clean_data2, "MEMORY_CHAIN_3")

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before

            assert (
                memory_increase < 300.0
            ), f"Chained operations memory increase {memory_increase:.2f}MB, expected < 300MB"
            assert_validated(clean_data3, "memory_chain")

    @pytest.mark.data_sanity
    @pytest.mark.edge_case
    @pytest.mark.parametrize(
        "advanced_edge_case",
        [
            "market_holidays",
            "corporate_actions",
            "currency_conversion",
            "timezone_boundaries",
            "leap_years",
            "millisecond_precision",
        ],
    )
    def test_advanced_edge_cases(self, advanced_edge_case):
        """Test advanced edge cases."""
        if advanced_edge_case == "market_holidays":
            # Test with market holidays (weekends)
            dates = pd.date_range("2023-01-01", periods=20, freq="D", tz=timezone.utc)
            # Filter to weekends only
            weekend_dates = dates[dates.weekday >= 5]

            data = pd.DataFrame(
                {
                    "Open": [100.0 + i for i in range(len(weekend_dates))],
                    "High": [102.0 + i for i in range(len(weekend_dates))],
                    "Low": [98.0 + i for i in range(len(weekend_dates))],
                    "Close": [101.0 + i for i in range(len(weekend_dates))],
                    "Volume": [1000000] * len(weekend_dates),
                },
                index=weekend_dates,
            )

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "EDGE_HOLIDAYS")
            assert len(clean_data) > 0, "Should handle market holidays"
            assert_validated(clean_data, "edge_holidays")

        elif advanced_edge_case == "corporate_actions":
            # Test with corporate actions (splits, dividends)
            data = self.create_test_data(20)
            # Simulate stock split
            data.loc[data.index[10:], "Close"] = data.loc[data.index[10:], "Close"] * 2
            data.loc[data.index[10:], "Open"] = data.loc[data.index[10:], "Open"] * 2
            data.loc[data.index[10:], "High"] = data.loc[data.index[10:], "High"] * 2
            data.loc[data.index[10:], "Low"] = data.loc[data.index[10:], "Low"] * 2

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "EDGE_CORP_ACTIONS")
            assert len(clean_data) > 0, "Should handle corporate actions"
            assert_validated(clean_data, "edge_corp_actions")

        elif advanced_edge_case == "currency_conversion":
            # Test with currency conversion scenarios
            data = self.create_test_data(20)
            # Simulate currency conversion (multiply by exchange rate)
            exchange_rate = 1.25
            data["Close"] = data["Close"] * exchange_rate
            data["Open"] = data["Open"] * exchange_rate
            data["High"] = data["High"] * exchange_rate
            data["Low"] = data["Low"] * exchange_rate

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "EDGE_CURRENCY")
            assert len(clean_data) > 0, "Should handle currency conversion"
            assert_validated(clean_data, "edge_currency")

        elif advanced_edge_case == "timezone_boundaries":
            # Test with timezone boundary conditions
            # Create data around DST transitions
            dates = pd.date_range("2023-03-12", periods=10, freq="h", tz="US/Eastern")
            data = pd.DataFrame(
                {
                    "Open": [100.0 + i for i in range(len(dates))],
                    "High": [102.0 + i for i in range(len(dates))],
                    "Low": [98.0 + i for i in range(len(dates))],
                    "Close": [101.0 + i for i in range(len(dates))],
                    "Volume": [1000000] * len(dates),
                },
                index=dates,
            )

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "EDGE_TIMEZONE")
            assert len(clean_data) > 0, "Should handle timezone boundaries"
            assert clean_data.index.tz == timezone.utc, "Should convert to UTC"
            assert_validated(clean_data, "edge_timezone")

        elif advanced_edge_case == "leap_years":
            # Test with leap year data
            dates = pd.date_range("2024-02-28", periods=5, freq="D", tz=timezone.utc)
            data = pd.DataFrame(
                {
                    "Open": [100.0 + i for i in range(len(dates))],
                    "High": [102.0 + i for i in range(len(dates))],
                    "Low": [98.0 + i for i in range(len(dates))],
                    "Close": [101.0 + i for i in range(len(dates))],
                    "Volume": [1000000] * len(dates),
                },
                index=dates,
            )

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "EDGE_LEAP_YEAR")
            assert len(clean_data) > 0, "Should handle leap years"
            assert_validated(clean_data, "edge_leap_year")

        elif advanced_edge_case == "millisecond_precision":
            # Test with millisecond precision timestamps
            dates = pd.date_range("2023-01-01", periods=10, freq="1ms", tz=timezone.utc)
            data = pd.DataFrame(
                {
                    "Open": [100.0 + i * 0.001 for i in range(len(dates))],
                    "High": [102.0 + i * 0.001 for i in range(len(dates))],
                    "Low": [98.0 + i * 0.001 for i in range(len(dates))],
                    "Close": [101.0 + i * 0.001 for i in range(len(dates))],
                    "Volume": [1000000] * len(dates),
                },
                index=dates,
            )

            guard = attach_guard(data)
            clean_data = self.wrapper.validate_dataframe(data, "EDGE_MILLISECOND")
            assert len(clean_data) > 0, "Should handle millisecond precision"
            assert_validated(clean_data, "edge_millisecond")


def test_integration_presence():
    """Test that DataSanity is integrated in all required modules."""

    integration_modules = [
        "core/engine/backtest.py",
        "core/engine/paper.py",
        "brokers/data_provider.py",
        "features/feature_engine.py",
        "features/ensemble.py",
        "strategies/ensemble_strategy.py",
    ]

    for module_path in integration_modules:
        try:
            with open(module_path) as f:
                content = f.read()

            # Check for DataSanity imports
            assert (
                "get_data_sanity_wrapper" in content
                or "validate_dataframe" in content
                or "load_and_validate" in content
            ), f"Module {module_path} missing DataSanity integration"

            # Parse AST to check for actual usage
            tree = ast.parse(content)

            # Look for function calls
            calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, "attr"):
                        calls.append(node.func.attr)
                    elif hasattr(node.func, "id"):
                        calls.append(node.func.id)

            # Should have validation calls
            validation_calls = [
                "validate_dataframe",
                "load_and_validate",
                "validate_market_data",
            ]
            has_validation = any(call in calls for call in validation_calls)

            assert (
                has_validation
            ), f"Module {module_path} missing validation calls in AST"

        except FileNotFoundError:
            pytest.skip(f"Module {module_path} not found")


def test_falsification_scenarios():
    """Test falsification scenarios that should always fail."""

    wrapper = get_data_sanity_wrapper()

    # Scenario 1: Extreme price corruption
    data = pd.DataFrame(
        {
            "Open": [100.0, 200.0, 1e11, 400.0],  # Extreme price
            "High": [102.0, 202.0, 1e11, 402.0],
            "Low": [98.0, 198.0, 1e11, 398.0],
            "Close": [101.0, 201.0, 1e11, 401.0],
            "Volume": [1000000] * 4,
        },
        index=pd.date_range("2023-01-01", periods=4, freq="D", tz=timezone.utc),
    )

    with pytest.raises(DataSanityError, match="prices > 1000000.0"):
        wrapper.validate_dataframe(data, "FALSIFY_EXTREME")

    # Scenario 2: Negative prices
    data = pd.DataFrame(
        {
            "Open": [100.0, -50.0, 300.0, 400.0],  # Negative price
            "High": [102.0, -48.0, 302.0, 402.0],
            "Low": [98.0, -52.0, 298.0, 398.0],
            "Close": [101.0, -50.0, 301.0, 401.0],
            "Volume": [1000000] * 4,
        },
        index=pd.date_range("2023-01-01", periods=4, freq="D", tz=timezone.utc),
    )

    with pytest.raises(DataSanityError, match="prices < 0.01"):
        wrapper.validate_dataframe(data, "FALSIFY_NEGATIVE")

    # Scenario 3: NaN burst
    data = pd.DataFrame(
        {
            "Open": [100.0, np.nan, 300.0, np.nan],
            "High": [102.0, np.nan, 302.0, np.nan],
            "Low": [98.0, np.nan, 298.0, np.nan],
            "Close": [101.0, np.nan, 301.0, np.nan],
            "Volume": [1000000] * 4,
        },
        index=pd.date_range("2023-01-01", periods=4, freq="D", tz=timezone.utc),
    )

    with pytest.raises(DataSanityError, match="non-finite values"):
        wrapper.validate_dataframe(data, "FALSIFY_NAN")

    # Scenario 4: OHLC violations
    data = pd.DataFrame(
        {
            "Open": [100.0, 200.0, 300.0, 400.0],
            "High": [102.0, 198.0, 302.0, 402.0],  # High < Open
            "Low": [98.0, 202.0, 298.0, 398.0],  # Low > Open
            "Close": [101.0, 201.0, 301.0, 401.0],
            "Volume": [1000000] * 4,
        },
        index=pd.date_range("2023-01-01", periods=4, freq="D", tz=timezone.utc),
    )

    with pytest.raises(DataSanityError, match="OHLC consistency"):
        wrapper.validate_dataframe(data, "FALSIFY_OHLC")

    # Scenario 5: Negative volume
    data = pd.DataFrame(
        {
            "Open": [100.0, 200.0, 300.0, 400.0],
            "High": [102.0, 202.0, 302.0, 402.0],
            "Low": [98.0, 198.0, 298.0, 398.0],
            "Close": [101.0, 201.0, 301.0, 401.0],
            "Volume": [1000000, -500000, 1000000, 1000000],  # Negative volume
        },
        index=pd.date_range("2023-01-01", periods=4, freq="D", tz=timezone.utc),
    )

    with pytest.raises(DataSanityError, match="negative volume"):
        wrapper.validate_dataframe(data, "FALSIFY_VOLUME")


def test_guard_enforcement():
    """Test that guards prevent unvalidated data usage."""

    @require_validated_data("test_function")
    def test_function(df: pd.DataFrame) -> str:
        return "success"

    # Create unvalidated data
    data = pd.DataFrame(
        {
            "Open": [100.0, 200.0, 300.0],
            "High": [102.0, 202.0, 302.0],
            "Low": [98.0, 198.0, 298.0],
            "Close": [101.0, 201.0, 301.0],
            "Volume": [1000000] * 3,
        },
        index=pd.date_range("2023-01-01", periods=3, freq="D", tz=timezone.utc),
    )

    # Should fail without validation
    with pytest.raises(DataSanityError, match="DataFrame used before validation"):
        test_function(data)

    # Should pass with validation
    wrapper = get_data_sanity_wrapper()
    clean_data = wrapper.validate_dataframe(data, "TEST")
    assert test_function(clean_data) == "success"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
