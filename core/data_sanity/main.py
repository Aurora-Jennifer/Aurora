"""
Data Sanity Layer
Comprehensive data validation and repair for all market data sources.
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""

import hashlib
import logging
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


from .errors import DataSanityError


@dataclass
class ValidationResult:
    """Result of data validation with detailed information."""

    repairs: list[str]  # List of repairs performed
    flags: list[str]  # List of flags raised (e.g., ["lookahead_detected"])
    outliers: int  # Number of outliers detected
    rows_in: int  # Number of input rows
    rows_out: int  # Number of output rows
    profile: str  # Profile used for validation
    validation_time: float  # Time taken for validation


# Lightweight CI-facing result (non-breaking: new types and methods)
@dataclass
class SanityViolation:
    code: str
    details: str


@dataclass
class SanityCheckResult:
    mode: str
    violations: list[SanityViolation]
    ok: bool

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "violations": [vi.__dict__ for vi in self.violations],
        }

    def summary(self) -> str:
        if self.ok or not self.violations:
            return "data_sanity_ok"
        v = self.violations[0]
        return f"data_sanity_violation[{v.code}]: {v.details}"


class DataSanityGuard:
    """
    Runtime guard to ensure DataFrames are validated before use.

    This guard tracks DataFrame objects and ensures they've been validated
    before being consumed by downstream components.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize guard for a DataFrame."""
        self._df_id = id(df)
        self._df_hash = self._compute_hash(df)
        self._validated = False
        self._validation_time = None
        self._symbol = None

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the DataFrame for tracking."""
        # Hash the shape, dtypes, and first/last few values
        hash_data = f"{df.shape}_{df.dtypes.to_dict()}"
        if len(df) > 0:
            hash_data += f"_{df.iloc[0].to_dict()}_{df.iloc[-1].to_dict()}"
        return hashlib.md5(hash_data.encode()).hexdigest()[:8]

    def mark_validated(self, df: pd.DataFrame, symbol: str = None):
        """Mark this DataFrame as validated."""
        if id(df) == self._df_id:
            self._validated = True
            self._validation_time = datetime.now()
            self._symbol = symbol
            logger.debug(
                f"DataSanityGuard: Marked DataFrame {self._df_id} as validated for {symbol}"
            )
        else:
            logger.warning("DataSanityGuard: DataFrame ID mismatch during validation")

    def assert_validated(self, context: str = "unknown"):
        """Assert that this DataFrame has been validated."""
        if not self._validated:
            raise DataSanityError(
                f"DataSanityGuard: DataFrame used before validation in {context}. "
                f"ID: {self._df_id}, Hash: {self._df_hash}"
            )
        logger.debug(f"DataSanityGuard: DataFrame {self._df_id} validated for {context}")

    def get_status(self) -> dict:
        """Get guard status information."""
        return {
            "df_id": self._df_id,
            "df_hash": self._df_hash,
            "validated": self._validated,
            "validation_time": self._validation_time,
            "symbol": self._symbol,
        }


# Global registry of guards
_guard_registry = weakref.WeakValueDictionary()


def attach_guard(df: pd.DataFrame) -> DataSanityGuard:
    """Attach a DataSanityGuard to a DataFrame."""
    guard = DataSanityGuard(df)
    _guard_registry[id(df)] = guard
    df._sanity_guard = guard
    return guard


def get_guard(df: pd.DataFrame) -> DataSanityGuard | None:
    """Get the DataSanityGuard for a DataFrame."""
    return getattr(df, "_sanity_guard", None)


def assert_validated(df: pd.DataFrame, context: str = "unknown"):
    """Assert that a DataFrame has been validated."""
    guard = get_guard(df)
    if guard:
        guard.assert_validated(context)
    else:
        # If no guard, create one and mark as validated (backward compatibility)
        logger.warning(f"DataSanityGuard: No guard found for DataFrame in {context}, creating one")
        attach_guard(df)


class DataSanityValidator:
    """
    Comprehensive data validation and repair for market data.

    Validates:
    - Time series integrity (monotonic, UTC, no duplicates)
    - Price data sanity (finite, positive, reasonable bounds)
    - OHLC consistency (low <= {open,close} <= high)
    - Volume data validity
    - Outlier detection and repair
    """

    def __init__(self, config_path: str = "config/data_sanity.yaml", profile: str = "default"):
        """
        Initialize DataSanityValidator.

        Args:
            config_path: Path to configuration file
            profile: Profile to use ("default" or "strict")
        """
        self.config = self._load_config(config_path)
        self.profile = profile
        self.profile_config = self._get_profile_config(profile)
        self.repair_count = 0
        self.outlier_count = 0
        self.validation_failures = []

        logger.info(
            f"Initialized DataSanityValidator with profile: {profile}, mode: {self.profile_config['mode']}"
        )

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded data sanity config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return self._get_default_config()

    def _get_profile_config(self, profile: str) -> dict:
        """Get profile-specific configuration."""
        profiles = self.config.get("profiles", {})
        if profile not in profiles:
            logger.warning(f"Profile '{profile}' not found, using default")
            profile = "default"

        return profiles.get(profile, {})

    @staticmethod
    def in_ci() -> bool:
        import os

        return str(os.getenv("CI", "")).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_tz(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
        if getattr(idx, "tz", None) is None:
            return idx.tz_localize(tz)
        return idx.tz_convert(tz)

    def validate_dataframe_fast(self, df: pd.DataFrame, profile: str) -> SanityCheckResult:
        import numpy as np

        vio: list[SanityViolation] = []
        cfg_raw = self._get_profile_config(profile)
        mode = cfg_raw.get("mode", "warn")
        enforced = mode == "enforce"
        if df is None or len(df) == 0:
            return SanityCheckResult(
                mode=mode, violations=[SanityViolation("EMPTY_SERIES", "no rows")], ok=False
            )
        cfg = cfg_raw
        # timezone
        tz = cfg.get("tz")
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        if tz and isinstance(idx, pd.DatetimeIndex):
            try:
                new_idx = self._normalize_tz(idx, tz)
                if not new_idx.equals(idx):
                    df = df.copy()
                    df.index = new_idx
                    idx = new_idx
            except Exception:
                pass
        # monotonic & duplicates
        require_monotonic = cfg.get("require_monotonic_dates", False) or enforced
        if require_monotonic and idx is not None and not idx.is_monotonic_increasing:
            vio.append(SanityViolation("NON_MONO_INDEX", "index not non-decreasing"))
        forbid_duplicates = cfg.get("forbid_duplicates", False) or enforced
        if forbid_duplicates and idx is not None and idx.has_duplicates:
            n_dupes = int(idx.duplicated().sum())
            vio.append(SanityViolation("DUP_TS", f"{n_dupes} duplicate stamps"))
        # numeric inf/nan
        nums = df.select_dtypes(include=[np.number])
        arr = (
            np.asarray(nums.to_numpy(dtype="float64"), dtype="float64")
            if nums.shape[1]
            else np.empty((len(df), 0))
        )
        forbid_inf = cfg.get("forbid_infinite", False) or enforced
        if forbid_inf and arr.size and np.isinf(arr).any():
            vio.append(SanityViolation("INF_VALUES", "infinite values present"))
        max_nan = cfg.get("max_nan_pct", (0.0 if enforced else None))
        if max_nan is not None and arr.size:
            nan_pct = float(np.isnan(arr).mean())
            if nan_pct > max_nan:
                vio.append(
                    SanityViolation("NAN_VALUES", f"NaN fraction {nan_pct:.4f} > {max_nan:.4f}")
                )
        return SanityCheckResult(mode=mode, violations=vio, ok=(len(vio) == 0))

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
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

    def validate_and_repair(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Validate and repair market data with strict invariants.

        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol name for logging

        Returns:
            Tuple of (cleaned DataFrame, ValidationResult)
        """
        import time

        start_time = time.time()

        if data.empty:
            if self.profile_config.get("allow_repairs", True):
                logger.warning(f"Empty data for {symbol}")
                result = ValidationResult(
                    repairs=[],
                    flags=[],
                    outliers=0,
                    rows_in=0,
                    rows_out=0,
                    profile=self.profile,
                    validation_time=time.time() - start_time,
                )
                return data, result
            else:
                raise DataSanityError(f"{symbol}: Empty data not allowed in strict mode")

        original_shape = data.shape
        logger.info(f"Validating {symbol}: {original_shape[0]} rows, {original_shape[1]} columns")

        # Initialize validation result
        repairs = []
        flags = []
        outliers = 0
        changed_time = False
        changed_price = False

        # Reset counters
        self.repair_count = 0
        self.outlier_count = 0
        self.validation_failures = []

        # Make a copy to avoid modifying original
        clean_data = data.copy()

        # Transfer guard from original to copy if it exists
        original_guard = get_guard(data)
        if original_guard:
            # Transfer the guard and update its DataFrame ID to match the copy
            clean_data._sanity_guard = original_guard
            original_guard._df_id = id(clean_data)  # Update to new DataFrame ID
            original_guard._df_hash = hash(str(clean_data.values.tobytes()))
        else:
            # Attach guard if not present
            attach_guard(clean_data)

        # --- 1) canonicalize datetime index (strict raises specific tz/mono/dup errors) ---
        from .datetime import canonicalize_datetime_index
        clean_data = canonicalize_datetime_index(clean_data, self.profile_config)

        # --- 2) coerce OHLCV to numeric / check finite (strict raises NONFINITE/INVALID_DTYPE first) ---
        from .clean import coerce_ohlcv_numeric
        clean_data = coerce_ohlcv_numeric(clean_data, self.profile_config)

        # --- 3) OHLC invariants (strict raises, lenient repairs) ---
        from .invariants import assert_ohlc_invariants
        clean_data = assert_ohlc_invariants(clean_data, self.profile_config)

        # --- 4) Volume validation (strict non-negative) ---
        clean_data, volume_repairs, volume_flags = self._validate_volume_data_strict(
            clean_data, symbol
        )
        repairs.extend(volume_repairs)
        flags.extend(volume_flags)

        # --- 5) Outlier detection (only in lenient mode) ---
        if self.profile_config.get("allow_repairs", True):
            (
                clean_data,
                outlier_repairs,
                outlier_count,
            ) = self._detect_and_repair_outliers_strict(clean_data, symbol)
            repairs.extend(outlier_repairs)
            outliers = outlier_count
        else:
            outliers = 0

        # --- 6) Returns calculation (strict bounds) ---
        clean_data, return_repairs, return_flags = self._calculate_returns_strict(
            clean_data, symbol
        )
        repairs.extend(return_repairs)
        flags.extend(return_flags)

        # --- 7) Final validation checks ---
        clean_data, final_repairs, final_flags = self._final_validation_checks_strict(
            clean_data, symbol
        )
        repairs.extend(final_repairs)
        flags.extend(final_flags)

        # Check for lookahead contamination
        lookahead_detected = self._detect_lookahead_contamination(clean_data)
        if lookahead_detected:
            flags.append("lookahead_contamination")
            # Strict fails if configured
            if self.profile_config.get("fail_on_lookahead_flag", False):
                raise DataSanityError(f"{symbol}: Lookahead contamination")

        # --- Only enforce repairs that actually changed data ---
        benign = {"canonicalized_datetime_index", "coerced_ohlcv_numeric"}
        effective_repairs = [r for r in repairs if r not in benign]  # others always count
        if changed_time and "canonicalized_datetime_index" not in effective_repairs:
            effective_repairs.append("canonicalized_datetime_index")
        if changed_price and "coerced_ohlcv_numeric" not in effective_repairs:
            effective_repairs.append("coerced_ohlcv_numeric")

        # In strict mode: fail if any effective repairs occurred
        if self.profile_config.get("fail_if_any_repair", False) and effective_repairs:
            raise DataSanityError(f"{symbol}: Repairs occurred in strict mode: {effective_repairs}")

        # Also fail if repairs are not allowed but repairs were attempted
        if not self.profile_config.get("allow_repairs", True) and repairs:
            raise DataSanityError(
                f"{symbol}: Repairs not allowed but repairs were attempted: {repairs}"
            )

        # Mark as validated - use the guard from the clean_data
        guard = get_guard(clean_data)
        if guard:
            guard.mark_validated(clean_data, symbol)
        else:
            # If no guard exists, create one and mark as validated
            attach_guard(clean_data)
            guard = get_guard(clean_data)
            guard.mark_validated(clean_data, symbol)

        # Create validation result
        validation_time = time.time() - start_time
        result = ValidationResult(
            repairs=repairs,
            flags=flags,
            outliers=outliers,
            rows_in=original_shape[0],
            rows_out=len(clean_data),
            profile=self.profile,
            validation_time=validation_time,
        )

        # Log summary
        self._log_validation_summary(symbol, original_shape, clean_data.shape)

        return clean_data, result

    def _validate_time_series_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate time series with strict profile rules."""
        repairs = []
        flags = []

        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if self.profile_config.get("allow_repairs", True):
                # Try to convert to datetime
                try:
                    data.index = pd.to_datetime(data.index)
                    repairs.append("converted_to_datetime_index")
                except Exception:
                    raise DataSanityError(f"{symbol}: No valid datetime index found") from None
            else:
                raise DataSanityError(f"{symbol}: No valid datetime index found")

        # Check for monotonic index
        if not data.index.is_monotonic_increasing:
            if self.profile_config.get("allow_drop_dupes", True):
                # Remove duplicates and sort
                data = data[~data.index.duplicated(keep="first")].sort_index()
                repairs.append("dropped_duplicates_and_sorted")
            else:
                raise DataSanityError(f"{symbol}: Index is not monotonic")

        # Check for timezone consistency
        if data.index.tz is None:
            if self.profile_config.get("allow_repairs", True):
                data.index = data.index.tz_localize(UTC)
                repairs.append("localized_to_utc")
            else:
                raise DataSanityError(f"{symbol}: Naive timezone not allowed in strict mode")
        elif data.index.tz != UTC:
            if self.profile_config.get("allow_repairs", True):
                try:
                    data.index = data.index.tz_convert(UTC)
                    repairs.append("converted_to_utc")
                except Exception as e:
                    # Handle mixed timezone data by localizing to UTC
                    logger.warning(
                        f"Timezone conversion failed for {symbol}: {e}, localizing to UTC"
                    )
                    data.index = data.index.tz_localize(UTC)
                    repairs.append("localized_mixed_timezone_to_utc")
            else:
                raise DataSanityError(f"{symbol}: Non-UTC timezone not allowed in strict mode")

        return data, repairs, flags

    def _validate_time_series(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and repair time series integrity."""
        ts_config = self.config["time_series"]

        # Ensure we have a proper index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "Date" in data.columns:
                data = data.set_index("Date")
            else:
                self._handle_validation_failure(f"{symbol}: No valid datetime index found")
                return data

        # Check for monotonic timestamps
        if ts_config["require_monotonic"]:
            if not data.index.is_monotonic_increasing:
                if ts_config["allow_duplicates"]:
                    # Sort and keep duplicates
                    data = data.sort_index()
                else:
                    # Remove duplicates and sort
                    data = data[~data.index.duplicated(keep="first")].sort_index()
                    self._log_repair(f"{symbol}: Removed duplicate timestamps")

            # Ensure monotonic after any repairs
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()

        # Check for UTC timezone
        if ts_config["require_utc"]:
            if data.index.tz is None:
                # Assume UTC if no timezone
                data.index = data.index.tz_localize("UTC")
                self._log_repair(f"{symbol}: Localized timestamps to UTC")
            elif data.index.tz != UTC:
                # Convert to UTC
                data.index = data.index.tz_convert("UTC")
                self._log_repair(f"{symbol}: Converted timestamps to UTC")

        # Check for large gaps
        if len(data) > 1:
            gaps = data.index.to_series().diff().dt.days
            large_gaps = gaps > ts_config["max_gap_days"]
            if large_gaps.any():
                gap_count = large_gaps.sum()
                max_gap = gaps.max()
                self._log_repair(
                    f"{symbol}: Found {gap_count} gaps > {ts_config['max_gap_days']} days (max: {max_gap} days)"
                )

        return data

    def _validate_price_data_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate price data with strict profile rules."""
        repairs = []
        flags = []

        price_cols = self._get_price_columns(data)

        for col in price_cols:
            if col not in data.columns:
                continue

            series = data[col]

            # Convert string data types to numeric if needed
            if series.dtype == "object":
                try:
                    series = pd.to_numeric(series, errors="coerce")
                    data[col] = series
                    if self.profile_config.get("allow_repairs", True):
                        repairs.append(f"converted_string_to_numeric_in_{col}")
                    else:
                        raise DataSanityError(
                            f"{symbol}: String data type in {col} not allowed in strict mode"
                        )
                except Exception:
                    if not self.profile_config.get("allow_repairs", True):
                        raise DataSanityError(
                            f"{symbol}: Cannot convert string data to numeric in {col}"
                        ) from None
                    else:
                        # Try to extract numeric values from strings
                        series = pd.to_numeric(
                            series.str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
                        )
                        data[col] = series
                        repairs.append(f"extracted_numeric_from_string_in_{col}")

            # Check for negative prices
            negative_prices = series <= 0
            if negative_prices.any():
                if self.profile_config.get("allow_clip_prices", True):
                    # Clip negative prices to minimum
                    min_price = self.config["price_limits"]["min_price"]
                    data.loc[negative_prices, col] = min_price
                    repairs.append(f"clipped_negative_prices_in_{col}")
                else:
                    raise DataSanityError(f"{symbol}: Negative prices in {col}")

            # Check for extreme prices
            max_price = self.profile_config.get(
                "price_max", self.config["price_limits"]["max_price"]
            )
            extreme_prices = series > max_price
            if extreme_prices.any():
                if self.profile_config.get("allow_clip_prices", True):
                    # Clip extreme prices
                    data.loc[extreme_prices, col] = max_price
                    repairs.append(f"clipped_extreme_prices_in_{col}")
                else:
                    raise DataSanityError(f"{symbol}: Prices > {max_price} in {col}")

            # Check for non-finite values
            non_finite = ~np.isfinite(series)
            if non_finite.any():
                if self.profile_config.get("allow_ffill_nans", True):
                    # Forward fill non-finite values
                    data[col] = series.ffill().bfill()
                    repairs.append(f"forward_filled_non_finite_in_{col}")
                else:
                    raise DataSanityError(f"{symbol}: Non-finite values in {col}")

        return data, repairs, flags

    def _validate_price_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and repair price data."""
        price_config = self.config["price_limits"]

        # Standardize column names
        price_cols = self._get_price_columns(data)

        for col in price_cols:
            if col not in data.columns:
                continue

            series = data[col]

            # Check for non-finite values
            non_finite = ~np.isfinite(series)
            if non_finite.any():
                count = non_finite.sum()
                if self.config["repair_mode"] == "fail":
                    self._handle_validation_failure(f"{symbol}: {count} non-finite values in {col}")
                elif self.config["repair_mode"] == "drop":
                    data = data[~non_finite]
                    self._log_repair(f"{symbol}: Dropped {count} non-finite values from {col}")
                else:
                    # Forward fill non-finite values
                    # Handle edge cases where NaN might be at the beginning or end
                    filled_series = series.ffill().bfill()
                    # If there are still NaN values, fill with a reasonable default
                    if filled_series.isna().any():
                        # Use median of non-NaN values as default
                        median_val = series.dropna().median()
                        if pd.isna(median_val):
                            median_val = 100.0  # Fallback default
                        filled_series = filled_series.fillna(median_val)
                        self._log_repair(
                            f"{symbol}: Used median ({median_val:.2f}) to fill remaining NaN values in {col}"
                        )
                    data[col] = filled_series
                    self._log_repair(f"{symbol}: Forward-filled {count} non-finite values in {col}")

            # Check price bounds
            too_high = series > price_config["max_price"]
            too_low = series < price_config["min_price"]

            if too_high.any() or too_low.any():
                high_count = too_high.sum()
                low_count = too_low.sum()

                if self.config["repair_mode"] == "fail":
                    self._handle_validation_failure(
                        f"{symbol}: {high_count} prices > {price_config['max_price']}, "
                        f"{low_count} prices < {price_config['min_price']} in {col}"
                    )
                elif self.config["repair_mode"] == "drop":
                    invalid_mask = too_high | too_low
                    data = data.loc[~invalid_mask]
                    self._log_repair(
                        f"{symbol}: Dropped {invalid_mask.sum()} invalid prices from {col}"
                    )
                else:
                    # Winsorize extreme values
                    if self.config["repair_mode"] == "winsorize":
                        data[col] = series.clip(
                            lower=price_config["min_price"],
                            upper=price_config["max_price"],
                        )
                        self._log_repair(
                            f"{symbol}: Winsorized {high_count + low_count} extreme prices in {col}"
                        )
                    else:
                        # Default warn mode - clip extreme values
                        data[col] = series.clip(
                            lower=price_config["min_price"],
                            upper=price_config["max_price"],
                        )
                        self._log_repair(
                            f"{symbol}: Clipped {high_count + low_count} extreme prices in {col}"
                        )

        # Final check: ensure all price columns are finite after all repairs
        for col in price_cols:
            if col in data.columns:
                series = data[col]
                non_finite = ~np.isfinite(series)
                if non_finite.any():
                    # Use median of finite values as fallback
                    finite_series = series[np.isfinite(series)]
                    median_val = finite_series.median() if len(finite_series) > 0 else 100.0  # Ultimate fallback

                    data.loc[non_finite, col] = median_val
                    self._log_repair(
                        f"{symbol}: Final cleanup: filled {non_finite.sum()} remaining non-finite values in {col} with median ({median_val:.2f})"
                    )

        return data

    def _validate_ohlc_consistency_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate OHLC consistency with strict profile rules."""
        repairs = []
        flags = []

        if not all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            return data, repairs, flags

        # Check OHLC invariants
        bad_high = (data["High"] < data[["Open", "Close"]].max(axis=1)).sum()
        bad_low = (data["Low"] > data[["Open", "Close"]].min(axis=1)).sum()

        if bad_high > 0 or bad_low > 0:
            if self.profile_config.get("allow_fix_ohlc", True):
                # Fix OHLC inconsistencies
                data = self._repair_ohlc_inconsistencies(data)
                repairs.append(f"fixed_ohlc_inconsistencies:{bad_high + bad_low}")
            else:
                raise DataSanityError(
                    f"{symbol}: OHLC invariant violation (High < max(Open,Close): {bad_high}, Low > min(Open,Close): {bad_low})"
                )

        return data, repairs, flags

    def _validate_ohlc_consistency(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and repair OHLC consistency."""
        ohlc_config = self.config["ohlc_validation"]

        if not ohlc_config["require_ohlc_consistency"]:
            return data

        price_cols = self._get_price_columns(data)
        if len(price_cols) < 4:  # Need OHLC
            return data

        # Check OHLC relationships
        inconsistencies = []

        # High should be >= Open, Close
        if "High" in data.columns and "Open" in data.columns:
            high_open_violations = data["High"] < data["Open"]
            if high_open_violations.any():
                inconsistencies.append(f"High < Open: {high_open_violations.sum()}")

        if "High" in data.columns and "Close" in data.columns:
            high_close_violations = data["High"] < data["Close"]
            if high_close_violations.any():
                inconsistencies.append(f"High < Close: {high_close_violations.sum()}")

        # Low should be <= Open, Close
        if "Low" in data.columns and "Open" in data.columns:
            low_open_violations = data["Low"] > data["Open"]
            if low_open_violations.any():
                inconsistencies.append(f"Low > Open: {low_open_violations.sum()}")

        if "Low" in data.columns and "Close" in data.columns:
            low_close_violations = data["Low"] > data["Close"]
            if low_close_violations.any():
                inconsistencies.append(f"Low > Close: {low_close_violations.sum()}")

        # Check high-low spread
        if "High" in data.columns and "Low" in data.columns and "Close" in data.columns:
            spread = (data["High"] - data["Low"]) / data["Close"]
            excessive_spread = spread > ohlc_config["max_high_low_spread"]
            if excessive_spread.any():
                inconsistencies.append(f"Excessive spread: {excessive_spread.sum()}")

        if inconsistencies:
            if self.config["repair_mode"] == "fail":
                self._handle_validation_failure(
                    f"{symbol}: OHLC inconsistencies: {', '.join(inconsistencies)}"
                )
            else:
                # Repair OHLC inconsistencies
                data = self._repair_ohlc_inconsistencies(data)
                self._log_repair(
                    f"{symbol}: Repaired OHLC inconsistencies: {', '.join(inconsistencies)}"
                )

        return data

    def _validate_volume_data_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate volume data with strict profile rules."""
        repairs = []
        flags = []

        volume_col = self._get_volume_column(data)
        if volume_col not in data.columns:
            return data, repairs, flags

        volume = data[volume_col]

        # Check for negative volume
        negative_volume = volume < 0
        if negative_volume.any():
            if self.profile_config.get("allow_clip_prices", True):
                # Make negative volume positive
                data.loc[negative_volume, volume_col] = volume[negative_volume].abs()
                repairs.append("made_negative_volume_positive")
            else:
                raise DataSanityError(f"{symbol}: Negative volume values")

        # Check for excessive volume
        max_volume = self.config["price_limits"]["max_volume"]
        excessive_volume = volume > max_volume
        if excessive_volume.any():
            if self.profile_config.get("allow_clip_prices", True):
                # Cap excessive volume
                data.loc[excessive_volume, volume_col] = max_volume
                repairs.append("capped_excessive_volume")
            else:
                raise DataSanityError(f"{symbol}: Excessive volume values > {max_volume}")

        # Check for zero volume
        if not self.config["ohlc_validation"]["allow_zero_volume"]:
            zero_volume = volume == 0
            if zero_volume.any():
                if self.profile_config.get("allow_repairs", True):
                    # Replace with median volume
                    median_volume = volume[volume > 0].median()
                    if pd.isna(median_volume):
                        median_volume = 1000000  # Default
                    data.loc[zero_volume, volume_col] = median_volume
                    repairs.append("replaced_zero_volume_with_median")
                else:
                    raise DataSanityError(f"{symbol}: Zero volume values")

        return data, repairs, flags

    def _validate_volume_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and repair volume data."""
        volume_col = "Volume"
        if volume_col not in data.columns:
            return data

        volume = data[volume_col]

        # Check for negative volume
        negative_volume = volume < 0
        if negative_volume.any():
            count = negative_volume.sum()
            if self.config["repair_mode"] == "fail":
                self._handle_validation_failure(f"{symbol}: {count} negative volume values")
            else:
                data[volume_col] = volume.abs()
                self._log_repair(f"{symbol}: Made {count} negative volume values positive")

        # Check for excessive volume
        max_volume = self.config["price_limits"]["max_volume"]
        excessive_volume = volume > max_volume
        if excessive_volume.any():
            count = excessive_volume.sum()
            if self.config["repair_mode"] == "fail":
                self._handle_validation_failure(
                    f"{symbol}: {count} excessive volume values > {max_volume}"
                )
            elif self.config["repair_mode"] == "drop":
                data = data.loc[~excessive_volume]
                self._log_repair(f"{symbol}: Dropped {count} excessive volume values")
            else:
                data[volume_col] = volume.clip(upper=max_volume)
                self._log_repair(f"{symbol}: Capped {count} excessive volume values")

        # Check for zero volume
        if not self.config["ohlc_validation"]["allow_zero_volume"]:
            zero_volume = volume == 0
            if zero_volume.any():
                count = zero_volume.sum()
                if self.config["repair_mode"] == "fail":
                    self._handle_validation_failure(f"{symbol}: {count} zero volume values")
                else:
                    # Replace with median volume
                    median_volume = volume[volume > 0].median()
                    if pd.isna(median_volume):
                        median_volume = 1000000  # Default
                    data.loc[zero_volume, volume_col] = median_volume
                    self._log_repair(f"{symbol}: Replaced {count} zero volume values with median")

        return data

    def _detect_and_repair_outliers_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], int]:
        """Detect and repair outliers with strict profile rules."""
        repairs = []
        outlier_count = 0

        outlier_config = self.config["outlier_detection"]

        if len(data) < outlier_config["min_obs_for_outlier"]:
            return data, repairs, outlier_count

        price_cols = self._get_price_columns(data)

        for col in price_cols:
            if col not in data.columns:
                continue

            series = data[col]

            # Use log prices for outlier detection
            log_prices = np.log(series)

            # Calculate robust statistics
            median = np.median(log_prices)
            mad = np.median(np.abs(log_prices - median))

            if mad == 0:
                continue

            # Calculate z-scores using MAD
            z_scores = 0.6745 * (log_prices - median) / mad

            # Detect outliers
            outliers = np.abs(z_scores) > outlier_config["z_score_threshold"]

            if outliers.any():
                outlier_count += outliers.sum()

                if self.profile_config.get("allow_winsorize", True):
                    # Winsorize outliers
                    q_low = self.config["winsorize_quantile"]
                    q_high = 1 - q_low
                    lower_bound = series.quantile(q_low)
                    upper_bound = series.quantile(q_high)
                    data[col] = series.clip(lower=lower_bound, upper=upper_bound)
                    repairs.append(f"winsorized_outliers_in_{col}")
                else:
                    raise DataSanityError(f"{symbol}: {outliers.sum()} outliers detected in {col}")

        return data, repairs, outlier_count

    def _detect_and_repair_outliers(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Detect and repair outliers using robust statistics."""
        outlier_config = self.config["outlier_detection"]

        if len(data) < outlier_config["min_obs_for_outlier"]:
            return data

        price_cols = self._get_price_columns(data)
        outliers_found = 0

        for col in price_cols:
            if col not in data.columns:
                continue

            series = data[col]

            # Use log prices for outlier detection
            log_prices = np.log(series)

            # Calculate robust statistics
            median = np.median(log_prices)
            mad = np.median(np.abs(log_prices - median))

            if mad == 0:
                continue

            # Calculate z-scores using MAD
            z_scores = 0.6745 * (log_prices - median) / mad

            # Detect outliers
            outliers = np.abs(z_scores) > outlier_config["z_score_threshold"]

            if outliers.any():
                outlier_count = outliers.sum()
                outliers_found += outlier_count

                if self.config["repair_mode"] == "fail":
                    self._handle_validation_failure(
                        f"{symbol}: {outlier_count} outliers detected in {col}"
                    )
                elif self.config["repair_mode"] == "drop":
                    data = data.loc[~outliers]
                    self._log_repair(f"{symbol}: Dropped {outlier_count} outliers from {col}")
                elif self.config["repair_mode"] == "winsorize":
                    # Winsorize outliers
                    q_low = self.config["winsorize_quantile"]
                    q_high = 1 - q_low
                    lower_bound = series.quantile(q_low)
                    upper_bound = series.quantile(q_high)
                    data[col] = series.clip(lower=lower_bound, upper=upper_bound)
                    self._log_repair(f"{symbol}: Winsorized {outlier_count} outliers in {col}")

        self.outlier_count = outliers_found
        return data

    def _calculate_returns_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Calculate returns with strict profile rules."""
        repairs = []
        flags = []

        returns_config = self.config["returns"]

        if "Close" not in data.columns:
            return data, repairs, flags

        close_prices = data["Close"]

        if len(close_prices) < returns_config["min_periods"]:
            return data, repairs, flags

        # Calculate log returns
        returns = np.log(close_prices / close_prices.shift(1)) if returns_config["method"] == "log_close_to_close" else close_prices.pct_change()

        # Handle missing values
        returns = returns.ffill().bfill() if returns_config["fill_method"] == "forward" else returns.fillna(0)

        # Check for extreme returns
        max_return = self.config["price_limits"]["max_daily_return"]
        extreme_returns = np.abs(returns) > max_return

        if extreme_returns.any():
            if self.profile_config.get("allow_winsorize", True):
                # Winsorize extreme returns
                returns = returns.clip(lower=-max_return, upper=max_return)
                repairs.append("winsorized_extreme_returns")
            else:
                raise DataSanityError(
                    f"{symbol}: {extreme_returns.sum()} extreme returns > {max_return}"
                )

        # Add returns to data
        data["Returns"] = returns

        return data, repairs, flags

    def _calculate_returns(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate clean returns from close prices."""
        returns_config = self.config["returns"]

        if "Close" not in data.columns:
            return data

        close_prices = data["Close"]

        if len(close_prices) < returns_config["min_periods"]:
            return data

        # Calculate log returns
        returns = np.log(close_prices / close_prices.shift(1)) if returns_config["method"] == "log_close_to_close" else close_prices.pct_change()

        # Handle missing values
        returns = returns.ffill().bfill() if returns_config["fill_method"] == "forward" else returns.fillna(0)

        # Check for extreme returns
        max_return = self.config["price_limits"]["max_daily_return"]
        extreme_returns = np.abs(returns) > max_return

        if extreme_returns.any():
            count = extreme_returns.sum()
            if self.config["repair_mode"] == "fail":
                self._handle_validation_failure(f"{symbol}: {count} extreme returns > {max_return}")
            elif self.config["repair_mode"] == "drop":
                data = data.loc[~extreme_returns]
                self._log_repair(f"{symbol}: Dropped {count} extreme returns")
            else:
                # Winsorize extreme returns
                returns = returns.clip(lower=-max_return, upper=max_return)
                self._log_repair(f"{symbol}: Winsorized {count} extreme returns")

        # Add returns to data
        data["Returns"] = returns

        return data

    def _final_validation_checks_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Perform final validation checks with strict profile rules."""
        repairs = []
        flags = []

        # 1. Ensure minimum data requirements
        if len(data) < 1:
            raise DataSanityError(f"{symbol}: Insufficient data (need >= 1 row, got {len(data)})")

        # 2. Verify column schema
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataSanityError(f"{symbol}: Missing required columns: {missing_cols}")

        # 3. Verify data types
        expected_dtypes = {
            "Open": np.floating,
            "High": np.floating,
            "Low": np.floating,
            "Close": np.floating,
            "Volume": (np.floating, np.integer),  # Allow both float and int for volume
        }

        for col, expected_type in expected_dtypes.items():
            if col in data.columns:
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not any(np.issubdtype(data[col].dtype, t) for t in expected_type):
                        raise DataSanityError(
                            f"{symbol}: Column {col} has wrong dtype: {data[col].dtype}"
                        )
                else:
                    # Single allowed type
                    if not np.issubdtype(data[col].dtype, expected_type):
                        raise DataSanityError(
                            f"{symbol}: Column {col} has wrong dtype: {data[col].dtype}"
                        )

        # 4. Final finite check
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in data.columns:
                non_finite = ~np.isfinite(data[col])
                if non_finite.any():
                    raise DataSanityError(
                        f"{symbol}: {non_finite.sum()} non-finite values in {col} after validation"
                    )

        # 5. Verify index integrity
        if not data.index.is_monotonic_increasing:
            raise DataSanityError(f"{symbol}: Index is not monotonic after validation")

        if data.index.has_duplicates:
            raise DataSanityError(f"{symbol}: Index has duplicates after validation")

        # 6. Check for corporate actions consistency (if Adj Close present)
        if "Adj Close" in data.columns and "Close" in data.columns:
            adj_returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
            close_returns = np.log(data["Close"] / data["Close"].shift(1))

            # Adj returns should be similar to close returns (within tolerance)
            diff = np.abs(adj_returns - close_returns)
            large_diff = diff > 0.1  # 10% tolerance

            if large_diff.any():
                repairs.append(f"large_adj_close_differences:{large_diff.sum()}")

        return data, repairs, flags

    def _detect_lookahead_contamination(self, data: pd.DataFrame) -> bool:
        """Detect potential lookahead contamination."""
        if "Returns" in data.columns and len(data) > 1:
            # Detect simple future leakage: value equals next-step value at t
            r = data["Returns"].to_numpy()
            # Compare r[t] vs r[t+1]
            eq_next = np.isfinite(r[:-1]) & np.isfinite(r[1:]) & (np.abs(r[:-1] - r[1:]) < 1e-12)
            if eq_next.any():
                return True
        return False

    def _final_validation_checks(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Perform final validation checks with strict invariants."""

        # 1. Ensure minimum data requirements
        if len(data) < 1:
            raise DataSanityError(f"{symbol}: Insufficient data (need >= 1 row, got {len(data)})")

        # 2. Check for lookahead contamination (basic check)
        if "Returns" in data.columns:
            # Returns should not have future information
            future_returns = data["Returns"].shift(-1).notna()
            if future_returns.any():
                self._log_repair(f"{symbol}: Potential lookahead contamination detected")

        # 3. Verify column schema
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataSanityError(f"{symbol}: Missing required columns: {missing_cols}")

        # 4. Verify data types
        expected_dtypes = {
            "Open": np.floating,
            "High": np.floating,
            "Low": np.floating,
            "Close": np.floating,
            "Volume": (np.floating, np.integer),  # Allow both float and int for volume
        }

        for col, expected_type in expected_dtypes.items():
            if col in data.columns:
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not any(np.issubdtype(data[col].dtype, t) for t in expected_type):
                        raise DataSanityError(
                            f"{symbol}: Column {col} has wrong dtype: {data[col].dtype}"
                        )
                else:
                    # Single allowed type
                    if not np.issubdtype(data[col].dtype, expected_type):
                        raise DataSanityError(
                            f"{symbol}: Column {col} has wrong dtype: {data[col].dtype}"
                        )

        # 5. Final finite check
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in data.columns:
                non_finite = ~np.isfinite(data[col])
                if non_finite.any():
                    raise DataSanityError(
                        f"{symbol}: {non_finite.sum()} non-finite values in {col} after validation"
                    )

        # 6. Verify index integrity
        if not data.index.is_monotonic_increasing:
            raise DataSanityError(f"{symbol}: Index is not monotonic after validation")

        if data.index.has_duplicates:
            raise DataSanityError(f"{symbol}: Index has duplicates after validation")

        # 7. Check for corporate actions consistency (if Adj Close present)
        if "Adj Close" in data.columns and "Close" in data.columns:
            adj_returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
            close_returns = np.log(data["Close"] / data["Close"].shift(1))

            # Adj returns should be similar to close returns (within tolerance)
            diff = np.abs(adj_returns - close_returns)
            large_diff = diff > 0.1  # 10% tolerance

            if large_diff.any():
                self._log_repair(
                    f"{symbol}: {large_diff.sum()} large differences between Adj Close and Close returns"
                )

        logger.info(f"{symbol}: Final validation checks passed")
        return data

    def _get_price_columns(self, data: pd.DataFrame) -> list[str]:
        """Get standardized price column names."""
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns - keep just the first level (OHLCV names)
            data.columns = [col[0] for col in data.columns]

        # Map common column name variations
        column_mapping = {
            "open": "Open",
            "OPEN": "Open",
            "Open": "Open",
            "high": "High",
            "HIGH": "High",
            "High": "High",
            "low": "Low",
            "LOW": "Low",
            "Low": "Low",
            "close": "Close",
            "CLOSE": "Close",
            "Close": "Close",
            "volume": "Volume",
            "VOLUME": "Volume",
            "Volume": "Volume",
        }

        # Standardize column names
        data.columns = [column_mapping.get(str(col).lower(), col) for col in data.columns]

        # Return price columns
        price_cols = []
        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                price_cols.append(col)

        return price_cols

    def _get_volume_column(self, data: pd.DataFrame) -> str:
        """Get standardized volume column name."""
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns - keep just the first level (OHLCV names)
            data.columns = [col[0] for col in data.columns]

        # Map common column name variations
        column_mapping = {"volume": "Volume", "VOLUME": "Volume", "Volume": "Volume"}

        # Standardize column names
        data.columns = [column_mapping.get(str(col).lower(), col) for col in data.columns]

        # Return volume column
        if "Volume" in data.columns:
            return "Volume"
        else:
            return "Volume"  # Default fallback

    def _repair_ohlc_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair OHLC inconsistencies."""
        # Ensure High >= max(Open, Close)
        if all(col in data.columns for col in ["High", "Open", "Close"]):
            data["High"] = data[["High", "Open", "Close"]].max(axis=1)

        # Ensure Low <= min(Open, Close)
        if all(col in data.columns for col in ["Low", "Open", "Close"]):
            data["Low"] = data[["Low", "Open", "Close"]].min(axis=1)

        # Additional check: ensure High >= Low
        if all(col in data.columns for col in ["High", "Low"]):
            # If High < Low, set High = Low + small amount
            high_low_violations = data["High"] < data["Low"]
            if high_low_violations.any():
                data.loc[high_low_violations, "High"] = data.loc[high_low_violations, "Low"] + 0.01

        return data

    def _handle_validation_failure(self, message: str):
        """Handle validation failure based on repair mode."""
        self.validation_failures.append(message)

        if self.config["repair_mode"] == "fail":
            raise DataSanityError(message)
        else:
            logger.warning(f"Validation failure: {message}")

    def _log_repair(self, message: str):
        """Log a repair action."""
        self.repair_count += 1
        if self.config["logging"]["log_repairs"]:
            logger.info(f"Data repair: {message}")

    def _log_validation_summary(self, symbol: str, original_shape: tuple, final_shape: tuple):
        """Log validation summary."""
        summary_msg = (
            f"{symbol} validation complete: "
            f"{original_shape[0]} → {final_shape[0]} rows, "
            f"{self.repair_count} repairs, "
            f"{self.outlier_count} outliers"
        )

        if self.validation_failures:
            summary_msg += f", {len(self.validation_failures)} failures"

        log_level = getattr(logging, self.config["logging"]["summary_level"])
        logger.log(log_level, summary_msg)

    def coerce_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert string data types to numeric for OHLCV columns.

        Args:
            data: DataFrame with potentially string data

        Returns:
            DataFrame with numeric data types
        """
        result = data.copy()

        # Define columns to convert
        price_cols = ["Open", "High", "Low", "Close"]
        volume_cols = ["Volume"]

        # Convert price columns
        for col in price_cols:
            if col in result.columns and result[col].dtype == "object":
                try:
                    # Remove currency symbols and whitespace
                    cleaned_series = (
                        result[col].astype(str).str.replace(r"[\$€£¥₹]", "", regex=True)
                    )
                    cleaned_series = cleaned_series.str.strip()

                    # Handle European decimal notation (comma as decimal separator)
                    # Convert 100,02 to 100.02
                    cleaned_series = cleaned_series.str.replace(
                        r"(\d+),(\d+)", r"\1.\2", regex=True
                    )

                    # Remove remaining commas (thousands separators)
                    cleaned_series = cleaned_series.str.replace(r",", "", regex=False)

                    # Handle scientific notation and currency suffixes
                    cleaned_series = cleaned_series.str.replace(
                        r"\s*[A-Z]{3}$", "", regex=True
                    )  # Remove USD, EUR, etc.

                    # Convert to numeric
                    result[col] = pd.to_numeric(cleaned_series, errors="coerce")

                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}")

        # Convert volume columns
        for col in volume_cols:
            if col in result.columns and result[col].dtype == "object":
                try:
                    # Remove thousands separators and whitespace
                    cleaned_series = result[col].astype(str).str.replace(r",", "", regex=False)
                    cleaned_series = cleaned_series.str.strip()

                    # Convert to numeric
                    result[col] = pd.to_numeric(cleaned_series, errors="coerce")

                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}")

        return result

    def compute_returns(self, close_prices: pd.Series) -> pd.Series:
        """
        Compute returns from close prices.

        Args:
            close_prices: Series of close prices

        Returns:
            Series of returns (first value is 0)
        """
        if len(close_prices) == 0:
            return pd.Series(dtype=float)

        # Calculate percentage change
        returns = close_prices.pct_change()

        # Fill NaN with 0 (first observation)
        returns = returns.fillna(0.0)

        return returns


class DataSanityWrapper:
    """
    Wrapper that applies data sanity validation to all data sources.
    """

    def __init__(self, config_path: str = "config/data_sanity.yaml", profile: str = "default"):
        """Initialize DataSanityWrapper."""
        self.validator = DataSanityValidator(config_path, profile)
        self.profile = profile
        logger.info(f"Initialized DataSanityWrapper with profile: {profile}")

    def load_and_validate(self, filepath: str, symbol: str = None) -> pd.DataFrame:
        """
        Load data from file and apply validation.

        Args:
            filepath: Path to data file
            symbol: Symbol name (inferred from filename if None)

        Returns:
            Validated DataFrame
        """
        if symbol is None:
            symbol = Path(filepath).stem

        # Load data based on file extension
        if filepath.endswith(".pkl"):
            data = pd.read_pickle(filepath)
        elif filepath.endswith(".csv"):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif filepath.endswith(".parquet"):
            data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # Apply validation
        clean_data, result = self.validator.validate_and_repair(data, symbol)
        return clean_data

    def validate_dataframe(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN", **kwargs
    ) -> pd.DataFrame:
        """
        Validate an existing DataFrame with back-compat support.

        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging (backward compatibility)
            **kwargs: Additional arguments (profile, etc.)

        Returns:
            Validated DataFrame
        """
        profile = kwargs.get("profile", "default")

        # Create validator with specified profile
        validator = DataSanityValidator(profile=profile)
        clean_data, result = validator.validate_and_repair(data, symbol)

        # Handle mode-based behavior
        mode = validator.profile_config.get("mode", "warn")
        if mode == "error" and (result.repairs or result.flags):
            raise DataSanityError(
                f"{symbol}: Validation failed with mode='error': repairs={result.repairs}, flags={result.flags}"
            )

        return clean_data

    def validate(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Alias for validate_dataframe for compatibility.

        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging

        Returns:
            Validated DataFrame
        """
        return self.validate_dataframe(data, symbol=symbol)

    def get_validation_stats(self) -> dict:
        """Get validation statistics."""
        return {
            "repair_count": self.validator.repair_count,
            "outlier_count": self.validator.outlier_count,
            "validation_failures": self.validator.validation_failures,
        }


# Global instance for easy access
_data_sanity_wrapper = None


def get_data_sanity_wrapper(
    config_path: str = "config/data_sanity.yaml", profile: str = "default"
) -> DataSanityWrapper:
    """Get global DataSanityWrapper instance."""
    global _data_sanity_wrapper
    if _data_sanity_wrapper is None:
        _data_sanity_wrapper = DataSanityWrapper(config_path, profile)
    return _data_sanity_wrapper


def validate_market_data(data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """
    Convenience function to validate market data.

    Args:
        data: DataFrame with market data
        symbol: Symbol name for logging

    Returns:
        Validated DataFrame
    """
    wrapper = get_data_sanity_wrapper()
    return wrapper.validate_dataframe(data, symbol)
