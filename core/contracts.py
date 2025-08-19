"""
Data contracts and runtime validation for trading system components.

This module provides runtime checks to ensure data integrity throughout
the trading pipeline.
"""

import logging
from functools import wraps

import numpy as np
import pandas as pd

from .data_sanity import DataSanityError, assert_validated

logger = logging.getLogger(__name__)


class DataFrameContract:
    """Contract for validated market data DataFrames."""

    @staticmethod
    def validate_market_data(df: pd.DataFrame, context: str = "unknown") -> bool:
        """
        Validate that a DataFrame meets market data requirements.

        Args:
            df: DataFrame to validate
            context: Context for error messages

        Returns:
            True if valid

        Raises:
            DataSanityError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise DataSanityError(f"{context}: Expected DataFrame, got {type(df)}")

        if df.empty:
            raise DataSanityError(f"{context}: DataFrame is empty")

        # Check for required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataSanityError(f"{context}: Missing required columns: {missing_cols}")

        # Check data types
        expected_dtypes = {
            "Open": np.floating,
            "High": np.floating,
            "Low": np.floating,
            "Close": np.floating,
            "Volume": (np.floating, np.integer),  # Allow both float and int for volume
        }

        for col, expected_type in expected_dtypes.items():
            if isinstance(expected_type, tuple):
                # Multiple allowed types
                if not any(np.issubdtype(df[col].dtype, t) for t in expected_type):
                    raise DataSanityError(
                        f"{context}: Column {col} has wrong dtype: {df[col].dtype}"
                    )
            else:
                # Single allowed type
                if not np.issubdtype(df[col].dtype, expected_type):
                    raise DataSanityError(
                        f"{context}: Column {col} has wrong dtype: {df[col].dtype}"
                    )

        # Check for finite values
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if not np.isfinite(df[col]).all():
                raise DataSanityError(f"{context}: Non-finite values in {col}")

        # Check index integrity
        if not df.index.is_monotonic_increasing:
            raise DataSanityError(f"{context}: Index is not monotonic")

        if df.index.has_duplicates:
            raise DataSanityError(f"{context}: Index has duplicates")

        # Assert that DataFrame has been validated by DataSanity
        assert_validated(df, context)

        return True


class SignalContract:
    """Contract for trading signals."""

    @staticmethod
    def validate_signals(signals: pd.Series, context: str = "unknown") -> bool:
        """
        Validate that a signal series meets requirements.

        Args:
            signals: Signal series to validate
            context: Context for error messages

        Returns:
            True if valid

        Raises:
            DataSanityError: If validation fails
        """
        if not isinstance(signals, pd.Series):
            raise DataSanityError(f"{context}: Expected Series, got {type(signals)}")

        if signals.empty:
            raise DataSanityError(f"{context}: Signal series is empty")

        # Check for finite values
        if not np.isfinite(signals).all():
            raise DataSanityError(f"{context}: Non-finite values in signals")

        # Check signal bounds (allow configurable bounds)
        signal_min, signal_max = -1.0, 1.0  # Default bounds
        out_of_bounds = (signals < signal_min) | (signals > signal_max)

        if out_of_bounds.any():
            raise DataSanityError(
                f"{context}: {out_of_bounds.sum()} signals outside bounds [{signal_min}, {signal_max}]"
            )

        return True

    @staticmethod
    def validate_confidence(confidence: pd.Series, context: str = "unknown") -> bool:
        """
        Validate confidence scores.

        Args:
            confidence: Confidence series to validate
            context: Context for error messages

        Returns:
            True if valid

        Raises:
            DataSanityError: If validation fails
        """
        if not isinstance(confidence, pd.Series):
            raise DataSanityError(f"{context}: Expected Series, got {type(confidence)}")

        if confidence.empty:
            raise DataSanityError(f"{context}: Confidence series is empty")

        # Check for finite values
        if not np.isfinite(confidence).all():
            raise DataSanityError(f"{context}: Non-finite values in confidence")

        # Check confidence bounds [0, 1]
        out_of_bounds = (confidence < 0) | (confidence > 1)
        if out_of_bounds.any():
            raise DataSanityError(
                f"{context}: {out_of_bounds.sum()} confidence values outside [0, 1]"
            )

        return True


def require_validated_data(context: str = "unknown"):
    """
    Decorator to require validated data for functions.

    Args:
        context: Context for error messages
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check first DataFrame argument
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    DataFrameContract.validate_market_data(arg, f"{func.__name__}:{context}")
                    break

            # Check DataFrame keyword arguments
            for _key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    DataFrameContract.validate_market_data(value, f"{func.__name__}:{context}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_validated_signals(context: str = "unknown"):
    """
    Decorator to require validated signals for functions.

    Args:
        context: Context for error messages
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check first Series argument
            for arg in args:
                if isinstance(arg, pd.Series):
                    SignalContract.validate_signals(arg, f"{func.__name__}:{context}")
                    break

            # Check Series keyword arguments
            for _key, value in kwargs.items():
                if isinstance(value, pd.Series):
                    SignalContract.validate_signals(value, f"{func.__name__}:{context}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


class FeatureContract:
    """Contract for feature DataFrames."""

    @staticmethod
    def validate_features(features: pd.DataFrame, context: str = "unknown") -> bool:
        """
        Validate that a feature DataFrame meets requirements.

        Args:
            features: Feature DataFrame to validate
            context: Context for error messages

        Returns:
            True if valid

        Raises:
            DataSanityError: If validation fails
        """
        if not isinstance(features, pd.DataFrame):
            raise DataSanityError(f"{context}: Expected DataFrame, got {type(features)}")

        if features.empty:
            raise DataSanityError(f"{context}: Feature DataFrame is empty")

        # Check for finite values
        non_finite = ~np.isfinite(features.select_dtypes(include=[np.number]))
        if non_finite.any().any():
            non_finite_cols = non_finite.any()[non_finite.any()].index.tolist()
            raise DataSanityError(f"{context}: Non-finite values in columns: {non_finite_cols}")

        # Check index integrity
        if not features.index.is_monotonic_increasing:
            raise DataSanityError(f"{context}: Feature index is not monotonic")

        if features.index.has_duplicates:
            raise DataSanityError(f"{context}: Feature index has duplicates")

        return True


def validate_feature_input(func):
    """Decorator to validate feature input and output."""

    @wraps(func)
    def wrapper(data: pd.DataFrame, *args, **kwargs):
        # Validate input data
        DataFrameContract.validate_market_data(data, f"{func.__name__}:input")

        # Call function
        result = func(data, *args, **kwargs)

        # Validate output features
        if isinstance(result, pd.DataFrame):
            FeatureContract.validate_features(result, f"{func.__name__}:output")

        return result

    return wrapper


def validate_strategy_output(func):
    """Decorator to validate strategy output."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Call function
        result = func(*args, **kwargs)

        # Validate output signals
        if isinstance(result, pd.Series):
            SignalContract.validate_signals(result, f"{func.__name__}:output")
        elif isinstance(result, dict) and "signals" in result:
            SignalContract.validate_signals(result["signals"], f"{func.__name__}:signals")
            if "confidence" in result:
                SignalContract.validate_confidence(
                    result["confidence"], f"{func.__name__}:confidence"
                )

        return result

    return wrapper
