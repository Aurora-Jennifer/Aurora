"""
Metamorphic tests for DataSanity validation.
Test invariance properties and transformations.
"""

import numpy as np
import pandas as pd
import pytest

from core.data_sanity import DataSanityError, DataSanityValidator
from core.meta.invariance import additive_invariance_score, invariance_score
from tests.factories import base_df
from tests.helpers.assertions import assert_data_integrity, assert_verdict

# Transformations that should preserve validation results
TRANSFORMS = [
    lambda d: d.assign(dummy=1),  # Add dummy column
    lambda d: d.reindex(d.index),  # Reindex same order
    lambda d: d.copy(),  # Deep copy
]


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_clean_invariance(transform):
    """Test that clean data remains valid after transformations."""
    validator = DataSanityValidator(profile="strict")
    original_df = base_df()

    # Original should pass
    original_clean, original_result = assert_verdict(validator, original_df, "PASS", "ORIGINAL")

    # Transformed should also pass
    transformed_df = transform(original_df)
    transformed_clean, transformed_result = assert_verdict(
        validator, transformed_df, "PASS", "TRANSFORMED"
    )

    # Data integrity should be preserved
    if original_clean is not None and transformed_clean is not None:
        assert_data_integrity(transformed_clean, transformed_df)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_failure_invariance(transform):
    """Test that failure cases remain invalid after transformations."""
    validator = DataSanityValidator(profile="strict")

    # Create a failure case (negative prices)
    original_df = base_df()
    original_df.loc[original_df.index[50], "Close"] = -50.0

    # Original should fail
    with pytest.raises(DataSanityError):
        validator.validate_and_repair(original_df, "ORIGINAL")

    # Transformed should also fail
    transformed_df = transform(original_df)
    with pytest.raises(DataSanityError):
        validator.validate_and_repair(transformed_df, "TRANSFORMED")


def test_scale_invariance():
    """Test that validation is invariant to price scaling."""
    validator = DataSanityValidator(profile="strict")
    base_data = base_df()

    # Test different price scales
    scales = [0.1, 1.0, 10.0, 100.0, 1000.0]

    for scale in scales:
        scaled_data = base_data.copy()
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            scaled_data[col] = scaled_data[col] * scale

        # All should pass
        clean_data, result = assert_verdict(validator, scaled_data, "PASS", f"SCALE_{scale}")

        if clean_data is not None and "Returns" in clean_data.columns:
            # Returns should be the same (scale invariant)
            original_returns = base_data["Close"].pct_change(fill_method=None).fillna(0.0)
            scaled_returns = clean_data["Returns"]
            # Use invariance_score for truly scale-invariant comparison
            score = invariance_score(original_returns.values, scaled_returns.values)
            assert score >= 0.999, f"Scale invariance failed: score={score}"


def test_time_invariance():
    """Test that validation is invariant to time shifts."""
    validator = DataSanityValidator(profile="strict")
    base_data = base_df()

    # Test different time shifts
    shifts = ["1D", "1W", "30D", "90D", "365D"]

    for shift in shifts:
        shifted_data = base_data.copy()
        shifted_data.index = shifted_data.index + pd.Timedelta(shift)

        # All should pass
        clean_data, result = assert_verdict(validator, shifted_data, "PASS", f"SHIFT_{shift}")

        if clean_data is not None:
            # Data should be identical except for timestamps
            assert len(clean_data) == len(base_data), "Row count should be preserved"
            assert "Returns" in clean_data.columns, "Returns should be added"


def test_volume_scale_invariance():
    """Test that validation is invariant to volume scaling."""
    validator = DataSanityValidator(profile="strict")
    base_data = base_df()

    # Test different volume scales
    scales = [0.1, 1.0, 10.0, 100.0, 1000.0]

    for scale in scales:
        scaled_data = base_data.copy()
        scaled_data["Volume"] = scaled_data["Volume"] * scale

        # All should pass
        clean_data, result = assert_verdict(validator, scaled_data, "PASS", f"VOL_SCALE_{scale}")

        if clean_data is not None:
            # Price data should be unchanged
            price_cols = ["Open", "High", "Low", "Close"]
            for col in price_cols:
                np.testing.assert_allclose(
                    clean_data[col].values,
                    base_data[col].values,
                    rtol=1e-10,
                    err_msg=f"Price data changed for volume scale {scale}",
                )


def test_ohlc_permutation_invariance():
    """Test that validation is invariant to OHLC column order."""
    validator = DataSanityValidator(profile="strict")
    base_data = base_df()

    # Test different column orders
    column_orders = [
        ["Open", "High", "Low", "Close", "Volume"],
        ["Volume", "Open", "High", "Low", "Close"],
        ["Close", "Open", "High", "Low", "Volume"],
        ["High", "Low", "Open", "Close", "Volume"],
    ]

    for order in column_orders:
        reordered_data = base_data[order]

        # All should pass
        clean_data, result = assert_verdict(
            validator, reordered_data, "PASS", f"ORDER_{'_'.join(order)}"
        )

        if clean_data is not None:
            # Data should be identical
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                np.testing.assert_allclose(
                    clean_data[col].values,
                    base_data[col].values,
                    rtol=1e-10,
                    err_msg=f"Data changed for column order {order}",
                )


def test_additive_invariance():
    """Test that adding constant values preserves validation."""
    validator = DataSanityValidator(profile="strict")
    base_data = base_df()

    # Test adding constants to prices
    constants = [0.01, 1.0, 10.0, 100.0]

    for const in constants:
        shifted_data = base_data.copy()
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            shifted_data[col] = shifted_data[col] + const

        # All should pass
        clean_data, result = assert_verdict(validator, shifted_data, "PASS", f"ADD_{const}")

        if clean_data is not None:
            # Use additive_invariance_score for mathematically correct additive invariance
            # This compares z-scored first differences of price columns, which are truly invariant to additive shifts
            score = additive_invariance_score(base_data, shifted_data)
            assert score >= 0.999, f"Additive invariance failed: score={score}"


def test_monotonic_transformations():
    """Test that monotonic transformations preserve validation."""
    validator = DataSanityValidator(profile="strict")
    base_data = base_df()

    # Test monotonic transformations
    transforms = [
        lambda x: x,  # Identity
        lambda x: x**2,  # Square
        lambda x: np.sqrt(x),  # Square root
        lambda x: np.log(x),  # Log
        lambda x: np.exp(x / 100),  # Scaled exponential
    ]

    for i, transform in enumerate(transforms):
        transformed_data = base_data.copy()
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            transformed_data[col] = transform(transformed_data[col])

        # All should pass (monotonic transformations preserve order)
        clean_data, result = assert_verdict(validator, transformed_data, "PASS", f"MONO_{i}")

        if clean_data is not None:
            # OHLC relationships should be preserved
            assert (clean_data["High"] >= clean_data["Low"]).all(), (
                "High >= Low should be preserved"
            )
            assert (clean_data["High"] >= clean_data["Open"]).all(), (
                "High >= Open should be preserved"
            )
            assert (clean_data["High"] >= clean_data["Close"]).all(), (
                "High >= Close should be preserved"
            )
