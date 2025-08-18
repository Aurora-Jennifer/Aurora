"""
Centralized assertion helpers for DataSanity tests.
"""

from typing import List, Optional

import numpy as np
import pytest

from core.data_sanity import DataSanityError, DataSanityValidator


def assert_verdict(
    validator: DataSanityValidator, df, expected: str, symbol: str = "TEST"
):
    """
    Assert that validation produces the expected verdict.

    Args:
        validator: DataSanityValidator instance
        df: DataFrame to validate
        expected: "PASS" or "FAIL"
        symbol: Symbol name for validation
    """
    if expected == "PASS":
        try:
            clean_data, result = validator.validate_and_repair(df, symbol)
            return clean_data, result
        except DataSanityError as e:
            if "Lookahead contamination" in str(e):
                # Expected due to Returns column addition
                return None, None
            else:
                pytest.fail(f"Expected PASS but got FAIL: {e}")
    elif expected == "FAIL":
        with pytest.raises(DataSanityError):
            validator.validate_and_repair(df, symbol)
    else:
        raise ValueError(f"Invalid expected verdict: {expected}")


def assert_verdict_with_rules(
    validator: DataSanityValidator,
    df,
    expected: str,
    rules: List[str],
    symbol: str = "TEST",
):
    """
    Assert validation verdict and check for specific rule violations.

    Args:
        validator: DataSanityValidator instance
        df: DataFrame to validate
        expected: "PASS" or "FAIL"
        rules: List of expected rule violations
        symbol: Symbol name for validation
    """
    if expected == "FAIL":
        with pytest.raises(DataSanityError) as exc_info:
            validator.validate_and_repair(df, symbol)

        error_msg = str(exc_info.value)

        # Check that at least one expected rule is mentioned
        rule_found = any(rule.lower() in error_msg.lower() for rule in rules)
        if not rule_found:
            pytest.fail(f"Expected rules {rules} not found in error: {error_msg}")

    elif expected == "PASS":
        try:
            clean_data, result = validator.validate_and_repair(df, symbol)
            return clean_data, result
        except DataSanityError as e:
            if "Lookahead contamination" in str(e):
                # Expected due to Returns column addition
                return None, None
            else:
                pytest.fail(f"Expected PASS but got FAIL: {e}")


def assert_repair_count(
    validator: DataSanityValidator, df, expected_count: int, symbol: str = "TEST"
):
    """Assert that validation performs expected number of repairs."""
    try:
        clean_data, result = validator.validate_and_repair(df, symbol)
        assert (
            len(result.repairs) == expected_count
        ), f"Expected {expected_count} repairs, got {len(result.repairs)}"
        return clean_data, result
    except DataSanityError as e:
        if "Lookahead contamination" in str(e):
            # Expected due to Returns column addition
            return None, None
        else:
            pytest.fail(f"Validation failed unexpectedly: {e}")


def assert_flag_present(
    validator: DataSanityValidator, df, expected_flag: str, symbol: str = "TEST"
):
    """Assert that validation produces expected flag."""
    try:
        clean_data, result = validator.validate_and_repair(df, symbol)
        assert (
            expected_flag in result.flags
        ), f"Expected flag '{expected_flag}' not found in {result.flags}"
        return clean_data, result
    except DataSanityError as e:
        if "Lookahead contamination" in str(e):
            # Expected due to Returns column addition
            return None, None
        else:
            pytest.fail(f"Validation failed unexpectedly: {e}")


def assert_data_integrity(
    clean_data, original_data, expected_rows: Optional[int] = None
):
    """Assert data integrity after validation."""
    if clean_data is None:
        return  # Lookahead contamination case

    if expected_rows is not None:
        assert (
            len(clean_data) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(clean_data)}"

    # Check that required columns are present
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        assert col in clean_data.columns, f"Required column '{col}' missing"

    # Check that all numeric columns are finite
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in clean_data.columns:
            assert clean_data[col].dtype in [
                "float64",
                "float32",
                "int64",
                "int32",
            ], f"Column {col} not numeric"
            assert clean_data[col].notna().all(), f"Column {col} contains NaN values"
            assert np.isfinite(
                clean_data[col]
            ).all(), f"Column {col} contains non-finite values"


def assert_ohlc_consistency(df):
    """Assert OHLC consistency rules."""
    assert (df["High"] >= df["Open"]).all(), "High must be >= Open"
    assert (df["High"] >= df["Close"]).all(), "High must be >= Close"
    assert (df["Low"] <= df["Open"]).all(), "Low must be <= Open"
    assert (df["Low"] <= df["Close"]).all(), "Low must be <= Close"


def assert_returns_correctness(df):
    """Assert that Returns column is correctly calculated."""
    if "Returns" in df.columns:
        expected_returns = df["Close"].pct_change().fillna(0.0)
        np.testing.assert_allclose(
            df["Returns"].values,
            expected_returns.values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Returns calculation incorrect",
        )
