"""
Test data type casting - verify robust string-to-numeric conversion.
"""

import numpy as np
import pandas as pd
import pytest


def test_coerce_dtypes_method_exists(strict_validator):
    """Test that coerce_dtypes method exists."""
    # Check if the method exists
    if not hasattr(strict_validator, "coerce_dtypes"):
        pytest.skip(
            "coerce_dtypes method not implemented - implement this method for robust string-to-numeric conversion"
        )

    # If it exists, test basic functionality
    data = pd.DataFrame(
        {
            "Open": ["100.01", "200.02"],
            "High": ["101.01", "201.02"],
            "Low": ["99.01", "199.02"],
            "Close": ["100.50", "200.50"],
            "Volume": ["1000000", "2000000"],
        }
    )

    result = strict_validator.coerce_dtypes(data)
    assert result["Open"].dtype in [np.float64, np.float32], "Open should be numeric"
    assert result["Volume"].dtype in [
        np.float64,
        np.float32,
        np.int64,
        np.int32,
    ], "Volume should be numeric"


def test_string_price_conversion(strict_validator, mk_ts):
    """Test conversion of various string price formats."""
    # Create base data
    data = mk_ts(n=5)

    # Replace with string formats
    data["Open"] = ["$100.01", "100,02", " 100.03 ", "1.0004e2", "100.05 USD"]
    data["High"] = ["$101.01", "101,02", " 101.03 ", "1.0104e2", "101.05 USD"]
    data["Low"] = ["$99.01", "99,02", " 99.03 ", "9.9004e1", "99.05 USD"]
    data["Close"] = ["$100.50", "100,50", " 100.50 ", "1.0050e2", "100.50 USD"]
    data["Volume"] = ["1,000,000", "2,000,000", "3,000,000", "4,000,000", "5,000,000"]

    # Test if coerce_dtypes method exists
    if hasattr(strict_validator, "coerce_dtypes"):
        result = strict_validator.coerce_dtypes(data)

        # Check that all columns are numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert result[col].dtype in [
                np.float64,
                np.float32,
                np.int64,
                np.int32,
            ], f"{col} should be numeric"

        # Check that no NaNs were introduced
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert not result[col].isna().any(), f"{col} should have no NaN values after conversion"

        # Check specific conversions
        assert abs(result["Open"].iloc[0] - 100.01) < 0.001, "Currency symbol should be stripped"
        assert abs(result["Open"].iloc[1] - 100.02) < 0.001, "Comma should be handled"
        assert abs(result["Open"].iloc[2] - 100.03) < 0.001, "Whitespace should be stripped"
        assert abs(result["Open"].iloc[3] - 100.04) < 0.001, "Scientific notation should work"
        assert abs(result["Open"].iloc[4] - 100.05) < 0.001, "Currency suffix should be stripped"

        # Check volume conversion
        assert result["Volume"].iloc[0] == 1000000, "Thousands separators should be handled"
    else:
        pytest.skip("coerce_dtypes method not implemented")


def test_mixed_numeric_string_conversion(strict_validator, mk_ts):
    """Test conversion of mixed numeric and string data."""
    # Create base data
    data = mk_ts(n=10)

    # Mix numeric and string values
    data.loc[data.index[0:5], "Open"] = [100.0, 200.0, 300.0, 400.0, 500.0]  # Numeric
    # Convert to object dtype first to avoid FutureWarning
    data["Open"] = data["Open"].astype(object)
    data.loc[data.index[5:10], "Open"] = [
        "$600.01",
        "700,02",
        " 800.03 ",
        "9.0004e2",
        "1000.05 USD",
    ]  # String

    if hasattr(strict_validator, "coerce_dtypes"):
        result = strict_validator.coerce_dtypes(data)

        # All values should be numeric
        assert result["Open"].dtype in [
            np.float64,
            np.float32,
        ], "Open should be numeric"
        assert not result["Open"].isna().any(), "Should have no NaN values"

        # Check that numeric values were preserved
        assert abs(result["Open"].iloc[0] - 100.0) < 0.001, "Numeric values should be preserved"
        assert abs(result["Open"].iloc[4] - 500.0) < 0.001, "Numeric values should be preserved"

        # Check that string values were converted
        assert abs(result["Open"].iloc[5] - 600.01) < 0.001, "String values should be converted"
        assert abs(result["Open"].iloc[9] - 1000.05) < 0.001, "String values should be converted"
    else:
        pytest.skip("coerce_dtypes method not implemented")


def test_invalid_string_handling(strict_validator, mk_ts):
    """Test handling of invalid strings that can't be converted."""
    # Create base data
    data = mk_ts(n=5)

    # Add invalid strings
    data["Open"] = ["$100.01", "invalid", " 100.03 ", "not_a_number", "100.05 USD"]

    if hasattr(strict_validator, "coerce_dtypes"):
        result = strict_validator.coerce_dtypes(data)

        # Check that valid strings were converted
        assert abs(result["Open"].iloc[0] - 100.01) < 0.001, "Valid string should be converted"
        assert abs(result["Open"].iloc[2] - 100.03) < 0.001, "Valid string should be converted"
        assert abs(result["Open"].iloc[4] - 100.05) < 0.001, "Valid string should be converted"

        # Check that invalid strings became NaN
        assert pd.isna(result["Open"].iloc[1]), "Invalid string should become NaN"
        assert pd.isna(result["Open"].iloc[3]), "Invalid string should become NaN"
    else:
        pytest.skip("coerce_dtypes method not implemented")


def test_scientific_notation_handling(strict_validator, mk_ts):
    """Test handling of scientific notation."""
    # Create base data
    data = mk_ts(n=5)

    # Add scientific notation
    data["Open"] = ["1.23e2", "4.56E-1", "7.89e+3", "1.00e0", "2.50e-2"]

    if hasattr(strict_validator, "coerce_dtypes"):
        result = strict_validator.coerce_dtypes(data)

        # Check conversions
        assert abs(result["Open"].iloc[0] - 123.0) < 0.001, "1.23e2 should be 123.0"
        assert abs(result["Open"].iloc[1] - 0.456) < 0.001, "4.56E-1 should be 0.456"
        assert abs(result["Open"].iloc[2] - 7890.0) < 0.001, "7.89e+3 should be 7890.0"
        assert abs(result["Open"].iloc[3] - 1.0) < 0.001, "1.00e0 should be 1.0"
        assert abs(result["Open"].iloc[4] - 0.025) < 0.001, "2.50e-2 should be 0.025"
    else:
        pytest.skip("coerce_dtypes method not implemented")


def test_thousands_separator_handling(strict_validator, mk_ts):
    """Test handling of thousands separators."""
    # Create base data
    data = mk_ts(n=5)

    # Add thousands separators
    data["Volume"] = ["1,000", "2,500,000", "3,456,789", "1,234,567,890", "999,999"]

    if hasattr(strict_validator, "coerce_dtypes"):
        result = strict_validator.coerce_dtypes(data)

        # Check conversions
        assert result["Volume"].iloc[0] == 1000, "1,000 should be 1000"
        assert result["Volume"].iloc[1] == 2500000, "2,500,000 should be 2500000"
        assert result["Volume"].iloc[2] == 3456789, "3,456,789 should be 3456789"
        assert result["Volume"].iloc[3] == 1234567890, "1,234,567,890 should be 1234567890"
        assert result["Volume"].iloc[4] == 999999, "999,999 should be 999999"
    else:
        pytest.skip("coerce_dtypes method not implemented")


def test_currency_symbol_handling(strict_validator, mk_ts):
    """Test handling of currency symbols."""
    # Create base data
    data = mk_ts(n=5)

    # Add currency symbols
    data["Open"] = ["$100.01", "€200.02", "£300.03", "¥400.04", "₹500.05"]

    if hasattr(strict_validator, "coerce_dtypes"):
        result = strict_validator.coerce_dtypes(data)

        # Check conversions (currency symbols should be stripped)
        assert abs(result["Open"].iloc[0] - 100.01) < 0.001, "$100.01 should be 100.01"
        assert abs(result["Open"].iloc[1] - 200.02) < 0.001, "€200.02 should be 200.02"
        assert abs(result["Open"].iloc[2] - 300.03) < 0.001, "£300.03 should be 300.03"
        assert abs(result["Open"].iloc[3] - 400.04) < 0.001, "¥400.04 should be 400.04"
        assert abs(result["Open"].iloc[4] - 500.05) < 0.001, "₹500.05 should be 500.05"
    else:
        pytest.skip("coerce_dtypes method not implemented")
