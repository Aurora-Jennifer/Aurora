"""
Tests for DataSanity Price Positivity Rule
"""

import pandas as pd
import pytest

from core.data_sanity.rules.prices import (
    PricePositivityRule,
    PricePositivityRuleConfig,
    create_price_positivity_rule,
)


def test_rejects_non_positive_prices():
    """Test that the rule rejects non-positive prices."""
    df = pd.DataFrame({"Close": [100, 99, 0.0, -1, 101]})
    rule = PricePositivityRule(PricePositivityRuleConfig(False, False), ["Close"])

    with pytest.raises(ValueError, match="non-positive prices detected"):
        rule.validate(df)


def test_accepts_positive_prices():
    """Test that the rule accepts positive prices."""
    df = pd.DataFrame({"Close": [100, 99, 0.1, 101]})
    rule = PricePositivityRule(PricePositivityRuleConfig(False, False), ["Close"])

    result = rule.validate(df)
    assert result.equals(df)


def test_allow_zero_prices():
    """Test that zero prices are allowed when configured."""
    df = pd.DataFrame({"Close": [100, 99, 0.0, 101]})
    rule = PricePositivityRule(PricePositivityRuleConfig(False, True), ["Close"])

    result = rule.validate(df)
    assert result.equals(df)


def test_allow_negative_prices():
    """Test that negative prices are allowed when configured."""
    df = pd.DataFrame({"Close": [100, 99, -1, 101]})
    rule = PricePositivityRule(PricePositivityRuleConfig(True, True), ["Close"])

    result = rule.validate(df)
    assert result.equals(df)


def test_validate_and_repair_drops_bad_rows():
    """Test that validate_and_repair drops rows with non-positive prices."""
    df = pd.DataFrame({"Close": [100, 99, 0.0, -1, 101]})
    rule = PricePositivityRule(PricePositivityRuleConfig(False, False), ["Close"])

    result, repairs = rule.validate_and_repair(df)

    # Should drop rows 2 and 3 (0.0 and -1)
    expected = pd.DataFrame({"Close": [100, 99, 101]}, index=[0, 1, 4])
    pd.testing.assert_frame_equal(result, expected)
    assert "dropped_2_non_positive_prices" in repairs


def test_create_price_positivity_rule_from_config():
    """Test factory function creates rule from config dict."""
    config = {
        "allow_negative_prices": False,
        "allow_zero_prices": False,
        "price_cols": ["Close", "Open"]
    }

    rule = create_price_positivity_rule(config)
    assert isinstance(rule, PricePositivityRule)
    assert not rule.cfg.allow_negative_prices
    assert not rule.cfg.allow_zero_prices
    assert rule.price_cols == ("Close", "Open")


def test_log_returns_masks_bad_rows():
    """Test that log returns calculation masks bad rows."""
    from core.ml.safe_math import log_returns_from_close

    s = pd.Series([100, 99, 0.0, -1, 101])
    lr = log_returns_from_close(s)

    # Rows 2 and 3 should be NaN (0.0 and -1)
    assert pd.isna(lr.iloc[2])
    assert pd.isna(lr.iloc[3])

    # Other rows should have finite values
    assert not pd.isna(lr.iloc[1])  # log(99/100)
    assert not pd.isna(lr.iloc[4])  # log(101/99)


def test_multiple_price_columns():
    """Test validation with multiple price columns."""
    df = pd.DataFrame({
        "Open": [100, 99, 0.0, 101],
        "High": [102, 100, 1.0, 103],
        "Low": [98, 97, -1.0, 99],
        "Close": [101, 99, 0.5, 102]
    })

    rule = PricePositivityRule(PricePositivityRuleConfig(False, False))

    with pytest.raises(ValueError, match="non-positive prices detected"):
        rule.validate(df)


if __name__ == "__main__":
    pytest.main([__file__])
