"""
Comprehensive tests for DataSanity validation suite.
Tests all rules and staged validation.
"""


import numpy as np
import pandas as pd
import pytest

from core.data_sanity.registry import RULES, get_rule
from core.data_sanity.rules.finite import FiniteNumbersRule, FiniteNumbersRuleConfig
from core.data_sanity.rules.ohlc import OHLCConsistencyRule, OHLCConsistencyRuleConfig
from core.data_sanity.rules.prices import PricePositivityRule, PricePositivityRuleConfig


def test_price_positivity_rejects_negatives_and_zeros():
    """Test that price positivity rule rejects non-positive prices."""
    df = pd.DataFrame({
        "Open": [100, 99, 0.0, -1, 101],
        "High": [102, 100, 1.0, 0.0, 103],
        "Low": [98, 97, -1.0, -2, 99],
        "Close": [101, 99, 0.5, -0.5, 102]
    })

    rule = PricePositivityRule(PricePositivityRuleConfig(False, False))

    with pytest.raises(ValueError, match="non-positive prices detected"):
        rule.validate(df)


def test_ohlc_consistency_rejects_invalid_relationships():
    """Test that OHLC consistency rule rejects invalid relationships."""
    df = pd.DataFrame({
        "Open": [100, 99, 101],
        "High": [102, 98, 103],  # High < Open in row 1
        "Low": [98, 97, 104],    # Low > High in row 2
        "Close": [101, 99, 102]
    })

    rule = OHLCConsistencyRule(OHLCConsistencyRuleConfig(True))

    with pytest.raises(ValueError, match="OHLC inconsistent"):
        rule.validate(df)


def test_finite_numbers_rejects_non_finite():
    """Test that finite numbers rule rejects non-finite values."""
    df = pd.DataFrame({
        "Open": [100, np.nan, 101],
        "High": [102, np.inf, 103],
        "Low": [98, -np.inf, 99],
        "Close": [101, 99, 102]
    })

    rule = FiniteNumbersRule(FiniteNumbersRuleConfig(False, False, False))

    with pytest.raises(ValueError, match="non-finite values detected"):
        rule.validate(df)


def test_rule_registry_has_all_required_rules():
    """Test that all required rules are registered."""
    required_rules = {"price_positivity", "ohlc_consistency", "finite_numbers"}
    available_rules = set(RULES.keys())

    assert required_rules.issubset(available_rules), f"Missing rules: {required_rules - available_rules}"


def test_get_rule_creates_valid_instances():
    """Test that get_rule creates valid rule instances."""
    config = {"allow_negative_prices": False, "allow_zero_prices": False}

    rule = get_rule("price_positivity", config)
    assert isinstance(rule, PricePositivityRule)


def test_two_stage_validation_catches_post_transform_issues():
    """Test that two-stage validation catches issues introduced after transforms."""
    # Valid initial data
    df = pd.DataFrame({
        "Open": [100, 99, 101],
        "High": [102, 100, 103],
        "Low": [98, 97, 99],
        "Close": [101, 99, 102]
    })

    # First stage should pass
    rule1 = PricePositivityRule(PricePositivityRuleConfig(False, False))
    df_valid = rule1.validate(df)

    # Simulate a transformation that introduces negative prices
    df_transformed = df_valid.copy()
    df_transformed.loc[1, "Close"] = -1  # Introduce negative price

    # Second stage should catch the issue
    rule2 = PricePositivityRule(PricePositivityRuleConfig(False, False))
    with pytest.raises(ValueError, match="non-positive prices detected"):
        rule2.validate(df_transformed)


def test_ohlc_consistency_accepts_valid_data():
    """Test that OHLC consistency rule accepts valid data."""
    df = pd.DataFrame({
        "Open": [100, 99, 101],
        "High": [102, 100, 103],
        "Low": [98, 97, 99],
        "Close": [101, 99, 102]
    })

    rule = OHLCConsistencyRule(OHLCConsistencyRuleConfig(True))
    result = rule.validate(df)

    assert result.equals(df)


def test_finite_numbers_accepts_valid_data():
    """Test that finite numbers rule accepts valid data."""
    df = pd.DataFrame({
        "Open": [100, 99, 101],
        "High": [102, 100, 103],
        "Low": [98, 97, 99],
        "Close": [101, 99, 102],
        "Volume": [1000, 1100, 900]
    })

    rule = FiniteNumbersRule(FiniteNumbersRuleConfig(False, False, False))
    result = rule.validate(df)

    assert result.equals(df)


def test_price_positivity_repair_drops_bad_rows():
    """Test that price positivity repair drops bad rows."""
    df = pd.DataFrame({
        "Open": [100, 99, 0.0, -1, 101],
        "High": [102, 100, 1.0, 0.0, 103],
        "Low": [98, 97, -1.0, -2, 99],
        "Close": [101, 99, 0.5, -0.5, 102]
    })

    rule = PricePositivityRule(PricePositivityRuleConfig(False, False))

    result, repairs = rule.validate_and_repair(df)

    # Should drop rows 2, 3, 4 (0.0, -1, and their related rows)
    expected = pd.DataFrame({
        "Open": [100, 101],
        "High": [102, 103],
        "Low": [98, 99],
        "Close": [101, 102]
    }, index=[0, 4])

    pd.testing.assert_frame_equal(result, expected)
    assert any("dropped" in r for r in repairs)


def test_rule_config_validation():
    """Test that rule configurations are properly validated."""
    # Test price positivity config
    config = {
        "allow_negative_prices": False,
        "allow_zero_prices": False,
        "cols": ["Open", "High", "Low", "Close"]
    }

    rule = get_rule("price_positivity", config)
    assert not rule.cfg.allow_negative_prices
    assert not rule.cfg.allow_zero_prices
    assert rule.price_cols == ("Open", "High", "Low", "Close")


def test_missing_rule_raises_error():
    """Test that requesting a missing rule raises an error."""
    with pytest.raises(ValueError, match="Unknown rule"):
        get_rule("nonexistent_rule", {})


if __name__ == "__main__":
    pytest.main([__file__])
