"""
Tests for returns calculation contract validation.
"""

import pytest
import pandas as pd
import numpy as np
from core.data_sanity.returns_contract import (
    calc_log_returns, 
    check_returns_no_future_dependency,
    check_returns_recompute_consistency,
    validate_returns_contract,
    create_test_series
)


def test_calc_log_returns_basic():
    """Test basic log returns calculation."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    prices = pd.Series([100, 102, 98, 105, 103], index=dates)
    
    returns = calc_log_returns(prices)
    
    # First value should be NaN (no previous price)
    assert pd.isna(returns.iloc[0])
    
    # Other values should be finite
    assert returns.iloc[1:].notna().all()
    
    # Check specific values
    expected_returns = [np.nan, np.log(102/100), np.log(98/102), np.log(105/98), np.log(103/105)]
    for i, (actual, expected) in enumerate(zip(returns, expected_returns)):
        if pd.isna(expected):
            assert pd.isna(actual)
        else:
            assert abs(actual - expected) < 1e-12


def test_calc_log_returns_no_future_dependency():
    """Test that returns calculation has no future dependency."""
    test_prices = create_test_series()
    
    result = check_returns_no_future_dependency(calc_log_returns, test_prices)
    assert result == True


def test_calc_log_returns_recompute_consistency():
    """Test that returns calculation is consistent when recomputed."""
    test_prices = create_test_series()
    
    result = check_returns_recompute_consistency(calc_log_returns, test_prices)
    assert result == True


def test_validate_returns_contract():
    """Test the complete returns contract validation."""
    test_prices = create_test_series()
    
    passed, message = validate_returns_contract(calc_log_returns, test_prices)
    assert passed == True
    assert "passed" in message.lower()


def test_returns_contract_with_negative_prices():
    """Test returns calculation with negative prices (should handle gracefully)."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    prices = pd.Series([100, 102, -98, 105, 103], index=dates)  # Negative price
    
    returns = calc_log_returns(prices)
    
    # Should handle negative prices gracefully
    assert len(returns) == len(prices)
    # The return for the negative price should be NaN
    assert pd.isna(returns.iloc[2])


def test_returns_contract_with_zero_prices():
    """Test returns calculation with zero prices (should handle gracefully)."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    prices = pd.Series([100, 102, 0, 105, 103], index=dates)  # Zero price
    
    returns = calc_log_returns(prices)
    
    # Should handle zero prices gracefully
    assert len(returns) == len(prices)
    # The return for the zero price should be NaN
    assert pd.isna(returns.iloc[2])


def test_returns_contract_empty_series():
    """Test returns calculation with empty series."""
    empty_prices = pd.Series(dtype=float)
    
    returns = calc_log_returns(empty_prices)
    
    assert len(returns) == 0
    assert returns.dtype == 'float64'


def test_returns_contract_single_value():
    """Test returns calculation with single value."""
    single_price = pd.Series([100])
    
    returns = calc_log_returns(single_price)
    
    assert len(returns) == 1
    assert pd.isna(returns.iloc[0])  # No previous price to calculate return


def test_returns_contract_identical_prices():
    """Test returns calculation with identical consecutive prices."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    prices = pd.Series([100, 100, 100, 100, 100], index=dates)  # All identical
    
    returns = calc_log_returns(prices)
    
    # First value should be NaN
    assert pd.isna(returns.iloc[0])
    
    # Other values should be 0 (log(100/100) = 0)
    assert (returns.iloc[1:] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__])
