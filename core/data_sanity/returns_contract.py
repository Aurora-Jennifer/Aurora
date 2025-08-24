"""
Returns Calculation Contract Tests
Ensures returns calculation never uses future information.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd


def calc_log_returns(close: pd.Series) -> pd.Series:
    """
    Calculate log returns with strict backward-only information.
    
    Args:
        close: Close price series
        
    Returns:
        Log returns series (first value will be NaN)
    """
    # Strictly backward info
    prev = close.shift(1)

    # Drop/repair non-positive prices upstream; here guard just in case
    mask = (close > 0) & (prev > 0)

    out = pd.Series(index=close.index, dtype="float64")
    out.loc[mask] = np.log(close.loc[mask] / prev.loc[mask])

    return out


def check_returns_no_future_dependency(returns_func: Callable[[pd.Series], pd.Series],
                                    close_prices: pd.Series) -> bool:
    """
    Property test: returns at time t must not change if we truncate future rows.
    
    Args:
        returns_func: Function that calculates returns
        close_prices: Close price series
        
    Returns:
        True if no future dependency detected
    """
    if len(close_prices) < 10:
        return True  # Need sufficient data for meaningful test

    # Calculate returns on full dataset
    full_returns = returns_func(close_prices)

    # Calculate returns on truncated dataset (remove last 20% of rows)
    truncate_idx = int(len(close_prices) * 0.8)
    truncated_prices = close_prices.iloc[:truncate_idx]
    truncated_returns = returns_func(truncated_prices)

    # Compare returns for common timestamps
    common_idx = full_returns.index.intersection(truncated_returns.index)

    if len(common_idx) < 5:
        return True  # Need sufficient overlap

    # Check that returns are identical for common timestamps
    full_common = full_returns.loc[common_idx]
    trunc_common = truncated_returns.loc[common_idx]

    # Drop NaN values for comparison
    valid_mask = full_common.notna() & trunc_common.notna()

    if not valid_mask.any():
        return True  # No valid values to compare

    full_valid = full_common[valid_mask]
    trunc_valid = trunc_common[valid_mask]

    # Check if values are identical (within numerical precision)
    diff = np.abs(full_valid - trunc_valid)
    max_diff = diff.max()

    # Allow for small numerical differences
    return max_diff < 1e-12


def check_returns_recompute_consistency(returns_func: Callable[[pd.Series], pd.Series],
                                     close_prices: pd.Series) -> bool:
    """
    Property test: returns must equal recompute on prefix-only data.
    
    Args:
        returns_func: Function that calculates returns
        close_prices: Close price series
        
    Returns:
        True if recompute consistency holds
    """
    if len(close_prices) < 10:
        return True

    # Calculate returns on full dataset
    full_returns = returns_func(close_prices)

    # Test multiple truncation points
    test_points = [0.5, 0.6, 0.7, 0.8, 0.9]

    for ratio in test_points:
        truncate_idx = int(len(close_prices) * ratio)
        if truncate_idx < 5:
            continue

        truncated_prices = close_prices.iloc[:truncate_idx]
        truncated_returns = returns_func(truncated_prices)

        # Compare at the truncation point
        if truncate_idx < len(full_returns):
            full_val = full_returns.iloc[truncate_idx - 1]  # Last value in truncated
            trunc_val = truncated_returns.iloc[-1]  # Last value in truncated

            if pd.notna(full_val) and pd.notna(trunc_val):
                if abs(full_val - trunc_val) > 1e-12:
                    return False

    return True


def validate_returns_contract(returns_func: Callable[[pd.Series], pd.Series],
                            close_prices: pd.Series) -> tuple[bool, str]:
    """
    Validate that a returns calculation function satisfies the no-future-dependency contract.
    
    Args:
        returns_func: Function that calculates returns
        close_prices: Close price series
        
    Returns:
        Tuple of (passed, error_message)
    """
    try:
        # Test 1: No future dependency
        if not check_returns_no_future_dependency(returns_func, close_prices):
            return False, "Returns calculation shows future dependency"

        # Test 2: Recompute consistency
        if not check_returns_recompute_consistency(returns_func, close_prices):
            return False, "Returns calculation fails recompute consistency"

        return True, "All contract tests passed"

    except Exception as e:
        return False, f"Contract test failed with exception: {str(e)}"


def create_test_series() -> pd.Series:
    """Create a test price series for contract testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    # Create realistic price series with some volatility
    np.random.seed(42)  # For reproducibility
    returns = np.random.randn(100) * 0.02  # 2% daily volatility
    prices = 100 * np.exp(np.cumsum(returns))  # Start at $100
    return pd.Series(prices, index=dates)


if __name__ == "__main__":
    # Test the contract with our implementation
    test_prices = create_test_series()

    print("Testing returns calculation contract...")
    passed, message = validate_returns_contract(calc_log_returns, test_prices)

    if passed:
        print("✅ Contract validation passed")
    else:
        print(f"❌ Contract validation failed: {message}")
