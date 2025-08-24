"""
Safe mathematical functions for financial calculations.
Handles edge cases like zero/negative values that could cause numerical issues.
"""


import numpy as np
import pandas as pd


def log_returns_from_close(close: pd.Series) -> pd.Series:
    """
    Calculate log returns from close prices with safe handling of edge cases.
    
    Args:
        close: Series of close prices
        
    Returns:
        Series of log returns with NaN for invalid values
    """
    # Drop/NA out any non-positive rows deterministically
    bad = (close <= 0) | (close.shift(1) <= 0)
    safe = close.copy()
    safe[bad] = np.nan  # consistent, not fill-forward

    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(safe / safe.shift(1))

    return lr


def log_returns_pct(close: pd.Series) -> pd.Series:
    """
    Calculate log returns using pct_change with safe handling.
    
    Args:
        close: Series of close prices
        
    Returns:
        Series of log returns with NaN for invalid values
    """
    r = close.pct_change()
    # if any leg is non-positive, null the result
    bad = (close <= 0) | (close.shift(1) <= 0)
    r[bad] = np.nan
    with np.errstate(invalid="ignore"):
        return np.log1p(r)


def safe_divide(numerator: pd.Series | np.ndarray,
                denominator: pd.Series | np.ndarray,
                fill_value: float = np.nan) -> pd.Series | np.ndarray:
    """
    Safe division that handles zero denominators.
    
    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result of division with fill_value for zero denominators
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator

    # Replace inf and -inf with fill_value
    if isinstance(result, pd.Series):
        result = result.replace([np.inf, -np.inf], fill_value)
    else:
        result = np.where(np.isinf(result), fill_value, result)

    return result


def safe_log(x: pd.Series | np.ndarray,
             fill_value: float = np.nan) -> pd.Series | np.ndarray:
    """
    Safe logarithm that handles non-positive values.
    
    Args:
        x: Input values
        fill_value: Value to use for non-positive inputs
        
    Returns:
        Natural log of x with fill_value for non-positive values
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(x)

    # Replace inf, -inf, and nan with fill_value
    if isinstance(result, pd.Series):
        result = result.replace([np.inf, -np.inf, np.nan], fill_value)
    else:
        result = np.where(~np.isfinite(result), fill_value, result)

    return result
