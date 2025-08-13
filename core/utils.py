"""
Utility functions for the trading system.
Consolidates common functionality to reduce code duplication.
"""

import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd


def setup_logging(
    log_file: str = "trading.log", level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def calculate_returns(close: pd.Series, shift: int = -1) -> pd.Series:
    """
    Calculate returns from close prices.

    Args:
        close: Close price series
        shift: Number of periods to shift (default -1 for next-period returns)

    Returns:
        Returns series
    """
    return close.pct_change().shift(shift)


def calculate_performance_metrics(
    equity: pd.Series, start_date: str, end_date: str
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity: Equity curve series
        start_date: Start date string
        end_date: End date string

    Returns:
        Dictionary of performance metrics
    """
    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "volatility": 0.0,
        }

    # Calculate returns
    rets = equity.pct_change().dropna()

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    # CAGR
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    # Sharpe ratio
    sharpe = rets.mean() / (rets.std() + 1e-6) * np.sqrt(252)

    # Max drawdown
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Volatility
    volatility = rets.std() * np.sqrt(252)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "volatility": volatility,
    }


def ensure_directories(base_dir: str = ".") -> None:
    """
    Ensure required directories exist.

    Args:
        base_dir: Base directory path
    """
    folders = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "features"),
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "signals"),
        os.path.join(base_dir, "dashboard"),
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Division result or default value
    """
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by handling common data issues.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Fill infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with 0 for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def validate_strategy_params(params: Dict[str, Any], required_params: list) -> bool:
    """
    Validate strategy parameters.

    Args:
        params: Parameter dictionary
        required_params: List of required parameter names

    Returns:
        True if valid, False otherwise
    """
    for param in required_params:
        if param not in params:
            return False
    return True


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.

    Args:
        value: Value to format
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}%}"


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format value as currency string.

    Args:
        value: Value to format
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    return f"${value:.{decimals}f}"
