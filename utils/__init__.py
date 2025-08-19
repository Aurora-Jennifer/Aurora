"""
Shared Utilities Package

This package contains reusable utility modules for the ML trading system.
All modules are pure functions with no side effects for maximum reusability.
"""

from .indicators import (
    adx,
    atr,
    bollinger_bands,
    calculate_all_indicators,
    cci,
    diff,
    ichimoku,
    lag,
    lead,
    macd,
    mfi,
    normalize,
    obv,
    pct_change,
    roc,
    rolling_mean,
    rolling_median,
    rolling_std,
    rsi,
    stochastic,
    vwap,
    williams_r,
    winsorize,
    zscore,
)
from .logging import get_logger, setup_logging
from .metrics import (
    calmar_ratio,
    cvar,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    var,
)

__version__ = "1.0.0"
__all__ = [
    # Indicators
    "rolling_mean",
    "rolling_std",
    "rolling_median",
    "zscore",
    "winsorize",
    "normalize",
    "rsi",
    "macd",
    "atr",
    "bollinger_bands",
    "pct_change",
    "lag",
    "lead",
    "diff",
    "adx",
    "roc",
    "mfi",
    "stochastic",
    "williams_r",
    "cci",
    "obv",
    "vwap",
    "ichimoku",
    "calculate_all_indicators",
    # Metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "var",
    "cvar",
    "turnover",
    "hit_rate",
    "profit_factor",
    # Logging
    "get_logger",
    "setup_logging",
]
