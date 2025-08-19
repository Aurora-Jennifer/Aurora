"""
Simple Moving Average (SMA) Crossover Strategy
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategyParams


@dataclass
class SMAParams(StrategyParams):
    """Parameters for SMA Crossover strategy."""

    fast_period: int = 10
    slow_period: int = 50


class SMACrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.

    Generates long signals when fast SMA > slow SMA,
    flat signals otherwise.
    """

    def __init__(self, params: SMAParams):
        super().__init__(params)
        self.fast_period = params.fast_period
        self.slow_period = params.slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate SMA crossover signals."""
        close = df["Close"].dropna()

        # Calculate moving averages
        fast_sma = close.rolling(self.fast_period).mean()
        slow_sma = close.rolling(self.slow_period).mean()

        # Generate signals (1 = long when fast > slow, 0 = flat)
        signals = (fast_sma > slow_sma).astype(int)

        return signals

    def get_default_params(self) -> SMAParams:
        """Return default SMA parameters."""
        return SMAParams(fast_period=10, slow_period=50)

    def get_param_ranges(self) -> dict[str, Any]:
        """Return parameter ranges for optimization."""
        return {"fast_period": range(5, 31, 5), "slow_period": range(40, 201, 20)}

    def validate_params(self, params: SMAParams) -> bool:
        """Validate SMA parameters."""
        return (
            params.fast_period < params.slow_period
            and params.fast_period > 0
            and params.slow_period > 0
        )

    def get_description(self) -> str:
        """Return strategy description."""
        return f"SMA Crossover (fast={self.fast_period}, slow={self.slow_period})"
