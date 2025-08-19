"""
Momentum Strategy
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .base import BaseStrategy, StrategyParams


@dataclass
class MomentumParams(StrategyParams):
    """Parameters for Momentum strategy."""

    lookback_period: int = 20
    threshold: float = 0.02  # 2% threshold


class Momentum(BaseStrategy):
    """
    Momentum Strategy.

    Generates long signals when price momentum exceeds threshold,
    flat signals otherwise.
    """

    def __init__(self, params: MomentumParams):
        super().__init__(params)
        self.lookback_period = params.lookback_period
        self.threshold = params.threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate momentum signals."""
        close = df["Close"].dropna()

        # Calculate momentum (percentage change over lookback period)
        momentum = close.pct_change(self.lookback_period)

        # Generate signals (1 = long when momentum > threshold, 0 = flat)
        signals = (momentum > self.threshold).astype(int)

        return signals

    def get_default_params(self) -> MomentumParams:
        """Return default momentum parameters."""
        return MomentumParams(lookback_period=20, threshold=0.02)

    def get_param_ranges(self) -> dict[str, Any]:
        """Return parameter ranges for optimization."""
        return {
            "lookback_period": range(10, 51, 5),
            "threshold": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04],
        }

    def validate_params(self, params: MomentumParams) -> bool:
        """Validate momentum parameters."""
        return params.lookback_period > 0 and params.threshold > 0

    def get_description(self) -> str:
        """Return strategy description."""
        return f"Momentum (lookback={self.lookback_period}, threshold={self.threshold:.1%})"
