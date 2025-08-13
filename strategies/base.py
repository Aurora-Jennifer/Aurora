"""
Base strategy class for backtesting system.
All strategies should inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class StrategyParams:
    """Base class for strategy parameters."""


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement:
    - generate_signals(): Generate trading signals
    - get_default_params(): Return default parameters
    - get_param_ranges(): Return parameter ranges for optimization
    """

    def __init__(self, params: StrategyParams):
        self.params = params
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from price data.

        Args:
            df: DataFrame with price data (must have 'Close' column)

        Returns:
            pd.Series: Trading signals (1 = long, 0 = flat, -1 = short)
        """

    @abstractmethod
    def get_default_params(self) -> StrategyParams:
        """Return default parameters for this strategy."""

    @abstractmethod
    def get_param_ranges(self) -> Dict[str, Any]:
        """Return parameter ranges for optimization."""

    def backtest(self, df: pd.DataFrame, fee_bps: float = 5.0) -> tuple[pd.Series, int]:
        """
        Run backtest for this strategy.

        Args:
            df: DataFrame with price data
            fee_bps: Transaction costs in basis points

        Returns:
            tuple: (equity_curve, trade_count)
        """
        # Generate signals
        signals = self.generate_signals(df)

        # Calculate returns
        close = df["Close"].dropna()
        returns = close.pct_change().fillna(0.0)

        # Apply signals (shift to trade on next bar)
        signals = signals.shift(1).fillna(0)

        # Calculate transaction costs
        position_changes = signals.diff().abs().fillna(signals.iloc[0])
        fees = (fee_bps / 10000.0) * position_changes

        # Net returns
        net_returns = signals * returns - fees

        # Equity curve
        equity = (1.0 + net_returns).cumprod()
        trade_count = int(
            position_changes.sum().iloc[0]
            if hasattr(position_changes.sum(), "iloc")
            else position_changes.sum()
        )

        # Handle MultiIndex if present
        if isinstance(equity.index, pd.MultiIndex):
            equity = equity.droplevel(0)  # Remove the ticker level

        return equity, trade_count

    def get_description(self) -> str:
        """Return strategy description."""
        return f"{self.name} strategy"

    def validate_params(self, params: StrategyParams) -> bool:
        """Validate strategy parameters."""
        return True  # Override in subclasses if needed
