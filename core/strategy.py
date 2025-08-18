"""
Simple Strategy Interface for MVB
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Strategy(ABC):
    """Abstract strategy interface"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def on_bar(self, ctx: Dict[str, Any], bar: Dict[str, Any]) -> Dict[str, float]:
        """
        Process a new bar and return signals.

        Args:
            ctx: Context with portfolio, positions, etc.
            bar: Market data bar

        Returns:
            Dictionary of symbol -> signal (-1 to 1)
        """
        pass


class SimpleMAStrategy(Strategy):
    """Simple moving average crossover strategy"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.short_window = config.get("short_window", 20)
        self.long_window = config.get("long_window", 50)
        self.price_history = {}

    def on_bar(self, ctx: Dict[str, Any], bar: Dict[str, Any]) -> Dict[str, float]:
        """Generate signals based on MA crossover"""
        signals = {}

        for symbol, data in bar.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            price = data.get("price", 0)
            if price > 0:
                self.price_history[symbol].append(price)

                # Keep only recent history
                max_history = max(self.short_window, self.long_window) * 2
                if len(self.price_history[symbol]) > max_history:
                    self.price_history[symbol] = self.price_history[symbol][
                        -max_history:
                    ]

                # Calculate moving averages
                if len(self.price_history[symbol]) >= self.long_window:
                    short_ma = np.mean(self.price_history[symbol][-self.short_window :])
                    long_ma = np.mean(self.price_history[symbol][-self.long_window :])

                    # Generate signal
                    if short_ma > long_ma:
                        signals[symbol] = 0.5  # Buy signal
                    elif short_ma < long_ma:
                        signals[symbol] = -0.5  # Sell signal
                    else:
                        signals[symbol] = 0.0  # No signal
                else:
                    signals[symbol] = 0.0  # Not enough data

        return signals


class RandomStrategy(Strategy):
    """Random strategy for testing"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.signal_strength = config.get("signal_strength", 0.3)

    def on_bar(self, ctx: Dict[str, Any], bar: Dict[str, Any]) -> Dict[str, float]:
        """Generate random signals"""
        signals = {}

        for symbol in bar.keys():
            # Random signal between -signal_strength and +signal_strength
            signal = np.random.uniform(-self.signal_strength, self.signal_strength)
            signals[symbol] = signal

        return signals


class RegimeAwareStrategy(Strategy):
    """Regime-aware strategy using existing regime detector"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from core.regime_detector import RegimeDetector

        self.regime_detector = RegimeDetector(lookback_period=252)
        self.price_history = {}

    def on_bar(self, ctx: Dict[str, Any], bar: Dict[str, Any]) -> Dict[str, float]:
        """Generate regime-aware signals"""
        signals = {}

        # Detect regime
        regime_data = {}
        for symbol, data in bar.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            price = data.get("price", 0)
            if price > 0:
                self.price_history[symbol].append(price)
                regime_data[symbol] = price

        if regime_data:
            # Use regime detector (simplified)
            regime_name = self.regime_detector.get_current_regime_name()

            # Generate signals based on regime
            for symbol in bar.keys():
                if regime_name == "trend":
                    signals[symbol] = 0.4  # Moderate buy in trend
                elif regime_name == "mean_reversion":
                    signals[symbol] = -0.3  # Moderate sell in mean reversion
                elif regime_name == "chop":
                    signals[symbol] = 0.1  # Small position in chop
                else:
                    signals[symbol] = 0.0  # No signal

        return signals


def create_strategy(strategy_type: str, config: Dict[str, Any]) -> Strategy:
    """Factory function to create strategies"""
    if strategy_type == "simple_ma":
        return SimpleMAStrategy(config)
    elif strategy_type == "random":
        return RandomStrategy(config)
    elif strategy_type == "regime_aware":
        return RegimeAwareStrategy(config)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
