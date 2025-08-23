"""
Composer Integration for Trading Engines
Integrates the composer system into walkforward and backtest engines.
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..composer.contracts import MarketState
from ..composer.registry import build_composer_system
from ..metrics.composite import composite_score, load_composite_config
from ..utils import _safe_len

logger = logging.getLogger(__name__)


class ComposerIntegration:
    """
    Integration layer for composer system in trading engines.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize composer integration.

        Args:
            config: Configuration dictionary with composer settings
        """
        self.config = config
        self.use_composer = config.get("composer", {}).get("use_composer", False)

        if not self.use_composer:
            logger.info("Composer system disabled")
            return

        # Load composite scoring configuration
        self.weights, self.penalties = load_composite_config(config)

        # Build composer system
        try:
            (
                self.strategies,
                self.regime_extractor,
                self.composer,
            ) = build_composer_system(config)
            logger.info(f"Composer system initialized with {len(self.strategies)} strategies")
            logger.info(f"Regime extractor: {self.regime_extractor.name}")
            logger.info(f"Composer: {self.composer.name}")

            # DEBUG logs for fold information
            min_history_bars = config.get("composer", {}).get("min_history_bars", 120)
            logger.debug(
                f"Composer fold info: min_history_bars={min_history_bars}, "
                f"strategies={[s.name for s in self.strategies]}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize composer system: {e}")
            self.use_composer = False

    def create_market_state(
        self, data: pd.DataFrame, symbol: str, current_idx: int, lookback: int = 50
    ) -> MarketState:
        """
        Create MarketState from DataFrame data.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            current_idx: Current index in data
            lookback: Number of bars to include in market state

        Returns:
            MarketState object
        """
        # Ensure we have enough data for the regime extractor
        min_required = lookback + 1

        if current_idx < min_required - 1:
            # Not enough data yet, return empty state
            return MarketState(
                prices=np.array([]),
                volumes=np.array([]),
                features={},
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
            )

        start_idx = 0 if current_idx < lookback else current_idx - lookback + 1

        # Extract price and volume data
        prices = data.iloc[start_idx : current_idx + 1]["Close"].values
        volumes = data.iloc[start_idx : current_idx + 1]["Volume"].values

        # Calculate basic features
        features = {}
        if len(prices) >= 20:
            # RSI
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features["rsi"] = 100 - (100 / (1 + rs))
            else:
                features["rsi"] = 100

            # MACD-like
            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])
            features["macd"] = ma_short - ma_long

            # Bollinger Band position
            bb_middle = np.mean(prices[-20:])
            bb_std = np.std(prices[-20:])
            if bb_std > 0:
                features["bb_position"] = (prices[-1] - bb_middle) / (2 * bb_std)
            else:
                features["bb_position"] = 0

            # ATR-like
            high = data.iloc[start_idx : current_idx + 1]["High"].values
            low = data.iloc[start_idx : current_idx + 1]["Low"].values
            tr = np.maximum(high - low, np.abs(high - np.roll(prices, 1)))
            features["atr"] = np.mean(tr[-14:]) if len(tr) >= 14 else 0

        return MarketState(
            prices=prices,
            volumes=volumes,
            features=features,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
        )

    def get_composer_decision(
        self,
        data: pd.DataFrame,
        symbol: str,
        current_idx: int,
        asset_class: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Get trading decision from composer system.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            current_idx: Current index in data
            asset_class: Asset class (crypto, equity, etc.) for conditional logic

        Returns:
            Tuple of (signal, metadata)
        """
        if not self.use_composer:
            return 0.0, {"composer_used": False}

        # Gate all composer calls behind min_history_bars
        min_history_bars = self.config.get("composer", {}).get("min_history_bars", 120)
        if current_idx < min_history_bars:
            return 0.0, {"composer_used": False, "reason": "warmup"}

        try:
            # Composer safety: Check for empty inputs
            if data is None or _safe_len(data) == 0:
                return 0.0, {"composer_used": False, "reason": "empty_data"}

            # Create market state
            market_state = self.create_market_state(data, symbol, current_idx)

            # Check if market state has sufficient data
            if _safe_len(market_state.prices) < 50:
                return 0.0, {
                    "composer_used": False,
                    "reason": "insufficient_data_for_regime",
                }

            # Apply asset-specific overrides
            if asset_class and asset_class in self.config.get("assets", {}):
                asset_config = self.config["assets"][asset_class]
                # Override composer parameters for this asset class
                if "composer_params" in asset_config:
                    # Create temporary composer with asset-specific params
                    from ..composer.simple_composer import SoftmaxSelector

                    temp_composer = SoftmaxSelector(**asset_config["composer_params"])
                else:
                    temp_composer = self.composer
            else:
                temp_composer = self.composer

            # Log first composer call with feature size and NaN count
            if not hasattr(self, "_first_composer_call_logged"):
                self._first_composer_call_logged = True

                # Calculate feature size and NaN counts
                feature_size = len(market_state.prices) if market_state.prices is not None else 0
                nan_counts = {}
                if data is not None:
                    for col in data.columns:
                        if col in ["Open", "High", "Low", "Close", "Volume"]:
                            nan_counts[col] = data[col].isna().sum()

                logger.debug(
                    f"First composer call: bar_idx={current_idx}, feature_size={feature_size}, "
                    f"NaN_counts={nan_counts}"
                )

            # Get composer decision
            result = temp_composer.compose(market_state, self.strategies, self.regime_extractor)

            # Validate weight vector length and finiteness
            if hasattr(temp_composer, "strategies"):
                expected_length = len(temp_composer.strategies)
                if len(result.strategy_weights) != expected_length:
                    raise ValueError(
                        f"Weight vector length mismatch: expected {expected_length}, got {len(result.strategy_weights)}"
                    )

                if not all(np.isfinite(w) for w in result.strategy_weights):
                    raise ValueError(f"Non-finite weights detected: {result.strategy_weights}")

            # Prepare metadata
            metadata = {
                "composer_used": True,
                "final_signal": result.final_signal,
                "confidence": result.confidence,
                "strategy_weights": result.strategy_weights,
                "regime_type": result.regime_features.regime_type,
                "regime_features": {
                    "trend_strength": result.regime_features.trend_strength,
                    "choppiness": result.regime_features.choppiness,
                    "volatility": result.regime_features.volatility,
                    "momentum": result.regime_features.momentum,
                },
                "composer_metadata": result.metadata,
            }

            if logger.isEnabledFor(logging.INFO):
                try:
                    fam = None
                    elig = None
                    if asset_class and asset_class in self.config.get("assets", {}):
                        fam = (
                            self.config["assets"][asset_class]
                            .get("composer_params", {})
                            .get("family")
                        )
                        elig = self.config["assets"][asset_class].get("eligible_strategies")
                    logger.info(
                        "composer: asset_class=%s family=%s elig=%s conf=%.3f signal=%.3f regime=%s",
                        asset_class or self.get_asset_class(symbol),
                        fam,
                        ",".join(elig) if isinstance(elig, list) else str(elig),
                        float(result.confidence),
                        float(result.final_signal),
                        result.regime_features.regime_type,
                    )
                except Exception:
                    # Fall back to debug if logging payload fails
                    logger.debug(
                        f"Composer decision: signal={result.final_signal:.4f}, "
                        f"regime={result.regime_features.regime_type}, "
                        f"weights={result.strategy_weights}"
                    )

            return result.final_signal, metadata

        except Exception as e:
            # Log only the first failure per fold with bar index and NaN counts
            if not hasattr(self, "_first_composer_failure_logged"):
                self._first_composer_failure_logged = True

                # Compact dump of NaN counts in inputs
                nan_counts = {}
                if data is not None:
                    for col in data.columns:
                        if col in ["Open", "High", "Low", "Close", "Volume"]:
                            nan_counts[col] = data[col].isna().sum()

                logger.exception(
                    f"First composer failure at bar_idx={current_idx}, symbol={symbol}, "
                    f"NaN_counts={nan_counts}, error={str(e)}"
                )
            else:
                logger.debug(f"Composer failure at bar_idx={current_idx}: {str(e)}")

            return 0.0, {
                "composer_used": False,
                "reason": "composer_exception",
                "error": str(e),
            }

    def evaluate_strategy_performance(
        self, metrics: dict[str, float], symbol: str, asset_class: str | None = None
    ) -> dict[str, Any]:
        """
        Evaluate strategy performance using composite scoring.

        Args:
            metrics: Performance metrics dictionary
            symbol: Trading symbol
            asset_class: Asset class for conditional evaluation

        Returns:
            Evaluation results with composite score
        """
        if not self.use_composer:
            return {"composite_score": 0.0, "composer_used": False}

        try:
            # Calculate composite score
            score = composite_score(metrics, self.weights, self.penalties)

            # Asset-specific adjustments
            if asset_class and asset_class in self.config.get("assets", {}):
                self.config["assets"][asset_class]
                # Could apply asset-specific scoring adjustments here

            return {
                "composite_score": score,
                "composer_used": True,
                "weights_used": {
                    "alpha": self.weights.alpha,
                    "beta": self.weights.beta,
                    "gamma": self.weights.gamma,
                    "delta": self.weights.delta,
                },
                "penalties_applied": {
                    "max_dd_cap": self.penalties.max_dd_cap,
                    "min_trades": self.penalties.min_trades,
                },
            }

        except Exception as e:
            logger.error(f"Error in strategy evaluation: {e}")
            return {"composite_score": 0.0, "composer_used": False, "error": str(e)}

    def get_asset_class(self, symbol: str) -> str:
        """
        Determine asset class from symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Asset class string
        """
        symbol_upper = symbol.upper()

        # Crypto detection
        if any(crypto in symbol_upper for crypto in ["BTC", "ETH", "ADA", "DOT", "LINK", "UNI"]):
            return "crypto"

        # ETF detection
        if any(etf in symbol_upper for etf in ["SPY", "QQQ", "IWM", "TLT", "GLD", "SLV"]):
            return "etf"

        # Stock detection (common patterns)
        if len(symbol) <= 5 and symbol_upper.isalpha():
            return "equity"

        # Default
        return "unknown"

    def should_use_composer(self, symbol: str, asset_class: str | None = None) -> bool:
        """
        Determine if composer should be used for this symbol/asset class.

        Args:
            symbol: Trading symbol
            asset_class: Asset class (if not provided, will be determined)

        Returns:
            True if composer should be used
        """
        if not self.use_composer:
            return False

        if asset_class is None:
            asset_class = self.get_asset_class(symbol)

        # Check if asset class is in eligible strategies
        if asset_class in self.config.get("assets", {}):
            asset_config = self.config["assets"][asset_class]
            if "eligible_strategies" in asset_config:
                return len(asset_config["eligible_strategies"]) > 0

        # Check global eligible strategies
        global_strategies = self.config.get("eligible_strategies", [])
        return len(global_strategies) > 0


def integrate_composer_into_walkforward(
    data: pd.DataFrame, symbol: str, current_idx: int, config: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """
    Integration function for walkforward framework.

    Args:
        data: OHLCV DataFrame
        symbol: Trading symbol
        current_idx: Current index in data
        config: Configuration dictionary

    Returns:
        Tuple of (signal, metadata)
    """
    composer = ComposerIntegration(config)

    if not composer.should_use_composer(symbol):
        return 0.0, {"composer_used": False, "reason": "not_eligible"}

    asset_class = composer.get_asset_class(symbol)
    return composer.get_composer_decision(data, symbol, current_idx, asset_class)


def integrate_composer_into_backtest(
    data: pd.DataFrame, symbol: str, current_date: str, config: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """
    Integration function for backtest engine.

    Args:
        data: OHLCV DataFrame
        symbol: Trading symbol
        current_date: Current trading date
        config: Configuration dictionary

    Returns:
        Tuple of (signal, metadata)
    """
    composer = ComposerIntegration(config)

    if not composer.should_use_composer(symbol):
        return 0.0, {"composer_used": False, "reason": "not_eligible"}

    # Find current index in data
    current_idx = len(data) - 1  # Assume we're at the end

    asset_class = composer.get_asset_class(symbol)
    return composer.get_composer_decision(data, symbol, current_idx, asset_class)
