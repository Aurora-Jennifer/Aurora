"""
Composer Contracts
Defines protocols for strategies, regime extractors, and composers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MarketState:
    """Market state information."""

    prices: np.ndarray
    volumes: np.ndarray
    features: dict[str, float]
    timestamp: str
    symbol: str


@dataclass
class RegimeFeatures:
    """Regime features extracted from market state."""

    trend_strength: float
    choppiness: float
    volatility: float
    momentum: float
    regime_type: str  # 'trend', 'chop', 'volatile', 'unknown'


@dataclass
class StrategyPrediction:
    """Strategy prediction output."""

    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    strategy_name: str
    metadata: dict[str, Any]


@dataclass
class ComposerOutput:
    """Composer output with blended predictions."""

    final_signal: float  # -1 to 1
    strategy_weights: dict[str, float]
    regime_features: RegimeFeatures
    confidence: float
    metadata: dict[str, Any]


class Strategy(ABC):
    """Base strategy protocol."""

    @abstractmethod
    def predict(self, market_state: MarketState) -> StrategyPrediction:
        """Generate prediction for given market state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Strategy description."""


class RegimeExtractor(ABC):
    """Base regime extractor protocol."""

    @abstractmethod
    def extract(self, market_state: MarketState) -> RegimeFeatures:
        """Extract regime features from market state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Extractor name."""


class Composer(ABC):
    """Base composer protocol."""

    @abstractmethod
    def compose(
        self,
        market_state: MarketState,
        strategies: list[Strategy],
        regime_extractor: RegimeExtractor,
    ) -> ComposerOutput:
        """Compose strategy predictions into final signal."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Composer name."""


# Concrete implementations for common patterns


class SimpleStrategy(Strategy):
    """Simple strategy wrapper for existing strategies."""

    def __init__(self, name: str, predict_func, description: str = ""):
        self._name = name
        self._predict_func = predict_func
        self._description = description

    def predict(self, market_state: MarketState) -> StrategyPrediction:
        """Generate prediction using wrapped function."""
        signal = self._predict_func(market_state)
        return StrategyPrediction(
            signal=signal,
            confidence=0.5,  # Default confidence
            strategy_name=self.name,
            metadata={},
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description


class BasicRegimeExtractor(RegimeExtractor):
    """Basic regime extractor using simple technical indicators."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def extract(self, market_state: MarketState) -> RegimeFeatures:
        """Extract basic regime features."""
        prices = market_state.prices

        if len(prices) < self.lookback:
            return RegimeFeatures(
                trend_strength=0.0,
                choppiness=0.0,
                volatility=0.0,
                momentum=0.0,
                regime_type="unknown",
            )

        # Import safety functions
        from ..utils import _last, _safe_len

        # Check if we have enough data
        if _safe_len(prices) < self.lookback + 1:
            return RegimeFeatures(
                trend_strength=0.0,
                choppiness=0.0,
                volatility=0.0,
                momentum=0.0,
                regime_type="unknown",
            )

        # Calculate basic features with safe access
        returns = np.diff(np.log(prices))
        volatility = 0.0 if _safe_len(returns) < self.lookback else np.std(returns[-self.lookback:])

        # Trend strength (linear regression slope)
        x = np.arange(self.lookback)
        y = prices[-self.lookback :]
        slope = np.polyfit(x, y, 1)[0]
        mean_price = np.mean(prices[-self.lookback :])
        trend_strength = np.tanh(slope / mean_price) if mean_price > 0 else 0.0

        # Choppiness (ADX-like measure)
        high_low_range = np.max(prices[-self.lookback :]) - np.min(prices[-self.lookback :])
        path_length = np.sum(np.abs(np.diff(prices[-self.lookback :])))
        choppiness = 1.0 - (high_low_range / path_length) if path_length > 0 else 0.0

        # Momentum with safe access
        start_price = prices[-self.lookback] if _safe_len(prices) > self.lookback else None
        end_price = _last(prices)
        momentum = (end_price - start_price) / start_price if start_price is not None and end_price is not None and start_price > 0 else 0.0

        # Determine regime type with safe percentile calculation
        if abs(trend_strength) > 0.3:
            regime_type = "trend"
        elif choppiness > 0.7:
            regime_type = "chop"
        elif _safe_len(returns) > 0 and volatility > np.percentile(returns, 75):
            regime_type = "volatile"
        else:
            regime_type = "unknown"

        return RegimeFeatures(
            trend_strength=trend_strength,
            choppiness=choppiness,
            volatility=volatility,
            momentum=momentum,
            regime_type=regime_type,
        )

    @property
    def name(self) -> str:
        return "basic_kpis"


class SoftmaxComposer(Composer):
    """Softmax-based composer for blending strategy predictions."""

    def __init__(
        self,
        temperature: float = 1.0,
        trend_bias: float = 1.0,
        chop_bias: float = 1.0,
        min_confidence: float = 0.1,
    ):
        self.temperature = temperature
        self.trend_bias = trend_bias
        self.chop_bias = chop_bias
        self.min_confidence = min_confidence

    def compose(
        self,
        market_state: MarketState,
        strategies: list[Strategy],
        regime_extractor: RegimeExtractor,
    ) -> ComposerOutput:
        """Compose predictions using softmax weighting."""
        # Extract regime features
        regime_features = regime_extractor.extract(market_state)

        # Get predictions from all strategies
        predictions = []
        for strategy in strategies:
            pred = strategy.predict(market_state)
            predictions.append(pred)

        # Calculate weights based on regime and confidence
        weights = self._calculate_weights(predictions, regime_features)

        # Blend predictions
        final_signal = 0.0
        strategy_weights = {}

        for i, pred in enumerate(predictions):
            weight = weights[i]
            final_signal += pred.signal * weight
            strategy_weights[pred.strategy_name] = weight

        # Calculate overall confidence
        confidence = np.mean([pred.confidence for pred in predictions])

        return ComposerOutput(
            final_signal=np.clip(final_signal, -1, 1),
            strategy_weights=strategy_weights,
            regime_features=regime_features,
            confidence=confidence,
            metadata={
                "temperature": self.temperature,
                "num_strategies": len(strategies),
            },
        )

    def _calculate_weights(
        self, predictions: list[StrategyPrediction], regime_features: RegimeFeatures
    ) -> list[float]:
        """Calculate softmax weights for strategies."""
        # Base scores on confidence and regime alignment
        scores = []

        for pred in predictions:
            # Start with confidence
            score = pred.confidence

            # Adjust based on regime with configurable biases
            if regime_features.regime_type == "trend":
                # Favor momentum strategies in trending markets
                if "momentum" in pred.strategy_name.lower():
                    score += 0.2 * self.trend_bias
            elif (regime_features.regime_type == "chop" and
                  ("mean" in pred.strategy_name.lower() or "reversion" in pred.strategy_name.lower())):
                    score += 0.2 * self.chop_bias

            scores.append(score)

        # Apply softmax
        scores = np.array(scores) / self.temperature
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        weights = exp_scores / np.sum(exp_scores)

        return weights.tolist()

    @property
    def name(self) -> str:
        return "softmax_blender"
