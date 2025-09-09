"""
Simple Composer Implementation
MVP composer with softmax-based strategy selection.
"""

import logging

import numpy as np

from .contracts import (
    Composer,
    ComposerOutput,
    MarketState,
    RegimeExtractor,
    RegimeFeatures,
    Strategy,
    StrategyPrediction,
)

logger = logging.getLogger(__name__)


class SoftmaxSelector(Composer):
    """
    Simple softmax-based strategy selector.

    Uses regime features to weight strategies and blend their predictions.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        trend_bias: float = 1.0,
        chop_bias: float = 1.0,
        min_confidence: float = 0.1,
    ):
        """
        Initialize softmax selector.

        Args:
            temperature: Softmax temperature (lower = more deterministic)
            trend_bias: Bias multiplier for trend-following strategies
            chop_bias: Bias multiplier for mean-reversion strategies
            min_confidence: Minimum confidence threshold
        """
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
        """
        Compose strategy predictions using softmax weighting.

        Args:
            market_state: Current market state
            strategies: List of available strategies
            regime_extractor: Regime feature extractor

        Returns:
            ComposerOutput with blended prediction
        """
        # Check if we have enough data for regime extraction
        if len(market_state.prices) < 50:  # Minimum required for regime extractor
            logger.debug("Insufficient data for regime extraction")
            return ComposerOutput(
                final_signal=0.0,
                strategy_weights={},
                regime_features=RegimeFeatures(
                    trend_strength=0.0,
                    choppiness=0.0,
                    volatility=0.0,
                    momentum=0.0,
                    regime_type="unknown",
                ),
                confidence=0.0,
                metadata={"error": "insufficient_data_for_regime"},
            )

        # Extract regime features
        regime_features = regime_extractor.extract(market_state)

        # Get predictions from all strategies
        predictions = []
        for strategy in strategies:
            try:
                pred = strategy.predict(market_state)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
                continue

        if not predictions:
            logger.warning("No valid predictions from strategies")
            return ComposerOutput(
                final_signal=0.0,
                strategy_weights={},
                regime_features=regime_features,
                confidence=0.0,
                metadata={"error": "no_valid_predictions"},
            )

        # Calculate strategy weights based on regime
        weights = self._calculate_weights(predictions, regime_features)

        # Blend predictions
        final_signal = 0.0
        strategy_weights = {}
        total_weight = 0.0

        for i, pred in enumerate(predictions):
            weight = weights[i]
            final_signal += pred.signal * weight
            strategy_weights[pred.strategy_name] = weight
            total_weight += weight

        # Normalize signal if weights don't sum to 1
        if total_weight > 0:
            final_signal /= total_weight

        # Calculate overall confidence
        confidence = np.mean([pred.confidence for pred in predictions])
        confidence = max(confidence, self.min_confidence)

        # Log composition details
        logger.debug(f"Regime: {regime_features.regime_type}")
        logger.debug(f"Strategy weights: {strategy_weights}")
        logger.debug(f"Final signal: {final_signal:.4f}")

        return ComposerOutput(
            final_signal=np.clip(final_signal, -1, 1),
            strategy_weights=strategy_weights,
            regime_features=regime_features,
            confidence=confidence,
            metadata={
                "temperature": self.temperature,
                "num_strategies": len(predictions),
                "regime_type": regime_features.regime_type,
            },
        )

    def _calculate_weights(
        self, predictions: list[StrategyPrediction], regime_features: RegimeFeatures
    ) -> list[float]:
        """
        Calculate softmax weights for strategies based on regime.

        Args:
            predictions: List of strategy predictions
            regime_features: Extracted regime features

        Returns:
            List of weights for each strategy
        """
        scores = []

        for pred in predictions:
            # Start with base confidence
            score = pred.confidence

            # Apply regime-specific biases
            if regime_features.regime_type == "trend":
                # Favor momentum/trend-following strategies
                if self._is_trend_strategy(pred.strategy_name):
                    score *= self.trend_bias
                elif self._is_mean_reversion_strategy(pred.strategy_name):
                    score *= 1.0 / self.trend_bias

            elif regime_features.regime_type == "chop":
                # Favor mean reversion strategies
                if self._is_mean_reversion_strategy(pred.strategy_name):
                    score *= self.chop_bias
                elif self._is_trend_strategy(pred.strategy_name):
                    score *= 1.0 / self.chop_bias

            # Adjust based on trend strength
            if abs(regime_features.trend_strength) > 0.5 and self._is_trend_strategy(pred.strategy_name):
                score *= 1.0 + abs(regime_features.trend_strength)

            # Adjust based on choppiness
            if regime_features.choppiness > 0.7 and self._is_mean_reversion_strategy(pred.strategy_name):
                score *= 1.0 + regime_features.choppiness

            scores.append(score)

        # Apply softmax
        scores = np.array(scores) / self.temperature
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        weights = exp_scores / np.sum(exp_scores)

        return weights.tolist()

    def _is_trend_strategy(self, strategy_name: str) -> bool:
        """Check if strategy is trend-following."""
        trend_keywords = ["momentum", "trend", "breakout", "follow"]
        return any(keyword in strategy_name.lower() for keyword in trend_keywords)

    def _is_mean_reversion_strategy(self, strategy_name: str) -> bool:
        """Check if strategy is mean reversion."""
        reversion_keywords = ["mean", "reversion", "contrarian", "oscillator"]
        return any(keyword in strategy_name.lower() for keyword in reversion_keywords)

    @property
    def name(self) -> str:
        return "softmax_selector"


class ThresholdSelector(Composer):
    """
    Threshold-based strategy selector.

    Uses regime features to select the best strategy based on thresholds.
    """

    def __init__(
        self,
        trend_threshold: float = 0.3,
        chop_threshold: float = 0.7,
        volatility_threshold: float = 0.02,
    ):
        """
        Initialize threshold selector.

        Args:
            trend_threshold: Threshold for trend regime
            chop_threshold: Threshold for choppy regime
            volatility_threshold: Threshold for volatile regime
        """
        self.trend_threshold = trend_threshold
        self.chop_threshold = chop_threshold
        self.volatility_threshold = volatility_threshold

    def compose(
        self,
        market_state: MarketState,
        strategies: list[Strategy],
        regime_extractor: RegimeExtractor,
    ) -> ComposerOutput:
        """
        Select best strategy based on regime thresholds.

        Args:
            market_state: Current market state
            strategies: List of available strategies
            regime_extractor: Regime feature extractor

        Returns:
            ComposerOutput with selected strategy prediction
        """
        # Extract regime features
        regime_features = regime_extractor.extract(market_state)

        # Select best strategy based on regime
        selected_strategy = self._select_strategy(strategies, regime_features)

        if not selected_strategy:
            logger.warning("No suitable strategy found for regime")
            return ComposerOutput(
                final_signal=0.0,
                strategy_weights={},
                regime_features=regime_features,
                confidence=0.0,
                metadata={"error": "no_suitable_strategy"},
            )

        # Get prediction from selected strategy
        try:
            pred = selected_strategy.predict(market_state)
        except Exception as e:
            logger.warning(f"Selected strategy {selected_strategy.name} failed: {e}")
            return ComposerOutput(
                final_signal=0.0,
                strategy_weights={},
                regime_features=regime_features,
                confidence=0.0,
                metadata={"error": "strategy_failed"},
            )

        # Create weights dict (only selected strategy has weight 1.0)
        strategy_weights = {pred.strategy_name: 1.0}

        logger.debug(f"Selected strategy: {selected_strategy.name}")
        logger.debug(f"Regime: {regime_features.regime_type}")

        return ComposerOutput(
            final_signal=pred.signal,
            strategy_weights=strategy_weights,
            regime_features=regime_features,
            confidence=pred.confidence,
            metadata={
                "selected_strategy": selected_strategy.name,
                "regime_type": regime_features.regime_type,
                "selection_method": "threshold",
            },
        )

    def _select_strategy(
        self, strategies: list[Strategy], regime_features: RegimeFeatures
    ) -> Strategy | None:
        """
        Select the best strategy for the current regime.

        Args:
            strategies: Available strategies
            regime_features: Current regime features

        Returns:
            Selected strategy or None
        """
        if regime_features.regime_type == "trend":
            # Select trend-following strategy
            for strategy in strategies:
                if self._is_trend_strategy(strategy.name):
                    return strategy

        elif regime_features.regime_type == "chop":
            # Select mean reversion strategy
            for strategy in strategies:
                if self._is_mean_reversion_strategy(strategy.name):
                    return strategy

        elif regime_features.regime_type == "volatile":
            # Select low-risk strategy or use momentum
            for strategy in strategies:
                if "momentum" in strategy.name.lower():
                    return strategy

        # Default: return first strategy
        return strategies[0] if strategies else None

    def _is_trend_strategy(self, strategy_name: str) -> bool:
        """Check if strategy is trend-following."""
        trend_keywords = ["momentum", "trend", "breakout", "follow"]
        return any(keyword in strategy_name.lower() for keyword in trend_keywords)

    def _is_mean_reversion_strategy(self, strategy_name: str) -> bool:
        """Check if strategy is mean reversion."""
        reversion_keywords = ["mean", "reversion", "contrarian", "oscillator"]
        return any(keyword in strategy_name.lower() for keyword in reversion_keywords)

    @property
    def name(self) -> str:
        return "threshold_selector"
