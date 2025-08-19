"""
Composer Registry
Registration and factory system for strategies, regime extractors, and composers.
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

# Import safety functions
from ..utils import _last, _safe_len
from .contracts import Composer, RegimeExtractor, Strategy, BasicRegimeExtractor, SoftmaxComposer

logger = logging.getLogger(__name__)


class Registry:
    """Registry for components."""

    def __init__(self):
        self._strategies: dict[str, type[Strategy]] = {}
        self._regime_extractors: dict[str, type[RegimeExtractor]] = {}
        self._composers: dict[str, type[Composer]] = {}
        self._factories: dict[str, Callable] = {}

    def register_strategy(self, name: str):
        """Decorator to register a strategy."""

        def decorator(cls: type[Strategy]):
            if not issubclass(cls, Strategy):
                raise ValueError(f"{cls.__name__} must inherit from Strategy")

            self._strategies[name] = cls
            logger.info(f"Registered strategy: {name}")
            return cls

        return decorator

    def register_regime_extractor(self, name: str):
        """Decorator to register a regime extractor."""

        def decorator(cls: type[RegimeExtractor]):
            if not issubclass(cls, RegimeExtractor):
                raise ValueError(f"{cls.__name__} must inherit from RegimeExtractor")

            self._regime_extractors[name] = cls
            logger.info(f"Registered regime extractor: {name}")
            return cls

        return decorator

    def register_composer(self, name: str):
        """Decorator to register a composer."""

        def decorator(cls: type[Composer]):
            if not issubclass(cls, Composer):
                raise ValueError(f"{cls.__name__} must inherit from Composer")

            self._composers[name] = cls
            logger.info(f"Registered composer: {name}")
            return cls

        return decorator

    def register_factory(self, name: str, factory_func: Callable):
        """Register a factory function for custom instantiation."""
        self._factories[name] = factory_func
        logger.info(f"Registered factory: {name}")

    def build_strategy(self, name: str, **kwargs) -> Strategy | None:
        """Build a strategy instance."""
        if name in self._strategies:
            cls = self._strategies[name]
            return cls(**kwargs)
        elif name in self._factories:
            factory = self._factories[name]
            return factory(**kwargs)
        else:
            logger.error(f"Strategy not found: {name}")
            return None

    def build_regime_extractor(self, name: str, **kwargs) -> RegimeExtractor | None:
        """Build a regime extractor instance."""
        if name in self._regime_extractors:
            cls = self._regime_extractors[name]
            return cls(**kwargs)
        elif name in self._factories:
            factory = self._factories[name]
            return factory(**kwargs)
        else:
            logger.error(f"Regime extractor not found: {name}")
            return None

    def build_composer(self, name: str, **kwargs) -> Composer | None:
        """Build a composer instance."""
        if name in self._composers:
            cls = self._composers[name]
            return cls(**kwargs)
        elif name in self._factories:
            factory = self._factories[name]
            return factory(**kwargs)
        else:
            logger.error(f"Composer not found: {name}")
            return None

    def list_strategies(self) -> list[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())

    def list_regime_extractors(self) -> list[str]:
        """List all registered regime extractor names."""
        return list(self._regime_extractors.keys())

    def list_composers(self) -> list[str]:
        """List all registered composer names."""
        return list(self._composers.keys())

    def list_factories(self) -> list[str]:
        """List all registered factory names."""
        return list(self._factories.keys())


# Global registry instance
registry = Registry()


# Convenience decorators
def register_strategy(name: str):
    """Decorator to register a strategy in the global registry."""
    return registry.register_strategy(name)


def register_regime_extractor(name: str):
    """Decorator to register a regime extractor in the global registry."""
    return registry.register_regime_extractor(name)


def register_composer(name: str):
    """Decorator to register a composer in the global registry."""
    return registry.register_composer(name)


def build_strategy(name: str, **kwargs) -> Strategy | None:
    """Build a strategy from the global registry."""
    return registry.build_strategy(name, **kwargs)


def build_regime_extractor(name: str, **kwargs) -> RegimeExtractor | None:
    """Build a regime extractor from the global registry."""
    return registry.build_regime_extractor(name, **kwargs)


def build_composer(name: str, **kwargs) -> Composer | None:
    """Build a composer from the global registry."""
    return registry.build_composer(name, **kwargs)


# Register built-in components

registry.register_regime_extractor("basic_kpis")(BasicRegimeExtractor)
registry.register_composer("softmax_blender")(SoftmaxComposer)


# Strategy adapters for existing strategies


def create_momentum_strategy_adapter():
    """Create a momentum strategy adapter."""

    def momentum_predict(market_state):
        # Simple momentum strategy
        prices = market_state.prices
        if len(prices) < 20:
            return 0.0

        # Safe access to prices
        if _safe_len(prices) == 0:
            return 0.0

        # Calculate momentum with safe slicing
        if _safe_len(prices) < 20:
            return 0.0

        short_ma = np.mean(prices[-10:]) if _safe_len(prices) >= 10 else 0.0
        long_ma = np.mean(prices[-20:]) if _safe_len(prices) >= 20 else 0.0

        if short_ma > long_ma:
            return 0.5  # Positive momentum
        else:
            return -0.5  # Negative momentum

    from .contracts import SimpleStrategy

    return SimpleStrategy(predict_func=momentum_predict, name="momentum")


def create_mean_reversion_strategy_adapter():
    """Create a mean reversion strategy adapter."""

    def mean_reversion_predict(market_state):
        # Simple mean reversion strategy
        prices = market_state.prices
        if len(prices) < 20:
            return 0.0

        # Safe access to last price
        if _safe_len(prices) == 0:
            return 0.0

        # Calculate mean reversion signal
        current_price = _last(prices)
        if current_price is None:
            return 0.0
        mean_price = np.mean(prices[-20:])
        std_price = np.std(prices[-20:])

        if std_price == 0:
            return 0.0

        z_score = (current_price - mean_price) / std_price

        if z_score > 1.0:
            return -0.5  # Overbought, sell
        elif z_score < -1.0:
            return 0.5  # Oversold, buy
        else:
            return 0.0  # Neutral

    from .contracts import SimpleStrategy

    return SimpleStrategy(predict_func=mean_reversion_predict, name="mean_reversion")


def create_breakout_strategy_adapter():
    """Create a breakout strategy adapter."""

    def breakout_predict(market_state):
        # Simple breakout strategy
        prices = market_state.prices
        if len(prices) < 20:
            return 0.0

        # Safe access to last price
        if _safe_len(prices) == 0:
            return 0.0

        # Calculate breakout signal
        current_price = _last(prices)
        if current_price is None:
            return 0.0
        high_20 = np.max(prices[-20:])
        low_20 = np.min(prices[-20:])

        if current_price >= high_20 * 0.99:  # Near high
            return 0.5  # Breakout up
        elif current_price <= low_20 * 1.01:  # Near low
            return -0.5  # Breakout down
        else:
            return 0.0  # No breakout

    from .contracts import SimpleStrategy

    return SimpleStrategy(predict_func=breakout_predict, name="breakout")


# Register strategy adapters
registry.register_factory("momentum", create_momentum_strategy_adapter)
registry.register_factory("mean_reversion", create_mean_reversion_strategy_adapter)
registry.register_factory("breakout", create_breakout_strategy_adapter)


def build_composer_system(
    config: dict[str, Any],
) -> tuple[list[Strategy], RegimeExtractor, Composer]:
    """
    Build a complete composer system from configuration.

    Args:
        config: Configuration dictionary with composer settings

    Returns:
        Tuple of (strategies, regime_extractor, composer)
    """
    # Build strategies with filtering
    strategy_names = config.get("eligible_strategies", ["momentum", "mean_reversion"])
    strategies = []
    missing_strategies = []

    # Filter to only registered strategies
    registered_strategies = registry.list_strategies() + registry.list_factories()

    for name in strategy_names:
        if name in registered_strategies:
            strategy = build_strategy(name)
            if strategy:
                strategies.append(strategy)
            else:
                logger.warning(f"Failed to build strategy: {name}")
                missing_strategies.append(name)
        else:
            logger.warning(f"Strategy not registered: {name}")
            missing_strategies.append(name)

    # Warn about missing strategies
    if missing_strategies:
        logger.warning(f"Missing strategies: {missing_strategies}")

    # Ensure we have at least 2 strategies
    if len(strategies) < 2:
        raise ValueError(
            f"Insufficient strategies: need >= 2, got {len(strategies)}. "
            f"Available: {registered_strategies}"
        )

    # Build regime extractor
    extractor_name = config.get("regime_extractor", "basic_kpis")
    regime_extractor = build_regime_extractor(extractor_name)

    if not regime_extractor:
        logger.error(f"Failed to build regime extractor: {extractor_name}")
        raise ValueError(f"Regime extractor not found: {extractor_name}")

    # Build composer
    composer_name = config.get("composer_mode", "softmax_blender")
    composer_params = config.get("composer_params", {})
    composer = build_composer(composer_name, **composer_params)

    if not composer:
        logger.error(f"Failed to build composer: {composer_name}")
        raise ValueError(f"Composer not found: {composer_name}")

    # Store strategies in composer for validation
    composer.strategies = strategies

    logger.info(
        f"Built composer system with {len(strategies)} strategies: {[s.name for s in strategies]}"
    )
    logger.info(f"Regime extractor: {regime_extractor.name}")
    logger.info(f"Composer: {composer.name}")

    return strategies, regime_extractor, composer
