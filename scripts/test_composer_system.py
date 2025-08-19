#!/usr/bin/env python3
"""
Test script for the composer system.
Demonstrates the composite scoring and composer functionality.
"""

import json
import logging
from datetime import datetime

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from core.composer.contracts import MarketState, StrategyPrediction
from core.composer.registry import build_composer_system
from core.composer.simple_composer import SoftmaxSelector

# Import our new modules
from core.metrics.composite import (
    CompositeWeights,
    composite_score,
    evaluate_strategy_performance,
    load_composite_config,
)
from core.regime.basic import BasicRegimeExtractor


def create_mock_market_data():
    """Create mock market data for testing."""
    # Generate some realistic price data
    np.random.seed(42)
    n_points = 100
    base_price = 100.0

    # Create trending data with some noise
    trend = np.linspace(0, 0.2, n_points)  # 20% trend over period
    noise = np.random.normal(0, 0.01, n_points)  # 1% daily volatility
    prices = base_price * (1 + trend + noise)

    # Create volume data
    volumes = np.random.lognormal(10, 0.5, n_points)

    # Create features
    features = {"rsi": 65.0, "macd": 0.5, "bb_position": 0.7, "atr": 1.2}

    return prices, volumes, features


def test_composite_scoring():
    """Test the composite scoring system."""
    logger.info("Testing composite scoring system...")

    # Test different strategy performances
    strategies = [
        {
            "name": "Perfect Strategy",
            "metrics": {
                "cagr": 0.5,  # 50% CAGR
                "sharpe": 2.0,  # Sharpe 2.0
                "win_rate": 0.8,  # 80% win rate
                "avg_trade_return": 0.01,  # 1% avg return
                "max_dd": 0.1,  # 10% max drawdown
                "trade_count": 500,  # 500 trades
            },
        },
        {
            "name": "Good Strategy",
            "metrics": {
                "cagr": 0.2,  # 20% CAGR
                "sharpe": 1.5,  # Sharpe 1.5
                "win_rate": 0.6,  # 60% win rate
                "avg_trade_return": 0.005,  # 0.5% avg return
                "max_dd": 0.15,  # 15% max drawdown
                "trade_count": 300,  # 300 trades
            },
        },
        {
            "name": "Poor Strategy",
            "metrics": {
                "cagr": -0.1,  # -10% CAGR
                "sharpe": 0.5,  # Sharpe 0.5
                "win_rate": 0.4,  # 40% win rate
                "avg_trade_return": -0.002,  # -0.2% avg return
                "max_dd": 0.3,  # 30% max drawdown
                "trade_count": 100,  # 100 trades
            },
        },
    ]

    # Test with default weights
    logger.info("Testing with default weights:")
    for strategy in strategies:
        score = composite_score(strategy["metrics"])
        logger.info(f"{strategy['name']}: {score:.4f}")

    # Test with custom weights (CAGR-focused)
    logger.info("\nTesting with CAGR-focused weights:")
    weights_cagr = CompositeWeights(alpha=0.7, beta=0.2, gamma=0.05, delta=0.05)
    for strategy in strategies:
        score = composite_score(strategy["metrics"], weights=weights_cagr)
        logger.info(f"{strategy['name']}: {score:.4f}")

    # Test detailed evaluation
    logger.info("\nDetailed evaluation of Good Strategy:")
    evaluation = evaluate_strategy_performance(strategies[1]["metrics"])
    logger.info(f"Composite Score: {evaluation['composite_score']:.4f}")
    logger.info(f"Weighted Sum: {evaluation['weighted_sum']:.4f}")
    logger.info(f"Total Penalty: {evaluation['total_penalty']:.4f}")
    logger.info(f"Components: {evaluation['components']}")
    logger.info(f"Penalties: {evaluation['penalties']}")


def test_regime_extraction():
    """Test the regime extraction system."""
    logger.info("\nTesting regime extraction system...")

    # Create mock market data
    prices, volumes, features = create_mock_market_data()

    # Create market state
    market_state = MarketState(
        prices=prices,
        volumes=volumes,
        features=features,
        timestamp=datetime.now().isoformat(),
        symbol="SPY",
    )

    # Test basic regime extractor
    extractor = BasicRegimeExtractor(lookback=20)
    regime_features = extractor.extract(market_state)

    logger.info(f"Regime Type: {regime_features.regime_type}")
    logger.info(f"Trend Strength: {regime_features.trend_strength:.4f}")
    logger.info(f"Choppiness: {regime_features.choppiness:.4f}")
    logger.info(f"Volatility: {regime_features.volatility:.4f}")
    logger.info(f"Momentum: {regime_features.momentum:.4f}")


def test_composer_system():
    """Test the composer system."""
    logger.info("\nTesting composer system...")

    # Create mock market data
    prices, volumes, features = create_mock_market_data()

    # Create market state
    market_state = MarketState(
        prices=prices,
        volumes=volumes,
        features=features,
        timestamp=datetime.now().isoformat(),
        symbol="SPY",
    )

    # Create simple strategies
    class MockMomentumStrategy:
        def predict(self, market_state):
            return StrategyPrediction(
                signal=0.3,  # Positive momentum signal
                confidence=0.7,
                strategy_name="momentum",
                metadata={},
            )

        @property
        def name(self):
            return "momentum"

    class MockMeanReversionStrategy:
        def predict(self, market_state):
            return StrategyPrediction(
                signal=-0.2,  # Negative mean reversion signal
                confidence=0.6,
                strategy_name="mean_reversion",
                metadata={},
            )

        @property
        def name(self):
            return "mean_reversion"

    class MockBreakoutStrategy:
        def predict(self, market_state):
            return StrategyPrediction(
                signal=0.1,  # Weak breakout signal
                confidence=0.5,
                strategy_name="breakout",
                metadata={},
            )

        @property
        def name(self):
            return "breakout"

    # Create strategies list
    strategies = [
        MockMomentumStrategy(),
        MockMeanReversionStrategy(),
        MockBreakoutStrategy(),
    ]

    # Create regime extractor
    regime_extractor = BasicRegimeExtractor(lookback=20)

    # Create composer
    composer = SoftmaxSelector(temperature=1.0, trend_bias=1.2, chop_bias=1.1, min_confidence=0.1)

    # Test composition
    result = composer.compose(market_state, strategies, regime_extractor)

    logger.info(f"Final Signal: {result.final_signal:.4f}")
    logger.info(f"Confidence: {result.confidence:.4f}")
    logger.info(f"Strategy Weights: {result.strategy_weights}")
    logger.info(f"Regime Type: {result.regime_features.regime_type}")
    logger.info(f"Metadata: {result.metadata}")


def test_config_loading():
    """Test configuration loading."""
    logger.info("\nTesting configuration loading...")

    # Load the composer config
    with open("config/composer_config.json") as f:
        config = json.load(f)

    # Test loading composite weights and penalties
    weights, penalties = load_composite_config(config)

    logger.info(
        f"Loaded Weights: alpha={weights.alpha:.2f}, beta={weights.beta:.2f}, "
        f"gamma={weights.gamma:.2f}, delta={weights.delta:.2f}"
    )
    logger.info(
        f"Loaded Penalties: max_dd_cap={penalties.max_dd_cap:.2f}, "
        f"min_trades={penalties.min_trades}"
    )

    # Test building composer system from config
    try:
        strategies, regime_extractor, composer = build_composer_system(config)
        logger.info(f"Built composer system with {len(strategies)} strategies")
        logger.info(f"Regime extractor: {regime_extractor.name}")
        logger.info(f"Composer: {composer.name}")
    except Exception as e:
        logger.warning(f"Could not build composer system: {e}")


def main():
    """Main test function."""
    logger.info("Starting composer system tests...")

    try:
        # Test composite scoring
        test_composite_scoring()

        # Test regime extraction
        test_regime_extraction()

        # Test composer system
        test_composer_system()

        # Test config loading
        test_config_loading()

        logger.info("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
