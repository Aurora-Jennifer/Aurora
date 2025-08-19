#!/usr/bin/env python3
"""
Test script for asset-class-aware composer system.
Demonstrates how the composer chooses different strategies based on asset class.
"""

import json
import logging

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import composer integration
from core.engine.composer_integration import ComposerIntegration


def create_asset_specific_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Create synthetic data with asset-specific characteristics."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")

    # Asset-specific characteristics
    if "BTC" in symbol.upper():
        # Crypto: high volatility, trending
        base_price = 50000
        volatility = 0.03
        trend = 0.001  # Slight upward trend
    elif "SPY" in symbol.upper():
        # ETF: moderate volatility, mean-reverting
        base_price = 400
        volatility = 0.015
        trend = 0.0005
    elif "TSLA" in symbol.upper():
        # Stock: high volatility, trending
        base_price = 200
        volatility = 0.025
        trend = 0.0008
    else:
        # Default
        base_price = 100
        volatility = 0.02
        trend = 0.0003

    # Generate price series
    returns = np.random.normal(trend, volatility, days)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.normal(0, 0.001, days)),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.002, days))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.002, days))),
            "Close": prices,
            "Volume": np.random.lognormal(10, 0.5, days),
        }
    )

    return data


def test_asset_class_detection():
    """Test asset class detection."""
    logger.info("Testing asset class detection...")

    composer = ComposerIntegration({})

    test_symbols = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "TSLA", "AAPL", "UNKNOWN"]

    for symbol in test_symbols:
        asset_class = composer.get_asset_class(symbol)
        logger.info(f"{symbol} -> {asset_class}")


def test_asset_class_composer_decisions():
    """Test composer decisions for different asset classes."""
    logger.info("\nTesting asset class composer decisions...")

    # Load configuration
    with open("config/composer_config.json") as f:
        config = json.load(f)

    composer = ComposerIntegration(config)

    # Test symbols for different asset classes
    test_cases = [
        ("SPY", "etf"),
        ("BTC-USD", "crypto"),
        ("TSLA", "equity"),
        ("UNKNOWN", "unknown"),
    ]

    for symbol, expected_asset_class in test_cases:
        logger.info(f"\n--- Testing {symbol} ({expected_asset_class}) ---")

        # Create asset-specific data
        data = create_asset_specific_data(symbol, days=100)

        # Test composer decision at different points
        test_points = [50, 75, 99]  # Different indices in the data

        for idx in test_points:
            signal, metadata = composer.get_composer_decision(
                data, symbol, idx, expected_asset_class
            )

            logger.info(f"  Point {idx}: Signal={signal:.4f}")
            if metadata.get("composer_used", False):
                logger.info(f"    Regime: {metadata.get('regime_type', 'unknown')}")
                logger.info(f"    Strategy Weights: {metadata.get('strategy_weights', {})}")
                logger.info(f"    Confidence: {metadata.get('confidence', 0.0):.3f}")


def test_walkforward_integration():
    """Test walkforward integration with asset classes."""
    logger.info("\nTesting walkforward integration...")

    # Load configuration
    with open("config/composer_config.json") as f:
        config = json.load(f)

    composer = ComposerIntegration(config)

    # Test different symbols
    test_symbols = ["SPY", "BTC-USD", "TSLA"]

    for symbol in test_symbols:
        logger.info(f"\n--- Walkforward test for {symbol} ---")

        # Create data
        data = create_asset_specific_data(symbol, days=200)

        # Simulate walkforward steps
        for step in range(150, 200, 10):
            data_up_to_step = data.iloc[:step]

            # Get composer decision
            signal, metadata = composer.get_composer_decision(data_up_to_step, symbol, step - 1)

            if metadata.get("composer_used", False):
                logger.info(
                    f"  Step {step}: Signal={signal:.4f}, "
                    f"Regime={metadata.get('regime_type', 'unknown')}, "
                    f"Weights={metadata.get('strategy_weights', {})}"
                )


def test_performance_evaluation():
    """Test performance evaluation for different asset classes."""
    logger.info("\nTesting performance evaluation...")

    # Load configuration
    with open("config/composer_config.json") as f:
        config = json.load(f)

    composer = ComposerIntegration(config)

    # Test different performance scenarios
    test_cases = [
        (
            "SPY",
            "etf",
            {
                "cagr": 0.15,
                "sharpe": 1.2,
                "win_rate": 0.6,
                "avg_trade_return": 0.005,
                "max_dd": 0.1,
                "trade_count": 250,
            },
        ),
        (
            "BTC-USD",
            "crypto",
            {
                "cagr": 0.4,
                "sharpe": 1.8,
                "win_rate": 0.55,
                "avg_trade_return": 0.008,
                "max_dd": 0.25,
                "trade_count": 300,
            },
        ),
        (
            "TSLA",
            "equity",
            {
                "cagr": 0.25,
                "sharpe": 1.5,
                "win_rate": 0.58,
                "avg_trade_return": 0.006,
                "max_dd": 0.18,
                "trade_count": 280,
            },
        ),
    ]

    for symbol, asset_class, metrics in test_cases:
        logger.info(f"\n--- Performance evaluation for {symbol} ({asset_class}) ---")

        evaluation = composer.evaluate_strategy_performance(metrics, symbol, asset_class)

        logger.info(f"  Composite Score: {evaluation.get('composite_score', 0.0):.4f}")
        logger.info(f"  Weights Used: {evaluation.get('weights_used', {})}")
        logger.info(f"  Penalties Applied: {evaluation.get('penalties_applied', {})}")


def test_configuration_overrides():
    """Test asset-specific configuration overrides."""
    logger.info("\nTesting configuration overrides...")

    # Load configuration
    with open("config/composer_config.json") as f:
        config = json.load(f)

    # Test asset-specific settings
    for asset_class, asset_config in config.get("assets", {}).items():
        logger.info(f"\n--- Configuration for {asset_class} ---")
        logger.info(f"  Eligible Strategies: {asset_config.get('eligible_strategies', [])}")
        logger.info(f"  Composer Params: {asset_config.get('composer_params', {})}")


def main():
    """Main test function."""
    logger.info("Starting asset-class-aware composer tests...")

    try:
        # Test asset class detection
        test_asset_class_detection()

        # Test composer decisions
        test_asset_class_composer_decisions()

        # Test walkforward integration
        test_walkforward_integration()

        # Test performance evaluation
        test_performance_evaluation()

        # Test configuration overrides
        test_configuration_overrides()

        logger.info("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
