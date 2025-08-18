#!/usr/bin/env python3
"""
Simple demo of asset-class-aware composer functionality.
"""

import json
import logging

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import composer integration
from core.engine.composer_integration import ComposerIntegration


def create_simple_data(symbol: str, days: int = 50) -> pd.DataFrame:
    """Create simple synthetic data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")

    # Asset-specific characteristics
    if "BTC" in symbol.upper():
        base_price = 50000
        volatility = 0.03
    elif "SPY" in symbol.upper():
        base_price = 400
        volatility = 0.015
    else:
        base_price = 100
        volatility = 0.02

    # Generate price series
    returns = np.random.normal(0, volatility, days)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.random.lognormal(10, 0.5, days),
        }
    )

    return data


def main():
    """Main demo function."""
    logger.info("=== Asset-Class-Aware Composer Demo ===")

    # Load configuration
    with open("config/composer_config.json") as f:
        config = json.load(f)

    # Test asset class detection
    logger.info("\n1. Asset Class Detection:")
    composer = ComposerIntegration(config)

    test_symbols = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "TSLA", "AAPL"]
    for symbol in test_symbols:
        asset_class = composer.get_asset_class(symbol)
        logger.info(f"  {symbol} -> {asset_class}")

    # Test composer decisions for different asset classes
    logger.info("\n2. Composer Decisions by Asset Class:")

    test_cases = [("SPY", "etf"), ("BTC-USD", "crypto"), ("TSLA", "equity")]

    for symbol, expected_asset_class in test_cases:
        logger.info(f"\n  --- {symbol} ({expected_asset_class}) ---")

        # Create data
        data = create_simple_data(symbol, days=50)

        # Test at different points
        for idx in [25, 35, 45]:
            signal, metadata = composer.get_composer_decision(
                data, symbol, idx, expected_asset_class
            )

            logger.info(f"    Point {idx}: Signal={signal:.4f}")
            if metadata.get("composer_used", False):
                logger.info(f"      Regime: {metadata.get('regime_type', 'unknown')}")
                logger.info(
                    f"      Strategy Weights: {metadata.get('strategy_weights', {})}"
                )
            else:
                logger.info(
                    f"      Composer not used: {metadata.get('reason', 'unknown')}"
                )

    # Test performance evaluation
    logger.info("\n3. Performance Evaluation by Asset Class:")

    test_performances = [
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
    ]

    for symbol, asset_class, metrics in test_performances:
        logger.info(f"\n  --- {symbol} ({asset_class}) ---")

        evaluation = composer.evaluate_strategy_performance(
            metrics, symbol, asset_class
        )

        logger.info(
            f"    Composite Score: {evaluation.get('composite_score', 0.0):.4f}"
        )
        logger.info(f"    Weights Used: {evaluation.get('weights_used', {})}")

    # Show configuration overrides
    logger.info("\n4. Asset-Specific Configuration:")

    for asset_class, asset_config in config.get("assets", {}).items():
        logger.info(f"\n  --- {asset_class} ---")
        logger.info(
            f"    Eligible Strategies: {asset_config.get('eligible_strategies', [])}"
        )
        logger.info(f"    Composer Params: {asset_config.get('composer_params', {})}")

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
