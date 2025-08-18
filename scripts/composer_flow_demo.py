#!/usr/bin/env python3
"""
Demo showing the exact flow of the composer system.
Shows both strategy selection AND weight optimization.
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

# Import composer components
from core.engine.composer_integration import ComposerIntegration
from core.metrics.composite import CompositeWeights, composite_score


def demonstrate_composer_flow():
    """Demonstrate the complete composer flow."""
    logger.info("=== Composer System Flow Demo ===\n")

    # Load configuration
    with open("config/composer_config.json") as f:
        config = json.load(f)

    # Step 1: Show the two-level system
    logger.info("🎯 TWO-LEVEL COMPOSER SYSTEM:")
    logger.info("")
    logger.info("LEVEL 1: Strategy Selection (Model Selection)")
    logger.info("  ├── Asset Class Detection (crypto/etf/equity)")
    logger.info("  ├── Market Regime Detection (trend/chop/volatile)")
    logger.info("  ├── Strategy Eligibility Check")
    logger.info("  └── Strategy Weighting (softmax)")
    logger.info("")
    logger.info("LEVEL 2: Performance Weight Optimization")
    logger.info("  ├── CAGR Weight (alpha)")
    logger.info("  ├── Sharpe Weight (beta)")
    logger.info("  ├── Win Rate Weight (gamma)")
    logger.info("  └── Avg Trade Return Weight (delta)")
    logger.info("")

    # Step 2: Show asset-specific strategy selection
    logger.info("📊 ASSET-SPECIFIC STRATEGY SELECTION:")

    for asset_class, asset_config in config.get("assets", {}).items():
        logger.info(f"\n  {asset_class.upper()}:")
        logger.info(
            f"    Eligible Strategies: {asset_config.get('eligible_strategies', [])}"
        )
        logger.info(f"    Composer Params: {asset_config.get('composer_params', {})}")

    # Step 3: Show performance weight optimization
    logger.info("\n⚖️ PERFORMANCE WEIGHT OPTIMIZATION:")

    # Default weights
    default_weights = CompositeWeights()
    logger.info("  Default Weights:")
    logger.info(f"    CAGR (alpha): {default_weights.alpha:.2f}")
    logger.info(f"    Sharpe (beta): {default_weights.beta:.2f}")
    logger.info(f"    Win Rate (gamma): {default_weights.gamma:.2f}")
    logger.info(f"    Avg Trade Return (delta): {default_weights.delta:.2f}")

    # Custom weight examples
    logger.info("\n  Custom Weight Examples:")

    # CAGR-focused weights
    cagr_weights = CompositeWeights(alpha=0.7, beta=0.2, gamma=0.05, delta=0.05)
    logger.info(f"    CAGR-Focused: {cagr_weights}")

    # Sharpe-focused weights
    sharpe_weights = CompositeWeights(alpha=0.2, beta=0.7, gamma=0.05, delta=0.05)
    logger.info(f"    Sharpe-Focused: {sharpe_weights}")

    # Balanced weights
    balanced_weights = CompositeWeights(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25)
    logger.info(f"    Balanced: {balanced_weights}")

    # Step 4: Show the complete decision flow
    logger.info("\n🔄 COMPLETE DECISION FLOW:")

    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = 100 + np.cumsum(np.random.normal(0, 0.02, 50))

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.random.lognormal(10, 0.5, 50),
        }
    )

    # Test with different symbols
    test_symbols = ["SPY", "BTC-USD", "TSLA"]

    for symbol in test_symbols:
        logger.info(f"\n  --- {symbol} ---")

        # Get composer decision
        composer = ComposerIntegration(config)
        signal, metadata = composer.get_composer_decision(data, symbol, 45)

        logger.info(f"    Asset Class: {composer.get_asset_class(symbol)}")
        logger.info(f"    Final Signal: {signal:.4f}")

        if metadata.get("composer_used", False):
            logger.info(f"    Regime: {metadata.get('regime_type', 'unknown')}")
            logger.info(f"    Strategy Weights: {metadata.get('strategy_weights', {})}")
            logger.info(f"    Confidence: {metadata.get('confidence', 0.0):.3f}")
        else:
            logger.info(f"    Composer not used: {metadata.get('reason', 'unknown')}")

    # Step 5: Show performance evaluation with different weights
    logger.info("\n📈 PERFORMANCE EVALUATION WITH DIFFERENT WEIGHTS:")

    # Sample performance metrics
    metrics = {
        "cagr": 0.25,
        "sharpe": 1.5,
        "win_rate": 0.65,
        "avg_trade_return": 0.006,
        "max_dd": 0.15,
        "trade_count": 280,
    }

    logger.info("  Sample Performance:")
    logger.info(f"    CAGR: {metrics['cagr']:.1%}")
    logger.info(f"    Sharpe: {metrics['sharpe']:.2f}")
    logger.info(f"    Win Rate: {metrics['win_rate']:.1%}")
    logger.info(f"    Avg Trade Return: {metrics['avg_trade_return']:.1%}")
    logger.info(f"    Max Drawdown: {metrics['max_dd']:.1%}")
    logger.info(f"    Trade Count: {metrics['trade_count']}")

    # Calculate scores with different weights
    logger.info("\n  Composite Scores with Different Weights:")

    scores = {}
    for name, weights in [
        ("Default", default_weights),
        ("CAGR-Focused", cagr_weights),
        ("Sharpe-Focused", sharpe_weights),
        ("Balanced", balanced_weights),
    ]:
        score = composite_score(metrics, weights)
        scores[name] = score
        logger.info(f"    {name}: {score:.4f}")

    # Step 6: Show the complete system in action
    logger.info("\n🎯 COMPLETE SYSTEM IN ACTION:")
    logger.info("")
    logger.info("1. Symbol 'SPY' detected as 'etf'")
    logger.info("2. Eligible strategies: ['momentum', 'mean_reversion']")
    logger.info("3. Market regime detected as 'chop'")
    logger.info("4. Mean reversion strategy gets higher weight")
    logger.info("5. Strategies generate predictions")
    logger.info("6. Predictions blended using softmax")
    logger.info("7. Final signal generated")
    logger.info("8. Performance evaluated using composite scoring")
    logger.info("9. Weights can be optimized via walkforward analysis")
    logger.info("")

    logger.info("=== Demo Complete ===")


def show_weight_optimization():
    """Show how weight optimization works."""
    logger.info("\n🔧 WEIGHT OPTIMIZATION PROCESS:")
    logger.info("")
    logger.info("1. Run walkforward analysis with different weight combinations")
    logger.info("2. Evaluate each combination using composite scoring")
    logger.info("3. Select weights that maximize composite score")
    logger.info("4. Save optimized weights to config")
    logger.info("")
    logger.info("Example optimization:")
    logger.info("  Trial 1: alpha=0.4, beta=0.3, gamma=0.2, delta=0.1 → Score: 0.75")
    logger.info("  Trial 2: alpha=0.6, beta=0.2, gamma=0.15, delta=0.05 → Score: 0.82")
    logger.info("  Trial 3: alpha=0.5, beta=0.25, gamma=0.15, delta=0.1 → Score: 0.79")
    logger.info("  → Select Trial 2 (highest score)")
    logger.info("")


if __name__ == "__main__":
    demonstrate_composer_flow()
    show_weight_optimization()
