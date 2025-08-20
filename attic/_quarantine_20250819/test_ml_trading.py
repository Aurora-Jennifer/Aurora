#!/usr/bin/env python3
"""
Test ML Profit Learning System
Demonstrates the machine learning system that learns from trade profits
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import builtins
import contextlib

import pandas as pd
import yaml

from core.engine.backtest import BacktestEngine
from core.ml.profit_learner import ProfitLearner, TradeOutcome

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ml_system():
    """Test the ML profit learning system."""
    print("🧠 Testing ML Profit Learning System")
    print("=" * 50)

    # Load ML configuration
    try:
        with open("config/ml_config.yaml") as f:
            ml_config = yaml.safe_load(f)
        print("✅ Loaded ML configuration")
    except Exception as e:
        print(f"❌ Failed to load ML config: {e}")
        return

    # Initialize profit learner
    try:
        profit_learner = ProfitLearner(ml_config.get("ml_profit_learner", {}))
        print("✅ Initialized ProfitLearner")
    except Exception as e:
        print(f"❌ Failed to initialize ProfitLearner: {e}")
        return

    # Test with synthetic trade data
    print("\n📊 Testing with synthetic trade data...")

    # Create some synthetic trade outcomes
    synthetic_trades = [
        TradeOutcome(
            timestamp=pd.Timestamp("2024-01-01"),
            symbol="SPY",
            strategy="regime_aware_ensemble",
            regime="trend",
            entry_price=450.0,
            exit_price=455.0,
            position_size=10000.0,
            hold_duration=5,
            profit_loss=500.0,
            profit_loss_pct=0.05,  # 5% profit
            market_features={
                "volatility": 0.02,
                "rsi": 65.0,
                "sma_ratio": 1.02,
                "volume_ratio": 1.1,
                "price_position": 0.7,
                "momentum_5": 0.02,
                "momentum_20": 0.05,
                "z_score": 0.5,
                "returns_1d": 0.01,
                "returns_5d": 0.03,
            },
            trade_features={
                "position_size": 10000.0,
                "entry_price": 450.0,
                "hold_duration": 5,
                "strategy_confidence": 0.8,
                "market_regime": 0.9,
            },
        ),
        TradeOutcome(
            timestamp=pd.Timestamp("2024-01-02"),
            symbol="SPY",
            strategy="momentum",
            regime="trend",
            entry_price=455.0,
            exit_price=452.0,
            position_size=8000.0,
            hold_duration=3,
            profit_loss=-240.0,
            profit_loss_pct=-0.03,  # -3% loss
            market_features={
                "volatility": 0.025,
                "rsi": 70.0,
                "sma_ratio": 1.01,
                "volume_ratio": 0.9,
                "price_position": 0.6,
                "momentum_5": -0.01,
                "momentum_20": 0.03,
                "z_score": 0.3,
                "returns_1d": -0.005,
                "returns_5d": 0.02,
            },
            trade_features={
                "position_size": 8000.0,
                "entry_price": 455.0,
                "hold_duration": 3,
                "strategy_confidence": 0.6,
                "market_regime": 0.7,
            },
        ),
        TradeOutcome(
            timestamp=pd.Timestamp("2024-01-03"),
            symbol="SPY",
            strategy="mean_reversion",
            regime="chop",
            entry_price=452.0,
            exit_price=454.0,
            position_size=6000.0,
            hold_duration=2,
            profit_loss=120.0,
            profit_loss_pct=0.02,  # 2% profit
            market_features={
                "volatility": 0.015,
                "rsi": 45.0,
                "sma_ratio": 0.99,
                "volume_ratio": 1.0,
                "price_position": 0.5,
                "momentum_5": 0.005,
                "momentum_20": 0.01,
                "z_score": -0.2,
                "returns_1d": 0.002,
                "returns_5d": 0.01,
            },
            trade_features={
                "position_size": 6000.0,
                "entry_price": 452.0,
                "hold_duration": 2,
                "strategy_confidence": 0.7,
                "market_regime": 0.8,
            },
        ),
    ]

    # Record trades for learning
    for i, trade in enumerate(synthetic_trades):
        profit_learner.record_trade_outcome(trade)
        print(f"📈 Recorded trade {i + 1}: {trade.profit_loss_pct:.1%} profit")

    # Get learning summary
    summary = profit_learner.get_learning_summary()
    print("\n📊 Learning Summary:")
    print(f"   Total trades: {summary['total_trades']}")
    print(f"   Models trained: {summary['models_trained']}")
    print(f"   Performance history: {summary['performance_history_length']} trades")

    # Get strategy performance
    strategies = ["regime_aware_ensemble", "momentum", "mean_reversion"]
    print("\n📈 Strategy Performance:")
    for strategy in strategies:
        perf = profit_learner.get_strategy_performance(strategy)
        if perf:
            print(f"   {strategy}:")
            print(f"     Trades: {perf['total_trades']}")
            print(f"     Avg Profit: {perf['avg_profit_pct']:.2%}")
            print(f"     Win Rate: {perf['win_rate']:.1%}")
            print(f"     Best Trade: {perf['best_trade']:.2%}")
            print(f"     Worst Trade: {perf['worst_trade']:.2%}")

    print("\n✅ ML system test completed!")


def test_ml_backtest():
    """Test ML integration with backtest engine."""
    print("\n🚀 Testing ML Integration with Backtest")
    print("=" * 50)

    # Create a config with ML enabled
    ml_config = {
        "ml_enabled": True,
        "strategy": "regime_aware_ensemble",
        "symbols": ["SPY"],
        "capital": 100000,
        "fast_mode": True,
    }

    # Save temporary config
    import json

    with open("config/test_ml_backtest.json", "w") as f:
        json.dump(ml_config, f, indent=2)

    try:
        # Initialize backtest engine with ML
        engine = BacktestEngine("config/test_ml_backtest.json")

        if engine.ml_enabled:
            print("✅ ML integration enabled in backtest engine")
            print(f"   ML enabled: {engine.ml_enabled}")
            print(f"   Profit learner: {engine.profit_learner is not None}")
        else:
            print("❌ ML integration not enabled")

    except Exception as e:
        print(f"❌ Failed to test ML backtest: {e}")

    # Clean up
    with contextlib.suppress(builtins.BaseException):
        Path("config/test_ml_backtest.json").unlink()


if __name__ == "__main__":
    import pandas as pd

    print("🧠 ML Profit Learning System Test")
    print("=" * 60)

    # Test basic ML system
    test_ml_system()

    # Test ML backtest integration
    test_ml_backtest()

    print("\n🎯 Test completed!")
    print("\nNext steps:")
    print("1. Run a longer backtest with ML enabled")
    print("2. Analyze ML predictions vs actual outcomes")
    print("3. Tune ML parameters for better performance")
    print("4. Implement more sophisticated ML models")
