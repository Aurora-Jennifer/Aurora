#!/usr/bin/env python3
"""
Test ML Recording
Simple test to verify ML trade recording works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import yaml

from core.ml.profit_learner import ProfitLearner, TradeOutcome


def test_ml_recording():
    """Test ML trade recording functionality."""
    print("🧪 Testing ML Trade Recording")
    print("=" * 40)

    # Load ML configuration
    with open("config/ml_config.yaml") as f:
        ml_config = yaml.safe_load(f)

    # Initialize profit learner
    profit_learner = ProfitLearner(ml_config.get("ml_profit_learner", {}))

    print(f"📊 Initial state: {len(profit_learner.performance_history)} trades")

    # Create a test trade
    test_trade = TradeOutcome(
        timestamp=datetime.now(),
        symbol="SPY",
        strategy="test_strategy",
        regime="test_regime",
        entry_price=100.0,
        exit_price=101.0,
        position_size=1000.0,
        hold_duration=1,
        profit_loss=10.0,
        profit_loss_pct=0.01,
        market_features={"test_feature": 0.5},
        trade_features={"test_trade_feature": 0.3},
    )

    # Record the trade
    print("📝 Recording test trade...")
    profit_learner.record_trade_outcome(test_trade)

    print(f"📊 After recording: {len(profit_learner.performance_history)} trades")

    # Check the summary
    summary = profit_learner.get_learning_summary()
    print(f"📊 Summary shows: {summary['total_trades']} trades")

    # Save to disk
    print("💾 Saving to disk...")
    profit_learner._save_models()

    # Create new instance and load
    print("🔄 Creating new instance and loading...")
    profit_learner2 = ProfitLearner(ml_config.get("ml_profit_learner", {}))

    print(f"📊 New instance: {len(profit_learner2.performance_history)} trades")
    summary2 = profit_learner2.get_learning_summary()
    print(f"📊 New instance summary: {summary2['total_trades']} trades")

    print("✅ Test complete!")


if __name__ == "__main__":
    test_ml_recording()
