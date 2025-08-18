#!/usr/bin/env python3
"""
Analyze ML Learning Progress
Shows what the ML system has learned and how it stores knowledge
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from core.ml.profit_learner import ProfitLearner


def analyze_ml_learning():
    """Analyze what the ML system has learned."""
    print("🧠 ML Learning Analysis")
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
        print("✅ Loaded existing ProfitLearner")
    except Exception as e:
        print(f"❌ Failed to load ProfitLearner: {e}")
        return

    # Show learning summary
    summary = profit_learner.get_learning_summary()
    print("\n📊 Learning Summary:")
    print(f"   Total trades recorded: {summary['total_trades']}")
    print(f"   Models trained: {summary['models_trained']}")
    print(f"   Performance history: {summary['performance_history_length']} trades")
    print(f"   Last update: {summary['last_update']}")
    print(f"   Min trades for learning: {summary['min_trades_for_learning']}")

    # Show strategy performance
    print("\n📈 Strategy Performance Analysis:")
    strategies = [
        "regime_aware_ensemble",
        "momentum",
        "mean_reversion",
        "sma_crossover",
        "ensemble_basic",
    ]

    for strategy in strategies:
        perf = profit_learner.get_strategy_performance(strategy)
        if perf:
            print(f"\n   {strategy.upper()}:")
            print(f"     📊 Total trades: {perf['total_trades']}")
            print(f"     💰 Avg profit: {perf['avg_profit_pct']:.2%}")
            print(f"     🎯 Win rate: {perf['win_rate']:.1%}")
            print(f"     📈 Best trade: {perf['best_trade']:.2%}")
            print(f"     📉 Worst trade: {perf['worst_trade']:.2%}")
            print(f"     📊 Profit std: {perf['profit_std']:.2%}")
        else:
            print(f"\n   {strategy.upper()}: No trades yet")

    # Show model details
    print("\n🤖 Model Details:")
    if profit_learner.models:
        for model_name, model in profit_learner.models.items():
            print(f"   Model: {model_name}")
            print(f"     Type: {type(model).__name__}")
            print(
                f"     Features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'}"
            )
            print(
                f"     Coefficients: {len(model.coef_) if hasattr(model, 'coef_') else 'Unknown'}"
            )

            # Show feature importance (if available)
            if hasattr(model, "coef_"):
                print("     Top features by importance:")
                feature_names = [
                    "strategy_regime_ensemble",
                    "strategy_momentum",
                    "strategy_mean_reversion",
                    "strategy_sma_crossover",
                    "strategy_ensemble_basic",
                    "regime_trend",
                    "regime_chop",
                    "regime_volatile",
                    "volatility",
                    "rsi",
                    "sma_ratio",
                    "volume_ratio",
                    "price_position",
                    "momentum_5",
                    "momentum_20",
                    "z_score",
                    "returns_1d",
                    "returns_5d",
                ]

                # Get top 5 features by absolute coefficient value
                coef_abs = abs(model.coef_)
                top_indices = coef_abs.argsort()[-5:][::-1]

                for i, idx in enumerate(top_indices):
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        importance = model.coef_[idx]
                        print(f"       {i+1}. {feature_name}: {importance:.4f}")
    else:
        print("   No models trained yet")

    # Show storage details
    print("\n💾 Storage Details:")
    state_dir = Path("state/ml_profit_learner")
    if state_dir.exists():
        files = list(state_dir.glob("*"))
        print(f"   Storage directory: {state_dir}")
        print(f"   Files stored: {len(files)}")
        for file in files:
            size = file.stat().st_size
            print(f"     {file.name}: {size} bytes")
    else:
        print("   No storage directory found")

    # Show what the system has learned
    print("\n🎓 What the System Has Learned:")
    if profit_learner.performance_history:
        print(
            f"   📈 Trade history available: {len(profit_learner.performance_history)} trades"
        )

        # Analyze recent trades
        recent_trades = profit_learner.performance_history[-5:]  # Last 5 trades
        print("   📊 Recent trades:")
        for i, trade in enumerate(recent_trades):
            print(
                f"     Trade {i+1}: {trade.profit_loss_pct:.2%} profit ({trade.strategy})"
            )

        # Show learning progress
        if (
            len(profit_learner.performance_history)
            >= profit_learner.min_trades_for_learning
        ):
            print("   ✅ System has enough data for learning")
            print("   🧠 Models are being trained and updated")
        else:
            needed = profit_learner.min_trades_for_learning - len(
                profit_learner.performance_history
            )
            print(f"   ⏳ Need {needed} more trades to start learning")
    else:
        print("   📝 No trade history yet - system needs more trades to learn")


def show_learning_process():
    """Show how the learning process works."""
    print("\n🔄 How the Learning Process Works:")
    print("   1. 📊 Market data comes in")
    print("   2. 🧠 ML system extracts features (RSI, volatility, etc.)")
    print("   3. 🎯 System predicts expected profit for each strategy")
    print("   4. 📈 Trading decision made based on prediction")
    print("   5. 💰 Actual trade outcome recorded")
    print("   6. 🎓 Model updated with new knowledge")
    print("   7. 💾 Knowledge saved to disk for next session")

    print("\n📚 What Gets Stored:")
    print("   • Trained ML models (Ridge regression)")
    print("   • Feature scaling parameters")
    print("   • Trade performance history")
    print("   • Strategy-specific performance data")
    print("   • Model training metadata")


def show_next_steps():
    """Show what to do next."""
    print("\n🎯 Next Steps:")
    print("   1. 🔄 Run more backtests to accumulate training data")
    print("   2. 📊 Analyze ML predictions vs actual outcomes")
    print("   3. ⚙️  Tune ML parameters for better performance")
    print("   4. 🛡️  Add more guardrails and safety checks")
    print("   5. 🧪 Test on different market conditions")
    print("   6. 📈 Monitor learning progress over time")


if __name__ == "__main__":
    print("🧠 ML Learning Analysis Tool")
    print("=" * 60)

    analyze_ml_learning()
    show_learning_process()
    show_next_steps()

    print("\n✅ Analysis complete!")
    print("\n💡 Tip: Run this script after each backtest to see learning progress")
