#!/usr/bin/env python3
"""
Auto ML Analysis
Automatically generates ML analysis plots after backtests
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from core.ml.profit_learner import ProfitLearner
from core.ml.visualizer import MLVisualizer

# Configure logging
from core.utils import setup_logging

logger = setup_logging("logs/auto_ml_analysis.log", logging.INFO)
logger = logging.getLogger(__name__)


def auto_analyze_ml():
    """Automatically analyze ML system and generate plots."""
    print("🤖 Auto ML Analysis")
    print("=" * 40)

    try:
        # Load ML configuration
        with open("config/ml_config.yaml") as f:
            ml_config = yaml.safe_load(f)

        # Initialize components
        profit_learner = ProfitLearner(ml_config.get("ml_profit_learner", {}))
        visualizer = MLVisualizer()

        # Get current status
        summary = profit_learner.get_learning_summary()

        print("📊 Current Status:")
        print(f"   Trades: {summary['total_trades']}")
        print(f"   Models: {summary['models_trained']}")
        print(
            f"   Learning: {'Active' if summary['total_trades'] >= summary['min_trades_for_learning'] else 'Collecting Data'}"
        )

        # Generate plots
        print("\n🎨 Generating analysis plots...")
        plot_paths = visualizer.create_comprehensive_report(profit_learner, save_plots=True)

        print(f"✅ Generated {len(plot_paths)} plots")
        print(f"📁 Saved to: {visualizer.output_dir}")

        # Provide recommendations
        print("\n💡 Recommendations:")
        if summary["total_trades"] < summary["min_trades_for_learning"]:
            needed = summary["min_trades_for_learning"] - summary["total_trades"]
            print(f"   • Run more backtests to collect {needed} more trades")
            print("   • Try different time periods or symbols")
            print("   • Adjust signal thresholds to generate more trades")
        else:
            print("   • ML system is learning! Monitor prediction accuracy")
            print("   • Consider tuning ML parameters")
            print("   • Test on different market conditions")

        return True

    except Exception as e:
        print(f"❌ Error in auto analysis: {e}")
        return False


def quick_status_check():
    """Quick status check of ML system."""
    print("🔍 Quick ML Status Check")
    print("=" * 30)

    try:
        # Load ML configuration
        with open("config/ml_config.yaml") as f:
            ml_config = yaml.safe_load(f)

        profit_learner = ProfitLearner(ml_config.get("ml_profit_learner", {}))
        summary = profit_learner.get_learning_summary()

        print("📊 ML System Status:")
        print(f"   Trades recorded: {summary['total_trades']}")
        print(f"   Models trained: {summary['models_trained']}")
        print(f"   Learning threshold: {summary['min_trades_for_learning']}")
        print(
            f"   Status: {'🟢 Learning' if summary['total_trades'] >= summary['min_trades_for_learning'] else '🟡 Collecting Data'}"
        )

        # Check storage
        state_dir = Path("state/ml_profit_learner")
        if state_dir.exists():
            files = list(state_dir.glob("*"))
            print(f"   Storage: {len(files)} files saved")
        else:
            print("   Storage: No files found")

        return True

    except Exception as e:
        print(f"❌ Error in status check: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto ML Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick status check only")
    parser.add_argument("--full", action="store_true", help="Full analysis with plots")

    args = parser.parse_args()

    if args.quick:
        quick_status_check()
    elif args.full:
        auto_analyze_ml()
    else:
        # Default: quick check
        quick_status_check()
