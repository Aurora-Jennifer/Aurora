#!/usr/bin/env python3
"""
Generate ML Analysis Plots
Creates comprehensive visualizations for the ML trading system
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_ml_plots():
    """Generate comprehensive ML analysis plots."""
    print("📊 Generating ML Analysis Plots")
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

    # Initialize visualizer
    try:
        visualizer = MLVisualizer()
        print("✅ Initialized ML Visualizer")
    except Exception as e:
        print(f"❌ Failed to initialize visualizer: {e}")
        return

    # Generate plots
    print("\n🎨 Generating plots...")

    try:
        # Generate comprehensive report
        plot_paths = visualizer.create_comprehensive_report(profit_learner, save_plots=True)

        print(f"\n✅ Generated {len(plot_paths)} plots:")
        for plot_name, plot_path in plot_paths.items():
            print(f"   📈 {plot_name}: {plot_path}")

        # Show summary
        summary = profit_learner.get_learning_summary()
        print("\n📊 ML System Summary:")
        print(f"   Total trades: {summary['total_trades']}")
        print(f"   Models trained: {summary['models_trained']}")
        print(
            f"   Learning status: {'Active' if summary['total_trades'] >= summary['min_trades_for_learning'] else 'Collecting Data'}"
        )

        print(f"\n📁 Plots saved to: {visualizer.output_dir}")
        print(f"📄 Report saved to: {visualizer.output_dir}/ml_analysis_report.md")

    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback

        traceback.print_exc()


def generate_specific_plots():
    """Generate specific types of plots."""
    print("🎯 Generating Specific ML Plots")
    print("=" * 50)

    # Load ML configuration
    try:
        with open("config/ml_config.yaml") as f:
            ml_config = yaml.safe_load(f)
        profit_learner = ProfitLearner(ml_config.get("ml_profit_learner", {}))
        visualizer = MLVisualizer()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Generate specific plot types
    plot_types = [
        ("Learning Progress", visualizer.create_learning_progress_plots),
        ("Strategy Performance", visualizer.create_strategy_performance_plots),
        ("Prediction Analysis", visualizer.create_prediction_analysis_plots),
        ("Risk Analysis", visualizer.create_risk_analysis_plots),
    ]

    for plot_name, plot_func in plot_types:
        try:
            print(f"\n📊 Generating {plot_name} plots...")
            plot_paths = plot_func(profit_learner, save_plots=True)
            print(f"   ✅ Generated {len(plot_paths)} {plot_name} plots")
        except Exception as e:
            print(f"   ❌ Failed to generate {plot_name} plots: {e}")


if __name__ == "__main__":
    print("📊 ML Analysis Plot Generator")
    print("=" * 60)

    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("✅ Matplotlib and Seaborn available")
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)

    # Generate comprehensive plots
    generate_ml_plots()

    print("\n🎯 Plot generation complete!")
    print("\n💡 Tips:")
    print("   • Run this after each backtest to see learning progress")
    print("   • Check the generated report for detailed analysis")
    print("   • Use plots to identify which strategies perform best")
    print("   • Monitor prediction accuracy over time")
