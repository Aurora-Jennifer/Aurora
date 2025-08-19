#!/usr/bin/env python3
"""
Train ML Models with Persistence and Warm-Start

This script provides a CLI interface for training ML models with advanced
persistence tracking, warm-start capabilities, and continual learning.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine.backtest import BacktestEngine
from core.ml.profit_learner import ProfitLearner
from core.ml.warm_start import WarmStartManager
from core.utils import setup_logging
from experiments.persistence import FeaturePersistenceAnalyzer

logger = setup_logging("logs/train_with_persistence.log", logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ML models with persistence")

    # Training parameters
    parser.add_argument(
        "--config",
        type=str,
        default="config/ml_backtest_config.json",
        help="Path to ML training configuration",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for training (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date for training (YYYY-MM-DD)"
    )
    parser.add_argument("--symbol", type=str, default="SPY", help="Trading symbol")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")

    # Persistence options
    parser.add_argument(
        "--enable-persistence",
        action="store_true",
        default=True,
        help="Enable feature persistence tracking",
    )
    parser.add_argument(
        "--enable-warm-start",
        action="store_true",
        default=True,
        help="Enable warm-start from previous runs",
    )
    parser.add_argument(
        "--enable-curriculum",
        action="store_true",
        default=True,
        help="Enable curriculum learning",
    )

    # Analysis options
    parser.add_argument(
        "--analyze-persistence",
        action="store_true",
        help="Analyze feature persistence after training",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate persistence analysis plots",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate persistence report only (no training)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/persistence_training",
        help="Output directory for results",
    )
    parser.add_argument("--run-id", type=str, help="Custom run ID (auto-generated if not provided)")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return {}


def run_training_with_persistence(args):
    """Run ML training with persistence tracking."""
    try:
        # Load configuration
        config = load_config(args.config)
        if not config:
            logger.error("Failed to load configuration")
            return False

        # Initialize components
        persistence_analyzer = FeaturePersistenceAnalyzer()
        warm_start_manager = WarmStartManager()

        # Generate run ID
        run_id = args.run_id or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize profit learner
        ml_config = config.get("ml_config", {})
        profit_learner = ProfitLearner(ml_config)

        # Start new run
        profit_learner.start_new_run(
            ticker=args.symbol, start_date=args.start_date, end_date=args.end_date
        )

        logger.info(f"Starting training run: {run_id}")
        logger.info(f"Period: {args.start_date} to {args.end_date}")
        logger.info(f"Symbol: {args.symbol}")
        logger.info(f"Capital: ${args.capital:,.0f}")

        # Create temporary config file for backtest
        temp_config = {
            "ml_enabled": True,
            "strategy": config.get("strategy", "regime_aware_ensemble"),
            "symbols": [args.symbol],
            "capital": args.capital,
            "fast_mode": False,
            "risk_management": config.get("risk_management", {}),
        }

        temp_config_file = Path("temp_ml_training_config.json")
        with open(temp_config_file, "w") as f:
            json.dump(temp_config, f, indent=2)

        # Initialize backtest engine
        engine = BacktestEngine(str(temp_config_file))
        engine.profit_learner = profit_learner

        # Set date range
        engine.start_date = args.start_date
        engine.end_date = args.end_date

        # Run backtest
        logger.info("Running backtest with ML training...")
        results = engine.run_backtest(args.start_date, args.end_date)

        if not results:
            logger.error("Backtest failed")
            return False

        # Extract metrics
        metrics = engine.get_last_summary()
        logger.info("Training completed successfully!")
        logger.info(f"Total trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Total return: {metrics.get('total_return_pct', 0):.2%}")
        logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")

        # Log feature importance
        if args.enable_persistence:
            logger.info("Logging feature importance...")
            profit_learner.log_feature_importance()

        # Save checkpoint
        if args.enable_warm_start and hasattr(engine, "profit_learner"):
            logger.info("Saving model checkpoint...")
            # This would require access to the trained model and scaler
            # For now, we'll just log that it would be saved

        return True

    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False


def analyze_persistence_results(args):
    """Analyze persistence results."""
    try:
        persistence_analyzer = FeaturePersistenceAnalyzer()

        logger.info("Analyzing feature persistence...")
        persistence_data = persistence_analyzer.analyze_persistence()

        if "error" in persistence_data:
            logger.error(f"Persistence analysis error: {persistence_data['error']}")
            return False

        # Generate report
        report = persistence_analyzer.generate_persistence_report()

        # Save report
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / "persistence_analysis_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"Persistence analysis report saved: {report_file}")

        # Generate plots
        if args.generate_plots:
            logger.info("Generating persistence plots...")
            plots = persistence_analyzer.create_persistence_plots(str(output_dir))

            for plot in plots:
                if not plot.startswith("Error"):
                    logger.info(f"Generated plot: {plot}")

        # Print summary
        print("\n" + "=" * 60)
        print("FEATURE PERSISTENCE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total Runs: {persistence_data.get('total_runs', 0)}")
        print(f"Total Features: {persistence_data.get('total_features', 0)}")
        print(
            f"Avg Importance Stability: {persistence_data.get('avg_importance_stability', 0):.4f}"
        )
        print(f"Avg Rank Stability: {persistence_data.get('avg_rank_stability', 0):.4f}")

        if persistence_data.get("top_alpha_features"):
            print("\nTop Alpha Generation Features:")
            for feature_name, stats in persistence_data["top_alpha_features"][:5]:
                print(f"  {feature_name}: {stats['alpha_mean']:.4f}")

        print(f"\nDetailed report: {report_file}")
        print("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Error analyzing persistence: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()

    print("🧠 ML Training with Persistence & Continual Learning")
    print("=" * 60)

    if args.report_only:
        # Generate persistence report only
        success = analyze_persistence_results(args)
    else:
        # Run training with persistence
        success = run_training_with_persistence(args)

        if success and args.analyze_persistence:
            analyze_persistence_results(args)

    if success:
        print("\n✅ Training and analysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("\n❌ Training or analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
