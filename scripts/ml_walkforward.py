#!/usr/bin/env python3
"""
ML Walkforward Analysis Script

This script performs walkforward analysis with ML learning, combining:
- Traditional walkforward testing
- ML profit learning with persistence tracking
- Feature importance analysis across folds
- Continual learning with warm-start
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.engine.backtest import BacktestEngine
from core.ml.profit_learner import ProfitLearner
from core.ml.warm_start import WarmStartManager
from experiments.persistence import FeaturePersistenceAnalyzer, RunMetadata

logger = logging.getLogger(__name__)


class MLWalkforwardAnalyzer:
    def __init__(self, config_file: str, output_dir: str):
        """Initialize ML walkforward analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open(config_file) as f:
            self.config = json.load(f)

        # Initialize components
        self.persistence_analyzer = FeaturePersistenceAnalyzer()
        self.warm_start_manager = WarmStartManager()

        # Initialize profit learner with unified model support
        self.profit_learner = ProfitLearner(config=self.config)

        # Track fold results
        self.fold_results = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Multi-asset support
        self.unified_model = self.config.get("unified_model", False)
        self.assets = self.config.get("symbols", ["SPY"])

    def run_ml_walkforward(
        self,
        start_date: str,
        end_date: str,
        fold_length: int = 252,
        step_size: int = 63,
        warm_start: bool = False,
    ) -> Dict:
        """Run ML walkforward analysis with unified model support."""
        self.logger.info(
            f"Starting ML walkforward analysis: {start_date} to {end_date}"
        )
        self.logger.info(f"Unified model: {self.unified_model}")
        self.logger.info(f"Assets: {self.assets}")

        # Generate folds
        folds = self._generate_folds(start_date, end_date, fold_length, step_size)
        self.logger.info(f"Generated {len(folds)} folds")

        # Initialize warm start if enabled
        if warm_start:
            warm_start_config = self.warm_start_manager.get_warm_start_configuration(
                feature_names=self.profit_learner._get_feature_names()
            )
            self.logger.info(f"Warm start configuration: {warm_start_config}")

        # Run each fold
        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            self.logger.info(f"\n=== Fold {i+1}/{len(folds)} ===")
            self.logger.info(f"Train: {train_start} to {train_end}")
            self.logger.info(f"Test: {test_start} to {test_end}")

            # Start new ML run
            run_id = self.profit_learner.start_new_run(
                ticker="MULTI" if self.unified_model else self.assets[0],
                start_date=train_start,
                end_date=train_end,
            )

            # Run training fold
            train_results = self._run_training_fold(train_start, train_end)

            # Run testing fold
            test_results = self._run_testing_fold(test_start, test_end, train_results)

            # Log feature importance
            self._log_fold_feature_importance(
                run_id, test_start, test_end, test_results
            )

            # Save checkpoint if warm start enabled
            if warm_start:
                try:
                    # Get current model and scaler
                    if self.profit_learner.models:
                        model_key = list(self.profit_learner.models.keys())[0]
                        model = self.profit_learner.models[model_key]
                        scaler = getattr(self.profit_learner, "scaler", None)
                        feature_names = self.profit_learner._get_feature_names()

                        # Create metadata
                        metadata = RunMetadata(
                            run_id=run_id,
                            timestamp=datetime.now().isoformat(),
                            ticker="MULTI" if self.unified_model else self.assets[0],
                            start_date=train_start,
                            end_date=train_end,
                            model_type="ridge",
                            seed=42,
                            total_trades=test_results.get("total_trades", 0),
                            avg_profit=test_results.get("total_return", 0.0),
                            win_rate=0.5,
                            sharpe_ratio=test_results.get("sharpe_ratio", 0.0),
                        )

                        self.warm_start_manager.save_checkpoint(
                            model=model,
                            scaler=scaler,
                            feature_names=feature_names,
                            run_id=run_id,
                            metadata=metadata,
                        )
                        self.logger.info(f"Saved checkpoint for run {run_id}")
                except Exception as e:
                    self.logger.warning(f"Could not save checkpoint: {e}")

            # Store results
            fold_result = {
                "fold": i + 1,
                "train_period": f"{train_start} to {train_end}",
                "test_period": f"{test_start} to {test_end}",
                "train_results": train_results,
                "test_results": test_results,
                "run_id": run_id,
            }
            self.fold_results.append(fold_result)

        # Create summary structure
        summary = {
            "overall_metrics": {
                "total_folds": len(self.fold_results),
                "avg_test_return": sum(
                    fold.get("test_results", {}).get("total_return", 0)
                    for fold in self.fold_results
                )
                / len(self.fold_results)
                if self.fold_results
                else 0,
                "std_test_return": np.std(
                    [
                        fold.get("test_results", {}).get("total_return", 0)
                        for fold in self.fold_results
                    ]
                )
                if self.fold_results
                else 0,
                "avg_test_sharpe": sum(
                    fold.get("test_results", {}).get("sharpe_ratio", 0)
                    for fold in self.fold_results
                )
                / len(self.fold_results)
                if self.fold_results
                else 0,
                "total_trades": sum(
                    fold.get("test_results", {}).get("total_trades", 0)
                    for fold in self.fold_results
                ),
                "win_rate": 0.0,  # TODO: Calculate from trade logs
            },
            "feature_importance_summary": {},
            "learning_progress": self.fold_results,
            "fold_results": self.fold_results,
        }

        # Generate summary report
        self._generate_summary_report(
            summary, self.output_dir / "ml_walkforward_summary.md"
        )

        # Save results
        self._save_results(summary)

        return summary

    def _generate_folds(
        self, start_date: str, end_date: str, fold_length: int, step_size: int
    ) -> List[Tuple]:
        """Generate fold dates for walkforward analysis."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        folds = []
        current_start = start

        while current_start + timedelta(days=fold_length) <= end:
            train_start = current_start
            train_end = current_start + timedelta(days=fold_length - 1)
            test_start = train_end + timedelta(days=1)
            test_end = min(test_start + timedelta(days=step_size - 1), end)

            folds.append(
                (
                    train_start.strftime("%Y-%m-%d"),
                    train_end.strftime("%Y-%m-%d"),
                    test_start.strftime("%Y-%m-%d"),
                    test_end.strftime("%Y-%m-%d"),
                )
            )

            current_start += timedelta(days=step_size)

        return folds

    def _run_training_fold(self, start_date: str, end_date: str) -> Dict:
        """Run training fold with unified model support."""
        self.logger.info(f"Running training fold: {start_date} to {end_date}")

        if self.unified_model:
            # Train on multiple assets simultaneously
            return self._run_unified_training(start_date, end_date)
        else:
            # Train on single asset (current approach)
            return self._run_single_asset_training(start_date, end_date)

    def _run_unified_training(self, start_date: str, end_date: str) -> Dict:
        """Train unified model on multiple assets simultaneously."""
        self.logger.info(f"Training unified model on {len(self.assets)} assets")

        # Create temporary config for unified training
        unified_config = self.config.copy()
        unified_config.update(
            {
                "start_date": start_date,
                "end_date": end_date,
                "ml_enabled": True,
                "unified_model": True,
                "symbols": self.assets,  # Use all assets
            }
        )

        # Save config to temporary file
        config_file = (
            self.output_dir / f"temp_unified_config_{start_date}_{end_date}.json"
        )
        with open(config_file, "w") as f:
            json.dump(unified_config, f, indent=2)

        try:
            # Run unified backtest
            engine = BacktestEngine(config_file=str(config_file))
            results = engine.run_backtest(start_date, end_date)

            return {
                "total_return": results.get("total_return", 0.0),
                "sharpe_ratio": results.get("sharpe_ratio", 0.0),
                "max_drawdown": results.get("max_drawdown", 0.0),
                "total_trades": results.get("total_trades", 0),
                "ml_trades_recorded": len(self.profit_learner.performance_history),
                "unified_model": True,
            }
        finally:
            config_file.unlink(missing_ok=True)

    def _run_single_asset_training(self, start_date: str, end_date: str) -> Dict:
        """Train model on single asset (current approach)."""
        # Create temporary config for training
        train_config = self.config.copy()
        train_config.update(
            {
                "start_date": start_date,
                "end_date": end_date,
                "ml_enabled": True,
                "unified_model": False,
            }
        )

        # Save config to temporary file
        config_file = (
            self.output_dir / f"temp_train_config_{start_date}_{end_date}.json"
        )
        with open(config_file, "w") as f:
            json.dump(train_config, f, indent=2)

        try:
            # Run backtest
            engine = BacktestEngine(config_file=str(config_file))
            results = engine.run_backtest(start_date, end_date)

            return {
                "total_return": results.get("total_return", 0.0),
                "sharpe_ratio": results.get("sharpe_ratio", 0.0),
                "max_drawdown": results.get("max_drawdown", 0.0),
                "total_trades": results.get("total_trades", 0),
                "ml_trades_recorded": len(self.profit_learner.performance_history),
                "unified_model": False,
            }
        finally:
            config_file.unlink(missing_ok=True)

    def _run_testing_fold(
        self, start_date: str, end_date: str, train_results: Dict
    ) -> Dict:
        """Run testing fold with trained ML model."""
        self.logger.info(f"Running testing fold: {start_date} to {end_date}")

        # Create temporary config for testing
        test_config = self.config.copy()
        test_config.update(
            {
                "start_date": start_date,
                "end_date": end_date,
                "ml_enabled": True,
                "use_trained_model": True,  # Use the model trained in previous fold
            }
        )

        # Save config to temporary file
        config_file = self.output_dir / f"temp_test_config_{start_date}_{end_date}.json"
        with open(config_file, "w") as f:
            json.dump(test_config, f, indent=2)

        try:
            # Run backtest with trained ML model
            engine = BacktestEngine(config_file=str(config_file))

            # IMPORTANT: Pass the trained profit_learner to the engine
            if hasattr(self, "profit_learner") and self.profit_learner:
                engine.profit_learner = self.profit_learner
                engine.ml_enabled = True
                self.logger.info("Using trained ML model for testing")

            results = engine.run_backtest(start_date, end_date)

            return {
                "total_return": results.get("total_return", 0.0),
                "sharpe_ratio": results.get("sharpe_ratio", 0.0),
                "max_drawdown": results.get("max_drawdown", 0.0),
                "total_trades": results.get("total_trades", 0),
                "ml_predictions_used": True,
            }
        finally:
            # Clean up temporary config
            config_file.unlink(missing_ok=True)

    def _log_fold_feature_importance(
        self, run_id: str, start_date: str, end_date: str, test_results: Dict
    ):
        """Log feature importance for the current fold."""
        try:
            # Get current feature importance
            feature_importance = self._get_current_feature_importance()

            # Create metadata
            metadata = RunMetadata(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                ticker="SPY",
                start_date=start_date,
                end_date=end_date,
                model_type="ridge",
                seed=42,
                total_trades=test_results.get("total_trades", 0),
                avg_profit=test_results.get("total_return", 0.0),
                win_rate=0.5,  # Placeholder
                sharpe_ratio=test_results.get("sharpe_ratio", 0.0),
            )

            # Log to persistence analyzer
            self.persistence_analyzer.log_feature_importance(
                run_id=run_id,
                feature_importances=feature_importance,
                coefficients=feature_importance,  # Using same for simplicity
                metadata=metadata,
                regime="walkforward",
            )

            self.logger.info(f"Logged feature importance for fold {run_id}")

        except Exception as e:
            self.logger.error(f"Error logging feature importance: {e}")

    def _get_current_feature_importance(self) -> Dict:
        """Get current feature importance from the profit learner."""
        try:
            if not self.profit_learner or not self.profit_learner.models:
                return {}

            # Get the first available model
            model_key = list(self.profit_learner.models.keys())[0]
            model = self.profit_learner.models[model_key]

            if not hasattr(model, "coef_") or model.coef_ is None:
                return {}

            # Get feature names and coefficients
            feature_names = self.profit_learner._get_feature_names()
            coefficients = dict(zip(feature_names, model.coef_))

            # Return absolute coefficients as importance
            return {name: abs(coeff) for name, coeff in coefficients.items()}

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}

    def _analyze_feature_persistence(self) -> Dict:
        """Analyze feature persistence across all folds."""
        try:
            # Run persistence analysis
            persistence_results = self.persistence_analyzer.analyze_persistence()

            return {
                "top_alpha_features": persistence_results.get("top_alpha_features", []),
                "most_stable_features": persistence_results.get(
                    "most_stable_features", []
                ),
                "rank_stability": persistence_results.get("rank_stability", 0.0),
                "importance_stability": persistence_results.get(
                    "importance_stability", 0.0
                ),
            }
        except Exception as e:
            self.logger.error(f"Error analyzing feature persistence: {e}")
            return {}

    def _update_learning_progress(self, fold_result: Dict):
        """Update learning progress tracking."""
        progress = {
            "fold": fold_result["fold"],
            "run_id": fold_result["run_id"],
            "train_return": fold_result["train_results"].get("total_return", 0.0),
            "test_return": fold_result["test_results"].get("total_return", 0.0),
            "train_sharpe": fold_result["train_results"].get("sharpe_ratio", 0.0),
            "test_sharpe": fold_result["test_results"].get("sharpe_ratio", 0.0),
            "ml_trades": fold_result["train_results"].get("ml_trades_recorded", 0),
            "feature_count": len(fold_result["feature_importance"]),
        }

        self.learning_progress.append(progress)

    def _calculate_overall_metrics(self) -> Dict:
        """Calculate overall metrics across all folds."""
        if not self.fold_results:
            return {}

        test_returns = [
            fold["test_results"].get("total_return", 0.0) for fold in self.fold_results
        ]
        test_sharpes = [
            fold["test_results"].get("sharpe_ratio", 0.0) for fold in self.fold_results
        ]

        return {
            "avg_test_return": np.mean(test_returns),
            "std_test_return": np.std(test_returns),
            "avg_test_sharpe": np.mean(test_sharpes),
            "total_folds": len(self.fold_results),
            "positive_folds": sum(1 for r in test_returns if r > 0),
            "win_rate": sum(1 for r in test_returns if r > 0) / len(test_returns),
        }

    def _save_results(self, results: Dict):
        """Save analysis results to files."""
        # Save detailed results
        results_file = self.output_dir / "ml_walkforward_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary report
        summary_file = self.output_dir / "ml_walkforward_summary.md"
        self._generate_summary_report(results, summary_file)

        # Save learning progress
        progress_file = self.output_dir / "learning_progress.csv"
        if hasattr(self, "learning_progress"):
            progress_df = pd.DataFrame(self.learning_progress)
        else:
            # Use fold results as learning progress
            progress_df = pd.DataFrame(self.fold_results)
        progress_df.to_csv(progress_file, index=False)

        self.logger.info(f"Results saved to {self.output_dir}")

    def _generate_summary_report(self, results: Dict, output_file: Path):
        """Generate a summary report of the ML walkforward analysis."""
        with open(output_file, "w") as f:
            f.write("# ML Walkforward Analysis Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall metrics
            if isinstance(results, list):
                # Aggregate metrics from all folds
                total_trades = sum(fold.get("total_trades", 0) for fold in results)
                total_return = sum(fold.get("total_return", 0) for fold in results)
                avg_return = total_return / len(results) if results else 0
                overall = {
                    "total_folds": len(results),
                    "avg_test_return": avg_return,
                    "total_trades": total_trades,
                }
            else:
                overall = results.get("overall_metrics", {})

            f.write("## Overall Performance\n\n")
            f.write(f"- **Total Folds**: {overall.get('total_folds', 0)}\n")
            f.write(
                f"- **Average Test Return**: {overall.get('avg_test_return', 0.0):.4f}\n"
            )
            f.write(
                f"- **Test Return Std**: {overall.get('std_test_return', 0.0):.4f}\n"
            )
            f.write(
                f"- **Average Test Sharpe**: {overall.get('avg_test_sharpe', 0.0):.4f}\n"
            )
            f.write(f"- **Win Rate**: {overall.get('win_rate', 0.0):.2%}\n\n")

            # Feature importance summary
            if isinstance(results, list):
                feature_summary = {}
            else:
                feature_summary = results.get("feature_importance_summary", {})
            f.write("## Feature Importance Summary\n\n")

            top_features = feature_summary.get("top_alpha_features", [])
            if top_features:
                f.write("### Top Alpha Generation Features\n\n")
                for feature in top_features[:10]:
                    try:
                        if (
                            isinstance(feature, dict)
                            and "name" in feature
                            and "score" in feature
                        ):
                            f.write(
                                f"- **{feature['name']}**: {feature['score']:.4f}\n"
                            )
                        elif isinstance(feature, (list, tuple)) and len(feature) >= 2:
                            name = str(feature[0])
                            score = (
                                float(feature[1])
                                if isinstance(feature[1], (int, float))
                                else 0.0
                            )
                            f.write(f"- **{name}**: {score:.4f}\n")
                        else:
                            f.write(f"- **{str(feature)}**: N/A\n")
                    except Exception:
                        f.write(f"- **{str(feature)}**: Error formatting\n")
                f.write("\n")

            stable_features = feature_summary.get("most_stable_features", [])
            if stable_features:
                f.write("### Most Stable Features\n\n")
                for feature in stable_features[:10]:
                    try:
                        if (
                            isinstance(feature, dict)
                            and "name" in feature
                            and "stability" in feature
                        ):
                            f.write(
                                f"- **{feature['name']}**: {feature['stability']:.4f}\n"
                            )
                        elif isinstance(feature, (list, tuple)) and len(feature) >= 2:
                            name = str(feature[0])
                            stability = (
                                float(feature[1])
                                if isinstance(feature[1], (int, float))
                                else 0.0
                            )
                            f.write(f"- **{name}**: {stability:.4f}\n")
                        else:
                            f.write(f"- **{str(feature)}**: N/A\n")
                    except Exception:
                        f.write(f"- **{str(feature)}**: Error formatting\n")
                f.write("\n")

            # Learning progress
            f.write("## Learning Progress\n\n")
            if isinstance(results, list):
                progress = results  # Use the fold results directly
            else:
                progress = results.get("learning_progress", [])
            if progress:
                f.write(
                    "| Fold | Train Return | Test Return | Train Sharpe | Test Sharpe | ML Trades |\n"
                )
                f.write(
                    "|------|--------------|-------------|--------------|-------------|-----------|\n"
                )
                for i, p in enumerate(progress):
                    train_results = p.get("train_results", {})
                    test_results = p.get("test_results", {})
                    f.write(
                        f"| {i+1} | {train_results.get('total_return', 0):.4f} | {test_results.get('total_return', 0):.4f} | "
                        f"{train_results.get('sharpe_ratio', 0):.4f} | {test_results.get('sharpe_ratio', 0):.4f} | {test_results.get('total_trades', 0)} |\n"
                    )
                f.write("\n")

            f.write("## Recommendations\n\n")
            f.write("### For Model Improvement:\n")
            f.write("- Monitor feature importance stability across folds\n")
            f.write("- Use warm-start for faster convergence\n")
            f.write("- Apply curriculum learning for underperforming periods\n\n")

            f.write("### For Production Deployment:\n")
            f.write("- Validate on out-of-sample data\n")
            f.write("- Monitor model drift\n")
            f.write("- Implement robust risk management\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ML Walkforward Analysis")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--fold-length", type=int, default=252, help="Fold length in days"
    )
    parser.add_argument("--step-size", type=int, default=63, help="Step size in days")
    parser.add_argument(
        "--warm-start", action="store_true", help="Enable warm-start between folds"
    )
    parser.add_argument(
        "--output-dir", default="results/ml_walkforward", help="Output directory"
    )
    parser.add_argument(
        "--config", default="config/ml_backtest_config.json", help="Config file"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load config
    config_file = args.config
    output_dir = args.output_dir

    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Run ML walkforward analysis
    analyzer = MLWalkforwardAnalyzer(config_file, output_dir)

    try:
        results = analyzer.run_ml_walkforward(
            start_date=args.start_date,
            end_date=args.end_date,
            fold_length=args.fold_length,
            step_size=args.step_size,
            warm_start=args.warm_start,
        )

        print("\n" + "=" * 60)
        print("ML WALKFORWARD ANALYSIS COMPLETED")
        print("=" * 60)

        overall = results.get("overall_metrics", {})
        print(f"Total Folds: {overall.get('total_folds', 0)}")
        print(f"Average Test Return: {overall.get('avg_test_return', 0.0):.4f}")
        print(f"Average Test Sharpe: {overall.get('avg_test_sharpe', 0.0):.4f}")
        print(f"Win Rate: {overall.get('win_rate', 0.0):.2%}")
        print(f"\nResults saved to: {output_dir_path}")

    except Exception as e:
        print(f"Error during ML walkforward analysis: {e}")
        raise


if __name__ == "__main__":
    main()
