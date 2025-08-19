#!/usr/bin/env python3
"""
Multi-Asset Walkforward Analysis Script

This script runs 5-year walkforward analysis for multiple assets:
- SPY (S&P 500 ETF)
- AAPL (Apple)
- TSLA (Tesla)
- GOOG (Google)
- BTC-USD (Bitcoin)

Features:
- 5-year analysis period (2019-2024)
- 252-day training folds, 63-day testing
- Warm-start between folds
- Comprehensive reporting
- Asset comparison analysis
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class MultiAssetWalkforwardRunner:
    """Runner for multi-asset walkforward analysis."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        fold_length: int = 252,
        step_size: int = 63,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.fold_length = fold_length
        self.step_size = step_size

        # Asset configurations
        self.assets = {
            "spy": {
                "name": "SPY",
                "config": "config/ml_backtest_spy.json",
                "output_dir": "results/ml_walkforward_spy",
                "description": "S&P 500 ETF",
            },
            "aapl": {
                "name": "AAPL",
                "config": "config/ml_backtest_aapl.json",
                "output_dir": "results/ml_walkforward_aapl",
                "description": "Apple Inc.",
            },
            "tsla": {
                "name": "TSLA",
                "config": "config/ml_backtest_tsla.json",
                "output_dir": "results/ml_walkforward_tsla",
                "description": "Tesla Inc.",
            },
            "goog": {
                "name": "GOOG",
                "config": "config/ml_backtest_goog.json",
                "output_dir": "results/ml_walkforward_goog",
                "description": "Alphabet Inc. (Google)",
            },
            "btc": {
                "name": "BTC-USD",
                "config": "config/ml_backtest_btc.json",
                "output_dir": "results/ml_walkforward_btc",
                "description": "Bitcoin",
            },
        }

        # Results storage
        self.results = {}

    def run_walkforward_for_asset(self, asset_key: str) -> dict:
        """Run walkforward analysis for a specific asset."""
        asset = self.assets[asset_key]

        print(f"\n{'=' * 60}")
        print(f"Running Walkforward Analysis for {asset['name']} ({asset['description']})")
        print(f"{'=' * 60}")

        # Build command
        cmd = [
            "python",
            "scripts/ml_walkforward.py",
            "--start-date",
            self.start_date,
            "--end-date",
            self.end_date,
            "--fold-length",
            str(self.fold_length),
            "--step-size",
            str(self.step_size),
            "--warm-start",
            "--config",
            asset["config"],
            "--output-dir",
            asset["output_dir"],
        ]

        print(f"Command: {' '.join(cmd)}")
        print(f"Output directory: {asset['output_dir']}")

        try:
            # Run the walkforward analysis
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            print(f"✅ Successfully completed walkforward for {asset['name']}")
            print(f"Output: {result.stdout[-500:]}")  # Last 500 chars

            # Load results
            results_file = Path(asset["output_dir"]) / "ml_walkforward_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    asset_results = json.load(f)
                self.results[asset_key] = asset_results
                return asset_results
            else:
                print(f"⚠️  Results file not found: {results_file}")
                return {}

        except subprocess.CalledProcessError as e:
            print(f"❌ Error running walkforward for {asset['name']}: {e}")
            print(f"Error output: {e.stderr}")
            return {}

    def run_all_assets(self, assets_to_run: list[str] = None) -> dict:
        """Run walkforward analysis for all specified assets."""
        if assets_to_run is None:
            assets_to_run = list(self.assets.keys())

        print("\n🚀 Starting Multi-Asset Walkforward Analysis")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Fold length: {self.fold_length} days")
        print(f"Step size: {self.step_size} days")
        print(f"Assets: {', '.join([self.assets[asset]['name'] for asset in assets_to_run])}")

        start_time = datetime.now()

        for asset_key in assets_to_run:
            if asset_key not in self.assets:
                print(f"⚠️  Unknown asset: {asset_key}")
                continue

            self.run_walkforward_for_asset(asset_key)

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n{'=' * 60}")
        print("Multi-Asset Walkforward Analysis Complete!")
        print(f"Duration: {duration}")
        print(f"{'=' * 60}")

        return self.results

    def generate_comparison_report(self) -> str:
        """Generate a comparison report for all assets."""
        if not self.results:
            return "No results available for comparison."

        report = []
        report.append("# Multi-Asset Walkforward Analysis Comparison")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Period: {self.start_date} to {self.end_date}")
        report.append("")

        # Summary table
        report.append("## Performance Summary")
        report.append("")
        report.append(
            "| Asset | Description | Avg Test Return | Avg Test Sharpe | Win Rate | Total Folds |"
        )
        report.append(
            "|-------|-------------|-----------------|-----------------|----------|-------------|"
        )

        for asset_key, asset in self.assets.items():
            if asset_key in self.results:
                overall_metrics = self.results[asset_key].get("overall_metrics", {})

                avg_return = overall_metrics.get("avg_test_return", 0.0)
                avg_sharpe = overall_metrics.get("avg_test_sharpe", 0.0)
                win_rate = overall_metrics.get("win_rate", 0.0)
                total_folds = overall_metrics.get("total_folds", 0)

                report.append(
                    f"| {asset['name']} | {asset['description']} | {avg_return:.4f} | {avg_sharpe:.4f} | {win_rate:.2%} | {total_folds} |"
                )

        report.append("")

        # Detailed results
        report.append("## Detailed Results")
        report.append("")

        for asset_key, asset in self.assets.items():
            if asset_key in self.results:
                report.append(f"### {asset['name']} ({asset['description']})")
                report.append("")

                overall_metrics = self.results[asset_key].get("overall_metrics", {})

                report.append(
                    f"- **Average Test Return**: {overall_metrics.get('avg_test_return', 0.0):.4f}"
                )
                report.append(
                    f"- **Test Return Std**: {overall_metrics.get('std_test_return', 0.0):.4f}"
                )
                report.append(
                    f"- **Average Test Sharpe**: {overall_metrics.get('avg_test_sharpe', 0.0):.4f}"
                )
                report.append(f"- **Win Rate**: {overall_metrics.get('win_rate', 0.0):.2%}")
                report.append(f"- **Total Folds**: {overall_metrics.get('total_folds', 0)}")
                report.append(f"- **Positive Folds**: {overall_metrics.get('positive_folds', 0)}")
                report.append("")

                # Feature importance summary
                feature_summary = self.results[asset_key].get("feature_importance_summary", {})
                top_features = feature_summary.get("top_alpha_features", [])

                if top_features:
                    report.append("**Top Alpha Features:**")
                    for i, feature in enumerate(top_features[:5], 1):
                        try:
                            if (
                                isinstance(feature, dict)
                                and "name" in feature
                                and "score" in feature
                            ):
                                report.append(f"{i}. {feature['name']}: {feature['score']:.4f}")
                            elif isinstance(feature, (list, tuple)) and len(feature) >= 2:
                                name = str(feature[0])
                                score = (
                                    float(feature[1])
                                    if isinstance(feature[1], (int, float))
                                    else 0.0
                                )
                                report.append(f"{i}. {name}: {score:.4f}")
                            else:
                                report.append(f"{i}. {str(feature)}: N/A")
                        except Exception:
                            report.append(f"{i}. {str(feature)}: Error")
                    report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("")

        # Find best performing assets
        performance_data = []
        for asset_key, asset in self.assets.items():
            if asset_key in self.results:
                overall_metrics = self.results[asset_key].get("overall_metrics", {})
                performance_data.append(
                    {
                        "asset": asset["name"],
                        "description": asset["description"],
                        "avg_return": overall_metrics.get("avg_test_return", 0.0),
                        "avg_sharpe": overall_metrics.get("avg_test_sharpe", 0.0),
                        "win_rate": overall_metrics.get("win_rate", 0.0),
                    }
                )

        if performance_data:
            # Sort by Sharpe ratio
            performance_data.sort(key=lambda x: x["avg_sharpe"], reverse=True)

            report.append("### Best Performing Assets (by Sharpe Ratio)")
            report.append("")
            for i, perf in enumerate(performance_data[:3], 1):
                report.append(f"{i}. **{perf['asset']}** ({perf['description']})")
                report.append(f"   - Sharpe: {perf['avg_sharpe']:.4f}")
                report.append(f"   - Return: {perf['avg_return']:.4f}")
                report.append(f"   - Win Rate: {perf['win_rate']:.2%}")
                report.append("")

            # Sort by return
            performance_data.sort(key=lambda x: x["avg_return"], reverse=True)

            report.append("### Best Performing Assets (by Return)")
            report.append("")
            for i, perf in enumerate(performance_data[:3], 1):
                report.append(f"{i}. **{perf['asset']}** ({perf['description']})")
                report.append(f"   - Return: {perf['avg_return']:.4f}")
                report.append(f"   - Sharpe: {perf['avg_sharpe']:.4f}")
                report.append(f"   - Win Rate: {perf['win_rate']:.2%}")
                report.append("")

        report.append("### Next Steps")
        report.append("")
        report.append(
            "1. **Focus on top performers**: Analyze the best performing assets in detail"
        )
        report.append("2. **Feature analysis**: Compare feature importance across assets")
        report.append("3. **Risk management**: Implement asset-specific risk controls")
        report.append("4. **Portfolio construction**: Consider combining multiple assets")
        report.append("5. **Parameter optimization**: Fine-tune parameters for each asset")

        return "\n".join(report)

    def save_comparison_report(self, output_file: str = "results/multi_asset_comparison.md"):
        """Save the comparison report to a file."""
        report = self.generate_comparison_report()

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            f.write(report)

        print(f"📊 Comparison report saved to: {output_file}")
        return output_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Asset Walkforward Analysis")
    parser.add_argument("--start-date", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--fold-length", type=int, default=252, help="Fold length in days")
    parser.add_argument("--step-size", type=int, default=63, help="Step size in days")
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=["spy", "aapl", "tsla", "goog", "btc"],
        default=["spy", "aapl", "tsla", "goog", "btc"],
        help="Assets to analyze",
    )
    parser.add_argument(
        "--output-file",
        default="results/multi_asset_comparison.md",
        help="Output file for comparison report",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    print("🚀 Multi-Asset Walkforward Analysis")
    print("=" * 50)
    print(f"Assets: {', '.join(args.assets)}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Fold length: {args.fold_length} days")
    print(f"Step size: {args.step_size} days")
    print("=" * 50)

    # Create runner
    runner = MultiAssetWalkforwardRunner(
        start_date=args.start_date,
        end_date=args.end_date,
        fold_length=args.fold_length,
        step_size=args.step_size,
    )

    try:
        # Run analysis for all specified assets
        results = runner.run_all_assets(args.assets)

        # Generate and save comparison report
        output_file = runner.save_comparison_report(args.output_file)

        print("\n✅ Multi-asset walkforward analysis completed!")
        print(f"📊 Results saved to: {output_file}")
        print("📁 Individual results in: results/ml_walkforward_*/")

        # Show quick summary
        print("\n📈 Quick Summary:")
        for asset_key in args.assets:
            if asset_key in results:
                overall_metrics = results[asset_key].get("overall_metrics", {})
                asset_name = runner.assets[asset_key]["name"]
                avg_return = overall_metrics.get("avg_test_return", 0.0)
                avg_sharpe = overall_metrics.get("avg_test_sharpe", 0.0)
                print(f"  {asset_name}: Return={avg_return:.4f}, Sharpe={avg_sharpe:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
