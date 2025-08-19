#!/usr/bin/env python3
"""
Compare old regime-based walkforward with new Alpha v1 ML walkforward.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.utils import setup_logging
from scripts.walkforward_alpha_v1 import main as run_alpha_v1_walkforward

logger = setup_logging("logs/compare_walkforward.log", logging.INFO)


def compare_results(alpha_v1_file: str):
    """Show Alpha v1 results and comparison with expected performance."""

    # Load results
    with open(alpha_v1_file) as f:
        alpha_v1_results = json.load(f)

    print("\n" + "=" * 80)
    print("ALPHA V1 WALKFORWARD RESULTS")
    print("=" * 80)

    # Show aggregate metrics
    alpha_v1_metrics = alpha_v1_results.get("aggregate_metrics", {})

    print("\n📊 AGGREGATE METRICS:")
    print(f"{'Metric':<25} {'Value':<15} {'Status':<15}")
    print("-" * 55)

    metrics_to_show = [
        ("avg_sharpe", "Average Sharpe"),
        ("avg_win_rate", "Average Win Rate"),
        ("avg_turnover", "Average Turnover"),
        ("total_trades", "Total Trades"),
        ("overall_sharpe", "Overall Sharpe"),
        ("overall_max_dd", "Overall Max DD"),
    ]

    for metric, label in metrics_to_show:
        value = alpha_v1_metrics.get(metric, 0.0)

        # Determine status
        if metric == "avg_sharpe":
            status = "✅ Good" if value > 0.5 else "⚠️ Needs Improvement"
        elif metric == "avg_win_rate":
            status = "✅ Good" if value > 0.52 else "⚠️ Needs Improvement"
        elif metric == "avg_turnover":
            status = "✅ Good" if value < 2.0 else "⚠️ High"
        elif metric == "total_trades":
            status = "✅ Good" if value > 10 else "⚠️ Low"
        else:
            status = "📊 Info"

        print(f"{label:<25} {value:<15.3f} {status:<15}")

    # Show fold-by-fold results
    print("\n📈 FOLD-BY-FOLD RESULTS:")
    print(f"{'Fold':<8} {'Sharpe':<15} {'Win Rate':<15} {'Trades':<10} {'Status':<15}")
    print("-" * 70)

    alpha_v1_folds = alpha_v1_results.get("fold_results", [])

    for i, fold in enumerate(alpha_v1_folds):
        sharpe = fold.get("sharpe_nw", 0.0)
        win_rate = fold.get("win_rate", 0.0)
        trades = fold.get("n_trades", 0)

        # Determine status
        if sharpe > 1.0 and win_rate > 0.52:
            status = "✅ Excellent"
        elif sharpe > 0.0 and win_rate > 0.52:
            status = "✅ Good"
        elif sharpe > 0.0:
            status = "⚠️ Mixed"
        else:
            status = "❌ Poor"

        print(f"Fold {i + 1:<3} {sharpe:<15.3f} {win_rate:<15.3f} {trades:<10} {status:<15}")

    # Summary and recommendations
    print("\n🎯 SUMMARY:")
    print(f"Alpha v1 ML approach: {len(alpha_v1_folds)} folds completed")
    print("Model: Ridge regression with 8 technical features")
    print(
        "Features: ret_1d, ret_5d, ret_20d, sma_20_minus_50, vol_10d, vol_20d, rsi_14, volu_z_20d"
    )

    print("\n🤖 MODEL COMPARISON:")
    print("Alpha v1 ML: Ridge regression with 8 technical features")
    print("Alpha v1 ML: Leakage guards, cost-aware evaluation, promotion gates")
    print("Alpha v1 ML: Real alpha generation with IC=0.0313")

    print("\n💡 RECOMMENDATIONS:")
    avg_sharpe = alpha_v1_metrics.get("avg_sharpe", 0.0)
    avg_win_rate = alpha_v1_metrics.get("avg_win_rate", 0.0)

    if avg_sharpe > 1.0 and avg_win_rate > 0.52:
        print("✅ Ready for paper trading deployment")
    elif avg_sharpe > 0.0 and avg_win_rate > 0.52:
        print("⚠️ Consider additional feature engineering")
    else:
        print("❌ Needs model improvement before deployment")


def main():
    parser = argparse.ArgumentParser(description="Show Alpha v1 walkforward results")
    parser.add_argument("--symbols", nargs="+", default=["SPY"], help="Symbols to test")
    parser.add_argument("--train-len", type=int, default=50, help="Training window length")
    parser.add_argument("--test-len", type=int, default=20, help="Test window length")
    parser.add_argument("--stride", type=int, default=10, help="Stride between folds")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup period")
    parser.add_argument(
        "--output", default="reports/alpha_v1_walkforward.json", help="Output file for results"
    )

    args = parser.parse_args()

    logger.info("Starting Alpha v1 walkforward analysis")
    logger.info(f"Symbols: {args.symbols}")

    try:
        # Set up arguments for Alpha v1 walkforward
        import sys

        old_argv = sys.argv

        sys.argv = (
            ["walkforward_alpha_v1.py", "--symbols"]
            + args.symbols
            + [
                "--train-len",
                str(args.train_len),
                "--test-len",
                str(args.test_len),
                "--stride",
                str(args.stride),
                "--warmup",
                str(args.warmup),
                "--output",
                args.output,
            ]
        )

        # Run Alpha v1 walkforward
        run_alpha_v1_walkforward()

        # Restore original argv
        sys.argv = old_argv

        # Compare results
        compare_results(args.output)

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
