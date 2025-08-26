#!/usr/bin/env python3
"""
Multi-symbol walk-forward test script.
Tests regime-aware ensemble strategy across multiple symbols.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def build_features_for_symbol(
    symbol: str, start_date: str = "2020-01-01", end_date: str = "2024-12-31"
):
    """Build features for a given symbol."""
    try:
        from importlib import import_module

        import yfinance as yf
        build_features_parquet = import_module("core.data.features").build_features_parquet

        print(f"Building features for {symbol}...")
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True).reset_index()

        if len(df) < 252:
            print(f"  Skipping {symbol}: insufficient data ({len(df)} days)")
            return None

        # Build features
        parquet_path = build_features_parquet(symbol, df, "results/features")
        print(f"  Features saved to {parquet_path}")
        return str(parquet_path)

    except Exception as e:
        print(f"  Error building features for {symbol}: {e}")
        return None


def run_walkforward_test(symbol: str, parquet_path: str, output_dir: str):
    """Run walk-forward test for a symbol."""
    try:
        cmd = [
            "python",
            "apps/walk_cli.py",
            "--parquet",
            parquet_path,
            "--train",
            "252",
            "--test",
            "63",
            "--stride",
            "63",
            "--min-live-months",
            "6",
            "--output-dir",
            output_dir,
        ]

        print(f"Running walk-forward for {symbol}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✅ {symbol} completed successfully")
            return True
        print(f"  ❌ {symbol} failed: {result.stderr}")
        return False

    except Exception as e:
        print(f"  ❌ Error running {symbol}: {e}")
        return False


def analyze_results(output_dir: str):
    """Analyze results from walk-forward tests."""
    results = {}

    for symbol_dir in Path(output_dir).glob("*"):
        if not symbol_dir.is_dir():
            continue

        symbol = symbol_dir.name
        artifacts_file = symbol_dir / "artifacts_walk.json"

        if artifacts_file.exists():
            try:
                with open(artifacts_file) as f:
                    data = json.load(f)

                # Extract aggregate metrics
                aggregate = data[-1].get("aggregate", {})
                results[symbol] = {
                    "stitched_sharpe": aggregate.get("stitched_sharpe", 0),
                    "stitched_max_dd": aggregate.get("stitched_max_dd", 0),
                    "weighted_sharpe": aggregate.get("weighted_sharpe", 0),
                    "weighted_win_rate": aggregate.get("weighted_win_rate", 0),
                    "trusted_folds": aggregate.get("trusted_folds", 0),
                    "total_folds": aggregate.get("total_folds", 0),
                    "avg_weight": aggregate.get("avg_weight", 0),
                    "weight_std": aggregate.get("weight_std", 0),
                }

            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")

    return results


def print_summary_report(results: dict):
    """Print comprehensive summary report."""
    print("\n" + "=" * 100)
    print("MULTI-SYMBOL WALK-FORWARD SUMMARY REPORT")
    print("=" * 100)

    if not results:
        print("No results to analyze")
        return

    # Calculate aggregate statistics
    symbols = list(results.keys())
    stitched_sharpes = [r["stitched_sharpe"] for r in results.values()]
    weighted_sharpes = [r["weighted_sharpe"] for r in results.values()]
    trusted_ratios = [
        r["trusted_folds"] / r["total_folds"] if r["total_folds"] > 0 else 0
        for r in results.values()
    ]

    print(f"Symbols tested: {len(symbols)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nPERFORMANCE SUMMARY:")
    print(f"  Average Stitched Sharpe: {np.mean(stitched_sharpes):.3f}")
    print(f"  Average Weighted Sharpe: {np.mean(weighted_sharpes):.3f}")
    print(f"  Average Trusted Fold Ratio: {np.mean(trusted_ratios):.3f}")
    print(
        f"  Symbols with Positive Sharpe: {sum(1 for s in stitched_sharpes if s > 0)}/{len(symbols)}"
    )

    print("\nPER-SYMBOL BREAKDOWN:")
    print(f"{'Symbol':<8} {'Stitched':<8} {'Weighted':<8} {'Trusted':<8} {'Win Rate':<8}")
    print("-" * 50)

    for symbol in sorted(symbols):
        r = results[symbol]
        trusted_pct = f"{r['trusted_folds']}/{r['total_folds']}"
        print(
            f"{symbol:<8} {r['stitched_sharpe']:<8.3f} {r['weighted_sharpe']:<8.3f} {trusted_pct:<8} {r['weighted_win_rate']:<8.3f}"
        )

    # Top performers
    positive_sharpe = [
        (s, r["stitched_sharpe"]) for s, r in results.items() if r["stitched_sharpe"] > 0
    ]
    positive_sharpe.sort(key=lambda x: x[1], reverse=True)

    if positive_sharpe:
        print("\nTOP PERFORMERS (by Stitched Sharpe):")
        for i, (symbol, sharpe) in enumerate(positive_sharpe[:5]):
            print(f"  {i + 1}. {symbol}: {sharpe:.3f}")

    # Regime analysis
    print("\nREGIME ANALYSIS:")
    regime_counts = {}
    for symbol_dir in Path("artifacts/multi_symbol").glob("*"):
        if symbol_dir.is_dir():
            artifacts_file = symbol_dir / "artifacts_walk.json"
            if artifacts_file.exists():
                try:
                    with open(artifacts_file) as f:
                        data = json.load(f)

                    for fold in data[:-1]:  # Exclude aggregate
                        regime = fold.get("regime", "unknown")
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1

                except Exception:
                    continue

    for regime, count in sorted(regime_counts.items()):
        print(f"  {regime}: {count} folds")


def main():
    """Main function for multi-symbol testing."""
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "AMZN"]

    print("MULTI-SYMBOL WALK-FORWARD TEST")
    print("=" * 50)

    # Build features for all symbols
    parquet_paths = {}
    for symbol in symbols:
        path = build_features_for_symbol(symbol)
        if path:
            parquet_paths[symbol] = path

    print(f"\nBuilt features for {len(parquet_paths)} symbols")

    # Run walk-forward tests
    success_count = 0
    for symbol, parquet_path in parquet_paths.items():
        output_dir = f"artifacts/multi_symbol/{symbol}"
        if run_walkforward_test(symbol, parquet_path, output_dir):
            success_count += 1

    print(f"\nCompleted walk-forward tests for {success_count}/{len(parquet_paths)} symbols")

    # Analyze results
    results = analyze_results("artifacts/multi_symbol")

    # Print summary report
    print_summary_report(results)

    # Save aggregate results
    with open("artifacts/multi_symbol/summary_results.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "symbols_tested": len(parquet_paths),
                "successful_runs": success_count,
                "results": results,
            },
            f,
            indent=2,
        )

    print("\nAggregate results saved to: artifacts/multi_symbol/summary_results.json")


if __name__ == "__main__":
    main()
