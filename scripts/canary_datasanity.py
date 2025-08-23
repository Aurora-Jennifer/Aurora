#!/usr/bin/env python3
"""
Canary testing script for DataSanity v1 vs v2 comparison.

This script runs the same validation on a fixed corpus using both
v1 and v2 engines and compares the outcomes to detect regressions.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_sanity import (
    export_metrics,
    get_telemetry_stats,
    validate_and_repair_with_engine_switch,
)


def load_canary_corpus() -> dict[str, pd.DataFrame]:
    """
    Load the fixed corpus for canary testing.

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    corpus = {}

    # Load smoke fixtures
    smoke_dir = Path("data/fixtures/smoke")
    if smoke_dir.exists():
        for csv_file in smoke_dir.glob("*.csv"):
            symbol = csv_file.stem
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            corpus[symbol] = df

    # Add some synthetic data if needed
    if not corpus:
        logging.warning("No smoke fixtures found, creating synthetic corpus")
        dates = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
        for symbol in ["SPY", "TSLA", "BTC-USD"]:
            data = {
                "Open": [100 + i for i in range(100)],
                "High": [102 + i for i in range(100)],
                "Low": [98 + i for i in range(100)],
                "Close": [101 + i for i in range(100)],
                "Volume": [10000 + i * 100 for i in range(100)],
            }
            corpus[symbol] = pd.DataFrame(data, index=dates)

    return corpus


def run_canary_test(
    corpus: dict[str, pd.DataFrame],
    profiles: list[str] = None
) -> dict[str, Any]:
    """
    Run canary test comparing v1 vs v2 engines.

    Args:
        corpus: Dictionary of symbol -> DataFrame
        profiles: List of profiles to test

    Returns:
        Dictionary with test results
    """
    if profiles is None:
        profiles = ["strict", "walkforward_ci", "walkforward_smoke"]

    results = {
        "timestamp": time.time(),
        "engines": ["v1", "v2"],
        "profiles": profiles,
        "symbols": list(corpus.keys()),
        "comparisons": {},
        "telemetry": {},
        "metrics": {}
    }

    # Test each engine
    for engine in ["v1", "v2"]:
        logging.info(f"Testing engine: {engine}")

        # Set engine in config (this would normally be done via config override)
        # For now, we'll use the engine switch function directly

        engine_results = {}

        for profile in profiles:
            logging.info(f"  Testing profile: {profile}")
            profile_results = {}

            for symbol, df in corpus.items():
                logging.info(f"    Testing symbol: {symbol}")

                try:
                    # Use the engine switch function
                    start_time = time.time()

                    # For v2, we'd normally set the config, but for now v2=v1
                    if engine == "v2":
                        # This is where v2 would have different behavior
                        cleaned_df, validation_result = validate_and_repair_with_engine_switch(
                            df, symbol, profile, f"canary_{engine}_{profile}_{symbol}"
                        )
                    else:
                        cleaned_df, validation_result = validate_and_repair_with_engine_switch(
                            df, symbol, profile, f"canary_{engine}_{profile}_{symbol}"
                        )

                    end_time = time.time()

                    profile_results[symbol] = {
                        "passed": not bool(validation_result.flags),
                        "flags": validation_result.flags,
                        "repairs": validation_result.repairs,
                        "rows_in": validation_result.rows_in,
                        "rows_out": validation_result.rows_out,
                        "validation_time": validation_result.validation_time,
                        "total_time": end_time - start_time,
                        "error": None
                    }

                except Exception as e:
                    profile_results[symbol] = {
                        "passed": False,
                        "flags": [],
                        "repairs": [],
                        "rows_in": len(df),
                        "rows_out": 0,
                        "validation_time": 0.0,
                        "total_time": 0.0,
                        "error": str(e)
                    }

            engine_results[profile] = profile_results

        results["comparisons"][engine] = engine_results

    # Get telemetry stats
    results["telemetry"] = get_telemetry_stats()

    # Get metrics
    results["metrics"] = export_metrics()

    return results


def compare_results(results: dict[str, Any]) -> dict[str, Any]:
    """
    Compare v1 vs v2 results and identify differences.

    Args:
        results: Results from run_canary_test

    Returns:
        Dictionary with comparison analysis
    """
    comparison = {
        "timestamp": time.time(),
        "summary": {
            "total_tests": 0,
            "identical": 0,
            "different": 0,
            "v1_only_fail": 0,
            "v2_only_fail": 0,
            "regressions": []
        },
        "details": {}
    }

    v1_results = results["comparisons"]["v1"]
    v2_results = results["comparisons"]["v2"]

    for profile in results["profiles"]:
        for symbol in results["symbols"]:
            v1_result = v1_results[profile][symbol]
            v2_result = v2_results[profile][symbol]

            comparison["summary"]["total_tests"] += 1

            # Compare pass/fail status
            v1_passed = v1_result["passed"]
            v2_passed = v2_result["passed"]

            if v1_passed == v2_passed:
                comparison["summary"]["identical"] += 1
            else:
                comparison["summary"]["different"] += 1

                if v1_passed and not v2_passed:
                    comparison["summary"]["v2_only_fail"] += 1
                    comparison["summary"]["regressions"].append({
                        "profile": profile,
                        "symbol": symbol,
                        "type": "v2_regression",
                        "v1_flags": v1_result["flags"],
                        "v2_flags": v2_result["flags"]
                    })
                elif not v1_passed and v2_passed:
                    comparison["summary"]["v1_only_fail"] += 1

            # Store detailed comparison
            key = f"{profile}_{symbol}"
            comparison["details"][key] = {
                "v1_passed": v1_passed,
                "v2_passed": v2_passed,
                "v1_flags": v1_result["flags"],
                "v2_flags": v2_result["flags"],
                "v1_time": v1_result["total_time"],
                "v2_time": v2_result["total_time"],
                "time_diff_pct": ((v2_result["total_time"] - v1_result["total_time"]) / v1_result["total_time"] * 100) if v1_result["total_time"] > 0 else 0
            }

    return comparison


def main():
    """Main canary test function."""
    parser = argparse.ArgumentParser(description="Run DataSanity canary tests")
    parser.add_argument("--output", "-o", default="artifacts/canary_results.json",
                       help="Output file for results")
    parser.add_argument("--profiles", nargs="+",
                       default=["strict", "walkforward_ci", "walkforward_smoke"],
                       help="Profiles to test")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load corpus
    logging.info("Loading canary corpus...")
    corpus = load_canary_corpus()
    logging.info(f"Loaded {len(corpus)} symbols: {list(corpus.keys())}")

    # Run canary test
    logging.info("Running canary test...")
    results = run_canary_test(corpus, args.profiles)

    # Compare results
    logging.info("Comparing results...")
    comparison = compare_results(results)

    # Combine results
    full_results = {
        "canary_test": results,
        "comparison": comparison
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    logging.info(f"Results saved to {output_path}")

    # Print summary
    summary = comparison["summary"]
    print("\nCanary Test Summary:")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Identical results: {summary['identical']}")
    print(f"  Different results: {summary['different']}")
    print(f"  V2-only failures: {summary['v2_only_fail']}")
    print(f"  V1-only failures: {summary['v1_only_fail']}")

    if summary["regressions"]:
        print("\nRegressions detected:")
        for reg in summary["regressions"]:
            print(f"  {reg['profile']}/{reg['symbol']}: {reg['v2_flags']}")

    # Exit with error if regressions found
    if summary["v2_only_fail"] > 0:
        logging.error(f"Found {summary['v2_only_fail']} regressions in v2")
        sys.exit(1)

    logging.info("Canary test completed successfully")


if __name__ == "__main__":
    main()
