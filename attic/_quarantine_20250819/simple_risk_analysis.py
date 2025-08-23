#!/usr/bin/env python3
"""
Simple Risk Profile Analysis Script
Runs walkforward analysis and analyzes results for different risk profiles.
"""

import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd


def run_walkforward_analysis(symbol, start_date, end_date, config_file):
    """Run walkforward analysis and return results."""
    try:
        cmd = [
            "python",
            "scripts/walkforward_framework.py",
            "--symbol",
            symbol,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--train-len",
            "252",  # 1 year training
            "--test-len",
            "126",  # 6 months testing
            "--stride",
            "63",  # 3 months stride
            "--perf-mode",
            "RELAXED",
            "--validate-data",
        ]

        print(f"Running walkforward for {symbol}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Load results from the saved file
            results_file = "results/walkforward/results.json"
            if os.path.exists(results_file):
                with open(results_file) as f:
                    return json.load(f)
        else:
            print(f"Error running walkforward for {symbol}: {result.stderr}")
            return None

    except Exception as e:
        print(f"Exception running walkforward for {symbol}: {e}")
        return None


def calculate_cagr(total_return, start_date, end_date):
    """Calculate Compound Annual Growth Rate."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    years = (end - start).days / 365.25
    return ((1 + total_return) ** (1 / years) - 1) * 100


def analyze_results(data, symbol, profile_name, start_date, end_date):
    """Analyze walkforward results."""
    if not data:
        return None

    aggregate = data.get("aggregate", {})
    fold_results = data.get("fold_results", [])

    # Calculate CAGR
    total_return = aggregate.get("mean_total_return", 0)
    cagr = calculate_cagr(total_return, start_date, end_date)

    # Calculate trade statistics
    total_trades = sum(fold.get("n_trades", 0) for fold in fold_results)
    avg_duration = np.mean([fold.get("metrics", {}).get("median_hold", 1) for fold in fold_results])

    # Calculate positive fold percentage
    positive_folds = aggregate.get("positive_sharpe_folds", 0)
    total_folds = aggregate.get("total_folds", 1)
    positive_fold_pct = (positive_folds / total_folds) * 100

    return {
        "profile": profile_name,
        "symbol": symbol,
        "sharpe": aggregate.get("mean_sharpe", 0),
        "cagr": cagr,
        "max_dd": aggregate.get("mean_max_dd", 0),
        "positive_fold_pct": positive_fold_pct,
        "total_trades": total_trades,
        "avg_duration": avg_duration,
        "hit_rate": aggregate.get("mean_hit_rate", 0),
        "total_folds": total_folds,
    }


def run_sensitivity_analysis(base_config, param_name, base_value, variations):
    """Run sensitivity analysis by varying a parameter."""
    results = []

    for variation in variations:
        # Create modified config
        modified_config = base_config.copy()
        modified_config["risk_params"][param_name] = base_value * variation

        # Save temporary config
        temp_config = f"temp_sensitivity_{variation:.2f}.json"
        with open(temp_config, "w") as f:
            json.dump(modified_config, f, indent=2)

        # Run analysis
        data = run_walkforward_analysis("SPY", "2019-01-01", "2023-12-31", temp_config)

        if data:
            aggregate = data.get("aggregate", {})
            results.append(
                {
                    "variation": variation,
                    "value": base_value * variation,
                    "sharpe": aggregate.get("mean_sharpe", 0),
                    "max_dd": aggregate.get("mean_max_dd", 0),
                }
            )

        # Clean up
        if os.path.exists(temp_config):
            os.remove(temp_config)

    return results


def main():
    """Main analysis function."""
    print("🚀 Starting Risk Profile Analysis")
    print("=" * 50)

    # Configuration
    symbols = ["SPY", "BTC-USD", "TSLA"]
    risk_profiles = ["risk_low", "risk_balanced", "risk_strict"]
    start_date = "2019-01-01"
    end_date = "2023-12-31"

    # Results storage
    all_results = []

    # 1. Profile Shootout
    print("\n📊 Profile Shootout Analysis")
    print("-" * 30)

    for profile in risk_profiles:
        print(f"\nAnalyzing {profile}...")

        for symbol in symbols:
            # Load config to pass to analysis
            config_file = f"config/{profile}.json"

            data = run_walkforward_analysis(symbol, start_date, end_date, config_file)
            result = analyze_results(data, symbol, profile, start_date, end_date)

            if result:
                all_results.append(result)
                print(
                    f"  {symbol}: Sharpe={result['sharpe']:.3f}, CAGR={result['cagr']:.2f}%, MaxDD={result['max_dd']:.3f}"
                )

    # 2. Generate Reports
    print("\n📋 Generating Reports...")

    if all_results:
        df_results = pd.DataFrame(all_results)

        print("\n" + "=" * 80)
        print("📊 RISK PROFILE SHOOTOUT SUMMARY")
        print("=" * 80)

        for profile in df_results["profile"].unique():
            profile_data = df_results[df_results["profile"] == profile]
            print(f"\n{profile.upper()} PROFILE:")
            print("-" * 40)

            for _, row in profile_data.iterrows():
                print(
                    f"{row['symbol']:8} | Sharpe: {row['sharpe']:6.3f} | CAGR: {row['cagr']:6.2f}% | MaxDD: {row['max_dd']:6.3f} | +Folds: {row['positive_fold_pct']:5.1f}% | Trades: {row['total_trades']:4.0f}"
                )

        print("\n" + "=" * 60)
        print("📈 TRADE DURATION ANALYSIS")
        print("=" * 60)

        for profile in df_results["profile"].unique():
            profile_data = df_results[df_results["profile"] == profile]
            print(f"\n{profile.upper()} PROFILE:")
            print("-" * 30)

            for _, row in profile_data.iterrows():
                print(
                    f"{row['symbol']:8} | Avg Trades: {row['total_trades']:6.1f} | Avg Duration: {row['avg_duration']:6.1f} days"
                )

    # 3. Sensitivity Analysis
    print("\n🔍 Sensitivity Analysis (Balanced Profile)")
    print("-" * 40)

    try:
        with open("config/risk_balanced.json") as f:
            base_config = json.load(f)

        base_position_size = base_config["risk_params"]["max_position_size"]
        variations = [0.75, 0.875, 1.0, 1.125, 1.25]  # ±25%

        sensitivity_results = run_sensitivity_analysis(
            base_config, "max_position_size", base_position_size, variations
        )

        if sensitivity_results:
            print("\n" + "=" * 60)
            print("🔍 SENSITIVITY ANALYSIS - MAX POSITION SIZE")
            print("=" * 60)
            print("Variation | Position Size | Sharpe | MaxDD")
            print("-" * 50)

            for result in sensitivity_results:
                print(
                    f"{result['variation']:8.2f} | {result['value']:12.3f} | {result['sharpe']:6.3f} | {result['max_dd']:6.3f}"
                )

    except Exception as e:
        print(f"Error in sensitivity analysis: {e}")

    # Save detailed results
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        os.makedirs("results/risk_analysis", exist_ok=True)

        # Save profile shootout results
        df_results.to_csv(f"results/risk_analysis/profile_shootout_{timestamp}.csv", index=False)

        print(
            f"\n✅ Analysis complete! Results saved to results/risk_analysis/profile_shootout_{timestamp}.csv"
        )


if __name__ == "__main__":
    main()
