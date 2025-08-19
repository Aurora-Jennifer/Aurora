#!/usr/bin/env python3
"""
Risk Profile Analysis Script
Runs comprehensive walkforward analysis across multiple risk profiles and symbols.
"""

import json
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd


def run_walkforward_analysis(symbol, start_date, end_date, config_file, output_dir):
    """Run walkforward analysis for a single symbol and config."""
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
            "--output-dir",
            output_dir,
        ]

        print(f"Running walkforward for {symbol} with {config_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Parse results from output
            lines = result.stdout.split("\n")
            metrics = {}

            for line in lines:
                if "Mean Sharpe:" in line:
                    metrics["sharpe"] = float(line.split(":")[1].strip())
                elif "Mean Max DD:" in line:
                    metrics["max_dd"] = float(line.split(":")[1].strip())
                elif "Mean Total Return:" in line:
                    metrics["total_return"] = float(line.split(":")[1].strip())
                elif "Folds with positive Sharpe:" in line:
                    parts = line.split(":")[1].strip().split("/")
                    metrics["positive_folds"] = int(parts[0])
                    metrics["total_folds"] = int(parts[1])
                elif "Mean Hit Rate:" in line:
                    metrics["hit_rate"] = float(line.split(":")[1].strip())

            return metrics
        else:
            print(f"Error running walkforward for {symbol}: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Timeout running walkforward for {symbol}")
        return None
    except Exception as e:
        print(f"Exception running walkforward for {symbol}: {e}")
        return None


def load_config(config_file):
    """Load risk profile configuration."""
    with open(config_file) as f:
        return json.load(f)


def calculate_cagr(total_return, start_date, end_date):
    """Calculate Compound Annual Growth Rate."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    years = (end - start).days / 365.25
    return ((1 + total_return) ** (1 / years) - 1) * 100


def analyze_trade_duration(results_file):
    """Analyze trade duration from results file."""
    try:
        with open(results_file) as f:
            data = json.load(f)

        # Extract trade information
        trades = []
        for fold in data.get("fold_results", []):
            # This would need to be enhanced based on actual results structure
            trades.append(
                {
                    "fold_id": fold.get("fold_id", 0),
                    "n_trades": fold.get("n_trades", 0),
                    "avg_duration": fold.get("metrics", {}).get("median_hold", 1),
                }
            )

        return trades
    except Exception as e:
        print(f"Error analyzing trade duration: {e}")
        return []


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
        metrics = run_walkforward_analysis(
            "SPY", "2019-01-01", "2023-12-31", temp_config, "results/sensitivity"
        )

        if metrics:
            results.append(
                {
                    "variation": variation,
                    "value": base_value * variation,
                    "sharpe": metrics.get("sharpe", 0),
                    "max_dd": metrics.get("max_dd", 0),
                }
            )

        # Clean up
        os.remove(temp_config)

    return results


def main():
    """Main analysis function."""
    print("🚀 Starting Risk Profile Analysis")
    print("=" * 50)

    # Configuration
    symbols = ["SPY", "BTC-USD", "TSLA"]
    risk_profiles = ["risk_low.json", "risk_balanced.json", "risk_strict.json"]
    start_date = "2019-01-01"
    end_date = "2023-12-31"

    # Results storage
    all_results = []
    trade_duration_results = []

    # 1. Profile Shootout
    print("\n📊 Profile Shootout Analysis")
    print("-" * 30)

    for profile in risk_profiles:
        config_file = f"config/{profile}"
        profile_name = profile.replace(".json", "")

        print(f"\nAnalyzing {profile_name}...")

        for symbol in symbols:
            output_dir = f"results/risk_analysis/{profile_name}/{symbol}"
            os.makedirs(output_dir, exist_ok=True)

            metrics = run_walkforward_analysis(
                symbol, start_date, end_date, config_file, output_dir
            )

            if metrics:
                cagr = calculate_cagr(metrics.get("total_return", 0), start_date, end_date)
                positive_fold_pct = (
                    metrics.get("positive_folds", 0) / metrics.get("total_folds", 1)
                ) * 100

                result = {
                    "profile": profile_name,
                    "symbol": symbol,
                    "sharpe": metrics.get("sharpe", 0),
                    "cagr": cagr,
                    "max_dd": metrics.get("max_dd", 0),
                    "positive_fold_pct": positive_fold_pct,
                    "total_trades": metrics.get("total_trades", 0),
                    "hit_rate": metrics.get("hit_rate", 0),
                }

                all_results.append(result)

                print(
                    f"  {symbol}: Sharpe={result['sharpe']:.3f}, CAGR={result['cagr']:.2f}%, MaxDD={result['max_dd']:.3f}"
                )

    # 2. Trade Duration Analysis
    print("\n📈 Trade Duration Analysis")
    print("-" * 30)

    for profile in risk_profiles:
        profile_name = profile.replace(".json", "")

        for symbol in symbols:
            results_file = f"results/risk_analysis/{profile_name}/{symbol}/results.json"

            if os.path.exists(results_file):
                trades = analyze_trade_duration(results_file)

                if trades:
                    avg_trades = np.mean([t["n_trades"] for t in trades])
                    avg_duration = np.mean([t["avg_duration"] for t in trades])

                    trade_duration_results.append(
                        {
                            "profile": profile_name,
                            "symbol": symbol,
                            "avg_trades": avg_trades,
                            "avg_duration": avg_duration,
                        }
                    )

    # 3. Sensitivity Analysis
    print("\n🔍 Sensitivity Analysis (Balanced Profile)")
    print("-" * 40)

    base_config = load_config("config/risk_balanced.json")
    base_position_size = base_config["risk_params"]["max_position_size"]
    variations = [0.75, 0.875, 1.0, 1.125, 1.25]  # ±25%

    sensitivity_results = run_sensitivity_analysis(
        base_config, "max_position_size", base_position_size, variations
    )

    # Generate Reports
    print("\n📋 Generating Reports...")

    # Profile Shootout Summary
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

    # Trade Duration Summary
    if trade_duration_results:
        df_trades = pd.DataFrame(trade_duration_results)

        print("\n" + "=" * 60)
        print("📈 TRADE DURATION ANALYSIS")
        print("=" * 60)

        for profile in df_trades["profile"].unique():
            profile_data = df_trades[df_trades["profile"] == profile]
            print(f"\n{profile.upper()} PROFILE:")
            print("-" * 30)

            for _, row in profile_data.iterrows():
                print(
                    f"{row['symbol']:8} | Avg Trades: {row['avg_trades']:6.1f} | Avg Duration: {row['avg_duration']:6.1f} days"
                )

    # Sensitivity Analysis Summary
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

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save profile shootout results
    df_results.to_csv(f"results/risk_analysis/profile_shootout_{timestamp}.csv", index=False)

    # Save trade duration results
    if trade_duration_results:
        df_trades.to_csv(f"results/risk_analysis/trade_duration_{timestamp}.csv", index=False)

    # Save sensitivity results
    if sensitivity_results:
        df_sensitivity = pd.DataFrame(sensitivity_results)
        df_sensitivity.to_csv(f"results/risk_analysis/sensitivity_{timestamp}.csv", index=False)

    print("\n✅ Analysis complete! Results saved to results/risk_analysis/")
    print("📁 Files created:")
    print(f"   - profile_shootout_{timestamp}.csv")
    if trade_duration_results:
        print(f"   - trade_duration_{timestamp}.csv")
    if sensitivity_results:
        print(f"   - sensitivity_{timestamp}.csv")


if __name__ == "__main__":
    main()
