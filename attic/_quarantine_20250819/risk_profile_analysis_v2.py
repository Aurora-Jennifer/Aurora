#!/usr/bin/env python3
"""
Risk Profile Analysis Script v2
Uses backtest CLI to test different risk profiles across multiple symbols.
"""

import json
import os
import subprocess
from datetime import datetime

import pandas as pd


def run_backtest_analysis(symbol, start_date, end_date, profile_config):
    """Run backtest analysis for a single symbol and profile."""
    try:
        cmd = [
            "python",
            "cli/backtest.py",
            "--start",
            start_date,
            "--end",
            end_date,
            "--symbols",
            symbol,
            "--profile",
            profile_config,
            "--fast",
        ]

        print(f"Running backtest for {symbol} with {profile_config}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Parse results from output
            lines = result.stdout.split("\n")
            metrics = {}

            for line in lines:
                if "Total Return:" in line:
                    metrics["total_return"] = (
                        float(line.split(":")[1].strip().replace("%", "")) / 100
                    )
                elif "Sharpe Ratio:" in line:
                    metrics["sharpe"] = float(line.split(":")[1].strip())
                elif "Max Drawdown:" in line:
                    metrics["max_dd"] = float(line.split(":")[1].strip().replace("%", "")) / 100
                elif "Total Trades:" in line:
                    metrics["total_trades"] = int(line.split(":")[1].strip())
                elif "Final Value:" in line:
                    final_value = line.split(":")[1].strip().replace("$", "").replace(",", "")
                    metrics["final_value"] = float(final_value)
                elif "Initial Capital:" in line:
                    initial_capital = line.split(":")[1].strip().replace("$", "").replace(",", "")
                    metrics["initial_capital"] = float(initial_capital)

            return metrics
        print(f"Error running backtest for {symbol}: {result.stderr}")
        return None

    except subprocess.TimeoutExpired:
        print(f"Timeout running backtest for {symbol}")
        return None
    except Exception as e:
        print(f"Exception running backtest for {symbol}: {e}")
        return None


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
        metrics = run_backtest_analysis("SPY", "2019-01-01", "2023-12-31", temp_config)

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
        if os.path.exists(temp_config):
            os.remove(temp_config)

    return results


def main():
    """Main analysis function."""
    print("🚀 Starting Risk Profile Analysis v2")
    print("=" * 50)

    # Configuration
    symbols = ["SPY", "BTC-USD", "TSLA"]
    risk_profiles = [
        "config/risk_low.json",
        "config/risk_balanced.json",
        "config/risk_strict.json",
    ]
    start_date = "2019-01-01"
    end_date = "2023-12-31"

    # Results storage
    all_results = []

    # 1. Profile Shootout
    print("\n📊 Profile Shootout Analysis")
    print("-" * 30)

    for profile in risk_profiles:
        profile_name = profile.replace("config/", "").replace(".json", "")
        print(f"\nAnalyzing {profile_name}...")

        for symbol in symbols:
            metrics = run_backtest_analysis(symbol, start_date, end_date, profile)

            if metrics:
                cagr = calculate_cagr(metrics.get("total_return", 0), start_date, end_date)

                result = {
                    "profile": profile_name,
                    "symbol": symbol,
                    "sharpe": metrics.get("sharpe", 0),
                    "cagr": cagr,
                    "max_dd": metrics.get("max_dd", 0),
                    "total_trades": metrics.get("total_trades", 0),
                    "final_value": metrics.get("final_value", 0),
                    "initial_capital": metrics.get("initial_capital", 100000),
                    "total_return": metrics.get("total_return", 0),
                }

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
                    f"{row['symbol']:8} | Sharpe: {row['sharpe']:6.3f} | CAGR: {row['cagr']:6.2f}% | MaxDD: {row['max_dd']:6.3f} | Trades: {row['total_trades']:4.0f} | Final: ${row['final_value']:,.0f}"
                )

        print("\n" + "=" * 60)
        print("📈 TRADE COUNT ANALYSIS")
        print("=" * 60)

        for profile in df_results["profile"].unique():
            profile_data = df_results[df_results["profile"] == profile]
            print(f"\n{profile.upper()} PROFILE:")
            print("-" * 30)

            for _, row in profile_data.iterrows():
                print(
                    f"{row['symbol']:8} | Total Trades: {row['total_trades']:6.0f} | Final Value: ${row['final_value']:,.0f}"
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
