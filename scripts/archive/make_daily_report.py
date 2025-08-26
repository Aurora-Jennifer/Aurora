#!/usr/bin/env python3
"""
Daily Trading Report Generator
Creates a summary report of the day's trading activity.
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def load_performance_data(results_dir: str = "results") -> dict:
    """Load performance data from results directory."""
    try:
        # Load performance report
        perf_file = Path(results_dir) / "performance_report.json"
        if not perf_file.exists():
            print(f"❌ Performance report not found: {perf_file}")
            return {}

        with open(perf_file) as f:
            perf_data = json.load(f)

        # Load trade history
        trades_file = Path(results_dir) / "trade_history.csv"
        trades = []
        if trades_file.exists():
            trades = pd.read_csv(trades_file)

        # Load daily returns
        returns_file = Path(results_dir) / "daily_returns.csv"
        returns = []
        if returns_file.exists():
            returns = pd.read_csv(returns_file)

        return {
            "performance": perf_data,
            "trades": trades.to_dict("records") if len(trades) > 0 else [],
            "returns": returns.to_dict("records") if len(returns) > 0 else [],
        }

    except Exception as e:
        print(f"❌ Error loading performance data: {e}")
        return {}


def generate_daily_report(data: dict) -> str:
    """Generate a formatted daily report."""
    perf = data.get("performance", {})
    trades = data.get("trades", [])
    returns = data.get("returns", [])

    today = date.today().strftime("%Y-%m-%d")

    report = f"""
📊 Daily Trading Report - {today}
{"=" * 50}

💰 PERFORMANCE SUMMARY
  Total Return: {perf.get("total_return", 0):.2%}
  Current Capital: ${perf.get("current_capital", 0):,.2f}
  Sharpe Ratio: {perf.get("sharpe_ratio", 0):.2f}
  Max Drawdown: {perf.get("max_drawdown", 0):.2%}
  Total Trades: {perf.get("total_trades", 0)}

📈 TRADING ACTIVITY
  Trades Today: {len(trades)}
"""

    if trades:
        report += "  Recent Trades:\n"
        for trade in trades[-5:]:  # Last 5 trades
            report += f"    {trade.get('symbol', 'N/A')} {trade.get('action', 'N/A')} {trade.get('size', 0):.2f} @ ${trade.get('price', 0):.2f}\n"
    else:
        report += "  No trades executed today\n"

    if returns:
        latest_return = returns[-1] if returns else {}
        report += f"\n📅 DAILY RETURNS\n  Today's Return: {latest_return.get('return', 0):.2%}\n"

    # Regime information
    regime_stats = perf.get("regime_stats", {})
    if regime_stats:
        report += "\n🎯 REGIME ANALYSIS\n"
        for regime, stats in regime_stats.items():
            report += f"  {regime.title()}: {stats.get('count', 0)} days, avg confidence: {stats.get('avg_confidence', 0):.2f}\n"

    report += f"\n{'=' * 50}\n"
    report += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    return report


def save_report(report: str, output_dir: str = "results"):
    """Save the report to file."""
    try:
        output_path = Path(output_dir) / f"daily_report_{date.today().strftime('%Y-%m-%d')}.txt"
        with open(output_path, "w") as f:
            f.write(report)
        print(f"✅ Daily report saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving report: {e}")


def main():
    """Main function."""
    try:
        # Load data
        data = load_performance_data()
        if not data:
            print("❌ No performance data available")
            sys.exit(1)

        # Generate report
        report = generate_daily_report(data)

        # Print report
        print(report)

        # Save report
        save_report(report)

    except Exception as e:
        print(f"❌ Error generating daily report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
