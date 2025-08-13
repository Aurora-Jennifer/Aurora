#!/usr/bin/env python3
"""
Simple Terminal Dashboard for Trading Bot
"""

import json
import os
import time
from datetime import datetime

import pandas as pd


def load_data():
    """Load all trading data."""
    data = {}

    # Load performance
    try:
        with open("results/performance_report.json") as f:
            data["performance"] = json.load(f)
    except:
        data["performance"] = {}

    # Load trades
    try:
        data["trades"] = pd.read_csv("results/trade_history.csv")
    except:
        data["trades"] = pd.DataFrame()

    # Load recent logs
    try:
        with open("logs/trading_bot.log") as f:
            lines = f.readlines()
            data["recent_logs"] = lines[-10:]  # Last 10 lines
    except:
        data["recent_logs"] = []

    return data


def display_dashboard(data):
    """Display dashboard in terminal."""
    os.system("clear")  # Clear screen

    # Title
    print("🚀 Trading Bot Dashboard")
    print("=" * 50)

    # Performance metrics
    perf = data.get("performance", {})

    print("📊 Performance Metrics:")
    total_return = perf.get("total_return", 0)
    sharpe = perf.get("sharpe_ratio", 0)
    trades = perf.get("total_trades", 0)

    print(f"  Total Return: {total_return:.1%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Total Trades: {trades}")
    print()

    # Recent trades
    if not data["trades"].empty:
        print("💰 Recent Trades:")
        recent_trades = data["trades"].tail(5)
        for _, trade in recent_trades.iterrows():
            symbol = trade.get("symbol", "N/A")
            action = trade.get("action", "N/A")
            pnl = trade.get("pnl", 0)
            timestamp = trade.get("timestamp", "N/A")

            trade_str = f"  {symbol} {action} ${pnl:.2f} ({timestamp})"
            print(trade_str)
        print()

    # Recent logs
    print("📝 Recent Logs:")
    for log in data["recent_logs"]:
        print(f"  {log.strip()}")

    # Footer
    print()
    print("=" * 50)
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main dashboard loop."""
    print("Starting Trading Bot Dashboard...")
    print("Press Ctrl+C to exit")
    print()

    try:
        while True:
            data = load_data()
            display_dashboard(data)
            time.sleep(5)  # Refresh every 5 seconds
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
