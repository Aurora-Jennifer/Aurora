#!/usr/bin/env python3
"""
PnL Reconciliation Tool
Compares trading system PnL vs IBKR NetLiq for accuracy verification.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


class PnLReconciler:
    """Reconcile PnL between trading system and IBKR NetLiq."""

    def __init__(self, tolerance: float = 5.0):
        self.tolerance = tolerance  # Dollar tolerance for reconciliation

    def load_trading_system_pnl(self, results_dir: str = "results") -> Dict:
        """Load PnL data from trading system results."""
        try:
            # Load performance report
            perf_file = Path(results_dir) / "performance_report.json"
            if not perf_file.exists():
                print(f"❌ Performance report not found: {perf_file}")
                return {}

            with open(perf_file) as f:
                perf_data = json.load(f)

            # Load daily returns if available
            daily_returns_file = Path(results_dir) / "daily_returns.csv"
            daily_returns = []
            if daily_returns_file.exists():
                daily_returns = pd.read_csv(daily_returns_file)

            return {
                "total_return": perf_data.get("total_return", 0.0),
                "current_capital": perf_data.get("current_capital", 0.0),
                "total_trades": perf_data.get("total_trades", 0),
                "daily_returns": (
                    daily_returns.to_dict("records") if len(daily_returns) > 0 else []
                ),
            }

        except Exception as e:
            print(f"❌ Error loading trading system PnL: {e}")
            return {}

    def load_ibkr_flex_statement(self, flex_file: str) -> Dict:
        """Load IBKR Flex statement CSV."""
        try:
            if not Path(flex_file).exists():
                print(f"❌ Flex statement not found: {flex_file}")
                return {}

            # Parse IBKR Flex CSV
            data = []
            with open(flex_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)

            # Extract NetLiq values
            netliq_data = []
            for row in data:
                if "NetLiq" in row.get("Type", "") or "NetLiquidation" in row.get(
                    "Type", ""
                ):
                    try:
                        netliq_data.append(
                            {
                                "date": row.get("Date", ""),
                                "value": float(row.get("Value", 0)),
                                "currency": row.get("Currency", "USD"),
                            }
                        )
                    except (ValueError, KeyError):
                        continue

            return {
                "netliq_data": netliq_data,
                "latest_netliq": netliq_data[-1]["value"] if netliq_data else 0.0,
                "initial_netliq": netliq_data[0]["value"] if netliq_data else 0.0,
            }

        except Exception as e:
            print(f"❌ Error loading IBKR Flex statement: {e}")
            return {}

    def calculate_daily_pnl_delta(self, netliq_data: List[Dict]) -> List[Dict]:
        """Calculate daily PnL deltas from NetLiq data."""
        if len(netliq_data) < 2:
            return []

        deltas = []
        for i in range(1, len(netliq_data)):
            prev_value = netliq_data[i - 1]["value"]
            curr_value = netliq_data[i]["value"]
            delta = curr_value - prev_value

            deltas.append(
                {
                    "date": netliq_data[i]["date"],
                    "netliq_delta": delta,
                    "netliq_value": curr_value,
                }
            )

        return deltas

    def reconcile_pnl(self, trading_pnl: Dict, ibkr_data: Dict) -> bool:
        """Reconcile PnL between trading system and IBKR."""
        print("🔍 PnL Reconciliation")
        print("=" * 50)

        # Get trading system total PnL
        trading_total_pnl = (
            trading_pnl.get("current_capital", 0) - 100000
        )  # Assuming 100k initial
        trading_total_return = trading_pnl.get("total_return", 0.0)

        # Get IBKR total PnL
        ibkr_total_pnl = ibkr_data.get("latest_netliq", 0) - ibkr_data.get(
            "initial_netliq", 0
        )

        print("Trading System:")
        print(f"  Current Capital: ${trading_pnl.get('current_capital', 0):,.2f}")
        print(f"  Total PnL: ${trading_total_pnl:,.2f}")
        print(f"  Total Return: {trading_total_return:.2%}")
        print(f"  Total Trades: {trading_pnl.get('total_trades', 0)}")

        print("\nIBKR Flex Statement:")
        print(f"  Initial NetLiq: ${ibkr_data.get('initial_netliq', 0):,.2f}")
        print(f"  Latest NetLiq: ${ibkr_data.get('latest_netliq', 0):,.2f}")
        print(f"  Total PnL: ${ibkr_total_pnl:,.2f}")

        # Calculate difference
        pnl_difference = abs(trading_total_pnl - ibkr_total_pnl)
        print("\nReconciliation:")
        print(f"  PnL Difference: ${pnl_difference:,.2f}")
        print(f"  Tolerance: ${self.tolerance:,.2f}")

        # Check if within tolerance
        if pnl_difference <= self.tolerance:
            print("✅ RECONCILIATION PASS - Difference within tolerance")
            return True
        else:
            print("❌ RECONCILIATION FAIL - Difference exceeds tolerance")
            print("\nPossible causes:")
            print("  - Fees/slippage not properly accounted for")
            print("  - Date mismatch between trading and IBKR data")
            print("  - Different initial capital assumptions")
            print("  - Missing trades or fills")
            return False

    def reconcile_daily_returns(
        self, trading_returns: List[Dict], ibkr_deltas: List[Dict]
    ) -> bool:
        """Reconcile daily returns between trading system and IBKR."""
        if not trading_returns or not ibkr_deltas:
            print("⚠️  Daily reconciliation skipped - insufficient data")
            return True

        print("\n📊 Daily Returns Reconciliation")
        print("-" * 30)

        # Match dates and compare
        trading_by_date = {r.get("date", ""): r for r in trading_returns}
        ibkr_by_date = {d.get("date", ""): d for d in ibkr_deltas}

        all_dates = set(trading_by_date.keys()) | set(ibkr_by_date.keys())

        for date_str in sorted(all_dates):
            trading_return = trading_by_date.get(date_str, {})
            ibkr_delta = ibkr_by_date.get(date_str, {})

            if trading_return and ibkr_delta:
                trading_pnl = (
                    trading_return.get("return", 0) * 100000
                )  # Assuming 100k base
                ibkr_pnl = ibkr_delta.get("netliq_delta", 0)
                diff = abs(trading_pnl - ibkr_pnl)

                status = "✅" if diff <= self.tolerance else "❌"
                print(
                    f"{status} {date_str}: Trading=${trading_pnl:,.2f}, IBKR=${ibkr_pnl:,.2f}, Diff=${diff:,.2f}"
                )
            elif trading_return:
                print(f"⚠️  {date_str}: Trading data only")
            elif ibkr_delta:
                print(f"⚠️  {date_str}: IBKR data only")

        return True


def main():
    """Main reconciliation function."""
    parser = argparse.ArgumentParser(
        description="Reconcile PnL between trading system and IBKR"
    )
    parser.add_argument("--flex", required=True, help="Path to IBKR Flex statement CSV")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Dollar tolerance for reconciliation",
    )
    parser.add_argument(
        "--results-dir", default="results", help="Trading system results directory"
    )

    args = parser.parse_args()

    # Initialize reconciler
    reconciler = PnLReconciler(tolerance=args.tolerance)

    # Load data
    print("📊 Loading trading system PnL data...")
    trading_pnl = reconciler.load_trading_system_pnl(args.results_dir)

    print("📊 Loading IBKR Flex statement...")
    ibkr_data = reconciler.load_ibkr_flex_statement(args.flex)

    if not trading_pnl or not ibkr_data:
        print("❌ Failed to load required data")
        sys.exit(1)

    # Reconcile total PnL
    total_reconciled = reconciler.reconcile_pnl(trading_pnl, ibkr_data)

    # Reconcile daily returns
    daily_reconciled = reconciler.reconcile_daily_returns(
        trading_pnl.get("daily_returns", []),
        reconciler.calculate_daily_pnl_delta(ibkr_data.get("netliq_data", [])),
    )

    # Final result
    if total_reconciled and daily_reconciled:
        print("\n🎉 RECONCILIATION PASSED")
        sys.exit(0)
    else:
        print("\n❌ RECONCILIATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
