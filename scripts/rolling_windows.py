#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure backtest import works when run from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backtest import BacktestEngine

START = "2019-01-01"
END = "2025-01-01"
SYMBOLS = ["SPY"]
WINDOW_DAYS = 30
STEP_DAYS = 5


def trading_days_between(dates: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    # Inclusive start, inclusive end
    s = dates.searchsorted(pd.Timestamp(start))
    e = dates.searchsorted(pd.Timestamp(end), side="right")
    return dates[s:e]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=WINDOW_DAYS)
    args = parser.parse_args()
    window_days = int(args.window)

    cfg_path = Path("config/enhanced_paper_trading_config.json")
    if not cfg_path.exists():
        print("Config file not found", file=sys.stderr)
        sys.exit(1)
    eng = BacktestEngine(str(cfg_path))

    # Generate a simple business-day calendar; replace with market calendar if available
    dates = pd.bdate_range(START, END)
    rows = []
    for i in range(0, len(dates) - window_days, STEP_DAYS):
        s = dates[i].date().isoformat()
        e = dates[i + window_days - 1].date().isoformat()
        eng.run_backtest(s, e, SYMBOLS)
        summary = eng.get_last_summary()
        rows.append(
            {
                "start": s,
                "end": e,
                "ann_return": summary.get("Annualized Return", 0.0),
                "sharpe": summary.get("Sharpe Ratio", 0.0),
                "max_dd": summary.get("Max Drawdown", 0.0),
                "trades": summary.get("Total Trades", 0),
            }
        )
    out_path = Path("results/rolling_30d.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        print("No rows produced", file=sys.stderr)


if __name__ == "__main__":
    main()
