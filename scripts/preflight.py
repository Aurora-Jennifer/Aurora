#!/usr/bin/env python3
import datetime as dt
import json
import os
import pathlib
import sys

import pandas as pd

# Add project root to sys.path when running from scripts/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.engine.backtest import BacktestEngine

CFG_PATH = "config/enhanced_paper_trading_config.json"

cfg = json.load(open(CFG_PATH))
assert not cfg.get(
    "carry_positions_from_warmup", False
), "Set carry_positions_from_warmup=false"

eng = BacktestEngine(CFG_PATH)
eng.run_backtest("2024-01-02", "2024-01-09", ["SPY"])
s = eng.get_last_summary()
needed = ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades"]
missing = [k for k in needed if k not in s]
if missing:
    print("Missing metrics:", missing)
    sys.exit(2)

p = pathlib.Path("results/backtest/trades.csv")
if p.exists():
    t = pd.read_csv(p, parse_dates=["timestamp"])
    if not t.empty:
        d0, d1 = dt.date(2024, 1, 2), dt.date(2024, 1, 9)
        ts = t["timestamp"].dt.date
        if ((ts < d0) | (ts > d1)).any():
            print("Timestamp leakage in trades.csv")
            sys.exit(3)

print("OK")
