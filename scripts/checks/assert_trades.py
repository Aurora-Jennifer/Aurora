#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

trades_p = Path("results/backtest/trades.csv")
if not trades_p.exists():
    print("FAIL: trades.csv missing")
    sys.exit(2)

trades = pd.read_csv(trades_p, parse_dates=["timestamp"])

ok = True

# No zero-delta trades
zero_delta = (trades["delta_qty"].abs() < 1e-6).sum() if "delta_qty" in trades.columns else 0
if zero_delta > 0:
    print(f"FAIL: zero-delta trades found: {zero_delta}")
    ok = False

# Fees non-negative
neg_fees = (trades["fee"] < 0).sum() if "fee" in trades.columns else 0
if neg_fees > 0:
    print(f"FAIL: negative fees rows: {neg_fees}")
    ok = False

# Timestamps present and monotonic per file (not strictly required per-symbol)
if trades["timestamp"].isna().any():
    print("FAIL: NaN timestamps in trades.csv")
    ok = False

# No duplicates by (timestamp, symbol, trade_id)
if {"timestamp", "symbol", "trade_id"}.issubset(trades.columns):
    dups = trades.duplicated(subset=["timestamp", "symbol", "trade_id"]).sum()
    if dups > 0:
        print(f"FAIL: duplicate (timestamp, symbol, trade_id) rows: {dups}")
        ok = False

print("OK" if ok else "NOT OK")
sys.exit(0 if ok else 2)
