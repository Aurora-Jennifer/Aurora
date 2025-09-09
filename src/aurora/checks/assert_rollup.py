#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

rollup_p = Path("results/backtest/rollup.csv")
if not rollup_p.exists():
    print("FAIL: rollup.csv not found")
    sys.exit(2)

df = pd.read_csv(rollup_p)

required = ["run_key", "start_date", "end_date", "symbols", "total_pnl", "sharpe_like"]
missing = [c for c in required if c not in df.columns]
if missing:
    print("FAIL: rollup missing columns:", ",".join(missing))
    sys.exit(2)

if len(df) == 0:
    print("FAIL: rollup.csv is empty")
    sys.exit(2)

if df["run_key"].duplicated().any():
    print("FAIL: duplicated run_key in rollup.csv")
    sys.exit(2)

print("OK")
