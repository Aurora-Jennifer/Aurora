#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

ledger_p = Path("results/backtest/ledger.csv")
if not ledger_p.exists():
    print("No ledger.csv found", file=sys.stderr)
    sys.exit(1)

ledger = pd.read_csv(ledger_p, parse_dates=["date"])
eq = float(ledger["equity"].iloc[-1])
# Optional column presence
mdd = (
    float(ledger["max_drawdown"].iloc[-1])
    if "max_drawdown" in ledger.columns
    else float("nan")
)
trades_p = Path("results/backtest/trades.csv")
trades = pd.read_csv(trades_p) if trades_p.exists() else pd.DataFrame()
print(f"DAILY: equity={eq:,.2f} mdd={mdd:.3f} trades={len(trades)}")

# hard guardrails (edit thresholds)
if mdd > 0.20:
    sys.exit(10)
