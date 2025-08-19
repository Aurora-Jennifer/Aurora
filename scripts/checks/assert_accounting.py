#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd

results_dir = Path("results/backtest")
ledger = pd.read_csv(results_dir / "ledger.csv")
results_json = json.loads((results_dir / "results.json").read_text())
trades = pd.read_csv(results_dir / "trades.csv")

final_equity = float(ledger["equity"].iloc[-1])
initial_capital = float(results_json.get("trading_summary", {}).get("initial_capital", 0.0))
# equity-based pnl
pnl_equity = final_equity - initial_capital

# fees non-negative
neg_fees = (trades["fee"] < 0).sum()

ok = True
if neg_fees > 0:
    print(f"FAIL: negative fees rows={neg_fees}")
    ok = False

# Print reconciliation
print(
    f"Accounting: final_equity={final_equity:.2f} initial={initial_capital:.2f} "
    f"equity_pnl={pnl_equity:.2f}"
)

sys.exit(0 if ok else 2)
