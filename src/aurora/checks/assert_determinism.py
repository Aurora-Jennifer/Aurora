#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

# Run baseline twice and compare summaries
cmd = [
    "python",
    "backtest.py",
    "--start-date",
    "2023-08-01",
    "--end-date",
    "2025-08-12",
    "--symbols",
    "SPY",
    "QQQ",
    "--config",
    "config/enhanced_paper_trading_config.json",
]

for i in range(2):
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print("FAIL: backtest run", i + 1, "failed")
        sys.exit(2)
    time.sleep(1)

s = Path("results/backtest/summary.txt").read_text().splitlines()
vals = {}
for ln in s:
    if ":" in ln:
        k, v = ln.split(":", 1)
        vals[k.strip()] = v.strip().replace(",", "").replace("$", "")
fe = float(vals.get("Final Equity", "nan"))
tp = float(vals.get("Total PnL", "nan"))
# Rerun once more and parse again
subprocess.run(cmd)
s2 = Path("results/backtest/summary.txt").read_text().splitlines()
vals2 = {}
for ln in s2:
    if ":" in ln:
        k, v = ln.split(":", 1)
        vals2[k.strip()] = v.strip().replace(",", "").replace("$", "")
fe2 = float(vals2.get("Final Equity", "nan"))
tp2 = float(vals2.get("Total PnL", "nan"))

ok = abs(fe - fe2) < 1e-6 and abs(tp - tp2) < 1e-6
print("Determinism:", "OK" if ok else "NOT OK", fe, fe2, tp, tp2)
sys.exit(0 if ok else 2)
