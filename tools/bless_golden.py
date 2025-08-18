import json
from pathlib import Path
import os, sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.engine import backtest as bt


BASE = Path("baselines/spy_golden.json")


def run():
    df = pd.read_parquet("tests/golden/SPY.parquet")
    cfg = {"walkforward": {"fold_length": 60, "step_size": 10, "allow_truncated_final_fold": True}}
    res = bt.run_backtest(df, cfg, sanity_profile="walkforward_ci")
    if res.get("status", "OK") != "OK":
        raise SystemExit(f"Backtest failed: {res}")
    base = {
        "sharpe": float(res.get("sharpe", 0.0)),
        "max_drawdown": float(res.get("max_drawdown", 0.0)),
        "trades": int(res.get("trades", 0)),
    }
    BASE.parent.mkdir(parents=True, exist_ok=True)
    BASE.write_text(json.dumps(base, indent=2))
    print("Blessed:", base)


if __name__ == "__main__":
    run()


