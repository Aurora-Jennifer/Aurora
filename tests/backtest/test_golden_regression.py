import json
from pathlib import Path

import pandas as pd

from core.engine import backtest as bt


EPS = {"sharpe": 0.03, "max_drawdown": 0.02, "trades": 2}


def test_golden_regression():
    df = pd.read_parquet(Path("tests/golden/SPY.parquet"))
    res = bt.run_backtest(
        df,
        {"walkforward": {"fold_length": 60, "step_size": 10, "allow_truncated_final_fold": True}},
        sanity_profile="walkforward_ci",
    )
    assert res.get("status", "OK") == "OK", res
    base = json.loads(Path("baselines/spy_golden.json").read_text())
    assert abs(res["sharpe"] - base["sharpe"]) <= EPS["sharpe"]
    assert base["max_drawdown"] - EPS["max_drawdown"] <= res["max_drawdown"] <= base["max_drawdown"] + EPS["max_drawdown"]
    assert abs(res["trades"] - base["trades"]) <= EPS["trades"]


