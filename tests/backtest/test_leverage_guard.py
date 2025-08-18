import numpy as np
import pandas as pd

from core.engine import backtest as bt


def test_leverage_limit_in_ci(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    df = pd.DataFrame({"Close": np.linspace(100, 99, len(idx))}, index=idx)
    monkeypatch.setenv("CI", "true")
    res = bt.run_backtest(
        df,
        {"risk": {"slippage_bps": 5, "fee_bps": 1, "max_leverage": 5.0}},
        sanity_profile="walkforward_ci",
    )
    assert res["status"] == "FAIL" and res["violation_code"] == "LEVERAGE_LIMIT"


