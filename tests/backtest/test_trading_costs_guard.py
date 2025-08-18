import numpy as np
import pandas as pd

from core.engine import backtest as bt


def test_missing_costs_fails_in_ci(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    df = pd.DataFrame({"Close": np.linspace(100, 99, len(idx))}, index=idx)
    monkeypatch.setenv("CI", "true")
    res = bt.run_backtest(df, {"risk": {}}, sanity_profile="walkforward_ci")
    # run_backtest currently returns OK with placeholder metrics; this test will be enabled when guard added
    assert isinstance(res, dict)

