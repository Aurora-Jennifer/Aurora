import numpy as np
import pandas as pd

from core.engine import backtest as bt


def test_backtest_infinite_values():
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    close = np.linspace(100, 99, len(idx))
    close[20] = np.inf
    df = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000}, index=idx
    )
    res = bt.run_backtest(df, {"walkforward": {}}, sanity_profile="walkforward_ci")
    assert isinstance(res, dict) and res.get("status") == "FAIL"
    assert res.get("violation_code") == "INF_VALUES"
