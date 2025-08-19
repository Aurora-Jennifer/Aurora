import numpy as np
import pandas as pd

from core.engine import backtest as bt


def test_backtest_duplicate_timestamps():
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    idx = idx.insert(10, idx[10])
    close = np.linspace(100, 99, len(idx))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    res = bt.run_backtest(df, {"walkforward": {}}, sanity_profile="walkforward_ci")
    assert isinstance(res, dict) and res.get("status") == "FAIL"
    assert res.get("violation_code") in {"DUP_TS", "NON_MONO_INDEX"}
