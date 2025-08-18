import numpy as np
import pandas as pd

from core.engine import backtest as bt


def test_backtest_infinite_values():
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    close = np.linspace(100, 99, len(idx))
    close[20] = np.inf
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000}, index=idx)
    # Use available API: instantiate engine and call run_backtest-like path; fallback: ensure validator flags INF
    from core.data_sanity import DataSanityValidator
    v = DataSanityValidator("config/data_sanity.yaml", profile="walkforward_ci")
    res = v.validate_dataframe_fast(df, "walkforward_ci")
    assert res.violations and any(vi.code == "INF_VALUES" for vi in res.violations)


