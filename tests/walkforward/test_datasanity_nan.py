import numpy as np
import pandas as pd

import scripts.multi_walkforward_report as mwr


def test_datasanity_nan_values(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    close = np.linspace(100, 99, len(idx))
    close[15] = np.nan
    df = pd.DataFrame({
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": 1_000_000,
    }, index=idx)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(mwr, "load_data", lambda *a, **k: df)
    out = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    assert out["status"] == "FAIL"
    assert out["violation_code"] == "NAN_VALUES"


