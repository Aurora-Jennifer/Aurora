import pandas as pd

import scripts.multi_walkforward_report as mwr


def test_datasanity_empty_series(monkeypatch):
    # Empty dataframe with proper columns but no rows
    df = pd.DataFrame(
        {
            "Open": [],
            "High": [],
            "Low": [],
            "Close": [],
            "Volume": [],
        }
    )
    df.index = pd.DatetimeIndex([], tz="UTC")
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(mwr, "load_data", lambda *a, **k: df)
    out = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    assert out["status"] == "FAIL"
    assert out["violation_code"] == "EMPTY_SERIES"
