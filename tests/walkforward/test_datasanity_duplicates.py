import numpy as np
import pandas as pd

import scripts.multi_walkforward_report as mwr


def test_datasanity_duplicate_timestamps(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    idx = idx.insert(10, idx[10])  # one duplicate
    df = pd.DataFrame({"Close": np.linspace(100, 99, len(idx))}, index=idx)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(mwr, "load_data", lambda *a, **k: df)
    out = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    assert out["status"] == "FAIL"
    assert out["violation_code"] in {"DUP_TS", "NON_MONO_INDEX"}


