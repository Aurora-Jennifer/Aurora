import numpy as np
import pandas as pd

import scripts.multi_walkforward_report as mwr


def test_short_fold_policy(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=75, tz="UTC")
    df = pd.DataFrame({"Close": np.linspace(100, 99, len(idx))}, index=idx)
    monkeypatch.setattr(mwr, "load_data", lambda *a, **k: df)
    out = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    assert out.get("folds", 0) == 1 or out.get("violation_code") == "SHORT_FOLD"
