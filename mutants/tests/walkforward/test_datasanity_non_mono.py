import numpy as np
import pandas as pd

import scripts.multi_walkforward_report as mwr


def test_datasanity_non_monotonic_index(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    # Make index non-monotonic by swapping two points
    idx = idx.insert(40, idx[10])
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
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(mwr, "load_data", lambda *a, **k: df)
    out = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    assert out["status"] == "FAIL"
    assert out["violation_code"] in {"NON_MONO_INDEX", "DUP_TS"}
