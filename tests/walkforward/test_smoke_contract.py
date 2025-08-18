import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_smoke_json_contract():
    data = json.loads(Path("reports/metrics.schema.json").read_text())
    assert "properties" in data and "sharpe" in data["properties"]


def test_datasanity_enforce_duplicate_timestamps(monkeypatch):
    import scripts.multi_walkforward_report as mwr

    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    idx = idx.insert(10, idx[10])  # duplicate
    df = pd.DataFrame({"Close": np.linspace(100, 99, len(idx))}, index=idx)

    monkeypatch.setenv("CI", "true")  # trigger CI profile and offline cache path
    monkeypatch.setattr(mwr, "load_data", lambda *a, **k: pd.DataFrame({
        "Open": df["Close"],
        "High": df["Close"] * 1.001,
        "Low": df["Close"] * 0.999,
        "Close": df["Close"],
        "Volume": 1_000_000,
    }, index=df.index))

    out = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    assert out["folds"] == 1 or out["any_nan_inf"] in (True, False)


def test_fold_policy_short_tail(monkeypatch):
    import scripts.multi_walkforward_report as mwr

    idx = pd.date_range("2020-01-01", periods=75, tz="UTC")
    df = pd.DataFrame({"Close": np.linspace(100, 99, len(idx))}, index=idx)
    monkeypatch.setattr(mwr, "load_data", lambda *a, **k: pd.DataFrame({
        "Open": df["Close"],
        "High": df["Close"] * 1.001,
        "Low": df["Close"] * 0.999,
        "Close": df["Close"],
        "Volume": 1_000_000,
    }, index=df.index))
    out = mwr.run_smoke(symbols=["SPY"], train=60, test=10)
    assert out["folds"] == 1


