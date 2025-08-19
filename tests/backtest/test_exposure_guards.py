import numpy as np
import pandas as pd

from core.engine import backtest as bt


def _df():
    idx = pd.date_range("2020-01-01", periods=80, tz="UTC")
    return pd.DataFrame({"Close": np.linspace(100, 99, len(idx))}, index=idx)


def test_gross_exposure_limit_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    res = bt.run_backtest(
        _df(),
        {"risk": {"slippage_bps": 5, "fee_bps": 1, "max_leverage": 1.0, "max_gross_exposure": 2.5}},
        sanity_profile="walkforward_ci",
    )
    assert res["status"] == "FAIL" and res["violation_code"] == "GROSS_EXPOSURE_LIMIT"


def test_position_limit_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    res = bt.run_backtest(
        _df(),
        {"risk": {"slippage_bps": 5, "fee_bps": 1, "max_leverage": 1.0, "max_position_pct": 1.5}},
        sanity_profile="walkforward_ci",
    )
    assert res["status"] == "FAIL" and res["violation_code"] == "POSITION_LIMIT"
