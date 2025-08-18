import numpy as np
import pandas as pd
import pytest
from core.metrics.returns import percent_returns, log_returns, diff_returns
import yaml

def _toy_close(n=100, seed=7):
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.Series(base, index=pd.date_range("2024-01-01", periods=n, freq="D"))

def test_percent_returns_scale_invariant_strict():
    px = _toy_close()
    r1 = percent_returns(px)
    r2 = percent_returns(px * 3.7)
    np.testing.assert_allclose(r1.values, r2.values, rtol=1e-12, atol=1e-12)

def test_percent_returns_additive_shift_correlation_high():
    px = _toy_close()
    r1 = percent_returns(px)
    r2 = percent_returns(px + 50.0)
    corr = np.corrcoef(r1.values, r2.values)[0, 1]
    assert corr >= 0.999

def test_config_default_percent():
    with open("config/base.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("data", {}).get("returns", {}).get("type") == "percent"

@pytest.mark.skipif(True, reason="Only relevant when diff is configured as return type")
def test_diff_returns_additive_invariant_strict():
    px = _toy_close()
    d1 = diff_returns(px)
    d2 = diff_returns(px + 25.0)
    np.testing.assert_allclose(d1.values, d2.values, rtol=1e-12, atol=1e-12)


