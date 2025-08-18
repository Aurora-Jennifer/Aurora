import numpy as np
import pandas as pd
from ml.feature_stats import stats, psi


def test_stats_finite_and_missing_bounds():
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": [0.0, 0.0, 0.0, 0.0],
    }, index=pd.date_range("2020-01-01", periods=4, tz="UTC"))
    s = stats(df)
    assert set(s.keys()) == {"a", "b"}
    assert 0.0 <= s["a"]["missing_pct"] <= 1.0
    assert s["b"]["std"] == 0.0


def test_psi_returns_nonnegative_and_finite():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 1000))
    df = pd.DataFrame({"x": x})
    ref = {"x": {"bin_edges": np.linspace(-4, 4, 11).tolist(), "ref_probs": [0.1] * 10}}
    p = psi(df, ref)
    assert "x" in p and p["x"] >= 0.0
    assert np.isfinite(p["x"]) or np.isnan(p["x"]) is False


