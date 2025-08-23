
import numpy as np
import pandas as pd
import pytest

from core.ml.build_features import build_matrix


def _synth_df(n: int = 600) -> pd.DataFrame:
    t = pd.date_range("2020-01-01", periods=n, freq="D")
    price = 100 + np.cumsum(np.random.default_rng(42).normal(0, 1, size=n)).astype(np.float32)
    return pd.DataFrame({"Open": price, "High": price * 1.001, "Low": price * 0.999, "Close": price, "Volume": np.full(n, 1_000, dtype=np.int64)}, index=t)


@pytest.mark.sanity
def test_schema_features_fp32_and_nonan():
    df = _synth_df(400)
    X, y = build_matrix(df)
    assert X.shape[0] == y.shape[0] and X.shape[0] > 0
    # dtypes are float32
    bad = [c for c in X.columns if str(X[c].dtype) != "float32"]
    assert not bad, f"non-float32 columns: {bad}"
    # no NaNs
    assert int(np.isnan(X.values).sum()) == 0


@pytest.mark.sanity
def test_negative_control_shuffled_labels_yields_zero_ic():
    try:
        from sklearn.linear_model import Ridge
    except Exception:  # pragma: no cover
        pytest.skip("scikit-learn not available")
    df = _synth_df(600)
    X, y = build_matrix(df)
    X = X.astype("float32")
    y = y.astype("float32")
    n = len(y)
    np.arange(n)
    split = int(n * 0.8)
    rng = np.random.default_rng(123)
    y_shuf = pd.Series(y.values.copy())
    rng.shuffle(y_shuf.values)
    mdl = Ridge(alpha=1.0)
    mdl.fit(X.iloc[:split], y_shuf.iloc[:split])
    preds = pd.Series(mdl.predict(X.iloc[split:]).astype(np.float32))
    from scipy.stats import spearmanr

    ic, _ = spearmanr(preds.values, y.iloc[split:].values)
    ic = 0.0 if np.isnan(ic) else float(ic)
    assert abs(ic) < 0.05, f"shuffled-label IC too large: {ic:.4f}"


