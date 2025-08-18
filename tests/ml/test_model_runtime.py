import numpy as np
import pandas as pd

from ml.runtime import set_seeds, build_features, infer_weights


class TinyModel:
    def predict(self, X):
        X = np.asarray(X)
        return X[-1:].sum(axis=1)


def test_infer_weights_deterministic():
    set_seeds(1337)
    idx = pd.date_range("2020-01-01", periods=200, tz="UTC")
    close = np.linspace(100, 110, len(idx))
    df = pd.DataFrame({"Close": close}, index=idx)
    feats = build_features(df, ["ret_1d", "ret_5d", "vol_10d"])
    w1 = infer_weights(TinyModel(), feats, ["ret_1d", "ret_5d", "vol_10d"], "tanh", 0.5, min_bars=60)
    w2 = infer_weights(TinyModel(), feats, ["ret_1d", "ret_5d", "vol_10d"], "tanh", 0.5, min_bars=60)
    assert isinstance(w1, dict) and w1 == w2


def test_bad_model_fallback():
    class BadModel:
        def predict(self, X):
            return [float("nan")]

    idx = pd.date_range("2020-01-01", periods=200, tz="UTC")
    df = pd.DataFrame({"Close": np.linspace(100, 110, len(idx))}, index=idx)
    feats = build_features(df, ["ret_1d", "ret_5d", "vol_10d"])
    w = infer_weights(BadModel(), feats, ["ret_1d", "ret_5d", "vol_10d"], "tanh", 0.5, min_bars=60)
    assert w.get("status") == "HOLD"


