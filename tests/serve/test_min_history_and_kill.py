import numpy as np
import pandas as pd
import pytest

from serve.adapter import _Predictor as Predictor


def test_min_history_guard(tmp_path, monkeypatch):
    side = tmp_path / "sidecar.json"
    side.write_text('{"features":["x","y"],"dtypes":["float32","float32"]}')
    df = pd.DataFrame({"x": [0.0] * 10, "y": [0.0] * 10})
    # Enable kill switch to avoid loading a real ONNX during this guard test
    monkeypatch.setenv("SERVE_DUMMY", "1")
    p = Predictor(onnx_path="/dev/null", sidecar_path=str(side), min_history=50)
    with pytest.raises(ValueError):
        _ = p.predict_batch(df)


def test_kill_switch_returns_zeros(tmp_path, monkeypatch):
    side = tmp_path / "sidecar.json"
    side.write_text('{"features":["x","y"],"dtypes":["float32","float32"]}')
    monkeypatch.setenv("SERVE_DUMMY", "1")
    p = Predictor(onnx_path="/dev/null", sidecar_path=str(side), min_history=1)
    df = pd.DataFrame({"x": np.zeros(2, np.float32), "y": np.zeros(2, np.float32)})
    out = p.predict_batch(df)
    assert (out == 0.0).all() and out.dtype == np.float32


