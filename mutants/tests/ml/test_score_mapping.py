import numpy as np
from ml.runtime import _map_scores_to_weights


def test_mapping_tanh_linear_softmax_clips_and_finite():
    arr = np.array([100.0, -100.0, 0.0])
    for map_name in ("tanh", "linear", "softmax"):
        w = _map_scores_to_weights(arr, map_name, max_abs=0.5)
        assert np.isfinite(w).all()
        assert np.max(np.abs(w)) <= 0.5 + 1e-9
