import numpy as np
import pytest

from core.ml.build_features import build_matrix
from core.ml.contracts import hash_array


@pytest.mark.sanity
def test_featurizer_determinism(load_clean_df):
    df = load_clean_df("SPY", size="tiny")
    X1df, y1s = build_matrix(df, horizon=1)
    X2df, y2s = build_matrix(df.copy(), horizon=1)
    X1, X2 = X1df.to_numpy(), X2df.to_numpy()
    y1, y2 = y1s.to_numpy(), y2s.to_numpy()
    assert hash_array(X1) == hash_array(X2)
    assert np.array_equal(y1, y2)


