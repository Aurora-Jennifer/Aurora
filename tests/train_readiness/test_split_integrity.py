import pandas as pd
import pytest

from core.ml.build_features import build_matrix


@pytest.mark.sanity
def test_forward_purged_splits():
    """Test that time series splits are forward-only with purge gaps"""
    # Load golden snapshot data
    import json
    with open("artifacts/snapshots/golden_ml_v1/manifest.json") as f:
        json.load(f)

    # Load SPY data
    df = pd.read_parquet("artifacts/snapshots/golden_ml_v1/SPY.parquet")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    X, y = build_matrix(df, horizon=1)

    # Ensure we have data
    assert len(X) > 0, "No features generated"
    assert len(y) > 0, "No targets generated"

    # Test TimeSeriesSplit with purge gap
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=3, test_size=100, gap=10)

    for train_idx, test_idx in tscv.split(X):
        # Check that test comes after train
        assert max(train_idx) < min(test_idx), "Test data overlaps with train data"

        # Check that there's a gap (purge)
        gap_size = min(test_idx) - max(train_idx) - 1
        assert gap_size >= 10, f"Gap size {gap_size} is less than required 10"

        # Check that indices are in order
        assert all(train_idx[i] <= train_idx[i+1] for i in range(len(train_idx)-1))
        assert all(test_idx[i] <= test_idx[i+1] for i in range(len(test_idx)-1))


