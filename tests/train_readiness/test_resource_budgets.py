import time

import psutil
import pytest

from core.ml.build_features import build_matrix


@pytest.mark.sanity
def test_train_time_and_memory():
    """Test that training stays within resource budgets"""
    # Load golden snapshot data
    import json
    with open("artifacts/snapshots/golden_ml_v1/manifest.json") as f:
        json.load(f)

    # Load SPY data
    import pandas as pd
    df = pd.read_parquet("artifacts/snapshots/golden_ml_v1/SPY.parquet")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    X, y = build_matrix(df, horizon=1)

    # Ensure we have data
    assert len(X) > 0, "No features generated"
    assert len(y) > 0, "No targets generated"

    # Time budget
    time_budget = 5.0  # seconds

    # Memory budget (MB)
    memory_budget = 512  # MB

    # Start monitoring
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()

    # Train a simple model
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)

    # Check time
    elapsed_time = time.time() - start_time
    assert elapsed_time < time_budget, f"Training took {elapsed_time:.2f}s, budget was {time_budget}s"

    # Check memory
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = end_memory - start_memory
    assert memory_used < memory_budget, f"Memory used {memory_used:.1f}MB, budget was {memory_budget}MB"


