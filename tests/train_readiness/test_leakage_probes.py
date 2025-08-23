import numpy as np
import pytest

from core.ml.build_features import build_matrix


@pytest.mark.sanity
def test_negative_controls():
    """Test that negative controls (leakage probes) work correctly"""
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

    # Test that leading the label (simulating leakage) causes performance to spike
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import TimeSeriesSplit

    # Normal case (no leakage)
    tscv = TimeSeriesSplit(n_splits=2, test_size=100, gap=10)
    normal_scores = []

    for tr, te in tscv.split(X):
        if len(tr) > 0 and len(te) > 0:
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            score = r2_score(y.iloc[te], pred)
            normal_scores.append(score)

    # Leakage case: use the actual target as a feature (perfect leakage)
    X_leak = X.copy()
    X_leak['leaked_target'] = y  # This is perfect leakage - we're using the target as a feature

    leakage_scores = []
    for tr, te in tscv.split(X_leak):
        if len(tr) > 0 and len(te) > 0:
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_leak.iloc[tr], y.iloc[tr])
            pred = model.predict(X_leak.iloc[te])
            score = r2_score(y.iloc[te], pred)
            leakage_scores.append(score)

    # Print debug info
    print(f"Normal scores: {normal_scores}")
    print(f"Leakage scores: {leakage_scores}")

    # Normal scores should be modest (financial data is noisy)
    if normal_scores:
        avg_normal = np.mean(normal_scores)
        print(f"Average normal R²: {avg_normal}")
        assert avg_normal < 0.3, f"Normal R² {avg_normal} is suspiciously high"

    # Leakage scores should be much higher (proving the probe works)
    if leakage_scores and normal_scores:
        avg_leakage = np.mean(leakage_scores)
        avg_normal = np.mean(normal_scores)
        print(f"Average leakage R²: {avg_leakage}")
        print(f"Difference: {avg_leakage - avg_normal}")

        # Perfect leakage should improve performance significantly
        improvement = avg_leakage - avg_normal
        assert improvement > 0.1, f"Leakage improvement {improvement} should be > 0.1, got {avg_leakage} vs {avg_normal}"


