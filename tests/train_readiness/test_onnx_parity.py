import json
import os

import numpy as np
import pandas as pd
import pytest

from core.ml.build_features import build_matrix


@pytest.mark.sanity
def test_xgboost_model_save_load():
    """Test that XGBoost model can be saved and loaded with matching predictions"""
    try:
        import xgboost as xgb
    except ImportError:
        pytest.skip("XGBoost not available")

    # Load golden snapshot data
    with open("artifacts/snapshots/golden_ml_v1/manifest.json") as f:
        json.load(f)

    # Load SPY data
    df = pd.read_parquet("artifacts/snapshots/golden_ml_v1/SPY.parquet")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    X, y = build_matrix(df, horizon=1)

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=10,  # small for speed
        max_depth=2,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)

    # Save model
    model_path = "test_model.json"
    model.save_model(model_path)

    try:
        # Load model
        loaded_model = xgb.XGBRegressor()
        loaded_model.load_model(model_path)

        # Test predictions
        X_test = X.iloc[-100:].to_numpy()

        # Original predictions
        orig_pred = model.predict(X_test)

        # Loaded predictions
        loaded_pred = loaded_model.predict(X_test)

        # Compare
        diff = np.abs(orig_pred - loaded_pred)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Assert parity
        assert max_diff < 1e-10, f"Max difference {max_diff} exceeds tolerance"
        assert mean_diff < 1e-11, f"Mean difference {mean_diff} exceeds tolerance"

    finally:
        # Cleanup
        if os.path.exists(model_path):
            os.remove(model_path)


@pytest.mark.sanity
def test_xgboost_training_smoke():
    """Test that XGBoost training completes and produces reasonable metrics"""
    try:
        import xgboost as xgb
    except ImportError:
        pytest.skip("XGBoost not available")

    # Load golden snapshot data
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

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)

    # Make predictions
    pred = model.predict(X)

    # Basic sanity checks
    assert len(pred) == len(y)
    assert not np.any(np.isnan(pred))
    assert not np.any(np.isinf(pred))

    # R² should be reasonable (not too high on noisy financial data)
    from sklearn.metrics import r2_score
    r2 = r2_score(y, pred)
    assert -1.0 <= r2 <= 0.8, f"R² {r2} outside reasonable range"


@pytest.mark.sanity
def test_onnx_export_flag():
    """Test that ONNX export is controlled by config flag"""
    import yaml

    with open("config/train_profiles.yaml") as f:
        cfg = yaml.safe_load(f)

    # Check that golden_linear doesn't export ONNX
    linear_cfg = cfg["train"]["profiles"]["golden_linear"]
    assert not linear_cfg.get("export_onnx", False)

    # Check that golden_xgb does export ONNX
    xgb_cfg = cfg["train"]["profiles"]["golden_xgb"]
    assert xgb_cfg.get("export_onnx", False)
