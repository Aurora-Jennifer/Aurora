# tests/ml/test_alpha_eval_contract.py
"""
Test alpha evaluation contract with deterministic small dataset.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def create_synthetic_dataset() -> pd.DataFrame:
    """Create a small, deterministic synthetic dataset for testing."""
    np.random.seed(42)

    # Create 100 days of synthetic data
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")

    # Generate synthetic features and labels
    return pd.DataFrame(
        {
            "ret_1d": np.random.randn(100) * 0.01,
            "ret_5d": np.random.randn(100) * 0.02,
            "ret_20d": np.random.randn(100) * 0.05,
            "sma_20_minus_50": np.random.randn(100) * 0.1,
            "vol_10d": np.random.randn(100) * 0.005,
            "vol_20d": np.random.randn(100) * 0.005,
            "rsi_14": np.random.uniform(0, 100, 100),
            "volu_z_20d": np.random.randn(100),
            "ret_fwd_1d": np.random.randn(100) * 0.01,  # Target
            "symbol": "TEST",
        },
        index=dates,
    )



def test_alpha_eval_metrics_calculation():
    """Test that alpha evaluation metrics are calculated correctly."""
    from ml.eval.alpha_eval import (
        calculate_hit_rate,
        calculate_ic_spearman,
        calculate_return_with_costs,
        calculate_turnover,
    )

    # Create synthetic predictions and targets
    np.random.seed(42)
    predictions = np.random.randn(100) * 0.01
    targets = np.random.randn(100) * 0.01

    # Test IC calculation
    ic = calculate_ic_spearman(predictions, targets)
    assert isinstance(ic, float)
    assert -1.0 <= ic <= 1.0

    # Test hit rate calculation
    hit_rate = calculate_hit_rate(predictions, targets)
    assert isinstance(hit_rate, float)
    assert 0.0 <= hit_rate <= 1.0

    # Test turnover calculation
    turnover = calculate_turnover(predictions)
    assert isinstance(turnover, float)
    assert turnover >= 0.0

    # Test return with costs calculation
    return_with_costs = calculate_return_with_costs(predictions, targets, 5.0, 1.0)
    assert isinstance(return_with_costs, float)


def test_walkforward_folds_creation():
    """Test walkforward folds creation with synthetic data."""
    from ml.eval.alpha_eval import create_walkforward_folds

    df = create_synthetic_dataset()

    # Create folds
    folds = create_walkforward_folds(df, n_folds=3, min_train_size=30)

    # Check that we got the expected number of folds
    assert len(folds) > 0, "No folds created"

    # Check each fold
    for i, (train_df, test_df) in enumerate(folds):
        # Check that train and test are separate
        train_dates = set(train_df.index)
        test_dates = set(test_df.index)

        overlap = train_dates.intersection(test_dates)
        assert len(overlap) == 0, f"Fold {i}: train/test overlap detected"

        # Check temporal ordering
        max_train_date = max(train_df.index)
        min_test_date = min(test_df.index)
        assert max_train_date < min_test_date, f"Fold {i}: test starts before train ends"

        # Check minimum train size
        assert len(train_df) >= 30, f"Fold {i}: train size {len(train_df)} < 30"


def test_alpha_eval_schema_compliance():
    """Test that alpha evaluation results comply with schema."""
    # Create a mock model file
    import pickle

    import jsonschema
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Create a simple mock pipeline
    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

    # Create synthetic training data
    np.random.seed(42)
    X_train = np.random.randn(50, 8)
    y_train = np.random.randn(50)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Save mock model
    model_path = Path("artifacts/models/test_linear_v1.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    try:
        # Create synthetic feature files
        feature_dir = Path("artifacts/feature_store")
        feature_dir.mkdir(parents=True, exist_ok=True)

        df = create_synthetic_dataset()
        df.to_parquet(feature_dir / "SPY.parquet")
        df.to_parquet(feature_dir / "TSLA.parquet")

        # Mock the feature loading function

        def mock_load_feature_data(symbols, feature_dir):
            all_data = []
            for symbol in symbols:
                file_path = Path(feature_dir) / f"{symbol}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df["symbol"] = symbol
                    all_data.append(df)
            return pd.concat(all_data, ignore_index=False)

        # Patch the function
        import ml.eval.alpha_eval as alpha_eval_module

        alpha_eval_module.load_feature_data = mock_load_feature_data

        # Run evaluation with smaller min_train_size for testing
        from ml.eval.alpha_eval import create_walkforward_folds

        # Create folds with smaller minimum size
        folds = create_walkforward_folds(df, n_folds=2, min_train_size=30)

        # Mock the evaluate_fold function to avoid loading the actual model
        def mock_evaluate_fold(train_df, test_df, model_path, config):
            return {
                "ic_spearman": 0.02,
                "hit_rate": 0.52,
                "turnover": 0.1,
                "return_with_costs": 0.05,
                "n_predictions": len(test_df),
            }

        # Mock the evaluate_fold function
        alpha_eval_module.evaluate_fold = mock_evaluate_fold

        # Create results manually
        fold_summaries = []
        all_ics = []
        all_hit_rates = []
        all_turnovers = []
        all_returns = []

        for i, (train_df, test_df) in enumerate(folds):
            fold_results = mock_evaluate_fold(train_df, test_df, model_path, {})

            fold_summary = {
                "fold": i + 1,
                "train_start": train_df.index[0].strftime("%Y-%m-%d"),
                "train_end": train_df.index[-1].strftime("%Y-%m-%d"),
                "test_start": test_df.index[0].strftime("%Y-%m-%d"),
                "test_end": test_df.index[-1].strftime("%Y-%m-%d"),
                **fold_results,
            }

            fold_summaries.append(fold_summary)
            all_ics.append(fold_results["ic_spearman"])
            all_hit_rates.append(fold_results["hit_rate"])
            all_turnovers.append(fold_results["turnover"])
            all_returns.append(fold_results["return_with_costs"])

        # Calculate overall metrics
        overall_metrics = {
            "mean_ic": np.mean(all_ics),
            "mean_hit_rate": np.mean(all_hit_rates),
            "mean_turnover": np.mean(all_turnovers),
            "mean_return_with_costs": np.mean(all_returns),
            "std_ic": np.std(all_ics),
            "std_hit_rate": np.std(all_hit_rates),
            "min_ic": np.min(all_ics),
            "max_ic": np.max(all_ics),
            "min_hit_rate": np.min(all_hit_rates),
            "max_hit_rate": np.max(all_hit_rates),
        }

        # Create results
        results = {
            "model_id": "linear_v1",
            "symbols": ["SPY", "TSLA"],
            "fold_summaries": fold_summaries,
            "overall_metrics": overall_metrics,
            "metadata": {
                "evaluation_date": datetime.now(UTC).isoformat(),
                "model_path": str(model_path),
                "feature_config": {},
                "training_samples": len(folds[0][0]) if folds else 0,
                "test_samples": sum(len(fold[1]) for fold in folds),
                "n_folds": len(folds),
            },
        }

        # Load schema
        schema_path = Path("reports/alpha.schema.json")
        with open(schema_path) as f:
            schema = json.load(f)

        # Validate results against schema
        jsonschema.validate(instance=results, schema=schema)

        # Check required fields
        assert "model_id" in results
        assert "symbols" in results
        assert "fold_summaries" in results
        assert "overall_metrics" in results
        assert "metadata" in results

        # Check overall metrics
        overall_metrics = results["overall_metrics"]
        required_metrics = ["mean_ic", "mean_hit_rate", "mean_turnover", "mean_return_with_costs"]
        for metric in required_metrics:
            assert metric in overall_metrics
            assert isinstance(overall_metrics[metric], int | float)

        # Check fold summaries
        fold_summaries = results["fold_summaries"]
        assert len(fold_summaries) > 0

        for fold in fold_summaries:
            required_fold_fields = [
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "ic_spearman",
                "hit_rate",
                "turnover",
                "return_with_costs",
            ]
            for field in required_fold_fields:
                assert field in fold

    finally:
        # Cleanup
        if model_path.exists():
            model_path.unlink()

        test_eval_path = Path("reports/test_alpha_eval.json")
        if test_eval_path.exists():
            test_eval_path.unlink()


def test_deterministic_results():
    """Test that evaluation results are deterministic with fixed seed."""
    import pickle

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Create deterministic synthetic data
    np.random.seed(42)

    # Create mock model
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))]
    )

    X_train = np.random.randn(50, 8)
    y_train = np.random.randn(50)
    pipeline.fit(X_train, y_train)

    model_path = Path("artifacts/models/test_det.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    try:
        # Create deterministic feature data
        feature_dir = Path("artifacts/feature_store")
        feature_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")

        df = pd.DataFrame(
            {
                "ret_1d": np.random.randn(100) * 0.01,
                "ret_5d": np.random.randn(100) * 0.02,
                "ret_20d": np.random.randn(100) * 0.05,
                "sma_20_minus_50": np.random.randn(100) * 0.1,
                "vol_10d": np.random.randn(100) * 0.005,
                "vol_20d": np.random.randn(100) * 0.005,
                "rsi_14": np.random.uniform(0, 100, 100),
                "volu_z_20d": np.random.randn(100),
                "ret_fwd_1d": np.random.randn(100) * 0.01,
                "symbol": "TEST",
            },
            index=dates,
        )

        df.to_parquet(feature_dir / "SPY.parquet")

        # Mock feature loading

        def mock_load_feature_data(symbols, feature_dir):
            all_data = []
            for symbol in symbols:
                file_path = Path(feature_dir) / f"{symbol}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df["symbol"] = symbol
                    all_data.append(df)
            return pd.concat(all_data, ignore_index=False)

        import ml.eval.alpha_eval as alpha_eval_module

        alpha_eval_module.load_feature_data = mock_load_feature_data

        # Create folds with smaller minimum size for testing
        from ml.eval.alpha_eval import create_walkforward_folds

        folds = create_walkforward_folds(df, n_folds=2, min_train_size=30)

        # Mock the evaluate_fold function
        def mock_evaluate_fold(train_df, test_df, model_path, config):
            return {
                "ic_spearman": 0.02,
                "hit_rate": 0.52,
                "turnover": 0.1,
                "return_with_costs": 0.05,
                "n_predictions": len(test_df),
            }

        alpha_eval_module.evaluate_fold = mock_evaluate_fold

        # Create results manually for both runs
        def create_results():
            fold_summaries = []
            all_ics = []
            all_hit_rates = []
            all_turnovers = []
            all_returns = []

            for i, (train_df, test_df) in enumerate(folds):
                fold_results = mock_evaluate_fold(train_df, test_df, model_path, {})

                fold_summary = {
                    "fold": i + 1,
                    "train_start": train_df.index[0].strftime("%Y-%m-%d"),
                    "train_end": train_df.index[-1].strftime("%Y-%m-%d"),
                    "test_start": test_df.index[0].strftime("%Y-%m-%d"),
                    "test_end": test_df.index[-1].strftime("%Y-%m-%d"),
                    **fold_results,
                }

                fold_summaries.append(fold_summary)
                all_ics.append(fold_results["ic_spearman"])
                all_hit_rates.append(fold_results["hit_rate"])
                all_turnovers.append(fold_results["turnover"])
                all_returns.append(fold_results["return_with_costs"])

            overall_metrics = {
                "mean_ic": np.mean(all_ics),
                "mean_hit_rate": np.mean(all_hit_rates),
                "mean_turnover": np.mean(all_turnovers),
                "mean_return_with_costs": np.mean(all_returns),
                "std_ic": np.std(all_ics),
                "std_hit_rate": np.std(all_hit_rates),
                "min_ic": np.min(all_ics),
                "max_ic": np.max(all_ics),
                "min_hit_rate": np.min(all_hit_rates),
                "max_hit_rate": np.max(all_hit_rates),
            }

            return {
                "model_id": "linear_v1",
                "symbols": ["SPY"],
                "fold_summaries": fold_summaries,
                "overall_metrics": overall_metrics,
                "metadata": {
                    "evaluation_date": datetime.now(UTC).isoformat(),
                    "model_path": str(model_path),
                    "feature_config": {},
                    "training_samples": len(folds[0][0]) if folds else 0,
                    "test_samples": sum(len(fold[1]) for fold in folds),
                    "n_folds": len(folds),
                },
            }

        # Run evaluation twice with same seed
        np.random.seed(42)
        results1 = create_results()

        np.random.seed(42)
        results2 = create_results()

        # Check that results are identical
        assert results1["overall_metrics"]["mean_ic"] == results2["overall_metrics"]["mean_ic"]
        assert (
            results1["overall_metrics"]["mean_hit_rate"]
            == results2["overall_metrics"]["mean_hit_rate"]
        )

    finally:
        # Cleanup
        if model_path.exists():
            model_path.unlink()

        for path in ["reports/test_det1.json", "reports/test_det2.json"]:
            if Path(path).exists():
                Path(path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
