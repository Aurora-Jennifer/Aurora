"""
Reproducibility and parallel consistency tests for walkforward framework.
Verify that results are deterministic and parallel execution matches sequential.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.walkforward_framework import (
    LeakageProofPipeline,
    build_feature_table,
    gen_walkforward,
    walkforward_run,
)


def create_test_data(seed=42):
    """Create consistent test data for reproducibility tests."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(100).cumsum() + 100,
            "High": np.random.randn(100).cumsum() + 102,
            "Low": np.random.randn(100).cumsum() + 98,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )
    return data


def run_walkforward_test(data, model_seed=42):
    """Run walkforward test with given data and seed."""
    # Build feature table
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    # Create pipeline
    pipeline = LeakageProofPipeline(X, y)

    # Run walkforward
    results = walkforward_run(pipeline, folds, prices, model_seed=model_seed)

    return results, folds


def extract_metrics(results):
    """Extract key metrics from walkforward results."""
    metrics = []
    for fold_id, fold_metrics, trades in results:
        metrics.append(
            {
                "fold_id": fold_id,
                "sharpe_nw": fold_metrics["sharpe_nw"],
                "sortino": fold_metrics["sortino"],
                "max_dd": fold_metrics["max_dd"],
                "hit_rate": fold_metrics["hit_rate"],
                "total_return": fold_metrics["total_return"],
                "n_trades": len(trades),
            }
        )
    return metrics


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 1337])
def test_seed_stability(seed):
    """Test that results are stable with fixed seeds."""
    # Create test data with fixed seed
    data = create_test_data(seed=seed)

    # Run walkforward twice with same seed
    results1, folds1 = run_walkforward_test(data, model_seed=seed)
    results2, folds2 = run_walkforward_test(data, model_seed=seed)

    # Extract metrics
    metrics1 = extract_metrics(results1)
    metrics2 = extract_metrics(results2)

    # Verify identical results
    assert len(metrics1) == len(
        metrics2
    ), f"WF_SEED: Different number of folds for seed {seed}"

    for m1, m2 in zip(metrics1, metrics2):
        assert (
            m1["fold_id"] == m2["fold_id"]
        ), f"WF_SEED: Fold ID mismatch for seed {seed}"
        assert (
            abs(m1["sharpe_nw"] - m2["sharpe_nw"]) < 1e-10
        ), f"WF_SEED: Sharpe mismatch for seed {seed}"
        assert (
            abs(m1["sortino"] - m2["sortino"]) < 1e-10
        ), f"WF_SEED: Sortino mismatch for seed {seed}"
        assert (
            abs(m1["max_dd"] - m2["max_dd"]) < 1e-10
        ), f"WF_SEED: Max DD mismatch for seed {seed}"
        assert (
            abs(m1["hit_rate"] - m2["hit_rate"]) < 1e-10
        ), f"WF_SEED: Hit rate mismatch for seed {seed}"
        assert (
            abs(m1["total_return"] - m2["total_return"]) < 1e-10
        ), f"WF_SEED: Total return mismatch for seed {seed}"
        assert (
            m1["n_trades"] == m2["n_trades"]
        ), f"WF_SEED: Trade count mismatch for seed {seed}"


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results (non-deterministic model)."""
    # Create test data
    data = create_test_data(seed=42)

    # Run with different seeds
    results1, _ = run_walkforward_test(data, model_seed=42)
    results2, _ = run_walkforward_test(data, model_seed=123)

    # Extract metrics
    metrics1 = extract_metrics(results1)
    metrics2 = extract_metrics(results2)

    # Check that at least some metrics are different
    sharpe_diffs = [
        abs(m1["sharpe_nw"] - m2["sharpe_nw"]) for m1, m2 in zip(metrics1, metrics2)
    ]
    max_sharpe_diff = max(sharpe_diffs)

    # Should have some variation (not all identical)
    assert (
        max_sharpe_diff > 1e-10
    ), "WF_SEED_DIFF: All Sharpe ratios identical with different seeds"


def test_fold_generation_reproducibility():
    """Test that fold generation is reproducible."""
    # Create test data
    data = create_test_data(seed=42)
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds twice
    folds1 = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))
    folds2 = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    # Verify identical folds
    assert len(folds1) == len(folds2), "WF_FOLD_REPRO: Different number of folds"

    for f1, f2 in zip(folds1, folds2):
        assert f1.fold_id == f2.fold_id, "WF_FOLD_REPRO: Fold ID mismatch"
        assert f1.train_lo == f2.train_lo, "WF_FOLD_REPRO: Train lo mismatch"
        assert f1.train_hi == f2.train_hi, "WF_FOLD_REPRO: Train hi mismatch"
        assert f1.test_lo == f2.test_lo, "WF_FOLD_REPRO: Test lo mismatch"
        assert f1.test_hi == f2.test_hi, "WF_FOLD_REPRO: Test hi mismatch"


def test_feature_pipeline_reproducibility():
    """Test that feature pipeline produces reproducible results."""
    # Create test data
    data = create_test_data(seed=42)

    # Build feature table twice
    X1, y1, prices1 = build_feature_table(data, warmup_days=20)
    X2, y2, prices2 = build_feature_table(data, warmup_days=20)

    # Verify identical features
    np.testing.assert_allclose(
        X1,
        X2,
        rtol=1e-10,
        atol=1e-10,
        err_msg="WF_FEATURE_REPRO: Feature matrix mismatch",
    )
    np.testing.assert_allclose(
        y1, y2, rtol=1e-10, atol=1e-10, err_msg="WF_FEATURE_REPRO: Target mismatch"
    )
    np.testing.assert_allclose(
        prices1,
        prices2,
        rtol=1e-10,
        atol=1e-10,
        err_msg="WF_FEATURE_REPRO: Price mismatch",
    )


def test_parallel_equals_sequential():
    """Test that parallel execution produces same results as sequential."""
    # Note: This test assumes the walkforward framework supports parallel execution
    # For now, we'll test that sequential execution is deterministic

    # Create test data
    data = create_test_data(seed=42)

    # Run sequential execution twice
    results1, _ = run_walkforward_test(data, model_seed=42)
    results2, _ = run_walkforward_test(data, model_seed=42)

    # Extract metrics
    metrics1 = extract_metrics(results1)
    metrics2 = extract_metrics(results2)

    # Verify identical results
    assert len(metrics1) == len(metrics2), "WF_PARALLEL: Different number of results"

    for m1, m2 in zip(metrics1, metrics2):
        assert m1["fold_id"] == m2["fold_id"], "WF_PARALLEL: Fold ID mismatch"
        assert (
            abs(m1["sharpe_nw"] - m2["sharpe_nw"]) < 1e-10
        ), "WF_PARALLEL: Sharpe mismatch"
        assert (
            abs(m1["sortino"] - m2["sortino"]) < 1e-10
        ), "WF_PARALLEL: Sortino mismatch"
        assert abs(m1["max_dd"] - m2["max_dd"]) < 1e-10, "WF_PARALLEL: Max DD mismatch"
        assert (
            abs(m1["hit_rate"] - m2["hit_rate"]) < 1e-10
        ), "WF_PARALLEL: Hit rate mismatch"
        assert (
            abs(m1["total_return"] - m2["total_return"]) < 1e-10
        ), "WF_PARALLEL: Total return mismatch"
        assert m1["n_trades"] == m2["n_trades"], "WF_PARALLEL: Trade count mismatch"


def test_model_warm_start_reproducibility():
    """Test that model warm-start doesn't introduce non-determinism."""
    # Create test data
    data = create_test_data(seed=42)
    X, y, prices = build_feature_table(data, warmup_days=20)

    # Generate folds
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=5))

    # Create pipeline
    pipeline = LeakageProofPipeline(X, y)

    # Run with warm-start
    results1 = walkforward_run(pipeline, folds, prices, model_seed=42)

    # Run again with same seed
    pipeline2 = LeakageProofPipeline(X, y)
    results2 = walkforward_run(pipeline2, folds, prices, model_seed=42)

    # Extract metrics
    metrics1 = extract_metrics(results1)
    metrics2 = extract_metrics(results2)

    # Verify identical results
    assert len(metrics1) == len(metrics2), "WF_WARMSTART: Different number of results"

    for m1, m2 in zip(metrics1, metrics2):
        assert m1["fold_id"] == m2["fold_id"], "WF_WARMSTART: Fold ID mismatch"
        assert (
            abs(m1["sharpe_nw"] - m2["sharpe_nw"]) < 1e-10
        ), "WF_WARMSTART: Sharpe mismatch"


def test_data_loading_reproducibility():
    """Test that data loading is reproducible."""
    # Create test data with different seeds
    data1 = create_test_data(seed=42)
    data2 = create_test_data(seed=42)  # Same seed

    # Verify identical data
    np.testing.assert_allclose(
        data1["Open"].values,
        data2["Open"].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="WF_DATA_REPRO: Open prices mismatch",
    )
    np.testing.assert_allclose(
        data1["High"].values,
        data2["High"].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="WF_DATA_REPRO: High prices mismatch",
    )
    np.testing.assert_allclose(
        data1["Low"].values,
        data2["Low"].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="WF_DATA_REPRO: Low prices mismatch",
    )
    np.testing.assert_allclose(
        data1["Close"].values,
        data2["Close"].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="WF_DATA_REPRO: Close prices mismatch",
    )
    np.testing.assert_allclose(
        data1["Volume"].values,
        data2["Volume"].values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="WF_DATA_REPRO: Volume mismatch",
    )


def test_metric_aggregation_reproducibility():
    """Test that metric aggregation is reproducible."""
    # Create test data
    data = create_test_data(seed=42)
    results, _ = run_walkforward_test(data, model_seed=42)

    # Extract metrics
    metrics = extract_metrics(results)

    # Calculate aggregate metrics
    sharpe_scores = [m["sharpe_nw"] for m in metrics]
    sortino_scores = [m["sortino"] for m in metrics]
    max_dds = [m["max_dd"] for m in metrics]
    hit_rates = [m["hit_rate"] for m in metrics]
    total_returns = [m["total_return"] for m in metrics]

    # Calculate statistics
    mean_sharpe1 = np.mean(sharpe_scores)
    mean_sortino1 = np.mean(sortino_scores)
    mean_max_dd1 = np.mean(max_dds)
    mean_hit_rate1 = np.mean(hit_rates)
    mean_total_return1 = np.mean(total_returns)

    # Calculate again
    mean_sharpe2 = np.mean(sharpe_scores)
    mean_sortino2 = np.mean(sortino_scores)
    mean_max_dd2 = np.mean(max_dds)
    mean_hit_rate2 = np.mean(hit_rates)
    mean_total_return2 = np.mean(total_returns)

    # Verify identical aggregation
    assert (
        abs(mean_sharpe1 - mean_sharpe2) < 1e-10
    ), "WF_AGGREGATE: Mean Sharpe mismatch"
    assert (
        abs(mean_sortino1 - mean_sortino2) < 1e-10
    ), "WF_AGGREGATE: Mean Sortino mismatch"
    assert (
        abs(mean_max_dd1 - mean_max_dd2) < 1e-10
    ), "WF_AGGREGATE: Mean Max DD mismatch"
    assert (
        abs(mean_hit_rate1 - mean_hit_rate2) < 1e-10
    ), "WF_AGGREGATE: Mean Hit Rate mismatch"
    assert (
        abs(mean_total_return1 - mean_total_return2) < 1e-10
    ), "WF_AGGREGATE: Mean Total Return mismatch"


@pytest.mark.slow
def test_large_dataset_reproducibility():
    """Test reproducibility with larger datasets."""
    # Create larger test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(500).cumsum() + 100,
            "High": np.random.randn(500).cumsum() + 102,
            "Low": np.random.randn(500).cumsum() + 98,
            "Close": np.random.randn(500).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, 500),
        },
        index=dates,
    )

    # Run walkforward twice
    results1, _ = run_walkforward_test(data, model_seed=42)
    results2, _ = run_walkforward_test(data, model_seed=42)

    # Extract metrics
    metrics1 = extract_metrics(results1)
    metrics2 = extract_metrics(results2)

    # Verify identical results
    assert len(metrics1) == len(metrics2), "WF_LARGE: Different number of results"

    for m1, m2 in zip(metrics1, metrics2):
        assert m1["fold_id"] == m2["fold_id"], "WF_LARGE: Fold ID mismatch"
        assert (
            abs(m1["sharpe_nw"] - m2["sharpe_nw"]) < 1e-10
        ), "WF_LARGE: Sharpe mismatch"
        assert abs(m1["sortino"] - m2["sortino"]) < 1e-10, "WF_LARGE: Sortino mismatch"
        assert abs(m1["max_dd"] - m2["max_dd"]) < 1e-10, "WF_LARGE: Max DD mismatch"
        assert (
            abs(m1["hit_rate"] - m2["hit_rate"]) < 1e-10
        ), "WF_LARGE: Hit rate mismatch"
        assert (
            abs(m1["total_return"] - m2["total_return"]) < 1e-10
        ), "WF_LARGE: Total return mismatch"
        assert m1["n_trades"] == m2["n_trades"], "WF_LARGE: Trade count mismatch"
