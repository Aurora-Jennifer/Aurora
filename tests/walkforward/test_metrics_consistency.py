"""
Metrics consistency tests for walkforward framework.
Verify that fold metrics aggregate correctly to summary metrics.
"""

import numpy as np
import pandas as pd

from scripts.walkforward_framework import (
    LeakageProofPipeline,
    build_feature_table,
    gen_walkforward,
    walkforward_run,
)


def create_test_data(seed=42):
    """Create consistent test data for metrics tests."""
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


def aggregate_fold_metrics(results):
    """Aggregate fold metrics to summary metrics."""
    if not results:
        return {}

    # Extract all metrics
    sharpe_scores = []
    sortino_scores = []
    max_dds = []
    hit_rates = []
    total_returns = []
    volatilities = []
    trade_counts = []

    for fold_id, metrics, trades in results:
        sharpe_scores.append(metrics["sharpe_nw"])
        sortino_scores.append(metrics["sortino"])
        max_dds.append(metrics["max_dd"])
        hit_rates.append(metrics["hit_rate"])
        total_returns.append(metrics["total_return"])
        volatilities.append(metrics["volatility"])
        trade_counts.append(len(trades))

    # Calculate aggregate statistics
    summary = {
        "mean_sharpe": np.mean(sharpe_scores),
        "mean_sortino": np.mean(sortino_scores),
        "mean_max_dd": np.mean(max_dds),
        "mean_hit_rate": np.mean(hit_rates),
        "mean_total_return": np.mean(total_returns),
        "mean_volatility": np.mean(volatilities),
        "total_trades": sum(trade_counts),
        "winning_trades": sum(int(hr * tc) for hr, tc in zip(hit_rates, trade_counts)),
        "positive_sharpe_folds": sum(1 for s in sharpe_scores if s > 0),
        "total_folds": len(results),
        "std_sharpe": np.std(sharpe_scores),
        "std_sortino": np.std(sortino_scores),
        "std_max_dd": np.std(max_dds),
        "std_hit_rate": np.std(hit_rates),
        "std_total_return": np.std(total_returns),
    }

    return summary


def test_fold_metrics_aggregate_to_summary():
    """Test that fold metrics aggregate correctly to summary metrics."""
    # Create test data
    data = create_test_data(seed=42)

    # Run walkforward
    results, folds = run_walkforward_test(data, model_seed=42)

    # Aggregate fold metrics
    summary = aggregate_fold_metrics(results)

    # Verify summary has expected structure
    expected_keys = [
        "mean_sharpe",
        "mean_sortino",
        "mean_max_dd",
        "mean_hit_rate",
        "mean_total_return",
        "mean_volatility",
        "total_trades",
        "winning_trades",
        "positive_sharpe_folds",
        "total_folds",
    ]

    for key in expected_keys:
        assert key in summary, f"WF_AGGREGATE: Missing key '{key}' in summary"

    # Verify basic sanity checks
    assert summary["total_folds"] == len(results), "WF_AGGREGATE: Total folds mismatch"
    assert (
        summary["total_trades"] >= 0
    ), "WF_AGGREGATE: Total trades should be non-negative"
    assert (
        summary["winning_trades"] >= 0
    ), "WF_AGGREGATE: Winning trades should be non-negative"
    assert (
        summary["positive_sharpe_folds"] >= 0
    ), "WF_AGGREGATE: Positive Sharpe folds should be non-negative"
    assert (
        summary["positive_sharpe_folds"] <= summary["total_folds"]
    ), "WF_AGGREGATE: Positive Sharpe folds cannot exceed total folds"


def test_metric_calculation_consistency():
    """Test that metric calculations are consistent across folds."""
    # Create test data
    data = create_test_data(seed=42)

    # Run walkforward
    results, folds = run_walkforward_test(data, model_seed=42)

    # Check that each fold has consistent metric structure
    expected_metric_keys = [
        "sharpe_nw",
        "sortino",
        "max_dd",
        "hit_rate",
        "turnover",
        "median_hold",
        "total_return",
        "volatility",
    ]

    for fold_id, metrics, trades in results:
        for key in expected_metric_keys:
            assert (
                key in metrics
            ), f"WF_METRIC: Missing metric '{key}' in fold {fold_id}"

        # Verify metric value types
        assert isinstance(
            metrics["sharpe_nw"], (int, float)
        ), f"WF_METRIC: Sharpe should be numeric in fold {fold_id}"
        assert isinstance(
            metrics["sortino"], (int, float)
        ), f"WF_METRIC: Sortino should be numeric in fold {fold_id}"
        assert isinstance(
            metrics["max_dd"], (int, float)
        ), f"WF_METRIC: Max DD should be numeric in fold {fold_id}"
        assert isinstance(
            metrics["hit_rate"], (int, float)
        ), f"WF_METRIC: Hit rate should be numeric in fold {fold_id}"
        assert isinstance(
            metrics["total_return"], (int, float)
        ), f"WF_METRIC: Total return should be numeric in fold {fold_id}"
        assert isinstance(
            metrics["volatility"], (int, float)
        ), f"WF_METRIC: Volatility should be numeric in fold {fold_id}"

        # Verify metric value ranges
        assert (
            -np.inf < metrics["sharpe_nw"] < np.inf
        ), f"WF_METRIC: Sharpe should be finite in fold {fold_id}"
        assert (
            -np.inf < metrics["sortino"] < np.inf
        ), f"WF_METRIC: Sortino should be finite in fold {fold_id}"
        assert (
            -10 <= metrics["max_dd"] <= 0
        ), f"WF_METRIC: Max DD should be in [-10, 0] in fold {fold_id}"
        assert (
            0 <= metrics["hit_rate"] <= 1
        ), f"WF_METRIC: Hit rate should be in [0, 1] in fold {fold_id}"
        assert (
            -np.inf < metrics["total_return"] < np.inf
        ), f"WF_METRIC: Total return should be finite in fold {fold_id}"
        assert (
            0 <= metrics["volatility"] < np.inf
        ), f"WF_METRIC: Volatility should be non-negative in fold {fold_id}"


def test_trade_consistency():
    """Test that trade data is consistent across folds."""
    # Create test data
    data = create_test_data(seed=42)

    # Run walkforward
    results, folds = run_walkforward_test(data, model_seed=42)

    # Check trade consistency
    for fold_id, metrics, trades in results:
        assert isinstance(
            trades, list
        ), f"WF_TRADE: Trades should be list in fold {fold_id}"

        # Check trade structure (simplified format)
        for trade in trades:
            expected_trade_keys = ["count", "wins", "losses", "median_hold"]
            for key in expected_trade_keys:
                assert (
                    key in trade
                ), f"WF_TRADE: Missing trade key '{key}' in fold {fold_id}"

            # Verify trade value types
            assert isinstance(
                trade["count"], int
            ), f"WF_TRADE: Count should be int in fold {fold_id}"
            assert isinstance(
                trade["wins"], int
            ), f"WF_TRADE: Wins should be int in fold {fold_id}"
            assert isinstance(
                trade["losses"], int
            ), f"WF_TRADE: Losses should be int in fold {fold_id}"
            assert isinstance(
                trade["median_hold"], int
            ), f"WF_TRADE: Median hold should be int in fold {fold_id}"

            # Verify trade value ranges
            assert (
                trade["count"] >= 0
            ), f"WF_TRADE: Count should be non-negative in fold {fold_id}"
            assert (
                trade["wins"] >= 0
            ), f"WF_TRADE: Wins should be non-negative in fold {fold_id}"
            assert (
                trade["losses"] >= 0
            ), f"WF_TRADE: Losses should be non-negative in fold {fold_id}"
            assert (
                trade["median_hold"] >= 0
            ), f"WF_TRADE: Median hold should be non-negative in fold {fold_id}"
            assert (
                trade["wins"] + trade["losses"] == trade["count"]
            ), f"WF_TRADE: Wins + losses should equal count in fold {fold_id}"


def test_metric_aggregation_accuracy():
    """Test that metric aggregation is mathematically accurate."""
    # Create test data
    data = create_test_data(seed=42)

    # Run walkforward
    results, folds = run_walkforward_test(data, model_seed=42)

    # Aggregate fold metrics
    summary = aggregate_fold_metrics(results)

    # Manually calculate aggregates
    sharpe_scores = [metrics["sharpe_nw"] for _, metrics, _ in results]
    sortino_scores = [metrics["sortino"] for _, metrics, _ in results]
    max_dds = [metrics["max_dd"] for _, metrics, _ in results]
    hit_rates = [metrics["hit_rate"] for _, metrics, _ in results]
    total_returns = [metrics["total_return"] for _, metrics, _ in results]
    volatilities = [metrics["volatility"] for _, metrics, _ in results]
    trade_counts = [len(trades) for _, _, trades in results]

    # Verify aggregation accuracy
    assert (
        abs(summary["mean_sharpe"] - np.mean(sharpe_scores)) < 1e-10
    ), "WF_ACCURACY: Mean Sharpe aggregation error"
    assert (
        abs(summary["mean_sortino"] - np.mean(sortino_scores)) < 1e-10
    ), "WF_ACCURACY: Mean Sortino aggregation error"
    assert (
        abs(summary["mean_max_dd"] - np.mean(max_dds)) < 1e-10
    ), "WF_ACCURACY: Mean Max DD aggregation error"
    assert (
        abs(summary["mean_hit_rate"] - np.mean(hit_rates)) < 1e-10
    ), "WF_ACCURACY: Mean Hit Rate aggregation error"
    assert (
        abs(summary["mean_total_return"] - np.mean(total_returns)) < 1e-10
    ), "WF_ACCURACY: Mean Total Return aggregation error"
    assert (
        abs(summary["mean_volatility"] - np.mean(volatilities)) < 1e-10
    ), "WF_ACCURACY: Mean Volatility aggregation error"
    assert summary["total_trades"] == sum(
        trade_counts
    ), "WF_ACCURACY: Total trades aggregation error"
    assert summary["positive_sharpe_folds"] == sum(
        1 for s in sharpe_scores if s > 0
    ), "WF_ACCURACY: Positive Sharpe folds aggregation error"


def test_metric_statistics_consistency():
    """Test that metric statistics are consistent."""
    # Create test data
    data = create_test_data(seed=42)

    # Run walkforward
    results, folds = run_walkforward_test(data, model_seed=42)

    # Aggregate fold metrics
    summary = aggregate_fold_metrics(results)

    # Extract metrics for manual calculation
    sharpe_scores = [metrics["sharpe_nw"] for _, metrics, _ in results]
    sortino_scores = [metrics["sortino"] for _, metrics, _ in results]
    max_dds = [metrics["max_dd"] for _, metrics, _ in results]
    hit_rates = [metrics["hit_rate"] for _, metrics, _ in results]
    total_returns = [metrics["total_return"] for _, metrics, _ in results]

    # Verify standard deviation calculations
    assert (
        abs(summary["std_sharpe"] - np.std(sharpe_scores)) < 1e-10
    ), "WF_STATS: Sharpe std aggregation error"
    assert (
        abs(summary["std_sortino"] - np.std(sortino_scores)) < 1e-10
    ), "WF_STATS: Sortino std aggregation error"
    assert (
        abs(summary["std_max_dd"] - np.std(max_dds)) < 1e-10
    ), "WF_STATS: Max DD std aggregation error"
    assert (
        abs(summary["std_hit_rate"] - np.std(hit_rates)) < 1e-10
    ), "WF_STATS: Hit Rate std aggregation error"
    assert (
        abs(summary["std_total_return"] - np.std(total_returns)) < 1e-10
    ), "WF_STATS: Total Return std aggregation error"


def test_winning_trades_calculation():
    """Test that winning trades calculation is accurate."""
    # Create test data
    data = create_test_data(seed=42)

    # Run walkforward
    results, folds = run_walkforward_test(data, model_seed=42)

    # Aggregate fold metrics
    summary = aggregate_fold_metrics(results)

    # Manually calculate winning trades
    manual_winning_trades = 0
    for _, metrics, trades in results:
        hit_rate = metrics["hit_rate"]
        n_trades = len(trades)
        manual_winning_trades += int(hit_rate * n_trades)

    assert (
        summary["winning_trades"] == manual_winning_trades
    ), "WF_WINNING: Winning trades calculation error"


def test_metric_bounds():
    """Test that metrics are within reasonable bounds."""
    # Create test data
    data = create_test_data(seed=42)

    # Run walkforward
    results, folds = run_walkforward_test(data, model_seed=42)

    # Aggregate fold metrics
    summary = aggregate_fold_metrics(results)

    # Check metric bounds
    assert (
        -10 < summary["mean_sharpe"] < 10
    ), "WF_BOUNDS: Mean Sharpe out of reasonable bounds"
    assert (
        -10 < summary["mean_sortino"] < 10
    ), "WF_BOUNDS: Mean Sortino out of reasonable bounds"
    assert (
        -1 <= summary["mean_max_dd"] <= 0
    ), "WF_BOUNDS: Mean Max DD out of reasonable bounds"
    assert (
        0 <= summary["mean_hit_rate"] <= 1
    ), "WF_BOUNDS: Mean Hit Rate out of reasonable bounds"
    assert (
        -10 < summary["mean_total_return"] < 10
    ), "WF_BOUNDS: Mean Total Return out of reasonable bounds"
    assert (
        0 <= summary["mean_volatility"] < 1000
    ), "WF_BOUNDS: Mean Volatility out of reasonable bounds"
    assert (
        summary["total_trades"] >= 0
    ), "WF_BOUNDS: Total trades should be non-negative"
    assert (
        summary["winning_trades"] >= 0
    ), "WF_BOUNDS: Winning trades should be non-negative"
    assert (
        summary["positive_sharpe_folds"] >= 0
    ), "WF_BOUNDS: Positive Sharpe folds should be non-negative"


def test_empty_results_handling():
    """Test that empty results are handled gracefully."""
    # Test with empty results
    empty_results = []
    summary = aggregate_fold_metrics(empty_results)

    # Should return empty summary
    assert summary == {}, "WF_EMPTY: Empty results should return empty summary"


def test_single_fold_consistency():
    """Test that single fold results are handled correctly."""
    # Create test data
    data = create_test_data(seed=42)

    # Run walkforward with single fold
    X, y, prices = build_feature_table(data, warmup_days=20)
    folds = list(gen_walkforward(n=len(X), train_len=30, test_len=10, stride=len(X)))
    pipeline = LeakageProofPipeline(X, y)
    results = walkforward_run(pipeline, folds, prices, model_seed=42)

    # Should have exactly one fold
    assert len(results) == 1, "WF_SINGLE: Should have exactly one fold"

    # Aggregate metrics
    summary = aggregate_fold_metrics(results)

    # Single fold metrics should match summary
    fold_id, metrics, trades = results[0]
    assert (
        abs(summary["mean_sharpe"] - metrics["sharpe_nw"]) < 1e-10
    ), "WF_SINGLE: Sharpe mismatch"
    assert (
        abs(summary["mean_sortino"] - metrics["sortino"]) < 1e-10
    ), "WF_SINGLE: Sortino mismatch"
    assert (
        abs(summary["mean_max_dd"] - metrics["max_dd"]) < 1e-10
    ), "WF_SINGLE: Max DD mismatch"
    assert (
        abs(summary["mean_hit_rate"] - metrics["hit_rate"]) < 1e-10
    ), "WF_SINGLE: Hit Rate mismatch"
    assert (
        abs(summary["mean_total_return"] - metrics["total_return"]) < 1e-10
    ), "WF_SINGLE: Total Return mismatch"
    assert summary["total_trades"] == len(trades), "WF_SINGLE: Trade count mismatch"
