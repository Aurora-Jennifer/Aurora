"""
Unit tests for composite scoring system.
"""


import pytest

from core.metrics.composite import (
    CompositePenalties,
    CompositeWeights,
    composite_score,
    evaluate_strategy_performance,
    load_composite_config,
    normalize_avg_trade_return,
    normalize_cagr,
    normalize_sharpe,
    normalize_win_rate,
)


class TestNormalizationFunctions:
    """Test metric normalization functions."""

    def test_normalize_cagr(self):
        """Test CAGR normalization."""
        # Test positive CAGR
        assert normalize_cagr(0.1) == pytest.approx(0.4, rel=1e-2)  # 10% CAGR
        assert normalize_cagr(0.5) == pytest.approx(0.667, rel=1e-2)  # 50% CAGR

        # Test negative CAGR
        assert normalize_cagr(-0.2) == pytest.approx(0.2, rel=1e-2)  # -20% CAGR
        assert normalize_cagr(-0.5) == pytest.approx(
            0.0, rel=1e-2
        )  # -50% CAGR (clipped)

        # Test bounds
        assert normalize_cagr(1.0) == pytest.approx(
            1.0, rel=1e-2
        )  # 100% CAGR (clipped)
        assert normalize_cagr(-0.6) == pytest.approx(
            0.0, rel=1e-2
        )  # -60% CAGR (clipped)

    def test_normalize_sharpe(self):
        """Test Sharpe ratio normalization."""
        # Test positive Sharpe
        assert normalize_sharpe(1.0) == pytest.approx(0.6, rel=1e-2)  # Sharpe 1.0
        assert normalize_sharpe(2.0) == pytest.approx(0.8, rel=1e-2)  # Sharpe 2.0

        # Test negative Sharpe
        assert normalize_sharpe(-1.0) == pytest.approx(0.2, rel=1e-2)  # Sharpe -1.0
        assert normalize_sharpe(-2.0) == pytest.approx(
            0.0, rel=1e-2
        )  # Sharpe -2.0 (clipped)

        # Test bounds
        assert normalize_sharpe(3.0) == pytest.approx(
            1.0, rel=1e-2
        )  # Sharpe 3.0 (clipped)
        assert normalize_sharpe(-3.0) == pytest.approx(
            0.0, rel=1e-2
        )  # Sharpe -3.0 (clipped)

    def test_normalize_win_rate(self):
        """Test win rate normalization."""
        assert normalize_win_rate(0.5) == 0.5  # 50% win rate
        assert normalize_win_rate(0.8) == 0.8  # 80% win rate
        assert normalize_win_rate(0.0) == 0.0  # 0% win rate
        assert normalize_win_rate(1.0) == 1.0  # 100% win rate

        # Test clipping
        assert normalize_win_rate(1.5) == 1.0  # Clipped to 1.0
        assert normalize_win_rate(-0.1) == 0.0  # Clipped to 0.0

    def test_normalize_avg_trade_return(self):
        """Test average trade return normalization."""
        # Test positive returns
        assert normalize_avg_trade_return(0.005) == pytest.approx(
            0.75, rel=1e-2
        )  # 0.5% return
        assert normalize_avg_trade_return(0.01) == pytest.approx(
            1.0, rel=1e-2
        )  # 1% return (clipped)

        # Test negative returns
        assert normalize_avg_trade_return(-0.005) == pytest.approx(
            0.25, rel=1e-2
        )  # -0.5% return
        assert normalize_avg_trade_return(-0.01) == pytest.approx(
            0.0, rel=1e-2
        )  # -1% return (clipped)

        # Test bounds
        assert normalize_avg_trade_return(0.02) == pytest.approx(
            1.0, rel=1e-2
        )  # 2% return (clipped)
        assert normalize_avg_trade_return(-0.02) == pytest.approx(
            0.0, rel=1e-2
        )  # -2% return (clipped)


class TestCompositeWeights:
    """Test CompositeWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = CompositeWeights()
        assert weights.alpha == 0.4
        assert weights.beta == 0.3
        assert weights.gamma == 0.2
        assert weights.delta == 0.1
        assert (
            abs(weights.alpha + weights.beta + weights.gamma + weights.delta - 1.0)
            < 1e-10
        )

    def test_custom_weights(self):
        """Test custom weight values."""
        weights = CompositeWeights(alpha=0.5, beta=0.3, gamma=0.15, delta=0.05)
        assert weights.alpha == 0.5
        assert weights.beta == 0.3
        assert weights.gamma == 0.15
        assert weights.delta == 0.05


class TestCompositePenalties:
    """Test CompositePenalties dataclass."""

    def test_default_penalties(self):
        """Test default penalty values."""
        penalties = CompositePenalties()
        assert penalties.max_dd_cap == 0.25
        assert penalties.min_trades == 200
        assert penalties.dd_penalty_factor == 2.0
        assert penalties.trade_penalty_factor == 1.5

    def test_custom_penalties(self):
        """Test custom penalty values."""
        penalties = CompositePenalties(
            max_dd_cap=0.15,
            min_trades=100,
            dd_penalty_factor=3.0,
            trade_penalty_factor=2.0,
        )
        assert penalties.max_dd_cap == 0.15
        assert penalties.min_trades == 100
        assert penalties.dd_penalty_factor == 3.0
        assert penalties.trade_penalty_factor == 2.0


class TestCompositeScore:
    """Test composite score calculation."""

    def test_perfect_strategy(self):
        """Test composite score for perfect strategy."""
        metrics = {
            "cagr": 0.5,  # 50% CAGR
            "sharpe": 2.0,  # Sharpe 2.0
            "win_rate": 0.8,  # 80% win rate
            "avg_trade_return": 0.01,  # 1% avg return
            "max_dd": 0.1,  # 10% max drawdown
            "trade_count": 500,  # 500 trades
        }

        score = composite_score(metrics)
        assert score > 0.8  # Should be high score
        assert score <= 1.0  # Should not exceed 1.0

    def test_poor_strategy(self):
        """Test composite score for poor strategy."""
        metrics = {
            "cagr": -0.2,  # -20% CAGR
            "sharpe": -1.0,  # Negative Sharpe
            "win_rate": 0.3,  # 30% win rate
            "avg_trade_return": -0.005,  # -0.5% avg return
            "max_dd": 0.4,  # 40% max drawdown
            "trade_count": 50,  # Only 50 trades
        }

        score = composite_score(metrics)
        assert score < 0.3  # Should be low score
        assert score >= 0.0  # Should not be negative

    def test_drawdown_penalty(self):
        """Test drawdown penalty application."""
        # Strategy with acceptable drawdown
        metrics_acceptable = {
            "cagr": 0.2,
            "sharpe": 1.5,
            "win_rate": 0.6,
            "avg_trade_return": 0.005,
            "max_dd": 0.2,  # 20% drawdown (under cap)
            "trade_count": 300,
        }

        # Strategy with excessive drawdown
        metrics_excessive = {
            "cagr": 0.2,
            "sharpe": 1.5,
            "win_rate": 0.6,
            "avg_trade_return": 0.005,
            "max_dd": 0.4,  # 40% drawdown (over cap)
            "trade_count": 300,
        }

        score_acceptable = composite_score(metrics_acceptable)
        score_excessive = composite_score(metrics_excessive)

        assert (
            score_acceptable > score_excessive
        )  # Excessive drawdown should be penalized

    def test_trade_count_penalty(self):
        """Test trade count penalty application."""
        # Strategy with sufficient trades
        metrics_sufficient = {
            "cagr": 0.2,
            "sharpe": 1.5,
            "win_rate": 0.6,
            "avg_trade_return": 0.005,
            "max_dd": 0.2,
            "trade_count": 300,  # Sufficient trades
        }

        # Strategy with insufficient trades
        metrics_insufficient = {
            "cagr": 0.2,
            "sharpe": 1.5,
            "win_rate": 0.6,
            "avg_trade_return": 0.005,
            "max_dd": 0.2,
            "trade_count": 50,  # Insufficient trades
        }

        score_sufficient = composite_score(metrics_sufficient)
        score_insufficient = composite_score(metrics_insufficient)

        assert (
            score_sufficient > score_insufficient
        )  # Insufficient trades should be penalized

    def test_missing_metrics(self):
        """Test handling of missing metrics."""
        metrics = {
            "cagr": 0.2,
            "sharpe": 1.5,
            # Missing other metrics
        }

        score = composite_score(metrics)
        assert score >= 0.0  # Should not crash
        assert score <= 1.0  # Should be bounded

    def test_custom_weights(self):
        """Test composite score with custom weights."""
        metrics = {
            "cagr": 0.3,
            "sharpe": 1.0,
            "win_rate": 0.7,
            "avg_trade_return": 0.005,
            "max_dd": 0.15,
            "trade_count": 250,
        }

        # Equal weights
        weights_equal = CompositeWeights(alpha=0.25, beta=0.25, gamma=0.25, delta=0.25)
        score_equal = composite_score(metrics, weights=weights_equal)

        # CAGR-focused weights
        weights_cagr = CompositeWeights(alpha=0.7, beta=0.1, gamma=0.1, delta=0.1)
        score_cagr = composite_score(metrics, weights=weights_cagr)

        # Scores should be different due to different weightings
        assert abs(score_equal - score_cagr) > 0.01


class TestEvaluateStrategyPerformance:
    """Test detailed strategy performance evaluation."""

    def test_performance_evaluation(self):
        """Test detailed performance evaluation."""
        results = {
            "cagr": 0.25,
            "sharpe": 1.8,
            "win_rate": 0.65,
            "avg_trade_return": 0.008,
            "max_dd": 0.18,
            "trade_count": 280,
        }

        evaluation = evaluate_strategy_performance(results)

        # Check structure
        assert "composite_score" in evaluation
        assert "weighted_sum" in evaluation
        assert "total_penalty" in evaluation
        assert "components" in evaluation
        assert "normalized_metrics" in evaluation
        assert "penalties" in evaluation

        # Check component scores
        components = evaluation["components"]
        assert "cagr_score" in components
        assert "sharpe_score" in components
        assert "win_rate_score" in components
        assert "avg_return_score" in components

        # Check normalized metrics
        norm_metrics = evaluation["normalized_metrics"]
        assert "norm_cagr" in norm_metrics
        assert "norm_sharpe" in norm_metrics
        assert "norm_win_rate" in norm_metrics
        assert "norm_avg_return" in norm_metrics

        # Check penalties
        penalties = evaluation["penalties"]
        assert "dd_penalty" in penalties
        assert "trade_penalty" in penalties

        # Check score bounds
        assert 0.0 <= evaluation["composite_score"] <= 1.0
        assert evaluation["weighted_sum"] >= 0.0
        assert evaluation["total_penalty"] >= 0.0


class TestLoadCompositeConfig:
    """Test configuration loading."""

    def test_load_default_config(self):
        """Test loading with default values."""
        config = {}
        weights, penalties = load_composite_config(config)

        # Check default weights
        assert weights.alpha == 0.4
        assert weights.beta == 0.3
        assert weights.gamma == 0.2
        assert weights.delta == 0.1

        # Check default penalties
        assert penalties.max_dd_cap == 0.25
        assert penalties.min_trades == 200
        assert penalties.dd_penalty_factor == 2.0
        assert penalties.trade_penalty_factor == 1.5

    def test_load_custom_config(self):
        """Test loading with custom values."""
        config = {
            "metric_weights": {"alpha": 0.5, "beta": 0.25, "gamma": 0.15, "delta": 0.1},
            "metric_weight_max_dd_cap": 0.15,
            "metric_weight_min_trades": 100,
            "metric_weight_dd_penalty_factor": 3.0,
            "metric_weight_trade_penalty_factor": 2.0,
        }

        weights, penalties = load_composite_config(config)

        # Check custom weights
        assert weights.alpha == 0.5
        assert weights.beta == 0.25
        assert weights.gamma == 0.15
        assert weights.delta == 0.1

        # Check custom penalties
        assert penalties.max_dd_cap == 0.15
        assert penalties.min_trades == 100
        assert penalties.dd_penalty_factor == 3.0
        assert penalties.trade_penalty_factor == 2.0

    def test_load_partial_config(self):
        """Test loading with partial configuration."""
        config = {
            "metric_weights": {
                "alpha": 0.6,
                "beta": 0.4
                # Missing gamma and delta
            },
            "metric_weight_max_dd_cap": 0.2
            # Missing other penalty parameters
        }

        weights, penalties = load_composite_config(config)

        # Check partial weights (should use defaults for missing)
        assert weights.alpha == 0.6
        assert weights.beta == 0.4
        assert weights.gamma == 0.2  # Default
        assert weights.delta == 0.1  # Default

        # Check partial penalties (should use defaults for missing)
        assert penalties.max_dd_cap == 0.2
        assert penalties.min_trades == 200  # Default
        assert penalties.dd_penalty_factor == 2.0  # Default
        assert penalties.trade_penalty_factor == 1.5  # Default


if __name__ == "__main__":
    pytest.main([__file__])
