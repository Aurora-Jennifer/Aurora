"""
End-to-end test for the refactored composer system.
"""

import numpy as np
import pandas as pd
import pytest

from core.config import load_config
from core.engine.composer_integration import ComposerIntegration
from core.walk.folds import gen_walkforward
from scripts.walkforward_with_composer import run_walkforward_with_composer


def test_composer_end_to_end():
    """Test the complete composer system end-to-end."""

    # Load configuration
    config = load_config(["config/base.yaml"])

    # Create test data
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)  # For reproducibility

    # Generate realistic price data
    returns = np.random.normal(0.0005, 0.015, len(dates))  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100

    data = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.normal(0, 0.002, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    # Test composer integration
    composer = ComposerIntegration(config)

    # Test warmup period
    signal, metadata = composer.get_composer_decision(data, "SPY", 50)
    assert signal == 0.0
    assert metadata["reason"] == "warmup"
    assert not metadata["composer_used"]

    # Test after warmup period
    signal, metadata = composer.get_composer_decision(data, "SPY", 150)
    # Should either use composer or have a valid reason for not using it
    assert "reason" in metadata or metadata["composer_used"]

    # Test fold generation
    n = len(data)
    train_len = config["walkforward"]["fold_length"]
    test_len = config["walkforward"]["step_size"]
    stride = config["walkforward"]["step_size"]

    folds = list(
        gen_walkforward(
            n,
            train_len,
            test_len,
            stride,
            allow_truncated_final_fold=config["walkforward"]["allow_truncated_final_fold"],
        )
    )

    assert len(folds) > 0, "Should generate at least one fold"

    # Test walkforward with composer
    results = run_walkforward_with_composer(
        data,
        "SPY",
        config,
        train_len=train_len,
        test_len=test_len,
        stride=stride,
        validate_data=False,  # Skip data validation for speed
    )

    assert len(results) > 0, "Should have results"

    # Check that results have expected structure
    for fold_id, metrics, trades in results:
        assert isinstance(fold_id, int)
        assert isinstance(metrics, dict)
        assert isinstance(trades, list)

        # Check that metrics have expected keys
        expected_keys = [
            "total_return",
            "sharpe_nw",
            "max_dd",
            "win_rate",
            "trade_count",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

        # Check that metrics are finite or have reason
        for key, value in metrics.items():
            if key != "reason":
                assert np.isfinite(value) or value == 0.0, f"Non-finite metric {key}: {value}"


def test_config_overlay():
    """Test configuration overlay functionality."""

    # Load base config
    base_config = load_config(["config/base.yaml"])
    assert base_config["risk"]["vol_target"] == 0.15

    # Load with risk overlay
    overlay_config = load_config(["config/base.yaml", "config/risk_low.yaml"])
    assert overlay_config["risk"]["vol_target"] == 0.10
    assert overlay_config["composer"]["params"]["temperature"] == 0.8

    # Verify other settings are preserved
    assert overlay_config["engine"]["min_history_bars"] == 120
    assert overlay_config["data"]["source"] == "yfinance"


def test_metrics_empty_handling():
    """Test that empty metrics are handled correctly."""
    from scripts.walkforward_with_composer import compute_metrics_from_pnl

    # Test with empty data
    pnl_series = np.array([])
    trades = []

    metrics = compute_metrics_from_pnl(pnl_series, trades)

    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_nw"] == 0.0
    assert metrics["max_dd"] == 0.0
    assert metrics["trade_count"] == 0
    assert metrics["reason"] == "no_trades"

    # Test with zero trades
    pnl_series = np.array([0.0, 0.0, 0.0])
    trades = [{"wins": 0, "losses": 0}]

    metrics = compute_metrics_from_pnl(pnl_series, trades)

    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_nw"] == 0.0
    assert metrics["max_dd"] == 0.0
    assert metrics["trade_count"] == 0
    assert metrics["reason"] == "no_trades"


if __name__ == "__main__":
    pytest.main([__file__])
