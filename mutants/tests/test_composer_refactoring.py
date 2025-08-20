"""
Test composer refactoring changes.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.composer.registry import build_composer_system
from core.config import get_cfg, load_config
from core.engine.composer_integration import ComposerIntegration
from core.walk.folds import gen_walkforward


def test_composer_integration_warmup_gating():
    """Test that composer calls are gated behind min_history_bars."""
    config = {
        "composer": {
            "use_composer": True,
            "min_history_bars": 120,
            "eligible_strategies": ["momentum", "mean_reversion"],
        }
    }

    composer = ComposerIntegration(config)

    # Create mock data
    data = pd.DataFrame(
        {
            "Open": np.random.randn(200) + 100,
            "High": np.random.randn(200) + 101,
            "Low": np.random.randn(200) + 99,
            "Close": np.random.randn(200) + 100,
            "Volume": np.random.randint(1000000, 10000000, 200),
        }
    )

    # Test warmup period (before min_history_bars)
    signal, metadata = composer.get_composer_decision(data, "SPY", 50)
    assert signal == 0.0
    assert metadata["reason"] == "warmup"
    assert not metadata["composer_used"]

    # Test after warmup period
    signal, metadata = composer.get_composer_decision(data, "SPY", 150)
    # Should either use composer or have a valid reason for not using it
    assert "reason" in metadata or metadata["composer_used"]


def test_composer_registry_strategy_filtering():
    """Test that only registered strategies are used."""
    config = {
        "eligible_strategies": ["momentum", "mean_reversion", "nonexistent_strategy"],
        "regime_extractor": "basic_kpis",
        "composer_mode": "softmax_blender",
    }

    # Should not raise an error, but should warn about missing strategy
    with patch("core.composer.registry.logger") as mock_logger:
        strategies, regime_extractor, composer = build_composer_system(config)

        # Should have 2 valid strategies
        assert len(strategies) == 2
        assert all(s.name in ["momentum", "mean_reversion"] for s in strategies)

        # Should have warned about missing strategy
        mock_logger.warning.assert_called()


def test_composer_registry_insufficient_strategies():
    """Test that insufficient strategies raise an error."""
    config = {
        "eligible_strategies": ["nonexistent_strategy"],
        "regime_extractor": "basic_kpis",
        "composer_mode": "softmax_blender",
    }

    with pytest.raises(ValueError, match="Insufficient strategies"):
        build_composer_system(config)


def test_fold_generation_short_window():
    """Test that short final test windows are handled correctly."""
    n = 1000
    train_len = 252
    test_len = 63
    stride = 63

    # Test without allowing truncated folds
    folds = list(gen_walkforward(n, train_len, test_len, stride, allow_truncated_final_fold=False))

    # All folds should have full test windows
    for fold in folds:
        test_window_size = fold.test_hi - fold.test_lo + 1
        assert test_window_size >= test_len

    # Test with allowing truncated folds
    folds = list(gen_walkforward(n, train_len, test_len, stride, allow_truncated_final_fold=True))

    # Should have more folds when allowing truncation
    assert len(folds) >= len(
        list(gen_walkforward(n, train_len, test_len, stride, allow_truncated_final_fold=False))
    )


def test_metrics_empty_equity_curve():
    """Test that empty equity curves return zero metrics with reason."""
    from scripts.walkforward_with_composer import compute_metrics_from_pnl

    # Test empty PnL series
    pnl_series = np.array([])
    trades = []

    metrics = compute_metrics_from_pnl(pnl_series, trades)

    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_nw"] == 0.0
    assert metrics["max_dd"] == 0.0
    assert metrics["trade_count"] == 0
    assert metrics["reason"] == "no_trades"

    # Test NaN PnL series
    pnl_series = np.array([np.nan, np.nan, np.nan])
    trades = []

    metrics = compute_metrics_from_pnl(pnl_series, trades)

    assert metrics["total_return"] == 0.0
    assert metrics["sharpe_nw"] == 0.0
    assert metrics["max_dd"] == 0.0
    assert metrics["trade_count"] == 0
    assert metrics["reason"] == "no_trades"


def test_config_loading():
    """Test configuration loading and merging."""
    # Test loading base config
    config = load_config(["config/base.yaml"])

    assert "engine" in config
    assert "composer" in config
    assert "walkforward" in config

    # Test deep merging

    merged = load_config(["config/base.yaml"])
    # Note: In a real test, we'd need to create a temporary overlay file
    # For now, just test the structure

    assert get_cfg(merged, "risk.vol_target") == 0.15  # From base config


def test_data_sanity_backcompat():
    """Test data sanity back-compat shim method."""
    from core.data_sanity import DataSanityWrapper

    wrapper = DataSanityWrapper()

    # Create test data with proper data types
    data = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1000000, 1000000, 1000000],
        }
    )

    # Test back-compat method
    result = wrapper.validate_dataframe(data, symbol="TEST")
    assert len(result) == 3

    # Test with kwargs
    result = wrapper.validate_dataframe(data, symbol="TEST", profile="default")
    assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__])
