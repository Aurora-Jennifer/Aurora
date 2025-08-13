"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from strategies.base import BaseStrategy
from strategies.regime_aware_ensemble import RegimeAwareEnsembleStrategy, RegimeAwareEnsembleParams


class TestBaseStrategy:
    """Test base strategy functionality."""

    def test_base_strategy_is_abstract(self):
        """Test that base strategy cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseStrategy()


class TestRegimeAwareEnsembleStrategy:
    """Test regime-aware ensemble strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = RegimeAwareEnsembleParams()
        self.strategy = RegimeAwareEnsembleStrategy(self.params)

    def test_strategy_initialization(self):
        """Test strategy can be initialized."""
        assert self.strategy is not None
        assert self.strategy.params == self.params

    def test_generate_signals_with_sufficient_data(self):
        """Test signal generation with sufficient data."""
        # Create test data with enough history
        dates = pd.date_range("2025-01-01", periods=300, freq="D")
        data = pd.DataFrame({
            "Close": np.random.randn(300).cumsum() + 100,
            "High": np.random.randn(300).cumsum() + 102,
            "Low": np.random.randn(300).cumsum() + 98,
            "Volume": np.random.randint(1000, 10000, 300)
        }, index=dates)
        
        signals = self.strategy.generate_signals(data)
        
        assert len(signals) == len(data)
        assert isinstance(signals, pd.Series)
        assert signals.dtype == float

    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data."""
        # Create test data with insufficient history
        dates = pd.date_range("2025-01-01", periods=30, freq="D")
        data = pd.DataFrame({
            "Close": np.random.randn(30).cumsum() + 100,
            "High": np.random.randn(30).cumsum() + 102,
            "Low": np.random.randn(30).cumsum() + 98,
            "Volume": np.random.randint(1000, 10000, 30)
        }, index=dates)
        
        signals = self.strategy.generate_signals(data)
        
        # Should return signals with some NaN values for insufficient data
        assert len(signals) == len(data)
        assert isinstance(signals, pd.Series)

    @patch('core.regime_detector.logger')
    def test_generate_signals_logs_warning_for_insufficient_data(self, mock_logger):
        """Test that warning is logged for insufficient data."""
        dates = pd.date_range("2025-01-01", periods=30, freq="D")
        data = pd.DataFrame({
            "Close": np.random.randn(30).cumsum() + 100,
            "High": np.random.randn(30).cumsum() + 102,
            "Low": np.random.randn(30).cumsum() + 98,
            "Volume": np.random.randint(1000, 10000, 30)
        }, index=dates)
        
        self.strategy.generate_signals(data)
        
        mock_logger.warning.assert_called_once()

    def test_regime_detection(self):
        """Test regime detection functionality."""
        dates = pd.date_range("2025-01-01", periods=300, freq="D")
        data = pd.DataFrame({
            "Close": np.random.randn(300).cumsum() + 100,
            "High": np.random.randn(300).cumsum() + 102,
            "Low": np.random.randn(300).cumsum() + 98,
            "Volume": np.random.randint(1000, 10000, 300)
        }, index=dates)
        
        regime_name, confidence, regime_params = self.strategy.regime_detector.detect_regime(data)
        
        assert regime_name in ["trend", "chop", "volatile"]
        assert 0 <= confidence <= 1
        assert hasattr(regime_params, 'regime_name')


class TestRegimeAwareEnsembleParams:
    """Test strategy parameters."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = RegimeAwareEnsembleParams()
        
        assert params.confidence_threshold == 0.3
        assert params.regime_lookback == 252
        assert params.min_periods == 30
        assert isinstance(params.trend_following_weight, float)
        assert isinstance(params.mean_reversion_weight, float)
        assert isinstance(params.rolling_window, int)

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = RegimeAwareEnsembleParams(
            confidence_threshold=0.5,
            regime_lookback=100,
            min_periods=100
        )
        
        assert params.confidence_threshold == 0.5
        assert params.regime_lookback == 100
        assert params.min_periods == 100
