"""
Unit tests for position sizing engine.
"""

import pytest
from core.execution.position_sizing import PositionSizer, PositionSizingConfig


class TestPositionSizingConfig:
    """Test position sizing configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PositionSizingConfig()
        
        assert config.max_position_size == 0.1
        assert config.max_total_exposure == 0.8
        assert config.min_trade_size == 100.0
        assert config.max_trade_size == 10000.0
        assert config.signal_threshold == 0.1
        assert config.volatility_adjustment is True
        assert config.portfolio_heat == 0.02
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PositionSizingConfig(
            max_position_size=0.15,
            min_trade_size=200.0,
            signal_threshold=0.2
        )
        
        assert config.max_position_size == 0.15
        assert config.min_trade_size == 200.0
        assert config.signal_threshold == 0.2


class TestPositionSizer:
    """Test position sizing functionality."""
    
    @pytest.fixture
    def position_sizer(self):
        """Create position sizer for testing."""
        config = PositionSizingConfig(
            max_position_size=0.1,
            max_total_exposure=0.8,
            min_trade_size=100.0,
            max_trade_size=10000.0,
            signal_threshold=0.1
        )
        return PositionSizer(config)
    
    def test_calculate_position_size_strong_signal(self, position_sizer):
        """Test position sizing with strong signal."""
        signal = 0.8
        symbol = "AAPL"
        current_price = 150.0
        portfolio_value = 100000.0
        current_positions = {}
        
        quantity, metadata = position_sizer.calculate_position_size(
            signal=signal,
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            current_positions=current_positions
        )
        
        assert quantity > 0
        assert metadata["signal"] == signal
        assert metadata["position_value"] == quantity * current_price
        assert metadata["position_pct"] == (quantity * current_price) / portfolio_value
    
    def test_calculate_position_size_weak_signal(self, position_sizer):
        """Test position sizing with weak signal below threshold."""
        signal = 0.05  # Below threshold
        symbol = "AAPL"
        current_price = 150.0
        portfolio_value = 100000.0
        current_positions = {}
        
        quantity, metadata = position_sizer.calculate_position_size(
            signal=signal,
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            current_positions=current_positions
        )
        
        assert quantity == 0
        assert metadata["reason"] == "signal_below_threshold"
    
    def test_calculate_position_size_zero_portfolio(self, position_sizer):
        """Test position sizing with zero portfolio value."""
        signal = 0.8
        symbol = "AAPL"
        current_price = 150.0
        portfolio_value = 0.0
        current_positions = {}
        
        quantity, metadata = position_sizer.calculate_position_size(
            signal=signal,
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            current_positions=current_positions
        )
        
        assert quantity == 0
        assert metadata["reason"] == "invalid_portfolio_value"
    
    def test_calculate_position_size_invalid_price(self, position_sizer):
        """Test position sizing with invalid price."""
        signal = 0.8
        symbol = "AAPL"
        current_price = 0.0
        portfolio_value = 100000.0
        current_positions = {}
        
        quantity, metadata = position_sizer.calculate_position_size(
            signal=signal,
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            current_positions=current_positions
        )
        
        assert quantity == 0
        assert metadata["reason"] == "invalid_price"
    
    def test_calculate_position_size_with_existing_positions(self, position_sizer):
        """Test position sizing with existing positions."""
        signal = 0.8
        symbol = "AAPL"
        current_price = 150.0
        portfolio_value = 100000.0
        current_positions = {
            "MSFT": {"quantity": 100, "value": 30000.0},
            "GOOGL": {"quantity": 50, "value": 15000.0}
        }
        
        quantity, metadata = position_sizer.calculate_position_size(
            signal=signal,
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            current_positions=current_positions
        )
        
        assert quantity > 0
        # Should be smaller due to existing exposure
        assert metadata["position_pct"] < 0.1  # Less than max position size
    
    def test_calculate_position_size_max_exposure_reached(self, position_sizer):
        """Test position sizing when max exposure is reached."""
        signal = 0.8
        symbol = "AAPL"
        current_price = 150.0
        portfolio_value = 100000.0
        # Existing positions already at 80% exposure
        current_positions = {
            "MSFT": {"quantity": 100, "value": 40000.0},
            "GOOGL": {"quantity": 100, "value": 40000.0}
        }
        
        quantity, metadata = position_sizer.calculate_position_size(
            signal=signal,
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            current_positions=current_positions
        )
        
        assert quantity == 0
        assert metadata["reason"] == "zero_quantity"
    
    def test_validate_position_valid(self, position_sizer):
        """Test position validation with valid position."""
        symbol = "AAPL"
        quantity = 50  # Smaller quantity to stay under max trade size
        current_price = 150.0
        portfolio_value = 100000.0
        current_positions = {}
        
        is_valid, reason = position_sizer.validate_position(
            symbol, quantity, current_price, portfolio_value, current_positions
        )
        
        assert is_valid
        assert reason == "valid"
    
    def test_validate_position_too_small(self, position_sizer):
        """Test position validation with position too small."""
        symbol = "AAPL"
        quantity = 1  # Very small position
        current_price = 50.0  # $50 total value
        portfolio_value = 100000.0
        current_positions = {}
        
        is_valid, reason = position_sizer.validate_position(
            symbol, quantity, current_price, portfolio_value, current_positions
        )
        
        assert not is_valid
        assert "below minimum" in reason
    
    def test_validate_position_too_large(self, position_sizer):
        """Test position validation with position too large."""
        symbol = "AAPL"
        quantity = 1000  # Large position
        current_price = 150.0  # $150,000 total value
        portfolio_value = 100000.0  # 150% of portfolio
        current_positions = {}
        
        is_valid, reason = position_sizer.validate_position(
            symbol, quantity, current_price, portfolio_value, current_positions
        )
        
        assert not is_valid
        assert "above maximum" in reason
    
    def test_calculate_rebalancing_orders(self, position_sizer):
        """Test rebalancing order calculation."""
        current_positions = {
            "AAPL": {"quantity": 100, "value": 15000.0},
            "MSFT": {"quantity": 50, "value": 15000.0}
        }
        target_weights = {
            "AAPL": 0.4,  # 40%
            "MSFT": 0.3,  # 30%
            "GOOGL": 0.3  # 30% (new position)
        }
        portfolio_value = 100000.0
        current_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2800.0
        }
        
        rebalancing_orders = position_sizer.calculate_rebalancing_orders(
            current_positions, target_weights, portfolio_value, current_prices
        )
        
        assert len(rebalancing_orders) > 0
        # Should include orders for all symbols that need rebalancing
        assert "AAPL" in rebalancing_orders or "MSFT" in rebalancing_orders or "GOOGL" in rebalancing_orders
    
    def test_get_position_sizing_summary(self, position_sizer):
        """Test getting position sizing summary."""
        summary = position_sizer.get_position_sizing_summary()
        
        assert "max_position_size" in summary
        assert "max_total_exposure" in summary
        assert "min_trade_size" in summary
        assert "max_trade_size" in summary
        assert "signal_threshold" in summary
        assert "portfolio_heat" in summary
        assert "volatility_adjustment" in summary
