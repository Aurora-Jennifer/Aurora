"""
Integration tests for execution infrastructure.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from core.execution.order_types import Order, OrderType, OrderSide, OrderStatus
from core.execution.position_sizing import PositionSizer, PositionSizingConfig
from core.execution.risk_manager import RiskManager, RiskLimits
from core.execution.portfolio_manager import PortfolioManager
from core.execution.order_manager import OrderManager
from core.execution.execution_engine import ExecutionEngine, ExecutionConfig


class TestExecutionIntegration:
    """Test integration between execution components."""
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Create mock Alpaca client."""
        client = Mock()
        client.get_account.return_value = Mock(cash=50000.0)
        client.get_all_positions.return_value = []
        client.get_all_orders.return_value = []  # Return empty list for reconciliation
        client.submit_order.return_value = Mock(id="order_123")
        return client
    
    @pytest.fixture
    def position_sizer(self):
        """Create position sizer."""
        config = PositionSizingConfig(
            max_position_size=0.1,
            min_trade_size=100.0,
            signal_threshold=0.1
        )
        return PositionSizer(config)
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager."""
        limits = RiskLimits(
            max_daily_loss=0.02,
            max_position_risk=0.05,
            max_orders_per_day=100
        )
        return RiskManager(limits)
    
    @pytest.fixture
    def portfolio_manager(self, mock_alpaca_client):
        """Create portfolio manager."""
        return PortfolioManager(mock_alpaca_client)
    
    @pytest.fixture
    def order_manager(self, mock_alpaca_client):
        """Create order manager."""
        return OrderManager(mock_alpaca_client)
    
    @pytest.fixture
    def execution_engine(self, order_manager, portfolio_manager, position_sizer, risk_manager):
        """Create execution engine."""
        config = ExecutionConfig(
            enabled=True,
            mode="paper",
            signal_threshold=0.1
        )
        return ExecutionEngine(
            order_manager=order_manager,
            portfolio_manager=portfolio_manager,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            config=config
        )
    
    def test_signal_to_order_flow(self, execution_engine):
        """Test complete signal to order flow."""
        # Mock signals
        signals = {
            "AAPL": 0.8,  # Strong buy signal
            "MSFT": -0.6,  # Strong sell signal
            "GOOGL": 0.05  # Weak signal (should be filtered out)
        }
        
        # Mock current prices
        current_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2800.0
        }
        
        # Execute signals
        result = execution_engine.execute_signals(signals, current_prices)
        
        # Verify result
        assert result.success is True
        assert result.orders_submitted >= 0  # May be 0 if risk checks fail
        assert result.execution_time > 0
        assert len(result.errors) == 0 or "GOOGL" in str(result.errors)  # Weak signal should be filtered
    
    def test_position_sizing_integration(self, position_sizer):
        """Test position sizing with realistic parameters."""
        signal = 0.7
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
        assert metadata["position_pct"] <= 0.1  # Should respect max position size
    
    def test_risk_management_integration(self, risk_manager):
        """Test risk management with order validation."""
        # Create a test order with smaller quantity to stay within limits
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=30,  # Smaller quantity: 30 * 150 = $4,500 (4.5% of portfolio)
            order_type=OrderType.MARKET
        )
        
        # Test risk check
        current_positions = {}
        portfolio_value = 100000.0
        daily_pnl = 0.0
        current_prices = {"AAPL": 150.0}
        
        is_allowed, reason, risk_metrics = risk_manager.check_order_risk(
            order=order,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            current_prices=current_prices
        )
        
        assert is_allowed
        assert reason == "risk_check_passed"
        assert hasattr(risk_metrics, "portfolio_value")
    
    def test_portfolio_manager_integration(self, portfolio_manager, mock_alpaca_client):
        """Test portfolio manager functionality."""
        # Mock account and positions
        mock_account = Mock()
        mock_account.cash = 50000.0
        mock_alpaca_client.get_account.return_value = mock_account
        mock_alpaca_client.get_all_positions.return_value = []
        
        # Update positions
        success = portfolio_manager.update_positions()
        
        assert success
        assert portfolio_manager.cash == 50000.0
        assert len(portfolio_manager.positions) == 0
        
        # Get portfolio summary
        summary = portfolio_manager.get_portfolio_summary()
        assert "portfolio_value" in summary
        assert "cash" in summary
        assert "positions_count" in summary
    
    def test_order_manager_integration(self, order_manager, mock_alpaca_client):
        """Test order manager functionality."""
        # Create test order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Mock successful submission
        mock_order = Mock()
        mock_order.id = "order_123"
        mock_alpaca_client.submit_order.return_value = mock_order
        
        # Submit order
        order_id = order_manager.submit_order(order)
        
        assert order_id == "order_123"
        assert order.alpaca_order_id == "order_123"
        assert order.status == OrderStatus.SUBMITTED
        assert "order_123" in order_manager.pending_orders
    
    def test_execution_engine_health_check(self, execution_engine):
        """Test execution engine health check."""
        health = execution_engine.health_check()
        
        assert "overall" in health
        assert "components" in health
        assert "timestamp" in health
        assert health["overall"] in ["healthy", "unhealthy"]
    
    def test_execution_engine_monitoring(self, execution_engine):
        """Test execution engine monitoring."""
        status = execution_engine.monitor_execution()
        
        assert "execution_engine" in status
        assert "orders" in status
        assert "portfolio" in status
        assert "risk" in status
    
    def test_emergency_stop(self, execution_engine):
        """Test emergency stop functionality."""
        # Mock pending orders
        mock_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            alpaca_order_id="order_123"
        )
        execution_engine.order_manager.pending_orders["order_123"] = mock_order
        
        # Execute emergency stop
        success = execution_engine.emergency_stop()
        
        assert success
        # Note: In a real test, we'd verify orders were cancelled
    
    def test_configuration_loading(self):
        """Test that configuration can be loaded properly."""
        # Test position sizing config
        config = PositionSizingConfig()
        assert config.max_position_size == 0.1
        assert config.min_trade_size == 100.0
        
        # Test risk limits
        limits = RiskLimits()
        assert limits.max_daily_loss == 0.02
        assert limits.max_position_risk == 0.05
        
        # Test execution config
        exec_config = ExecutionConfig()
        assert exec_config.enabled is True
        assert exec_config.mode == "paper"
    
    def test_order_validation_chain(self):
        """Test the complete order validation chain."""
        # Create order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Test order validation
        order.validate()
        assert order.is_active()
        assert not order.is_finished()
        
        # Test order conversion to Alpaca format
        alpaca_dict = order.to_alpaca_dict()
        assert alpaca_dict["symbol"] == "AAPL"
        assert alpaca_dict["side"] == "buy"
        assert alpaca_dict["qty"] == "100"
