"""
Unit tests for order types and validation.
"""

import pytest
from datetime import datetime
from core.execution.order_types import Order, OrderType, OrderSide, OrderStatus


class TestOrderTypes:
    """Test order type enums."""
    
    def test_order_type_values(self):
        """Test order type enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
    
    def test_order_side_values(self):
        """Test order side enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_status_values(self):
        """Test order status enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"


class TestOrder:
    """Test Order class functionality."""
    
    def test_market_order_creation(self):
        """Test creating a market order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.limit_price is None
        assert order.stop_price is None
    
    def test_limit_order_creation(self):
        """Test creating a limit order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )
        
        assert order.limit_price == 150.0
        assert order.order_type == OrderType.LIMIT
    
    def test_stop_order_creation(self):
        """Test creating a stop order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=140.0
        )
        
        assert order.stop_price == 140.0
        assert order.order_type == OrderType.STOP
    
    def test_stop_limit_order_creation(self):
        """Test creating a stop-limit order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP_LIMIT,
            limit_price=139.0,
            stop_price=140.0
        )
        
        assert order.limit_price == 139.0
        assert order.stop_price == 140.0
        assert order.order_type == OrderType.STOP_LIMIT
    
    def test_order_validation_invalid_symbol(self):
        """Test order validation with invalid symbol."""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            Order(
                symbol="",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET
            )
    
    def test_order_validation_invalid_quantity(self):
        """Test order validation with invalid quantity."""
        with pytest.raises(ValueError, match="Quantity must be a positive integer"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=0,
                order_type=OrderType.MARKET
            )
    
    def test_order_validation_limit_price_required(self):
        """Test order validation requires limit price for limit orders."""
        with pytest.raises(ValueError, match="Limit orders require a positive limit_price"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT
            )
    
    def test_order_validation_stop_price_required(self):
        """Test order validation requires stop price for stop orders."""
        with pytest.raises(ValueError, match="Stop orders require a positive stop_price"):
            Order(
                symbol="AAPL",
                side=OrderSide.SELL,
                quantity=100,
                order_type=OrderType.STOP
            )
    
    def test_order_validation_invalid_time_in_force(self):
        """Test order validation with invalid time in force."""
        with pytest.raises(ValueError, match="time_in_force must be one of"):
            Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
                time_in_force="invalid"
            )
    
    def test_order_is_active(self):
        """Test order active status check."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        assert order.is_active()
        
        order.status = OrderStatus.SUBMITTED
        assert order.is_active()
        
        order.status = OrderStatus.FILLED
        assert not order.is_active()
    
    def test_order_is_finished(self):
        """Test order finished status check."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        assert not order.is_finished()
        
        order.status = OrderStatus.FILLED
        assert order.is_finished()
        
        order.status = OrderStatus.CANCELLED
        assert order.is_finished()
    
    def test_get_remaining_quantity(self):
        """Test getting remaining quantity."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        assert order.get_remaining_quantity() == 100
        
        order.filled_quantity = 30
        assert order.get_remaining_quantity() == 70
        
        order.filled_quantity = 100
        assert order.get_remaining_quantity() == 0
    
    def test_update_fill(self):
        """Test updating order with fill."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        # Partial fill
        order.update_fill(30, 150.0)
        assert order.filled_quantity == 30
        assert order.filled_price == 150.0
        assert order.status == OrderStatus.PARTIALLY_FILLED
        
        # Complete fill
        order.update_fill(70, 151.0)
        assert order.filled_quantity == 100
        assert order.filled_price == 150.7  # Weighted average
        assert order.status == OrderStatus.FILLED
        assert order.filled_at is not None
    
    def test_update_fill_invalid(self):
        """Test updating fill with invalid data."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        with pytest.raises(ValueError, match="fill_quantity must be positive"):
            order.update_fill(0, 150.0)
        
        with pytest.raises(ValueError, match="fill_price must be positive"):
            order.update_fill(50, -150.0)
        
        with pytest.raises(ValueError, match="fill_quantity would exceed order quantity"):
            order.update_fill(150, 150.0)
    
    def test_to_alpaca_dict(self):
        """Test converting order to Alpaca format."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            time_in_force="day"
        )
        
        alpaca_dict = order.to_alpaca_dict()
        
        assert alpaca_dict["symbol"] == "AAPL"
        assert alpaca_dict["qty"] == "100"
        assert alpaca_dict["side"] == "buy"
        assert alpaca_dict["type"] == "limit"
        assert alpaca_dict["limit_price"] == "150.0"
        assert alpaca_dict["time_in_force"] == "day"
    
    def test_from_alpaca_response(self):
        """Test creating order from Alpaca response."""
        alpaca_response = {
            "id": "order_123",
            "symbol": "AAPL",
            "qty": "100",
            "side": "buy",
            "order_type": "market",
            "status": "filled",
            "time_in_force": "day",
            "created_at": "2024-01-01T10:00:00Z",
            "filled_at": "2024-01-01T10:01:00Z",
            "filled_qty": "100",
            "filled_avg_price": "150.50"
        }
        
        order = Order.from_alpaca_response(alpaca_response)
        
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.FILLED
        assert order.alpaca_order_id == "order_123"
        assert order.filled_quantity == 100
        assert order.filled_price == 150.50
    
    def test_string_representation(self):
        """Test string representation of order."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )
        
        str_repr = str(order)
        assert "AAPL" in str_repr
        assert "buy" in str_repr
        assert "100" in str_repr
        assert "market" in str_repr
