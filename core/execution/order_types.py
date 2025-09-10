"""
Order Types and Data Structures

Defines the core data structures for order management including order types,
sides, statuses, and the main Order class with validation logic.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order execution types supported by Alpaca."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status throughout its lifecycle."""
    PENDING = "pending"                    # Created but not submitted
    SUBMITTED = "submitted"               # Submitted to broker
    PARTIALLY_FILLED = "partially_filled" # Partially executed
    FILLED = "filled"                     # Fully executed
    CANCELLED = "cancelled"               # Cancelled by user
    REJECTED = "rejected"                 # Rejected by broker
    EXPIRED = "expired"                   # Expired (time in force)


@dataclass
class Order:
    """
    Represents a trading order with all necessary fields for execution.
    
    Attributes:
        symbol: Trading symbol (e.g., 'AAPL')
        side: Buy or sell
        quantity: Number of shares (positive integer)
        order_type: Market, limit, stop, or stop_limit
        limit_price: Price for limit orders (optional)
        stop_price: Price for stop orders (optional)
        time_in_force: Order duration ('day', 'gtc', 'ioc', 'fok')
        status: Current order status
        alpaca_order_id: Alpaca's order ID (set after submission)
        created_at: Order creation timestamp
        filled_at: Order fill timestamp (set when filled)
        filled_price: Average fill price
        filled_quantity: Number of shares filled
        metadata: Additional order metadata
    """
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    status: OrderStatus = OrderStatus.PENDING
    alpaca_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate order after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate order parameters and raise ValueError if invalid.
        
        Raises:
            ValueError: If order parameters are invalid
        """
        # Symbol validation
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Quantity validation
        if not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("Quantity must be a positive integer")
        
        # Price validation based on order type
        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("Limit orders require a positive limit_price")
        
        elif self.order_type == OrderType.STOP:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError("Stop orders require a positive stop_price")
        
        elif self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("Stop-limit orders require a positive limit_price")
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError("Stop-limit orders require a positive stop_price")
        
        # Time in force validation
        valid_tif = {"day", "gtc", "ioc", "fok"}
        if self.time_in_force not in valid_tif:
            raise ValueError(f"time_in_force must be one of {valid_tif}")
        
        # Filled quantity validation
        if self.filled_quantity < 0 or self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity must be between 0 and quantity")
        
        # Status validation
        if self.status == OrderStatus.FILLED and self.filled_quantity != self.quantity:
            raise ValueError("Filled orders must have filled_quantity equal to quantity")
    
    def is_active(self) -> bool:
        """Check if order is in an active state (can be cancelled)."""
        return self.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
    
    def is_finished(self) -> bool:
        """Check if order is in a finished state (no further action needed)."""
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
    
    def get_remaining_quantity(self) -> int:
        """Get remaining quantity to be filled."""
        return max(0, self.quantity - self.filled_quantity)
    
    def update_fill(self, fill_quantity: int, fill_price: float, fill_time: Optional[datetime] = None) -> None:
        """
        Update order with a fill.
        
        Args:
            fill_quantity: Number of shares filled
            fill_price: Price at which shares were filled
            fill_time: Time of fill (defaults to now)
        """
        if fill_quantity <= 0:
            raise ValueError("fill_quantity must be positive")
        
        if fill_price <= 0:
            raise ValueError("fill_price must be positive")
        
        if self.filled_quantity + fill_quantity > self.quantity:
            raise ValueError("fill_quantity would exceed order quantity")
        
        # Update filled quantity
        self.filled_quantity += fill_quantity
        
        # Update average fill price
        if self.filled_price is None:
            self.filled_price = fill_price
        else:
            # Calculate weighted average
            total_value = (self.filled_quantity - fill_quantity) * self.filled_price + fill_quantity * fill_price
            self.filled_price = total_value / self.filled_quantity
        
        # Update status
        if self.filled_quantity == self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = fill_time or datetime.now()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def to_alpaca_dict(self) -> Dict[str, Any]:
        """
        Convert order to Alpaca API format.
        
        Returns:
            Dictionary in Alpaca order format
        """
        order_dict = {
            "symbol": self.symbol,
            "qty": str(self.quantity),
            "side": self.side.value,
            "type": self.order_type.value,
            "time_in_force": self.time_in_force
        }
        
        if self.limit_price is not None:
            order_dict["limit_price"] = str(self.limit_price)
        
        if self.stop_price is not None:
            order_dict["stop_price"] = str(self.stop_price)
        
        return order_dict
    
    @classmethod
    def from_alpaca_response(cls, alpaca_order: Dict[str, Any]) -> 'Order':
        """
        Create Order from Alpaca API response.
        
        Args:
            alpaca_order: Order data from Alpaca API
            
        Returns:
            Order instance
        """
        # Parse basic fields
        symbol = alpaca_order["symbol"]
        side = OrderSide(alpaca_order["side"])
        # Handle both qty and quantity fields, with fallback to 0
        quantity = 0
        if "qty" in alpaca_order and alpaca_order["qty"] is not None:
            quantity = int(alpaca_order["qty"])
        elif "quantity" in alpaca_order and alpaca_order["quantity"] is not None:
            quantity = int(alpaca_order["quantity"])
        order_type = OrderType(alpaca_order["order_type"])
        
        # Parse optional fields
        limit_price = None
        if "limit_price" in alpaca_order and alpaca_order["limit_price"]:
            limit_price = float(alpaca_order["limit_price"])
        
        stop_price = None
        if "stop_price" in alpaca_order and alpaca_order["stop_price"]:
            stop_price = float(alpaca_order["stop_price"])
        
        time_in_force = alpaca_order.get("time_in_force", "day")
        
        # Parse status
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED
        }
        status = status_map.get(alpaca_order["status"], OrderStatus.PENDING)
        
        # Parse fill information
        filled_quantity = int(alpaca_order.get("filled_qty", "0"))
        filled_price = None
        if "filled_avg_price" in alpaca_order and alpaca_order["filled_avg_price"]:
            filled_price = float(alpaca_order["filled_avg_price"])
        
        filled_at = None
        if "filled_at" in alpaca_order and alpaca_order["filled_at"]:
            filled_at = datetime.fromisoformat(alpaca_order["filled_at"].replace("Z", "+00:00"))
        
        # Parse timestamps
        created_at = datetime.fromisoformat(alpaca_order["created_at"].replace("Z", "+00:00"))
        
        return cls(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=status,
            alpaca_order_id=alpaca_order["id"],
            created_at=created_at,
            filled_at=filled_at,
            filled_price=filled_price,
            filled_quantity=filled_quantity
        )
    
    def __str__(self) -> str:
        """String representation of order."""
        return (f"Order({self.symbol} {self.side.value} {self.quantity} {self.order_type.value} "
                f"@ {self.limit_price or 'MKT'} - {self.status.value})")
    
    def __repr__(self) -> str:
        """Detailed string representation of order."""
        return (f"Order(symbol='{self.symbol}', side={self.side.value}, quantity={self.quantity}, "
                f"order_type={self.order_type.value}, limit_price={self.limit_price}, "
                f"stop_price={self.stop_price}, status={self.status.value}, "
                f"alpaca_order_id='{self.alpaca_order_id}')")
