from typing import Dict, Any
from datetime import datetime, timezone
from brokers.interface import Broker


class MockBroker(Broker):
    """Mock broker for testing the interface."""
    
    def submit_order(self, symbol: str, side: str, qty: float, order_type: str = "market") -> Dict[str, Any]:
        return {
            "order_id": "mock_123",
            "status": "submitted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty
        }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return {
            "order_id": order_id,
            "status": "cancelled",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_positions(self) -> Dict[str, float]:
        return {"SPY": 100.0, "TSLA": -50.0}
    
    def get_cash(self) -> float:
        return 100000.0
    
    def get_fills(self, since: datetime | None = None) -> list[Dict[str, Any]]:
        return [
            {
                "symbol": "SPY",
                "side": "BUY",
                "qty": 100.0,
                "price": 450.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": "mock_123"
            }
        ]
    
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def test_broker_interface():
    """Test that the broker interface works correctly."""
    broker = MockBroker()
    
    # Test submit_order
    order = broker.submit_order("SPY", "BUY", 100.0)
    assert order["order_id"] == "mock_123"
    assert order["symbol"] == "SPY"
    assert order["side"] == "BUY"
    
    # Test cancel_order
    cancel = broker.cancel_order("mock_123")
    assert cancel["status"] == "cancelled"
    
    # Test get_positions
    positions = broker.get_positions()
    assert positions["SPY"] == 100.0
    assert positions["TSLA"] == -50.0
    
    # Test get_cash
    cash = broker.get_cash()
    assert cash == 100000.0
    
    # Test get_fills
    fills = broker.get_fills()
    assert len(fills) == 1
    assert fills[0]["symbol"] == "SPY"
    
    # Test now
    now = broker.now()
    assert isinstance(now, datetime)
