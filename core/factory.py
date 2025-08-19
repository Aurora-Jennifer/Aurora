"""
Core Factory for MVB Components
Wires components based on mode (backtest/shadow/paper)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.risk.guardrails import RiskGuardrails


class Clock:
    """Abstract clock interface"""

    def now(self) -> datetime:
        raise NotImplementedError

    def is_market_open(self) -> bool:
        raise NotImplementedError


class SimClock(Clock):
    """Simulation clock for backtesting"""

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date

    def now(self) -> datetime:
        return self.current_date

    def is_market_open(self) -> bool:
        # Simple business day check
        return self.current_date.weekday() < 5

    def advance(self, days: int = 1):
        """Advance the clock"""
        self.current_date += timedelta(days=days)


class RealClock(Clock):
    """Real-time clock for live trading"""

    def now(self) -> datetime:
        return datetime.now()

    def is_market_open(self) -> bool:
        # Simple market hours check (9:30 AM - 4:00 PM ET)
        now = self.now()
        if now.weekday() >= 5:  # Weekend
            return False

        # Convert to ET (simplified)
        hour = now.hour
        return 9 <= hour < 16


class DataFeed:
    """Abstract data feed interface"""

    def get_latest_data(self, symbols: list) -> dict[str, Any]:
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError


class CsvParquetFeed(DataFeed):
    """File-based data feed for backtesting"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def get_latest_data(self, symbols: list) -> dict[str, Any]:
        # Simplified - in real implementation would read from files
        data = {}
        for symbol in symbols:
            data[symbol] = {
                "price": 100.0,  # Placeholder
                "volume": 1000000,
                "timestamp": datetime.now(),
            }
        return data

    def is_connected(self) -> bool:
        return self.data_dir.exists()


class IBMarketDataFeed(DataFeed):
    """Interactive Brokers market data feed"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connected = False

    def get_latest_data(self, symbols: list) -> dict[str, Any]:
        # Simplified - would use IBKR API
        data = {}
        for symbol in symbols:
            data[symbol] = {
                "price": 100.0,  # Placeholder
                "volume": 1000000,
                "timestamp": datetime.now(),
            }
        return data

    def is_connected(self) -> bool:
        return self.connected


class Broker:
    """Abstract broker interface"""

    def submit_order(self, order: dict[str, Any]) -> str:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def get_positions(self) -> dict[str, int]:
        raise NotImplementedError


class SimBroker(Broker):
    """Simulation broker for backtesting/shadow mode"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.orders = {}
        self.order_id_counter = 0
        self.logger = logging.getLogger(__name__)

    def submit_order(self, order: dict[str, Any]) -> str:
        order_id = f"sim_{self.order_id_counter}"
        self.order_id_counter += 1

        # Simulate order execution
        symbol = order["symbol"]
        quantity = order["quantity"]
        price = order.get("price", 100.0)

        if order["side"] == "BUY":
            cost = quantity * price
            if cost <= self.cash:
                self.cash -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                self.logger.info(f"Simulated BUY {quantity} {symbol} @ ${price}")
            else:
                self.logger.warning(f"Insufficient cash for order: {cost} > {self.cash}")
        else:  # SELL
            if symbol in self.positions and self.positions[symbol] >= quantity:
                self.positions[symbol] -= quantity
                self.cash += quantity * price
                self.logger.info(f"Simulated SELL {quantity} {symbol} @ ${price}")
            else:
                self.logger.warning(f"Insufficient shares for sell: {symbol}")

        self.orders[order_id] = {
            "status": "FILLED",
            "filled_quantity": quantity,
            "filled_price": price,
        }

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id]["status"] = "CANCELLED"
            return True
        return False

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return self.orders.get(order_id, {"status": "NOT_FOUND"})

    def get_positions(self) -> dict[str, int]:
        return dict(self.positions)


class IBBroker(Broker):
    """Interactive Brokers broker for paper/live trading"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connected = False

    def submit_order(self, order: dict[str, Any]) -> str:
        # Simplified - would use IBKR API
        order_id = f"ib_{datetime.now().timestamp()}"
        self.logger.info(f"IBKR order submitted: {order}")
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        self.logger.info(f"IBKR order cancelled: {order_id}")
        return True

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        # Simplified - would query IBKR
        return {"status": "SUBMITTED"}

    def get_positions(self) -> dict[str, int]:
        # Simplified - would query IBKR
        return {}


class Portfolio:
    """Portfolio accounting and reconciliation"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.logger = logging.getLogger(__name__)

    def update_position(self, symbol: str, quantity: int, price: float, side: str):
        """Update position after trade"""
        if side == "BUY":
            if symbol not in self.positions:
                self.positions[symbol] = 0
            self.positions[symbol] += quantity
            self.cash -= quantity * price
        else:  # SELL
            if symbol in self.positions:
                self.positions[symbol] -= quantity
                self.cash += quantity * price
            else:
                self.logger.error(f"Attempting to sell {symbol} without position")

        # Record trade
        self.trades.append(
            {
                "timestamp": datetime.now(),
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "side": side,
            }
        )

    def get_total_value(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total = self.cash
        for symbol, quantity in self.positions.items():
            if symbol in prices:
                total += quantity * prices[symbol]
        return total

    def reconcile(self) -> bool:
        """Reconcile portfolio accounting"""
        # Simple reconciliation check
        expected_cash = self.initial_capital
        for trade in self.trades:
            if trade["side"] == "BUY":
                expected_cash -= trade["quantity"] * trade["price"]
            else:
                expected_cash += trade["quantity"] * trade["price"]

        tolerance = 1e-6
        if abs(self.cash - expected_cash) > tolerance:
            self.logger.error(f"Reconciliation failed: {self.cash} != {expected_cash}")
            return False

        return True


class Telemetry:
    """Telemetry and event logging"""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def log_event(self, event_type: str, data: dict[str, Any]):
        """Log an event to NDJSON file"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
        }

        with open(self.log_file, "a") as f:
            f.write(f"{json.dumps(event)}\n")

        self.logger.info(f"Logged {event_type}: {data}")


def make_mvb_components(mode: str, config: dict[str, Any]) -> dict[str, Any]:
    """
    Factory function to create MVB components based on mode.

    Args:
        mode: "backtest", "shadow", or "paper"
        config: Configuration dictionary

    Returns:
        Dictionary of initialized components
    """
    components = {}

    # Create clock based on mode
    if mode == "backtest":
        start_date = datetime.strptime(config.get("start_date", "2024-01-01"), "%Y-%m-%d")
        end_date = datetime.strptime(config.get("end_date", "2024-12-31"), "%Y-%m-%d")
        components["clock"] = SimClock(start_date, end_date)
    else:
        components["clock"] = RealClock()

    # Create data feed based on mode
    if mode == "backtest":
        components["feed"] = CsvParquetFeed(config.get("data_dir", "data"))
    else:
        components["feed"] = IBMarketDataFeed(config.get("ibkr_config", {}))

    # Create broker based on mode
    if mode in ["backtest", "shadow"]:
        components["broker"] = SimBroker(config.get("initial_capital", 100000))
    else:
        components["broker"] = IBBroker(config.get("ibkr_config", {}))

    # Create portfolio
    components["portfolio"] = Portfolio(config.get("initial_capital", 100000))

    # Create risk guardrails
    components["risk"] = RiskGuardrails(config)

    # Create telemetry
    log_file = config.get(
        "log_file", f"results/{mode}_{datetime.now().strftime('%Y%m%d_%H%M')}.ndjson"
    )
    components["telemetry"] = Telemetry(log_file)

    # Create trading engine (simplified for MVB)
    # We'll use the MVB runner instead of the full engines
    pass

    return components
