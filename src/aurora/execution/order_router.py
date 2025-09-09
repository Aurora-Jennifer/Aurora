"""
Order Router - Idempotent order execution with retry logic

Handles order routing with:
- Idempotent retries
- Timeout handling
- Position reconciliation
- Execution quality tracking
"""

import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


@dataclass
class OrderResponse:
    """Response from order execution"""
    client_order_id: str
    broker_order_id: str | None
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    timestamp: float = 0.0
    error_message: str | None = None


@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    timeout_orders: int = 0
    total_slippage: float = 0.0
    total_commission: float = 0.0
    avg_fill_time: float = 0.0


class MockBroker:
    """Mock broker for testing - simulates realistic execution"""
    
    def __init__(self, 
                 avg_slippage_bps: float = 2.0,
                 commission_bps: float = 1.0,
                 timeout_rate: float = 0.01,
                 reject_rate: float = 0.005):
        self.avg_slippage_bps = avg_slippage_bps
        self.commission_bps = commission_bps
        self.timeout_rate = timeout_rate
        self.reject_rate = reject_rate
        self.order_book = {}  # client_order_id -> order details
        self.execution_log = []
    
    def send_order(self, order_payload: dict[str, Any]) -> OrderResponse:
        """Simulate order execution"""
        client_order_id = order_payload.get("client_order_id")
        symbol = order_payload.get("symbol")
        side = order_payload.get("side")
        quantity = order_payload.get("quantity")
        price = order_payload.get("price", 100.0)  # Default price
        
        # Simulate random failures
        import random
        if random.random() < self.reject_rate:
            return OrderResponse(
                client_order_id=client_order_id,
                broker_order_id=None,
                status=OrderStatus.REJECTED,
                error_message="Simulated rejection"
            )
        
        if random.random() < self.timeout_rate:
            return OrderResponse(
                client_order_id=client_order_id,
                broker_order_id=None,
                status=OrderStatus.TIMEOUT,
                error_message="Simulated timeout"
            )
        
        # Simulate execution
        broker_order_id = f"BRK_{uuid.uuid4().hex[:8]}"
        
        # Simulate slippage
        slippage_factor = 1.0 + (random.gauss(0, self.avg_slippage_bps / 10000))
        filled_price = price * slippage_factor
        
        # Simulate commission
        commission = abs(quantity * filled_price * self.commission_bps / 10000)
        
        response = OrderResponse(
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=filled_price,
            commission=commission,
            timestamp=time.time()
        )
        
        # Store in order book
        self.order_book[client_order_id] = {
            "payload": order_payload,
            "response": response,
            "timestamp": time.time()
        }
        
        self.execution_log.append(response)
        logger.info(f"Order executed: {client_order_id} -> {broker_order_id}")
        
        return response
    
    def get_position(self, symbol: str) -> float:
        """Get current position for symbol"""
        # Mock position tracking
        return 0.0  # In production, this would query the broker


class OrderRouter:
    """Idempotent order router with retry logic"""
    
    def __init__(self, 
                 broker,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: float = 30.0):
        self.broker = broker
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.execution_metrics = ExecutionMetrics()
        self.order_history = {}  # client_order_id -> execution details
    
    def send_order(self, 
                   symbol: str,
                   side: str,
                   quantity: float,
                   price: float = None,
                   order_type: str = "MARKET",
                   client_order_id: str = None) -> OrderResponse:
        """
        Send order with idempotent retry logic
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Order price (None for market orders)
            order_type: Order type
            client_order_id: Client order ID (auto-generated if None)
            
        Returns:
            OrderResponse with execution details
        """
        if client_order_id is None:
            client_order_id = f"CLT_{uuid.uuid4().hex[:12]}"
        
        # Check if we've already processed this order
        if client_order_id in self.order_history:
            logger.info(f"Order already processed: {client_order_id}")
            return self.order_history[client_order_id]["response"]
        
        # Create order payload
        order_payload = {
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "timestamp": time.time()
        }
        
        # Attempt execution with retries
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Sending order (attempt {attempt + 1}): {client_order_id}")
                
                response = self.broker.send_order(order_payload)
                
                # Update metrics
                self.execution_metrics.total_orders += 1
                
                if response.status == OrderStatus.FILLED:
                    self.execution_metrics.successful_orders += 1
                    self.execution_metrics.total_slippage += abs(response.filled_price - (price or 100.0))
                    self.execution_metrics.total_commission += response.commission
                elif response.status == OrderStatus.TIMEOUT:
                    self.execution_metrics.timeout_orders += 1
                else:
                    self.execution_metrics.failed_orders += 1
                
                # Store in history
                self.order_history[client_order_id] = {
                    "payload": order_payload,
                    "response": response,
                    "attempts": attempt + 1,
                    "timestamp": time.time()
                }
                
                # If successful or permanently failed, return
                if response.status in [OrderStatus.FILLED, OrderStatus.REJECTED]:
                    return response
                
                # If timeout, retry after delay
                if response.status == OrderStatus.TIMEOUT and attempt < self.max_retries:
                    logger.warning(f"Order timeout, retrying in {self.retry_delay}s: {client_order_id}")
                    time.sleep(self.retry_delay)
                    continue
                
                return response
                
            except Exception as e:
                logger.error(f"Order execution error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    # Final attempt failed
                    error_response = OrderResponse(
                        client_order_id=client_order_id,
                        broker_order_id=None,
                        status=OrderStatus.REJECTED,
                        error_message=str(e)
                    )
                    self.execution_metrics.failed_orders += 1
                    self.order_history[client_order_id] = {
                        "payload": order_payload,
                        "response": error_response,
                        "attempts": attempt + 1,
                        "timestamp": time.time()
                    }
                    return error_response
        
        # Should never reach here
        raise RuntimeError(f"Order execution failed after {self.max_retries} retries")
    
    def reconcile_positions(self, expected_positions: dict[str, float]) -> dict[str, Any]:
        """
        Reconcile expected vs actual positions
        
        Args:
            expected_positions: Expected positions by symbol
            
        Returns:
            Reconciliation report
        """
        reconciliation = {
            "timestamp": time.time(),
            "expected": expected_positions.copy(),
            "actual": {},
            "discrepancies": {},
            "alerts": []
        }
        
        for symbol in expected_positions:
            try:
                actual_position = self.broker.get_position(symbol)
                reconciliation["actual"][symbol] = actual_position
                
                expected = expected_positions[symbol]
                discrepancy = actual_position - expected
                
                if abs(discrepancy) > 0.001:  # Tolerance for rounding
                    reconciliation["discrepancies"][symbol] = discrepancy
                    reconciliation["alerts"].append(
                        f"Position discrepancy for {symbol}: "
                        f"expected {expected:.3f}, actual {actual_position:.3f}"
                    )
                    logger.warning(f"Position discrepancy: {symbol} "
                                 f"expected {expected:.3f}, actual {actual_position:.3f}")
                
            except Exception as e:
                reconciliation["alerts"].append(f"Failed to get position for {symbol}: {e}")
                logger.error(f"Position reconciliation error for {symbol}: {e}")
        
        return reconciliation
    
    def get_execution_metrics(self) -> dict[str, Any]:
        """Get execution quality metrics"""
        metrics = {
            "total_orders": self.execution_metrics.total_orders,
            "successful_orders": self.execution_metrics.successful_orders,
            "failed_orders": self.execution_metrics.failed_orders,
            "timeout_orders": self.execution_metrics.timeout_orders,
            "success_rate": (self.execution_metrics.successful_orders / 
                           max(self.execution_metrics.total_orders, 1)),
            "avg_slippage": (self.execution_metrics.total_slippage / 
                           max(self.execution_metrics.successful_orders, 1)),
            "total_commission": self.execution_metrics.total_commission,
            "order_history_size": len(self.order_history)
        }
        
        return metrics
    
    def get_order_status(self, client_order_id: str) -> OrderResponse | None:
        """Get status of a specific order"""
        if client_order_id in self.order_history:
            return self.order_history[client_order_id]["response"]
        return None
    
    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel a pending order"""
        # In production, this would call broker cancel API
        logger.info(f"Cancelling order: {client_order_id}")
        return True  # Mock success
