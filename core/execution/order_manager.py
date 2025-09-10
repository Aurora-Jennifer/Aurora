"""
Order Manager

Handles order submission, tracking, and reconciliation with Alpaca API.
Provides order lifecycle management and status monitoring.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import asdict
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, OrderType as AlpacaOrderType, TimeInForce, QueryOrderStatus
from alpaca.common.exceptions import APIError

from .order_types import Order, OrderType, OrderSide, OrderStatus
from .pretrade_gate import PreTradeGate, GateResult


def _og(o, name, default=None):
    """Safe getattr or dict get."""
    if hasattr(o, name):
        return getattr(o, name)
    if isinstance(o, dict):
        return o.get(name, default)
    return default


def normalize_order(o):
    """Normalize Alpaca order object/dict to consistent format."""
    # Alpaca v2/v3/obj/dict/partial all supported
    oid = _og(o, "id") or _og(o, "client_order_id")
    sym = _og(o, "symbol")
    side = _og(o, "side")
    stat = _og(o, "status")

    # qty may be None for notional orders
    sub_qty = _og(o, "qty")
    # some SDKs expose strings
    try:
        sub_qty = int(sub_qty) if sub_qty is not None else None
    except Exception:
        sub_qty = None

    sub_notional = _og(o, "notional")
    try:
        sub_notional = float(sub_notional) if sub_notional is not None else None
    except Exception:
        sub_notional = None

    filled_qty = _og(o, "filled_qty", 0)
    try:
        filled_qty = int(filled_qty or 0)
    except Exception:
        filled_qty = 0

    avg_px = _og(o, "filled_avg_price", 0.0)
    try:
        avg_px = float(avg_px or 0.0)
    except Exception:
        avg_px = 0.0

    return {
        "id": oid,
        "symbol": sym,
        "side": side,
        "status": stat,
        "submitted_qty": sub_qty,             # can be None
        "submitted_notional": sub_notional,   # can be None
        "filled_qty": filled_qty,
        "filled_notional": filled_qty * avg_px,
        "avg_fill_price": avg_px,
    }

logger = logging.getLogger(__name__)

# Order status buckets
ACK_SET = {"new", "accepted", "partially_filled"}
PENDING_SET = {"pending_new", "pending_cancel", "pending_replace"}
TERMINAL_SET = {"filled", "canceled", "expired", "rejected", "stopped", "replaced"}


def get_bp_snapshot(client) -> Dict[str, Any]:
    """Get broker account snapshot for buying power analysis."""
    try:
        account = client.get_account()
        
        def safe_decimal(x, default="0"):
            try:
                return Decimal(str(x or default))
            except:
                return Decimal(default)
        
        return {
            "bp": safe_decimal(getattr(account, "buying_power", "0")),
            "dtbp": safe_decimal(getattr(account, "daytrading_buying_power", getattr(account, "buying_power", "0"))),
            "cash": safe_decimal(getattr(account, "cash", "0")),
            "equity": safe_decimal(getattr(account, "equity", "0")),
            "mult": int(getattr(account, "multiplier", "1") or 1),
            "shorting": bool(getattr(account, "shorting_enabled", False)),
            "blocked": bool(getattr(account, "trading_blocked", False)),
        }
    except Exception as e:
        logger.error(f"Failed to get account snapshot: {e}")
        return {
            "bp": Decimal("0"),
            "dtbp": Decimal("0"),
            "cash": Decimal("0"),
            "equity": Decimal("0"),
            "mult": 1,
            "shorting": False,
            "blocked": True,
        }


def broker_bp_allows(order_notional: float, side: str, current_pos: int, snap: Dict[str, Any], buffer: float = 1.03) -> tuple[bool, str]:
    """
    Check if broker buying power allows the order.
    
    Args:
        order_notional: Order notional value
        side: Order side ('buy' or 'sell')
        current_pos: Current position quantity
        snap: Account snapshot from get_bp_snapshot()
        
    Returns:
        Tuple of (allowed, reason)
    """
    need = Decimal(str(order_notional)) * Decimal(str(buffer))
    avail = max(snap["bp"], snap["dtbp"])
    
    # Check for trading blocks
    if snap["blocked"]:
        return False, "trading_blocked"
    
    # Check shorting requirements
    if side == "sell" and current_pos == 0:
        # Short to open: require margin and shorting enabled
        if not snap["shorting"]:
            return False, "shorting_disabled"
        if snap["mult"] <= 1:
            return False, "cash_only_no_margin"
    
    # Check buying power
    if need > avail:
        return False, f"need={need}, avail={avail}"
    
    return True, "ok"


# Two-phase batching constants
HAIRCUT = Decimal("0.98")
CASH_BUFFER = Decimal("500")


def split_delta(current_sh: int, delta_sh: int) -> tuple[int, int]:
    """
    Split a cross-zero delta into reduce-to-flat and open phases.
    
    Args:
        current_sh: Current position quantity (signed)
        delta_sh: Order delta quantity (signed)
        
    Returns:
        Tuple of (reduce_sh, open_sh) - both signed
    """
    target_sh = current_sh + delta_sh
    
    # Reducer leg to flat
    reduce_sh = 0
    if current_sh > 0 and target_sh < 0:   # long → short
        reduce_sh = -current_sh            # sell current to 0
    elif current_sh < 0 and target_sh > 0: # short → long
        reduce_sh = -current_sh            # buy to 0
    else:
        # same side or shrinking → reducer is min(|delta|, |current|)
        if (current_sh > 0 and delta_sh < 0) or (current_sh < 0 and delta_sh > 0):
            reduce_sh = max(min(delta_sh, -current_sh), -abs(current_sh))  # keep sign
        else:
            reduce_sh = 0

    open_sh = delta_sh - reduce_sh  # whatever remains after flattening
    return reduce_sh, open_sh

def plan_batches(current_positions: Dict[str, int], target_shares: Dict[str, int], ref_prices: Dict[str, float]) -> tuple[List[Dict], List[Dict]]:
    """
    Plan two-phase order batching: reducers first, then openers.
    
    Args:
        current_positions: Current position quantities {symbol: qty}
        target_shares: Target position quantities {symbol: qty}
        ref_prices: Reference prices {symbol: price}
        
    Returns:
        Tuple of (reducers, openers) order plans
    """
    reducers, openers = [], []
    
    for symbol, target_qty in target_shares.items():
        current_qty = current_positions.get(symbol, 0)
        delta = target_qty - current_qty
        
        if delta == 0:
            continue
            
        ref_price = ref_prices.get(symbol, 0.0)
        if ref_price <= 0:
            continue
        
        # Split cross-zero deltas into reduce-to-flat and open phases
        reduce_delta, open_delta = split_delta(current_qty, delta)
        
        # Create reducer order if needed
        if reduce_delta != 0:
            reduce_notional = abs(Decimal(reduce_delta) * Decimal(str(ref_price)))
            reduce_side = "buy" if reduce_delta > 0 else "sell"
            
            # Sanity guard: reducer cannot exceed current position
            assert not (reduce_side == "sell" and abs(reduce_delta) > abs(current_qty)), \
                f"Reducer cannot exceed current position: {symbol} reduce={reduce_delta} current={current_qty}"
            
            reducer_plan = {
                "symbol": symbol,
                "qty": abs(reduce_delta),
                "side": reduce_side,
                "notional": float(reduce_notional),
                "current_qty": current_qty,
                "target_qty": current_qty + reduce_delta,  # After reducer
                "ref_price": ref_price,
                "intent": "reduce",
                "leg": "reduce_to_flat"
            }
            reducers.append(reducer_plan)
        
        # Create opener order if needed
        if open_delta != 0:
            open_notional = abs(Decimal(open_delta) * Decimal(str(ref_price)))
            open_side = "buy" if open_delta > 0 else "sell"
            
            opener_plan = {
                "symbol": symbol,
                "qty": abs(open_delta),
                "side": open_side,
                "notional": float(open_notional),
                "current_qty": current_qty + reduce_delta,  # After reducer
                "target_qty": target_qty,
                "ref_price": ref_price,
                "intent": "open",
                "leg": "open_position"
            }
            openers.append(opener_plan)
    
    # Sort reducers by notional (biggest first to free most cash)
    reducers.sort(key=lambda o: o["notional"], reverse=True)
    
    # Sort openers by notional (biggest first by priority)
    openers.sort(key=lambda o: o["notional"], reverse=True)
    
    return reducers, openers


def estimate_proceeds(reducers: List[Dict]) -> Decimal:
    """Estimate cash proceeds from reducer orders."""
    proceeds = Decimal("0")
    for order in reducers:
        # Only sells of existing longs add cash in a cash account
        if order["side"] == "sell" and order["current_qty"] > 0:
            proceeds += Decimal(str(order["notional"]))
    return proceeds


class OrderManager:
    """
    Manages order submission, tracking, and reconciliation with Alpaca.
    
    Responsibilities:
    - Submit orders to Alpaca API
    - Track order status and updates
    - Handle order cancellations and modifications
    - Reconcile local state with Alpaca
    - Provide order history and reporting
    """
    
    def __init__(self, alpaca_client: TradingClient, config: Optional[Dict] = None):
        """
        Initialize Order Manager.
        
        Args:
            alpaca_client: Alpaca TradingClient instance
            config: Configuration dictionary
        """
        # Constructor instrumentation
        try:
            from core.utils.constructor_guard import construct_once
            construct_once("OrderManager")
        except ImportError:
            pass  # Skip instrumentation if not available
        
        self.alpaca_client = alpaca_client
        self.config = config or {}
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}  # alpaca_order_id -> Order
        self.filled_orders: Dict[str, Order] = {}   # alpaca_order_id -> Order
        self.cancelled_orders: Dict[str, Order] = {} # alpaca_order_id -> Order
        self.inflight_orders: Set[str] = set()  # Orders awaiting ACK
        
        # Initialize pre-trade gate
        # Self.config should contain risk and order settings; pass clients for richer checks
        self.pretrade_gate = PreTradeGate(self.config, trading_client=self.alpaca_client, data_client=self.config.get('data_client'))
        
        # Telemetry counters
        self.metrics = {
            'gate_reject_small': 0,
            'gate_reject_price_unknown': 0
        }
        
        # Configuration
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.reconciliation_interval = self.config.get('reconciliation_interval', 30)
        self.last_reconciliation = datetime.now()
        
        logger.info("OrderManager initialized")
    
    def wait_for_ack_or_terminal(self, order_id: str, timeout_s: float = 2.0, backoff_s: float = 0.05) -> str:
        """
        Wait for an order to transition from pending to ACK or terminal state.
        
        Args:
            order_id: Alpaca order ID to check
            timeout_s: Maximum time to wait
            backoff_s: Sleep between checks
            
        Returns:
            Final status
        """
        import time
        t0 = time.monotonic()
        last_status = None
        
        while time.monotonic() - t0 < timeout_s:
            try:
                order = self.alpaca_client.get_order_by_id(order_id)
                status = getattr(order, 'status', 'unknown')
                if hasattr(status, 'value'):
                    status = status.value
                status = str(status).lower()
                last_status = status
                
                if status in ACK_SET | TERMINAL_SET:
                    return status
                    
            except Exception as e:
                logger.debug(f"Error checking order {order_id}: {e}")
                
            time.sleep(backoff_s)
            
        return last_status or "unknown"
    
    def has_open_or_pending(self, symbol: str, side: str) -> bool:
        """
        Check if there's already an open or pending order for the given symbol and side.
        
        Args:
            symbol: Symbol to check
            side: Side to check ('buy' or 'sell')
            
        Returns:
            True if there's an open or pending order
        """
        # Check pending orders
        for order in self.pending_orders.values():
            if order.symbol == symbol and order.side.value.lower() == side.lower():
                return True
        
        # Check in-flight orders (these are pending ACK)
        for order_id in self.inflight_orders:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                if order.symbol == symbol and order.side.value.lower() == side.lower():
                    return True
        
        return False
    
    def get_account_snapshot(self) -> Dict[str, Any]:
        """Get current account snapshot for debugging."""
        return get_bp_snapshot(self.alpaca_client)
    
    def execute_two_phase_cycle(self, reducers: List[Dict], openers: List[Dict], metrics: Dict[str, Any]) -> Dict[str, int]:
        """
        Execute two-phase order cycle: reducers first, then openers within budget.
        
        Args:
            reducers: List of reducer order plans
            openers: List of opener order plans
            metrics: Metrics dictionary to update
            
        Returns:
            Dictionary with execution statistics
        """
        stats = {"reducers_submitted": 0, "openers_submitted": 0, "openers_skipped": 0}
        
        # Get initial account snapshot
        snap = get_bp_snapshot(self.alpaca_client)
        if snap["blocked"]:
            metrics["risk_skips"]["trading_blocked"] = metrics["risk_skips"].get("trading_blocked", 0) + 1
            logger.warning("Trading blocked, skipping two-phase cycle")
            return stats
        
        # Phase A: Submit reducers (partitioned: SELLs first, then BUYs)
        reducers_sell = [o for o in reducers if o["side"] == "sell"]
        reducers_buy = [o for o in reducers if o["side"] == "buy"]
        
        logger.info(f"Phase A1: Submitting {len(reducers_sell)} SELL reducers (free cash)")
        for order_plan in reducers_sell:
            try:
                # Create order from plan
                order = Order(
                    symbol=order_plan["symbol"],
                    side=OrderSide.SELL,
                    quantity=order_plan["qty"],
                    order_type=OrderType.MARKET,
                    limit_price=order_plan["ref_price"]  # Use reference price for pre-trade gate
                )
                
                # Submit with gate
                order_id, broker_status = self.submit_order_with_gate(
                    order=order,
                    signal_id=f"reducer_{order_plan['symbol']}_{order_plan['side']}",
                    current_position=order_plan["current_qty"]
                )
                
                if order_id:
                    stats["reducers_submitted"] += 1
                    logger.info(f"SELL reducer submitted: {order_plan['symbol']} {order_plan['side']} {order_plan['qty']}")
                else:
                    logger.warning(f"SELL reducer rejected: {order_plan['symbol']} {order_plan['side']} {order_plan['qty']}")
                    
            except Exception as e:
                logger.error(f"Failed to submit SELL reducer {order_plan['symbol']}: {e}")
        
        # Quick reconcile to update cash/BP after SELLs
        if len(reducers_sell) > 0:
            logger.info("Reconciling after SELL reducers...")
            self.reconcile_orders()
        
        # Phase A2: Submit BUY reducers (cover shorts) now that BP is freed
        logger.info(f"Phase A2: Submitting {len(reducers_buy)} BUY reducers (cover shorts)")
        for order_plan in reducers_buy:
            try:
                # Create order from plan
                order = Order(
                    symbol=order_plan["symbol"],
                    side=OrderSide.BUY,
                    quantity=order_plan["qty"],
                    order_type=OrderType.MARKET,
                    limit_price=order_plan["ref_price"]  # Use reference price for pre-trade gate
                )
                
                # Submit with gate
                order_id, broker_status = self.submit_order_with_gate(
                    order=order,
                    signal_id=f"reducer_{order_plan['symbol']}_{order_plan['side']}",
                    current_position=order_plan["current_qty"]
                )
                
                if order_id:
                    stats["reducers_submitted"] += 1
                    logger.info(f"BUY reducer submitted: {order_plan['symbol']} {order_plan['side']} {order_plan['qty']}")
                else:
                    logger.warning(f"BUY reducer rejected: {order_plan['symbol']} {order_plan['side']} {order_plan['qty']}")
                    
            except Exception as e:
                logger.error(f"Failed to submit BUY reducer {order_plan['symbol']}: {e}")
        
        # Final reconcile after all reducers
        if stats["reducers_submitted"] > 0:
            logger.info("Reconciling after all reducers...")
            self.reconcile_orders()
        
        # Phase B: Submit openers within budget
        snap2 = get_bp_snapshot(self.alpaca_client)
        proceeds_est = estimate_proceeds(reducers) * HAIRCUT
        budget = max(snap2["cash"], snap2["bp"], snap2["dtbp"]) + proceeds_est - CASH_BUFFER
        if budget < 0:
            budget = Decimal("0")
        
        logger.info(
            f"Phase B: Budget=${budget} (cash=${snap2['cash']}, proceeds_est=${proceeds_est}, buffer=${CASH_BUFFER})"
        )
        
        spent = Decimal("0")
        for order_plan in openers:
            need = Decimal(str(order_plan["notional"]))
            if spent + need > budget:
                metrics["risk_skips"]["insufficient_funding"] = metrics["risk_skips"].get("insufficient_funding", 0) + 1
                stats["openers_skipped"] += 1
                logger.info(f"Opener skipped (insufficient funding): {order_plan['symbol']} need=${need}, budget=${budget - spent}")
                continue
            
            try:
                # Create order from plan
                order = Order(
                    symbol=order_plan["symbol"],
                    side=OrderSide.BUY if order_plan["side"] == "buy" else OrderSide.SELL,
                    quantity=order_plan["qty"],
                    order_type=OrderType.MARKET,
                    limit_price=order_plan["ref_price"]  # Use reference price for pre-trade gate
                )
                
                # Submit with gate
                order_id, broker_status = self.submit_order_with_gate(
                    order=order,
                    signal_id=f"opener_{order_plan['symbol']}_{order_plan['side']}",
                    current_position=order_plan["current_qty"]
                )
                
                if order_id:
                    stats["openers_submitted"] += 1
                    spent += need
                    logger.info(f"Opener submitted: {order_plan['symbol']} {order_plan['side']} {order_plan['qty']} (spent=${spent})")
                else:
                    logger.warning(f"Opener rejected: {order_plan['symbol']} {order_plan['side']} {order_plan['qty']}")
                    
            except Exception as e:
                logger.error(f"Failed to submit opener {order_plan['symbol']}: {e}")
        
        logger.info(
            f"Two-phase cycle complete: reducers={stats['reducers_submitted']}, "
            f"openers={stats['openers_submitted']}, skipped={stats['openers_skipped']}"
        )
        
        return stats
    
    def submit_order_with_gate(
        self,
        order: Order,
        signal_id: str,
        current_position: int = 0,
        signal_timestamp: Optional[datetime] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Submit order with comprehensive pre-trade safety checks.
        
        Args:
            order: Order to submit
            signal_id: Unique signal identifier
            current_position: Current position quantity
            signal_timestamp: When the signal was generated
            
        Returns:
            Alpaca order ID if approved and submitted, None if rejected
        """
        # Run pre-trade gate
        decision = self.pretrade_gate.check_order(
            signal_id=signal_id,
            symbol=order.symbol,
            side=order.side.value.lower(),
            qty_target=order.quantity,  # Pass actual order quantity, not target position
            px_ref=order.limit_price or 0.0,
            current_position=current_position,
            broker_client=self.alpaca_client,
            signal_timestamp=signal_timestamp
        )
        
        # Log the decision
        self.pretrade_gate.log_decision(decision, signal_id, order.symbol)
        
        if decision.result == GateResult.REJECT:
            logger.warning(f"Order rejected by pre-trade gate: {decision.reason}")
            # Track reject reasons for telemetry
            if decision.reason == "TOO_SMALL":
                self.metrics['gate_reject_small'] = self.metrics.get('gate_reject_small', 0) + 1
            elif decision.reason == "PRICE_UNKNOWN":
                self.metrics['gate_reject_price_unknown'] = self.metrics.get('gate_reject_price_unknown', 0) + 1
            return None, None
        
        # Update order quantity if gate adjusted it
        if decision.shares is not None and decision.shares != order.quantity:
            logger.info(f"Gate adjusted order quantity: {order.quantity} -> {decision.shares}")
            order.quantity = abs(decision.shares)
            # Keep the original side - don't flip it based on shares sign
        
        # Submit the approved order
        order_id, broker_status = self.submit_order(order)
        return order_id, broker_status
    
    def submit_order(self, order: Order) -> tuple[str, str]:
        """
        Submit order to Alpaca and track it.
        
        Args:
            order: Order to submit
            
        Returns:
            Tuple of (Alpaca order ID, broker status)
            
        Raises:
            ValueError: If order is invalid
            APIError: If Alpaca API call fails
        """
        if not order.is_active():
            raise ValueError(f"Cannot submit order in status: {order.status}")
        
        # Pre-check buying power before submission (skip for reducer orders)
        # Note: We can't easily determine if this is a reducer here since we don't have current position
        # The two-phase system should handle BP checks at the planning level, not individual orders
        # For now, skip BP checks in submit_order since the gate and two-phase system handle this
        logger.debug(f"Submitting order to Alpaca: {order.symbol} {order.side.value} {order.quantity}")
        
        logger.info(f"Submitting order: {order}")
        
        try:
            # Convert to Alpaca order request
            alpaca_request = self._create_alpaca_request(order)
            
            # Submit to Alpaca
            alpaca_order = self.alpaca_client.submit_order(order_data=alpaca_request)
            
            # Update order with Alpaca ID
            order.alpaca_order_id = alpaca_order.id
            order.status = OrderStatus.SUBMITTED
            
            # Track the order
            self.pending_orders[alpaca_order.id] = order
            
            # Get broker status
            broker_status = getattr(alpaca_order, 'status', 'unknown')
            if hasattr(broker_status, 'value'):
                broker_status = broker_status.value
            broker_status = str(broker_status).lower()
            
            logger.info(f"Order submitted successfully: {alpaca_order.id} (status: {broker_status})")
            return alpaca_order.id, broker_status
            
        except APIError as e:
            logger.error(f"Failed to submit order: {e}")
            order.status = OrderStatus.REJECTED
            raise
        except Exception as e:
            logger.error(f"Unexpected error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: Alpaca order ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        if order_id not in self.pending_orders:
            logger.warning(f"Order {order_id} not found in pending orders")
            return False
        
        order = self.pending_orders[order_id]
        
        if not order.is_active():
            logger.warning(f"Cannot cancel order {order_id} in status: {order.status}")
            return False
        
        try:
            # Cancel on Alpaca
            self.alpaca_client.cancel_order_by_id(order_id)
            
            # Update local state
            order.status = OrderStatus.CANCELLED
            
            # Move to cancelled orders
            self.cancelled_orders[order_id] = order
            del self.pending_orders[order_id]
            
            logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except APIError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error cancelling order {order_id}: {e}")
            return False
    
    def cancel_open_orders(self, restrict_to_ids: Optional[Set[str]] = None) -> Dict[str, int]:
        """
        Cancel all open orders, optionally restricted to specific IDs.
        
        Args:
            restrict_to_ids: Set of order IDs to cancel (None = cancel all open)
            
        Returns:
            Dictionary with cancellation statistics
        """
        stats = {"cancelled": 0, "already_closed": 0, "errors": 0}
        
        try:
            # Get open orders from Alpaca
            open_orders = self.alpaca_client.get_orders(status="open")
            
            to_cancel = []
            for raw_order in open_orders:
                norm = normalize_order(raw_order)
                if norm["status"] in {"new", "accepted", "partially_filled"}:
                    if restrict_to_ids is None or norm["id"] in restrict_to_ids:
                        to_cancel.append(norm["id"])
            
            # Cancel each order
            for order_id in to_cancel:
                try:
                    self.alpaca_client.cancel_order_by_id(order_id)
                    stats["cancelled"] += 1
                    logger.info(f"Cancelled order {order_id}")
                except Exception as e:
                    msg = str(e)
                    if 'filled' in msg and '42210000' in msg:
                        stats["already_closed"] += 1
                        logger.info(f"Cancel benign: {order_id} already filled")
                    else:
                        stats["errors"] += 1
                        logger.warning(f"Cancel failed {order_id}: {msg}")
                        
        except Exception as e:
            logger.error(f"Error during bulk cancel: {e}")
            stats["errors"] += 1
            
        logger.info(f"Cancel open orders completed: {stats}")
        return stats
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get current order status from Alpaca.
        
        Args:
            order_id: Alpaca order ID
            
        Returns:
            Current order status or None if not found
        """
        try:
            alpaca_order = self.alpaca_client.get_order_by_id(order_id)
            
            # Map Alpaca status to our status
            status_map = {
                "new": OrderStatus.SUBMITTED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "filled": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
                "expired": OrderStatus.EXPIRED
            }
            
            return status_map.get(alpaca_order.status, OrderStatus.PENDING)
            
        except APIError as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting order status for {order_id}: {e}")
            return None
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID from local tracking.
        
        Args:
            order_id: Alpaca order ID
            
        Returns:
            Order instance or None if not found
        """
        # Check all tracking dictionaries
        for order_dict in [self.pending_orders, self.filled_orders, self.cancelled_orders]:
            if order_id in order_dict:
                return order_dict[order_id]
        return None
    
    def reconcile_orders(self) -> Dict[str, int]:
        """
        Sync local order state with Alpaca.
        
        Returns:
            Dictionary with reconciliation statistics
        """
        stats = {
            'updated': 0,
            'moved_to_filled': 0,
            'moved_to_cancelled': 0,
            'errors': 0
        }
        
        logger.info("Starting order reconciliation")
        
        # Get orders from Alpaca - only reconcile relevant orders
        try:
            alpaca_order_map = {}
            
            # Method 1: Query by specific pending order IDs (most robust)
            for order_id in self.pending_orders:
                try:
                    order = self.alpaca_client.get_order_by_id(order_id)
                    alpaca_order_map[order_id] = order
                except Exception as e:
                    logger.warning(f"Could not fetch order {order_id}: {e}")
                    alpaca_order_map[order_id] = None
            
            # Method 2: Refresh in-flight orders
            for order_id in list(self.inflight_orders):
                try:
                    order = self.alpaca_client.get_order_by_id(order_id)
                    alpaca_order_map[order_id] = order
                except Exception as e:
                    logger.warning(f"Could not fetch in-flight order {order_id}: {e}")
                    alpaca_order_map[order_id] = None
            
            # Method 3: Get open orders for additional context (limit scope)
            try:
                request = GetOrdersRequest(
                    status=QueryOrderStatus.OPEN,
                    limit=20,  # Reduced limit
                    nested=True
                )
                open_orders = self.alpaca_client.get_orders(request)
                # Add any open orders not in our pending list
                for order in open_orders:
                    oid = _og(order, 'id')
                    if oid and oid not in alpaca_order_map:
                        alpaca_order_map[oid] = order
            except Exception as e:
                logger.warning(f"Could not fetch open orders: {e}")
            
        except Exception as e:
            logger.error(f"Failed to fetch orders from Alpaca: {e}")
            stats['errors'] += 1
            return stats
        
        # Reconcile pending orders
        for order_id, local_order in list(self.pending_orders.items()):
            try:
                if order_id in alpaca_order_map and alpaca_order_map[order_id] is not None:
                    alpaca_order = alpaca_order_map[order_id]
                    
                    # Normalize the Alpaca order
                    norm = normalize_order(alpaca_order)
                    
                    # Update local order from normalized data
                    try:
                        alpaca_dict = {
                            'id': norm['id'],
                            'symbol': norm['symbol'],
                            'side': norm['side'],
                            'quantity': str(norm['submitted_qty']) if norm['submitted_qty'] is not None else None,
                            'notional': str(norm['submitted_notional']) if norm['submitted_notional'] is not None else None,
                            'order_type': _og(alpaca_order, 'order_type', 'market'),
                            'status': norm['status'],
                            'created_at': _og(alpaca_order, 'created_at', None),
                            'filled_at': _og(alpaca_order, 'filled_at', None),
                            'filled_qty': str(norm['filled_qty']),
                            'filled_avg_price': str(norm['avg_fill_price']) if norm['avg_fill_price'] > 0 else None
                        }
                        # Convert datetime objects to strings
                        if alpaca_dict['created_at'] and hasattr(alpaca_dict['created_at'], 'isoformat'):
                            alpaca_dict['created_at'] = alpaca_dict['created_at'].isoformat()
                        if alpaca_dict['filled_at'] and hasattr(alpaca_dict['filled_at'], 'isoformat'):
                            alpaca_dict['filled_at'] = alpaca_dict['filled_at'].isoformat()
                    except Exception as e:
                        logger.warning(f"Error normalizing order {order_id}: {e}")
                        continue
                    
                    updated_order = Order.from_alpaca_response(alpaca_dict)
                    
                    # Update local tracking based on normalized status
                    alpaca_status = norm['status']
                    
                    if alpaca_status in ['filled']:
                        self.filled_orders[order_id] = updated_order
                        del self.pending_orders[order_id]
                        stats['moved_to_filled'] += 1
                        logger.info(f"Order {order_id} filled: {alpaca_status}")
                    elif alpaca_status in ['canceled', 'rejected', 'expired']:
                        self.cancelled_orders[order_id] = updated_order
                        del self.pending_orders[order_id]
                        stats['moved_to_cancelled'] += 1
                        logger.info(f"Order {order_id} cancelled/rejected: {alpaca_status}")
                    else:
                        self.pending_orders[order_id] = updated_order
                        stats['updated'] += 1
                        logger.debug(f"Order {order_id} still pending: {alpaca_status}")
                else:
                    # Order not found in Alpaca - might be very old
                    logger.warning(f"Order {order_id} not found in Alpaca")
                    
            except Exception as e:
                logger.error(f"Error reconciling order {order_id}: {e}")
                stats['errors'] += 1
        
        # Reconcile in-flight orders
        for order_id in list(self.inflight_orders):
            try:
                if order_id in alpaca_order_map and alpaca_order_map[order_id] is not None:
                    alpaca_order = alpaca_order_map[order_id]
                    norm = normalize_order(alpaca_order)
                    status = norm['status']
                    
                    if status in ACK_SET:
                        # Order was promoted to ACK - move to pending orders
                        self.inflight_orders.discard(order_id)
                        if order_id in self.pending_orders:
                            self.pending_orders[order_id].status = OrderStatus.SUBMITTED
                        logger.info(f"In-flight order {order_id} promoted to ACK: {status}")
                    elif status in TERMINAL_SET:
                        # Order terminated while in-flight
                        self.inflight_orders.discard(order_id)
                        if order_id in self.pending_orders:
                            del self.pending_orders[order_id]
                        logger.info(f"In-flight order {order_id} terminated: {status}")
                    # If still pending, leave it in inflight_orders
                else:
                    # In-flight order not found - might have been filled/cancelled
                    self.inflight_orders.discard(order_id)
                    logger.warning(f"In-flight order {order_id} not found in Alpaca")
                    
            except Exception as e:
                logger.error(f"Error reconciling in-flight order {order_id}: {e}")
                stats['errors'] += 1
        
        self.last_reconciliation = datetime.now()
        logger.info(f"Order reconciliation completed: {stats}")
        return stats
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return list(self.pending_orders.values())
    
    def get_filled_orders(self, since: Optional[datetime] = None) -> List[Order]:
        """
        Get filled orders, optionally filtered by date.
        
        Args:
            since: Only return orders filled after this date
            
        Returns:
            List of filled orders
        """
        filled_orders = list(self.filled_orders.values())
        
        if since:
            filled_orders = [order for order in filled_orders 
                           if order.filled_at and order.filled_at >= since]
        
        return filled_orders
    
    def get_order_summary(self) -> Dict[str, int]:
        """
        Get summary of order counts by status.
        
        Returns:
            Dictionary with order counts
        """
        return {
            'pending': len(self.pending_orders),
            'filled': len(self.filled_orders),
            'cancelled': len(self.cancelled_orders),
            'total': len(self.pending_orders) + len(self.filled_orders) + len(self.cancelled_orders)
        }
    
    def should_reconcile(self) -> bool:
        """Check if reconciliation is needed based on time interval."""
        return (datetime.now() - self.last_reconciliation).total_seconds() > self.reconciliation_interval
    
    def _create_alpaca_request(self, order: Order):
        """
        Convert Order to Alpaca order request.
        
        Args:
            order: Order instance
            
        Returns:
            Alpaca order request object
        """
        # Map order side
        side_map = {
            OrderSide.BUY: AlpacaOrderSide.BUY,
            OrderSide.SELL: AlpacaOrderSide.SELL
        }
        
        # Map time in force
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK
        }
        
        # Get extended hours setting from config
        allow_extended_hours = self.config.get('allow_extended_hours', False)
        
        # Create base request
        base_kwargs = {
            "symbol": order.symbol,
            "qty": order.quantity,
            "side": side_map[order.side],
            "time_in_force": tif_map.get(order.time_in_force, TimeInForce.DAY),
            "extended_hours": allow_extended_hours
        }
        
        # Create specific request based on order type
        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(**base_kwargs)
        
        elif order.order_type == OrderType.LIMIT:
            return LimitOrderRequest(limit_price=order.limit_price, **base_kwargs)
        
        elif order.order_type == OrderType.STOP:
            return StopOrderRequest(stop_price=order.stop_price, **base_kwargs)
        
        elif order.order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                **base_kwargs
            )
        
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")
    
    def cleanup_old_orders(self, days_to_keep: int = 30) -> int:
        """
        Remove old orders from tracking to prevent memory growth.
        
        Args:
            days_to_keep: Number of days to keep orders
            
        Returns:
            Number of orders cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned = 0
        
        # Clean up filled orders
        for order_id in list(self.filled_orders.keys()):
            order = self.filled_orders[order_id]
            if order.filled_at and order.filled_at < cutoff_date:
                del self.filled_orders[order_id]
                cleaned += 1
        
        # Clean up cancelled orders
        for order_id in list(self.cancelled_orders.keys()):
            order = self.cancelled_orders[order_id]
            if order.created_at < cutoff_date:
                del self.cancelled_orders[order_id]
                cleaned += 1
        
        logger.info(f"Cleaned up {cleaned} old orders")
        return cleaned
