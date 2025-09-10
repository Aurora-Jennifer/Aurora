"""
Execution Engine

Orchestrates the complete execution flow from signals to orders.
Coordinates between all execution components and manages the execution lifecycle.
"""

import logging
import time
import threading
import atexit
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass

from .order_types import Order, OrderSide, OrderType, OrderStatus
from .order_manager import OrderManager
from .position_sizing import PositionSizer, PositionSizingConfig
from .risk_manager import RiskManager, RiskLimits
from .portfolio_manager import PortfolioManager
from .time_utils import ensure_aware_utc, now_utc
from .order_manager import ACK_SET, PENDING_SET, TERMINAL_SET, plan_batches


def pos_qty(pos) -> int:
    """Extract quantity from Position object or return 0."""
    if pos is None:
        return 0
    for attr in ("qty", "quantity", "qty_available"):
        v = getattr(pos, attr, None)
        if v is not None:
            try:
                return int(v)
            except (ValueError, TypeError):
                pass
    return 0

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    enabled: bool = True
    mode: str = "paper"  # paper, live
    signal_threshold: float = 0.1
    max_orders_per_execution: int = 10
    execution_timeout: int = 30  # seconds
    reconciliation_interval: int = 60  # seconds
    allow_extended_hours: bool = False  # Allow trading during extended hours


@dataclass
class ExecutionResult:
    """Result of execution operation."""
    success: bool
    orders_submitted: int
    orders_filled: int
    orders_rejected: int
    total_pnl: float
    execution_time: float
    errors: List[str]
    metadata: Dict[str, Any]


class ExecutionEngine:
    """
    Main execution engine that orchestrates the complete trading flow.
    
    Responsibilities:
    - Convert signals to orders
    - Apply position sizing and risk checks
    - Submit orders to broker
    - Monitor execution and update positions
    - Handle errors and recovery
    - Provide execution reporting
    """
    
    def __init__(
        self,
        order_manager: OrderManager,
        portfolio_manager: PortfolioManager,
        position_sizer: PositionSizer,
        risk_manager: RiskManager,
        config: ExecutionConfig
    ):
        """
        Initialize Execution Engine.
        
        Args:
            order_manager: Order management instance
            portfolio_manager: Portfolio management instance
            position_sizer: Position sizing instance
            risk_manager: Risk management instance
            config: Execution configuration
        """
        # Constructor instrumentation
        try:
            from core.utils.constructor_guard import construct_once
            construct_once("ExecutionEngine")
        except ImportError:
            pass  # Skip instrumentation if not available
        
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.config = config
        
        # Execution state
        self.is_running = False
        self.last_execution = None
        self.execution_count = 0
        
        # Telemetry counters
        self.metrics = {
            'skips_price_cap': 0,
            'skips_size_zero': 0,
            'gate_reject_small': 0,
            'gate_reject_price_unknown': 0
        }
        
        # Thread management for clean shutdown
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        
        # Register cleanup on exit
        atexit.register(self.stop)
        
        logger.info(f"ExecutionEngine initialized with config: {config}")
    
    def stop(self):
        """Stop all background threads for clean shutdown."""
        if not self.is_running:
            # Engine was never started, just clean up threads
            self._stop_event.set()
            for t in self._threads:
                t.join(timeout=5)
            return
            
        self.is_running = False
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5)
        logger.info("Execution engine stopped")
    
    def start(self):
        """Start the execution engine."""
        if self.is_running:
            logger.warning("Execution engine is already running")
            return
            
        self.is_running = True
        logger.info("Execution engine started")
    
    def execute_signals(self, signals: Dict[str, float], current_prices: Dict[str, float]) -> ExecutionResult:
        """
        Execute trading signals by converting them to orders.
        
        Args:
            signals: Trading signals {symbol: signal_strength}
            current_prices: Current market prices {symbol: price}
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        logger.info(f"Starting signal execution with {len(signals)} signals")
        
        # OH-SHIT GUARDRAIL: Kill switch check
        import os
        if os.getenv('KILL_SWITCH') == '1':
            logger.critical("KILL_SWITCH=1 - Refusing all new orders")
            return ExecutionResult(
                success=False,
                orders_submitted=0,
                orders_filled=0,
                orders_rejected=0,
                total_pnl=0.0,
                execution_time=time.time() - start_time,
                errors=["kill_switch_activated"],
                metadata={}
            )
        
        if not self.config.enabled:
            logger.warning("Execution engine is disabled")
            return ExecutionResult(
                success=False,
                orders_submitted=0,
                orders_filled=0,
                orders_rejected=0,
                total_pnl=0.0,
                execution_time=time.time() - start_time,
                errors=["execution_disabled"],
                metadata={}
            )
        
        # Update portfolio state
        if not self.portfolio_manager.update_positions():
            logger.error("Failed to update portfolio positions")
            return ExecutionResult(
                success=False,
                orders_submitted=0,
                orders_filled=0,
                orders_rejected=0,
                total_pnl=0.0,
                execution_time=time.time() - start_time,
                errors=["portfolio_update_failed"],
                metadata={}
            )
        
        # Get current portfolio metrics
        portfolio_metrics = self.portfolio_manager.calculate_portfolio_metrics()
        current_positions = self.portfolio_manager.get_all_positions()
        
        # Convert positions to dict format for compatibility
        positions_dict = {
            symbol: {
                "quantity": pos.quantity,
                "value": pos.market_value,
                "avg_cost": pos.avg_cost
            }
            for symbol, pos in current_positions.items()
        }
        
        # Process signals and generate target positions
        target_shares = {}
        errors = []
        orders_to_submit = []  # Ensure it always exists
        skipped_symbols = set()  # Track symbols skipped this loop
        
        # Log account snapshot at start of execution
        try:
            snap = self.order_manager.get_account_snapshot()
            logger.info(
                "Account snapshot: bp=$%s dtbp=$%s cash=$%s equity=$%s mult=%d shorting=%s blocked=%s",
                snap["bp"], snap["dtbp"], snap["cash"], snap["equity"], 
                snap["mult"], snap["shorting"], snap["blocked"]
            )
        except Exception as e:
            logger.warning(f"Failed to get account snapshot: {e}")
        
        # Calculate target positions for all signals
        for symbol, signal in signals.items():
            try:
                # Skip if no price available
                if symbol not in current_prices or current_prices[symbol] <= 0:
                    errors.append(f"No valid price for {symbol}")
                    continue
                
                # Check shorting policy
                order_side = "buy" if signal >= 0 else "sell"
                if order_side == "sell" and current_positions.get(symbol, 0) == 0:
                    allow_shorts = self.risk_manager.risk_limits.allow_shorts
                    if not allow_shorts:
                        logger.info("Skip shorting %s (policy long-only)", symbol)
                        continue
                
                # Calculate position size using professional-grade buffer method
                ref_price = current_prices[symbol]
                min_notional = self.risk_manager.risk_limits.min_order_notional
                max_order_notional = self.risk_manager.risk_limits.max_order_notional
                
                # Get current position for buffer-based sizing
                current_position = pos_qty(current_positions.get(symbol, 0))
                
                logger.debug(f"Computing size for {symbol}: signal={signal:.4f}, price={ref_price:.2f}, current_pos={current_position}, min_notional={min_notional}")
                
                # Get capital utilization factor from config
                capital_utilization_factor = getattr(self.config, 'capital_utilization_factor', 1.0)
                
                # Use buffer-based position sizing to prevent micro-rebalancing
                size_decision = self.position_sizer.compute_size_with_buffer(
                    symbol=symbol,
                    target_weight=signal,
                    price=ref_price,
                    portfolio_value=portfolio_metrics.total_value,
                    current_position=current_position,
                    min_notional=min_notional,
                    capital_utilization_factor=capital_utilization_factor
                )
                
                if size_decision is None:
                    # No action needed (within buffer zone or below thresholds)
                    logger.debug(f"HOLD {symbol}: within buffer zone or below thresholds (signal={signal:.4f}, current_pos={current_position})")
                    continue
                
                # Store order delta (not target position)
                target_shares[symbol] = size_decision.qty  # This is now the ORDER DELTA
                logger.debug(f"Order delta for {symbol}: {size_decision.qty} shares (current={current_position}, new_target={current_position + size_decision.qty})")
                
                
            except Exception as e:
                logger.error(f"Error processing signal for {symbol}: {e}")
                errors.append(f"Signal processing error for {symbol}: {str(e)}")
        
        # Execute two-phase order cycle
        if target_shares:
            logger.info(f"Planning two-phase execution for {len(target_shares)} symbols")
            
            # Debug: log order deltas (target_shares now contains order deltas, not target positions)
            min_shares = getattr(self.config, 'min_shares', 1)
            min_notional = self.risk_manager.risk_limits.min_order_notional
            
            for symbol, order_delta in target_shares.items():
                current_qty = pos_qty(current_positions.get(symbol, 0))
                new_target_qty = current_qty + order_delta
                
                # Apply minimum trade threshold to prevent churn
                if abs(order_delta) < min_shares:
                    order_delta = 0
                elif abs(order_delta * current_prices.get(symbol, 0)) < min_notional:
                    order_delta = 0
                
                logger.info(f"PLAN_DELTA {symbol} cur={current_qty:+d} order_delta={order_delta:+d} new_tgt={new_target_qty:+d}")
            
            # Plan batches: reducers first, then openers
            # Convert order deltas to target positions for plan_batches
            target_positions = {}
            for symbol, order_delta in target_shares.items():
                current_qty = pos_qty(current_positions.get(symbol, 0))
                target_positions[symbol] = current_qty + order_delta
            
            reducers, openers = plan_batches(
                current_positions={symbol: pos_qty(pos) for symbol, pos in current_positions.items()},
                target_shares=target_positions,  # Now contains target positions, not order deltas
                ref_prices=current_prices
            )
            
            # Early exit if nothing to do
            if not reducers and not openers:
                skips_price_cap = self.metrics.get('skips_price_cap', 0)
                logger.info(f"Two-phase plan: 0 reducers, 0 openers (holding). skips_price_cap={skips_price_cap}")
                return ExecutionResult(
                    success=True,
                    orders_submitted=0,
                    orders_filled=0,
                    orders_rejected=0,
                    total_pnl=portfolio_metrics.total_pnl,
                    execution_time=time.time() - start_time,
                    errors=errors,
                    metadata={
                        "portfolio_metrics": portfolio_metrics,
                        "signals_processed": len(signals),
                        "signals_skipped": len(signals) - len(target_shares),
                        "orders_created": 0,
                        "skips_price_cap": skips_price_cap
                    }
                )
            
            logger.info(f"Two-phase plan: {len(reducers)} reducers, {len(openers)} openers")
            
            # Execute the two-phase cycle
            cycle_stats = self.order_manager.execute_two_phase_cycle(reducers, openers, self.metrics)
            
            submitted_orders_count = cycle_stats["reducers_submitted"] + cycle_stats["openers_submitted"]
            rejected_orders_count = cycle_stats["openers_skipped"]
        else:
            logger.info("No target positions to execute")
            submitted_orders_count = 0
            rejected_orders_count = 0
        
        # Old order submission loop removed - now using two-phase system
        if False:  # Disable old loop
            try:
                # Generate unique signal ID for this order
                signal_id = f"exec_{int(time.time())}_{order.symbol}_{order.side.value}"
                
                # Get current position for this symbol
                try:
                    current_positions = self.portfolio_manager.get_positions_dict()
                    current_position = pos_qty(current_positions.get(order.symbol, 0))
                except AttributeError:
                    # Fallback if method missing; do not crash submissions
                    positions = getattr(self.portfolio_manager, "positions", {}) or {}
                    current_position = pos_qty(positions.get(order.symbol, 0))
                
                # Submit with pre-trade gate
                from datetime import timezone
                order_id, broker_status = self.order_manager.submit_order_with_gate(
                    order=order,
                    signal_id=signal_id,
                    current_position=current_position,
                    signal_timestamp=datetime.now(timezone.utc)
                )
                
                if order_id:
                    submitted_orders.append(order_id)
                    
                    # Record order for risk tracking (legacy)
                    self.risk_manager.record_order(order)
                    
                    # Handle different status buckets
                    if broker_status in ACK_SET:
                        # Order is ACKed - count toward limits and metrics
                        self.risk_manager.on_order_ack(order.symbol)
                        bar_submits += 1
                        logger.info(f"Order ACKed for {order.symbol}: {order_id} (status: {broker_status})")
                    elif broker_status in PENDING_SET:
                        # Order is pending - add to in-flight tracking
                        self.order_manager.inflight_orders.add(order_id)
                        logger.info(f"Order pending ACK for {order.symbol}: {order_id} (status: {broker_status})")
                        
                        # Try to promote to ACK with a quick poll
                        final_status = self.order_manager.wait_for_ack_or_terminal(order_id, timeout_s=1.0)
                        if final_status in ACK_SET:
                            self.risk_manager.on_order_ack(order.symbol)
                            bar_submits += 1
                            self.order_manager.inflight_orders.discard(order_id)
                            logger.info(f"Order promoted to ACK for {order.symbol}: {order_id} (status: {final_status})")
                        elif final_status in TERMINAL_SET:
                            self.order_manager.inflight_orders.discard(order_id)
                            logger.warning(f"Order terminated while pending for {order.symbol}: {order_id} (status: {final_status})")
                    elif broker_status in TERMINAL_SET:
                        # Order terminated immediately (rare)
                        logger.warning(f"Order terminated immediately for {order.symbol}: {order_id} (status: {broker_status})")
                    else:
                        # Unknown status
                        logger.warning(f"Unknown order status for {order.symbol}: {order_id} (status: {broker_status})")
                else:
                    logger.warning(f"Order rejected by pre-trade gate for {order.symbol}")
                    rejected_orders.append(order.symbol)
                    errors.append(f"Order rejected by pre-trade gate for {order.symbol}")
                
            except Exception as e:
                logger.error(f"Failed to submit order for {order.symbol}: {e}")
                rejected_orders.append(order.symbol)
                errors.append(f"Order submission failed for {order.symbol}: {str(e)}")
        
        # Wait for execution (simplified - in practice you'd monitor asynchronously)
        time.sleep(2)  # Give orders time to execute
        
        # Reconcile orders
        reconciliation_stats = self.order_manager.reconcile_orders()
        
        # Update portfolio with fills
        self._update_portfolio_from_fills()
        
        # Calculate execution metrics
        execution_time = time.time() - start_time
        filled_orders = self.order_manager.get_filled_orders()
        orders_filled = len(filled_orders)
        
        # Update execution state
        self.last_execution = now_utc()
        self.execution_count += 1
        
        # Collect telemetry from order manager
        order_manager_metrics = getattr(self.order_manager, 'metrics', {})
        
        # Prepare result
        result = ExecutionResult(
            success=len(errors) == 0,
            orders_submitted=submitted_orders_count,
            orders_filled=orders_filled,
            orders_rejected=rejected_orders_count,
            total_pnl=portfolio_metrics.total_pnl,
            execution_time=execution_time,
            errors=errors,
            metadata={
                "signals_processed": len(signals),
                "orders_created": submitted_orders_count,
                "reconciliation_stats": reconciliation_stats,
                "portfolio_metrics": {
                    "total_value": portfolio_metrics.total_value,
                    "daily_pnl": portfolio_metrics.daily_pnl,
                    "positions_count": portfolio_metrics.positions_count
                },
                # Add telemetry counters
                "skips_price_cap": self.metrics['skips_price_cap'],
                "skips_size_zero": self.metrics['skips_size_zero'],
                "gate_reject_small": order_manager_metrics.get('gate_reject_small', 0),
                "gate_reject_price_unknown": order_manager_metrics.get('gate_reject_price_unknown', 0),
                # Add risk skip counters
                "risk_skips": self.risk_manager.order_limit_metrics.counters["risk_skips"]
            }
        )
        
        logger.info(f"Signal execution completed: {result}")
        return result
    
    def _update_portfolio_from_fills(self) -> None:
        """Update portfolio with recent order fills."""
        try:
            # Get recent fills (use UTC-aware datetime)
            session_start = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)
            recent_fills = self.order_manager.get_filled_orders(since=session_start)
            
            # Update portfolio for each fill
            for order in recent_fills:
                if order.filled_at and order.filled_price and order.filled_quantity > 0:
                    self.portfolio_manager.update_position_from_fill(
                        symbol=order.symbol,
                        quantity=order.filled_quantity,
                        price=order.filled_price,
                        side=order.side.value
                    )
            
            logger.debug(f"Updated portfolio with {len(recent_fills)} fills")
            
        except Exception as e:
            logger.error(f"Error updating portfolio from fills: {e}")
    
    def monitor_execution(self) -> Dict[str, Any]:
        """
        Monitor current execution state and provide status.
        
        Returns:
            Dictionary with execution status
        """
        # Check if reconciliation is needed
        if self.order_manager.should_reconcile():
            reconciliation_stats = self.order_manager.reconcile_orders()
            logger.info(f"Performed reconciliation: {reconciliation_stats}")
        
        # Check if portfolio update is needed
        if self.portfolio_manager.should_update():
            self.portfolio_manager.update_positions()
        
        # Get current status
        order_summary = self.order_manager.get_order_summary()
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        risk_summary = self.risk_manager.get_risk_summary()
        
        return {
            "execution_engine": {
                "enabled": self.config.enabled,
                "mode": self.config.mode,
                "is_running": self.is_running,
                "last_execution": self.last_execution.isoformat() if self.last_execution else None,
                "execution_count": self.execution_count
            },
            "orders": order_summary,
            "portfolio": portfolio_summary,
            "risk": risk_summary
        }
    
    def emergency_stop(self) -> bool:
        """
        Execute emergency stop - cancel all pending orders.
        
        Returns:
            True if emergency stop was successful
        """
        logger.warning("Executing emergency stop")
        
        try:
            # Cancel all pending orders
            pending_orders = self.order_manager.get_pending_orders()
            cancelled_count = 0
            
            for order in pending_orders:
                if self.order_manager.cancel_order(order.alpaca_order_id):
                    cancelled_count += 1
            
            # Update portfolio state
            self.portfolio_manager.update_positions()
            
            # Check emergency stop conditions
            portfolio_metrics = self.portfolio_manager.calculate_portfolio_metrics()
            should_stop, reason = self.risk_manager.check_emergency_stop(
                portfolio_metrics.total_value,
                portfolio_metrics.daily_pnl
            )
            
            logger.warning(f"Emergency stop completed: {cancelled_count} orders cancelled, reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive execution summary.
        
        Returns:
            Dictionary with execution summary
        """
        return {
            "config": {
                "enabled": self.config.enabled,
                "mode": self.config.mode,
                "signal_threshold": self.config.signal_threshold,
                "max_orders_per_execution": self.config.max_orders_per_execution
            },
            "status": self.monitor_execution(),
            "performance": {
                "execution_count": self.execution_count,
                "last_execution": self.last_execution.isoformat() if self.last_execution else None,
                "uptime": (datetime.now() - self.last_execution).total_seconds() if self.last_execution else 0
            }
        }
    
    def start(self) -> None:
        """Start the execution engine."""
        if self.is_running:
            logger.warning("Execution engine is already running")
            return
        
        self.is_running = True
        logger.info("Execution engine started")
    
    def stop(self) -> None:
        """Stop the execution engine."""
        if not self.is_running:
            logger.info("Execution engine is not running")
            return
        
        self.is_running = False
        logger.info("Execution engine stopped")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status
        """
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check order manager
        try:
            order_summary = self.order_manager.get_order_summary()
            health_status["components"]["order_manager"] = {
                "status": "healthy",
                "pending_orders": order_summary["pending"]
            }
        except Exception as e:
            health_status["components"]["order_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall"] = "unhealthy"
        
        # Check portfolio manager
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            health_status["components"]["portfolio_manager"] = {
                "status": "healthy",
                "portfolio_value": portfolio_summary["portfolio_value"]
            }
        except Exception as e:
            health_status["components"]["portfolio_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall"] = "unhealthy"
        
        # Check risk manager
        try:
            risk_summary = self.risk_manager.get_risk_summary()
            health_status["components"]["risk_manager"] = {
                "status": "healthy",
                "daily_orders": risk_summary["daily_order_count"]
            }
        except Exception as e:
            health_status["components"]["risk_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall"] = "unhealthy"
        
        return health_status
