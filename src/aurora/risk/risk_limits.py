"""
Risk Limits Module - Hard risk controls for live trading

Implements institutional-grade risk controls including:
- Position limits
- Daily loss limits  
- Order rate limits
- Kill switches
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Order representation for risk checking"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    order_type: str = 'MARKET'
    client_order_id: str | None = None


@dataclass
class Portfolio:
    """Portfolio state for risk checking"""
    positions: dict[str, float]  # symbol -> position
    cash: float
    total_value: float
    
    def total_notional_after(self, order: Order) -> float:
        """Calculate total notional after order execution"""
        # Simplified calculation - in production would use proper position sizing
        return abs(order.quantity * order.price)
    
    def weight_after(self, order: Order, symbol: str) -> float:
        """Calculate position weight after order execution"""
        if self.total_value == 0:
            return 0.0
        new_position = self.positions.get(symbol, 0.0) + order.quantity
        return abs(new_position * order.price) / self.total_value


class RiskLimits:
    """Hard risk controls for trading system"""
    
    def __init__(self, 
                 max_notional: float = 100000.0,
                 max_daily_loss: float = 5000.0,
                 max_symbol_weight: float = 0.3,
                 max_orders_per_min: int = 10,
                 max_drawdown: float = 0.15):
        """
        Initialize risk limits
        
        Args:
            max_notional: Maximum total notional exposure
            max_daily_loss: Maximum daily loss before kill switch
            max_symbol_weight: Maximum weight per symbol (0.3 = 30%)
            max_orders_per_min: Maximum orders per minute
            max_drawdown: Maximum drawdown before kill switch
        """
        self.max_notional = max_notional
        self.max_daily_loss = max_daily_loss
        self.max_symbol_weight = max_symbol_weight
        self.max_orders_per_min = max_orders_per_min
        self.max_drawdown = max_drawdown
        
        # State tracking
        self._pnl_open = 0.0
        self._pnl_peak = 0.0
        self._order_timestamps = []
        self._daily_start_value = 0.0
        self._kill_switch_triggered = False
        
        logger.info(f"Risk limits initialized: "
                   f"notional={max_notional}, "
                   f"daily_loss={max_daily_loss}, "
                   f"symbol_weight={max_symbol_weight}, "
                   f"orders/min={max_orders_per_min}")
    
    def check_order(self, portfolio: Portfolio, order: Order) -> bool:
        """
        Check if order passes risk limits
        
        Args:
            portfolio: Current portfolio state
            order: Order to check
            
        Returns:
            True if order passes risk checks
            
        Raises:
            SystemExit: If kill switch is triggered
            RuntimeError: If risk limit is breached
        """
        if self._kill_switch_triggered:
            raise SystemExit("KILL-SWITCH: System is in kill mode")
        
        # Check position limits
        if not self._check_position_limits(portfolio, order):
            logger.warning("Order rejected: position limits breached")
            return False
        
        # Check order rate limits
        if not self._check_rate_limits():
            logger.warning("Order rejected: rate limit breached")
            return False
        
        # Check drawdown limits
        if not self._check_drawdown_limits():
            logger.warning("Order rejected: drawdown limit breached")
            return False
        
        logger.debug(f"Order passed risk checks: {order.symbol} {order.side} {order.quantity}")
        return True
    
    def update_pnl(self, realized_today: float, current_value: float):
        """
        Update PnL tracking and check daily loss limits
        
        Args:
            realized_today: Realized PnL for today
            current_value: Current portfolio value
            
        Raises:
            SystemExit: If daily loss limit is breached
        """
        self._pnl_open = realized_today
        
        # Update peak tracking
        if current_value > self._pnl_peak:
            self._pnl_peak = current_value
        
        # Check daily loss limit
        if self._pnl_open <= -self.max_daily_loss:
            self._kill_switch_triggered = True
            logger.critical(f"KILL-SWITCH TRIGGERED: Daily loss limit breached "
                          f"({self._pnl_open:.2f} <= -{self.max_daily_loss:.2f})")
            raise SystemExit("KILL-SWITCH: Daily loss limit breached")
        
        # Check drawdown limit
        if self._pnl_peak > 0:
            drawdown = (self._pnl_peak - current_value) / self._pnl_peak
            if drawdown > self.max_drawdown:
                self._kill_switch_triggered = True
                logger.critical(f"KILL-SWITCH TRIGGERED: Drawdown limit breached "
                              f"({drawdown:.2%} > {self.max_drawdown:.2%})")
                raise SystemExit("KILL-SWITCH: Drawdown limit breached")
    
    def _check_position_limits(self, portfolio: Portfolio, order: Order) -> bool:
        """Check position and notional limits"""
        # Check total notional
        total_notional = portfolio.total_notional_after(order)
        if total_notional > self.max_notional:
            logger.warning(f"Notional limit breached: {total_notional:.2f} > {self.max_notional:.2f}")
            return False
        
        # Check symbol weight
        symbol_weight = portfolio.weight_after(order, order.symbol)
        if symbol_weight > self.max_symbol_weight:
            logger.warning(f"Symbol weight limit breached: {symbol_weight:.2%} > {self.max_symbol_weight:.2%}")
            return False
        
        return True
    
    def _check_rate_limits(self) -> bool:
        """Check order rate limits"""
        now = time.time()
        
        # Clean old timestamps (older than 1 minute)
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60]
        
        # Check if we're at the limit
        if len(self._order_timestamps) >= self.max_orders_per_min:
            logger.warning(f"Rate limit breached: {len(self._order_timestamps)} orders in last minute")
            return False
        
        # Add current timestamp
        self._order_timestamps.append(now)
        return True
    
    def _check_drawdown_limits(self) -> bool:
        """Check drawdown limits"""
        if self._pnl_peak > 0:
            current_drawdown = (self._pnl_peak - (self._pnl_peak + self._pnl_open)) / self._pnl_peak
            if current_drawdown > self.max_drawdown:
                logger.warning(f"Drawdown limit breached: {current_drawdown:.2%} > {self.max_drawdown:.2%}")
                return False
        return True
    
    def reset_daily_limits(self, starting_value: float):
        """Reset daily limits (call at start of trading day)"""
        self._pnl_open = 0.0
        self._pnl_peak = starting_value
        self._daily_start_value = starting_value
        self._order_timestamps = []
        logger.info(f"Daily limits reset. Starting value: {starting_value:.2f}")
    
    def get_risk_status(self) -> dict[str, Any]:
        """Get current risk status"""
        return {
            "kill_switch_triggered": self._kill_switch_triggered,
            "pnl_open": self._pnl_open,
            "pnl_peak": self._pnl_peak,
            "orders_last_minute": len(self._order_timestamps),
            "daily_start_value": self._daily_start_value,
            "limits": {
                "max_notional": self.max_notional,
                "max_daily_loss": self.max_daily_loss,
                "max_symbol_weight": self.max_symbol_weight,
                "max_orders_per_min": self.max_orders_per_min,
                "max_drawdown": self.max_drawdown
            }
        }
    
    def force_kill_switch(self, reason: str = "Manual kill switch"):
        """Manually trigger kill switch"""
        self._kill_switch_triggered = True
        logger.critical(f"KILL-SWITCH TRIGGERED MANUALLY: {reason}")
        raise SystemExit(f"KILL-SWITCH: {reason}")


class RiskMonitor:
    """Real-time risk monitoring and alerting"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.alerts = []
    
    def check_risk_metrics(self, portfolio: Portfolio, current_time: float = None) -> dict[str, Any]:
        """Check all risk metrics and return status"""
        if current_time is None:
            current_time = time.time()
        
        status = self.risk_limits.get_risk_status()
        
        # Add additional monitoring
        status.update({
            "timestamp": current_time,
            "portfolio_value": portfolio.total_value,
            "cash": portfolio.cash,
            "positions": portfolio.positions.copy(),
            "alerts": self.alerts.copy()
        })
        
        return status
    
    def add_alert(self, level: str, message: str):
        """Add risk alert"""
        alert = {
            "timestamp": time.time(),
            "level": level,
            "message": message
        }
        self.alerts.append(alert)
        logger.warning(f"RISK ALERT [{level}]: {message}")
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
