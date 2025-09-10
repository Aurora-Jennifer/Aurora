"""
Risk Manager

Enforces risk limits and safety checks before order execution.
Provides comprehensive risk monitoring and portfolio risk calculations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import zoneinfo

from .order_types import Order, OrderSide

logger = logging.getLogger(__name__)

# NY timezone for trading day calculations
NY = zoneinfo.ZoneInfo("America/New_York")


class OrderLimitMetrics:
    """Track order limit metrics with proper trading day boundaries."""
    def __init__(self):
        self.trading_day_start = None          # datetime in NY tz
        self.acked_today = 0                   # broker-ACKed orders today
        self.acked_today_by_symbol = {}        # {symbol: int}
        self.last_submit_at = {}               # {symbol: datetime}
        self.window_submits = []               # [(ts, symbol)] for rate limiting
        self.counters = {"risk_skips": {}}     # Track skip reasons


def get_trading_day_start(now_utc: datetime) -> datetime:
    """Get trading day start in NY timezone."""
    now_ny = now_utc.astimezone(NY)
    day_start_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    return day_start_ny


def maybe_reset_day(metrics: OrderLimitMetrics, now_utc: datetime):
    """Reset daily counters if we've crossed into a new trading day."""
    start = get_trading_day_start(now_utc)
    if metrics.trading_day_start != start:
        metrics.trading_day_start = start
        metrics.acked_today = 0
        metrics.acked_today_by_symbol.clear()
        metrics.window_submits.clear()
        metrics.last_submit_at.clear()
        logger.info(f"Reset daily risk counters for new trading day: {start}")


def check_limits(symbol: str, metrics: OrderLimitMetrics, cfg, now_utc: datetime) -> Tuple[bool, Optional[str]]:
    """Check all risk limits for a symbol."""
    maybe_reset_day(metrics, now_utc)

    # Daily order limit
    if metrics.acked_today >= cfg.max_daily_orders:
        return False, "daily_order_limit_exceeded"

    # Per-symbol daily limit
    if metrics.acked_today_by_symbol.get(symbol, 0) >= cfg.max_orders_per_symbol_per_day:
        return False, "per_symbol_daily_limit_exceeded"

    # Sliding window rate limit
    window_sec = getattr(cfg, "max_submits_window_sec", 60)
    window_cap = getattr(cfg, "max_submits_per_window", 10)
    cutoff = now_utc.timestamp() - window_sec
    metrics.window_submits = [(t, s) for (t, s) in metrics.window_submits if t >= cutoff]
    if len(metrics.window_submits) >= window_cap:
        return False, "rate_limit_window_exceeded"

    # Per-symbol cooldown
    cooldown = getattr(cfg, "per_symbol_cooldown_sec", 3)
    last = metrics.last_submit_at.get(symbol)
    if last and (now_utc - last).total_seconds() < cooldown:
        return False, "per_symbol_cooldown"

    return True, None


def on_order_ack(symbol: str, metrics: OrderLimitMetrics, now_utc: datetime):
    """Record a broker-ACKed order."""
    maybe_reset_day(metrics, now_utc)
    metrics.acked_today += 1
    metrics.acked_today_by_symbol[symbol] = metrics.acked_today_by_symbol.get(symbol, 0) + 1
    metrics.window_submits.append((now_utc.timestamp(), symbol))
    metrics.last_submit_at[symbol] = now_utc


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Core risk limits
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_position_risk: float = 0.05  # 5% per position risk
    max_sector_exposure: float = 0.3  # 30% sector exposure limit
    max_correlation_exposure: float = 0.4  # 40% correlated exposure limit
    max_leverage: float = 1.0  # No leverage
    max_drawdown: float = 0.10  # 10% maximum drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss
    
    # Order throttling
    max_daily_orders: int = 1000  # Maximum orders per day (increased for testing)
    max_orders_per_symbol_per_day: int = 50  # Maximum orders per symbol per day
    max_submits_per_window: int = 10  # Max submits in sliding window (ACKed only)
    max_submits_window_sec: int = 60  # Sliding window duration
    per_symbol_cooldown_sec: int = 2  # Cooldown between orders for same symbol
    max_submits_per_bar: int = 4  # Max submits per execution bar
    
    # Oh-shit guardrails
    allow_shorts: bool = False  # Start long-only
    
    # Split caps by intent (reducer vs opener)
    max_notional_opener: float = 600      # applies to BUY-to-open / SELL-to-open
    max_notional_reducer: Optional[float] = None    # None = unlimited; or set e.g. 100000
    min_notional_opener: float = 10       # keep existing min
    min_notional_reducer: float = 0       # allow tiny closes ("dust")
    allow_dust_close: bool = True         # let reducers < min go if they fully close
    max_pos_pct: float = 0.05  # Per-symbol cap = 5% of equity
    max_gross_exposure: float = 0.60  # Sum(|positions|) ≤ 60% of equity
    max_order_notional: float = 10000  # Hard cap per order
    min_order_notional: float = 50  # Ignore dust
    backstop_max_ticket: float = 12000  # Gate backstop (higher than max_order_notional)
    stale_signal_secs: int = 120  # Reject old signals
    max_slip_pct: float = 0.5  # Reject if price deviates >0.5% from ref
    
    # Circuit breakers
    session_drawdown_limit: float = 0.01  # 1% session drawdown limit
    symbol_move_limit: float = 0.10  # 10% intraday move limit
    spread_limit_bps: int = 50  # 50 bps spread limit
    
    # Sector exposure checks (temporarily disabled until GICS data available)
    sector_exposure_checks: bool = False  # temp; re-enable when GICS wired


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    total_exposure: float
    leverage: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    sharpe_ratio: float
    sector_exposures: Dict[str, float]
    position_risks: Dict[str, float]


class RiskManager:
    """
    Manages risk limits and portfolio risk monitoring.
    
    Responsibilities:
    - Enforce daily loss limits
    - Validate position size limits
    - Monitor sector and correlation exposure
    - Calculate portfolio risk metrics
    - Provide risk alerts and warnings
    """
    
    def __init__(self, risk_limits: RiskLimits, sector_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize Risk Manager.
        
        Args:
            risk_limits: Risk limits configuration
            sector_mapping: Symbol to sector mapping {symbol: sector}
        """
        self.risk_limits = risk_limits
        self.order_limit_metrics = OrderLimitMetrics()  # Track daily metrics
        self.sector_mapping = sector_mapping or {}
        
        # Risk tracking
        self.daily_orders: Dict[str, int] = defaultdict(int)  # symbol -> count
        self.daily_order_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Portfolio tracking
        self.portfolio_history: List[Dict] = []
        self.max_portfolio_value = 0.0
        
        logger.info(f"RiskManager initialized with limits: {risk_limits}")
    
    def check_order_risk(
        self,
        order: Order,
        current_positions: Dict[str, Dict],
        portfolio_value: float,
        daily_pnl: float,
        current_prices: Dict[str, float]
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Check if order violates risk limits.
        
        Args:
            order: Order to check
            current_positions: Current positions
            portfolio_value: Total portfolio value
            daily_pnl: Daily P&L
            current_prices: Current prices
            
        Returns:
            Tuple of (is_allowed, reason, risk_metrics)
        """
        logger.debug(f"Checking risk for order: {order}")
        
        # Reset daily counters if new day
        self._reset_daily_counters_if_needed()
        
        # 1. Check daily order limits
        if not self._check_daily_order_limits(order):
            return False, "daily_order_limit_exceeded", {}
        
        # 2. Check daily loss limit
        if not self._check_daily_loss_limit(daily_pnl, portfolio_value):
            return False, "daily_loss_limit_exceeded", {}
        
        # 3. Check position size limits
        if not self._check_position_size_limit(order, portfolio_value):
            return False, "position_size_limit_exceeded", {}
        
        # 4. Check sector exposure limits
        if not self._check_sector_exposure_limit(order, current_positions, portfolio_value):
            return False, "sector_exposure_limit_exceeded", {}
        
        # 5. Check correlation exposure limits
        if not self._check_correlation_exposure_limit(order, current_positions, portfolio_value):
            return False, "correlation_exposure_limit_exceeded", {}
        
        # 6. Check leverage limits
        if not self._check_leverage_limit(order, current_positions, portfolio_value):
            return False, "leverage_limit_exceeded", {}
        
        # Calculate risk metrics
        risk_metrics = self.calculate_portfolio_risk(
            current_positions, portfolio_value, daily_pnl, current_prices
        )
        
        logger.info(f"Order risk check passed for {order.symbol}")
        return True, "risk_check_passed", risk_metrics
    
    def check_order_limits(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """Check all order limits for a symbol using proper risk metrics."""
        now_utc = datetime.now(timezone.utc)
        return check_limits(symbol, self.order_limit_metrics, self.risk_limits, now_utc)
    
    def on_order_ack(self, symbol: str):
        """Record a broker-ACKed order."""
        now_utc = datetime.now(timezone.utc)
        on_order_ack(symbol, self.order_limit_metrics, now_utc)
    
    def _check_daily_order_limits(self, order: Order) -> bool:
        """Check daily order count limits (legacy method - use check_order_limits instead)."""
        # Check total daily orders
        if self.daily_order_count >= self.risk_limits.max_daily_orders:
            logger.warning(f"Daily order limit exceeded: {self.daily_order_count}")
            return False
        
        # Check symbol-specific daily orders
        if self.daily_orders[order.symbol] >= self.risk_limits.max_orders_per_symbol_per_day:
            logger.warning(f"Daily order limit exceeded for {order.symbol}: {self.daily_orders[order.symbol]}")
            return False
        
        return True
    
    def _check_daily_loss_limit(self, daily_pnl: float, portfolio_value: float) -> bool:
        """Check daily loss limit."""
        if portfolio_value <= 0:
            return False
        
        daily_pnl_pct = daily_pnl / portfolio_value
        max_daily_loss_pct = self.risk_limits.max_daily_loss
        
        if daily_pnl_pct < -max_daily_loss_pct:
            logger.warning(f"Daily loss limit exceeded: {daily_pnl_pct:.2%} < -{max_daily_loss_pct:.2%}")
            return False
        
        return True
    
    def _check_position_size_limit(self, order: Order, portfolio_value: float) -> bool:
        """Check position size limit."""
        if portfolio_value <= 0:
            return False
        
        # Calculate position value (approximate)
        position_value = order.quantity * 100  # Rough estimate, should use actual price
        position_pct = position_value / portfolio_value
        
        if position_pct > self.risk_limits.max_position_risk:
            logger.warning(f"Position size limit exceeded: {position_pct:.2%} > {self.risk_limits.max_position_risk:.2%}")
            return False
        
        return True
    
    def _check_sector_exposure_limit(self, order: Order, current_positions: Dict[str, Dict], portfolio_value: float) -> bool:
        """Check sector exposure limit."""
        if not self.sector_mapping or portfolio_value <= 0:
            return True  # Skip if no sector mapping
        
        # Get sector for the order symbol
        sector = self.sector_mapping.get(order.symbol)
        if not sector:
            return True  # Skip if no sector mapping for symbol
        
        # Calculate current sector exposure
        sector_exposure = 0.0
        for symbol, position in current_positions.items():
            if self.sector_mapping.get(symbol) == sector:
                sector_exposure += position.get('value', 0)
        
        # Add new position to sector exposure
        new_position_value = order.quantity * 100  # Rough estimate
        new_sector_exposure = (sector_exposure + new_position_value) / portfolio_value
        
        if new_sector_exposure > self.risk_limits.max_sector_exposure:
            logger.warning(f"Sector exposure limit exceeded for {sector}: {new_sector_exposure:.2%} > {self.risk_limits.max_sector_exposure:.2%}")
            return False
        
        return True
    
    def _check_correlation_exposure_limit(self, order: Order, current_positions: Dict[str, Dict], portfolio_value: float) -> bool:
        """Check correlation exposure limit."""
        # This is a simplified check - in practice, you'd use correlation matrices
        # For now, we'll check if we have too many positions in similar assets
        if portfolio_value <= 0:
            return True
        
        # Count positions in similar assets (same prefix for now)
        symbol_prefix = order.symbol[:3]  # First 3 characters
        similar_exposure = 0.0
        
        for symbol, position in current_positions.items():
            if symbol.startswith(symbol_prefix):
                similar_exposure += position.get('value', 0)
        
        new_position_value = order.quantity * 100  # Rough estimate
        new_correlation_exposure = (similar_exposure + new_position_value) / portfolio_value
        
        if new_correlation_exposure > self.risk_limits.max_correlation_exposure:
            logger.warning(f"Correlation exposure limit exceeded for {symbol_prefix}: {new_correlation_exposure:.2%} > {self.risk_limits.max_correlation_exposure:.2%}")
            return False
        
        return True
    
    def _check_leverage_limit(self, order: Order, current_positions: Dict[str, Dict], portfolio_value: float) -> bool:
        """Check leverage limit."""
        if portfolio_value <= 0:
            return False
        
        # Calculate current leverage (simplified)
        total_exposure = sum(pos.get('value', 0) for pos in current_positions.values())
        current_leverage = total_exposure / portfolio_value
        
        if current_leverage > self.risk_limits.max_leverage:
            logger.warning(f"Leverage limit exceeded: {current_leverage:.2f} > {self.risk_limits.max_leverage:.2f}")
            return False
        
        return True
    
    def calculate_portfolio_risk(
        self,
        current_positions: Dict[str, Dict],
        portfolio_value: float,
        daily_pnl: float,
        current_prices: Dict[str, float]
    ) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            current_positions: Current positions
            portfolio_value: Total portfolio value
            daily_pnl: Daily P&L
            current_prices: Current prices
            
        Returns:
            RiskMetrics object
        """
        if portfolio_value <= 0:
            return RiskMetrics(
                portfolio_value=0,
                daily_pnl=0,
                daily_pnl_pct=0,
                total_exposure=0,
                leverage=0,
                max_drawdown=0,
                var_95=0,
                sharpe_ratio=0,
                sector_exposures={},
                position_risks={}
            )
        
        # Basic metrics
        daily_pnl_pct = daily_pnl / portfolio_value
        total_exposure = sum(pos.get('value', 0) for pos in current_positions.values())
        leverage = total_exposure / portfolio_value
        
        # Update max portfolio value for drawdown calculation
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        max_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0
        
        # Calculate sector exposures
        sector_exposures = defaultdict(float)
        for symbol, position in current_positions.items():
            sector = self.sector_mapping.get(symbol, "Unknown")
            sector_exposures[sector] += position.get('value', 0)
        
        # Convert to percentages
        sector_exposures = {sector: exposure / portfolio_value for sector, exposure in sector_exposures.items()}
        
        # Calculate position risks (simplified)
        position_risks = {}
        for symbol, position in current_positions.items():
            position_value = position.get('value', 0)
            position_risks[symbol] = position_value / portfolio_value
        
        # Simplified VaR calculation (95% confidence)
        # In practice, you'd use historical returns or Monte Carlo simulation
        var_95 = portfolio_value * 0.02  # 2% of portfolio value
        
        # Simplified Sharpe ratio (would need historical returns)
        sharpe_ratio = 0.0  # Placeholder
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_exposure=total_exposure,
            leverage=leverage,
            max_drawdown=max_drawdown,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            sector_exposures=dict(sector_exposures),
            position_risks=position_risks
        )
    
    def record_order(self, order: Order) -> None:
        """Record an order for daily tracking."""
        self._reset_daily_counters_if_needed()
        self.daily_orders[order.symbol] += 1
        self.daily_order_count += 1
        logger.debug(f"Recorded order for {order.symbol}, daily count: {self.daily_orders[order.symbol]}")
    
    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if it's a new day."""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_orders.clear()
            self.daily_order_count = 0
            self.last_reset_date = current_date
            logger.info("Reset daily order counters for new day")
    
    def get_risk_summary(self) -> Dict[str, any]:
        """
        Get summary of current risk status.
        
        Returns:
            Dictionary with risk summary
        """
        return {
            "daily_orders": dict(self.daily_orders),
            "daily_order_count": self.daily_order_count,
            "max_daily_orders": self.risk_limits.max_daily_orders,
            "max_orders_per_symbol_per_day": self.risk_limits.max_orders_per_symbol_per_day,
            "risk_limits": {
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "max_position_risk": self.risk_limits.max_position_risk,
                "max_sector_exposure": self.risk_limits.max_sector_exposure,
                "max_correlation_exposure": self.risk_limits.max_correlation_exposure,
                "max_leverage": self.risk_limits.max_leverage,
                "max_drawdown": self.risk_limits.max_drawdown,
                "stop_loss_pct": self.risk_limits.stop_loss_pct
            }
        }
    
    def check_emergency_stop(self, portfolio_value: float, daily_pnl: float) -> Tuple[bool, str]:
        """
        Check if emergency stop conditions are met.
        
        Args:
            portfolio_value: Total portfolio value
            daily_pnl: Daily P&L
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if portfolio_value <= 0:
            return True, "portfolio_value_zero"
        
        # Check maximum drawdown
        if self.max_portfolio_value > 0:
            current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            if current_drawdown > self.risk_limits.max_drawdown:
                return True, f"max_drawdown_exceeded: {current_drawdown:.2%}"
        
        # Check daily loss limit
        daily_pnl_pct = daily_pnl / portfolio_value
        if daily_pnl_pct < -self.risk_limits.max_daily_loss:
            return True, f"daily_loss_limit_exceeded: {daily_pnl_pct:.2%}"
        
        return False, "no_emergency_stop"
    
    def update_portfolio_history(self, portfolio_value: float, timestamp: Optional[datetime] = None) -> None:
        """Update portfolio history for risk calculations."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.portfolio_history.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value
        })
        
        # Keep only last 30 days of history
        cutoff_date = timestamp - timedelta(days=30)
        self.portfolio_history = [
            entry for entry in self.portfolio_history
            if entry["timestamp"] >= cutoff_date
        ]
