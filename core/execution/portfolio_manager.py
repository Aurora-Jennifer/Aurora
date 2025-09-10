"""
Portfolio Manager

Tracks positions, portfolio value, and P&L. Manages portfolio state
and provides portfolio analytics and reporting.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    cash: float
    net_position_value: float  # sum of position market values (longs - shorts)
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    daily_pnl: float
    daily_pnl_pct: float
    positions_count: int
    largest_position: Optional[str]
    largest_position_pct: float
    diversification_ratio: float


class PortfolioManager:
    """
    Manages portfolio state, positions, and performance tracking.
    
    Responsibilities:
    - Track current positions and cash
    - Calculate portfolio value and P&L
    - Manage position updates and reconciliation
    - Generate portfolio reports and analytics
    - Handle position rebalancing
    """
    
    def __init__(self, alpaca_client: TradingClient, config: Optional[Dict] = None):
        """
        Initialize Portfolio Manager.
        
        Args:
            alpaca_client: Alpaca TradingClient instance
            config: Configuration dictionary
        """
        # Constructor instrumentation
        try:
            from core.utils.constructor_guard import construct_once
            construct_once("PortfolioManager")
        except ImportError:
            pass  # Skip instrumentation if not available
        
        self.alpaca_client = alpaca_client
        self.config = config or {}
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash: float = 0.0
        self.portfolio_value: float = 0.0
        self.last_updated: Optional[datetime] = None
        
        # Performance tracking
        self.daily_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.portfolio_history: List[Dict] = []
        
        # Configuration
        self.update_interval = self.config.get('update_interval', 60)  # seconds
        self.last_sync = datetime.now()
        
        logger.info("PortfolioManager initialized")
    
    def get_positions(self) -> Dict[str, Position]:
        """Authoritative view after last reconcile."""
        return self.positions
    
    def get_position(self, symbol: str) -> Position:
        """Get position for a specific symbol."""
        return self.positions.get(symbol, Position(
            symbol=symbol,
            quantity=0,
            avg_cost=0.0,
            current_price=0.0,
            market_value=0.0,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0
        ))
    
    def get_positions_dict(self) -> Dict[str, int]:
        """Get positions as symbol -> quantity dict for compatibility."""
        return {symbol: pos.quantity for symbol, pos in self.positions.items()}
    
    def update_positions(self) -> bool:
        """
        Sync positions with Alpaca.
        
        Returns:
            True if update was successful
        """
        try:
            logger.debug("Updating positions from Alpaca")
            
            # Get account information
            account = self.alpaca_client.get_account()
            self.cash = float(account.cash)
            
            # Get positions
            alpaca_positions = self.alpaca_client.get_all_positions()
            
            # Update local positions
            new_positions = {}
            for alpaca_pos in alpaca_positions:
                position = self._create_position_from_alpaca(alpaca_pos)
                new_positions[position.symbol] = position
            
            self.positions = new_positions
            
            # Calculate portfolio value
            self._calculate_portfolio_value()
            
            # Update tracking
            self.last_updated = datetime.now()
            self.last_sync = datetime.now()
            
            logger.info(f"Updated {len(self.positions)} positions, portfolio value: ${self.portfolio_value:.2f}")
            return True
            
        except APIError as e:
            logger.error(f"Failed to update positions: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating positions: {e}")
            return False
    
    def _create_position_from_alpaca(self, alpaca_position) -> Position:
        """Create Position object from Alpaca position data."""
        return Position(
            symbol=alpaca_position.symbol,
            quantity=int(alpaca_position.qty),
            avg_cost=float(alpaca_position.avg_entry_price),
            current_price=float(alpaca_position.current_price),
            market_value=float(alpaca_position.market_value),
            unrealized_pnl=float(alpaca_position.unrealized_pl),
            unrealized_pnl_pct=float(alpaca_position.unrealized_plpc),
            last_updated=datetime.now()
        )
    
    def _calculate_portfolio_value(self) -> None:
        """Calculate total portfolio value."""
        net_position_value = sum(pos.market_value for pos in self.positions.values())
        self.portfolio_value = self.cash + net_position_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self.positions.copy()
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Returns:
            PortfolioMetrics object
        """
        if self.portfolio_value <= 0:
            return PortfolioMetrics(
                total_value=0,
                cash=0,
                net_position_value=0,
                unrealized_pnl=0,
                realized_pnl=0,
                total_pnl=0,
                daily_pnl=0,
                daily_pnl_pct=0,
                positions_count=0,
                largest_position=None,
                largest_position_pct=0,
                diversification_ratio=0
            )
        
        # Basic metrics
        net_position_value = sum(pos.market_value for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_pnl = self.realized_pnl + unrealized_pnl
        daily_pnl_pct = self.daily_pnl / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Position analysis
        positions_count = len(self.positions)
        largest_position = None
        largest_position_pct = 0.0
        
        if self.positions:
            largest_pos = max(self.positions.values(), key=lambda p: p.market_value)
            largest_position = largest_pos.symbol
            largest_position_pct = largest_pos.market_value / self.portfolio_value
        
        # Diversification ratio (simplified)
        diversification_ratio = self._calculate_diversification_ratio()
        
        return PortfolioMetrics(
            total_value=self.portfolio_value,
            cash=self.cash,
            net_position_value=net_position_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_pnl=total_pnl,
            daily_pnl=self.daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            positions_count=positions_count,
            largest_position=largest_position,
            largest_position_pct=largest_position_pct,
            diversification_ratio=diversification_ratio
        )
    
    def _calculate_diversification_ratio(self) -> float:
        """Calculate portfolio diversification ratio."""
        if not self.positions or self.portfolio_value <= 0:
            return 0.0
        
        # Calculate Herfindahl index (sum of squared weights)
        weights = [pos.market_value / self.portfolio_value for pos in self.positions.values()]
        herfindahl_index = sum(w**2 for w in weights)
        
        # Diversification ratio = 1 / Herfindahl index
        return 1.0 / herfindahl_index if herfindahl_index > 0 else 0.0
    
    def update_position_from_fill(self, symbol: str, quantity: int, price: float, side: str) -> None:
        """
        Update position based on order fill.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Fill price
            side: Order side ('buy' or 'sell')
        """
        logger.debug(f"Updating position for {symbol}: {side} {quantity} @ ${price:.2f}")
        
        current_position = self.positions.get(symbol)
        
        if side.lower() == 'buy':
            if current_position:
                # Update existing position
                new_quantity = current_position.quantity + quantity
                new_avg_cost = ((current_position.quantity * current_position.avg_cost) + 
                               (quantity * price)) / new_quantity
                
                current_position.quantity = new_quantity
                current_position.avg_cost = new_avg_cost
                current_position.current_price = price
                current_position.market_value = new_quantity * price
                current_position.unrealized_pnl = (price - new_avg_cost) * new_quantity
                current_position.unrealized_pnl_pct = (price - new_avg_cost) / new_avg_cost
                current_position.last_updated = datetime.now()
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    last_updated=datetime.now()
                )
        
        elif side.lower() == 'sell':
            if current_position:
                # Update existing position
                if quantity >= current_position.quantity:
                    # Selling entire position
                    realized_pnl = (price - current_position.avg_cost) * current_position.quantity
                    self.realized_pnl += realized_pnl
                    
                    del self.positions[symbol]
                else:
                    # Partial sale
                    realized_pnl = (price - current_position.avg_cost) * quantity
                    self.realized_pnl += realized_pnl
                    
                    current_position.quantity -= quantity
                    current_position.market_value = current_position.quantity * price
                    current_position.current_price = price
                    current_position.unrealized_pnl = (price - current_position.avg_cost) * current_position.quantity
                    current_position.unrealized_pnl_pct = (price - current_position.avg_cost) / current_position.avg_cost
                    current_position.last_updated = datetime.now()
            else:
                logger.warning(f"Attempted to sell {quantity} shares of {symbol} but no position exists")
        
        # Recalculate portfolio value
        self._calculate_portfolio_value()
        
        logger.info(f"Position updated for {symbol}, portfolio value: ${self.portfolio_value:.2f}")
    
    def rebalance_portfolio(self, target_weights: Dict[str, float]) -> List[Dict]:
        """
        Generate rebalancing orders to achieve target weights.
        
        Args:
            target_weights: Target weights {symbol: weight}
            
        Returns:
            List of rebalancing orders
        """
        if self.portfolio_value <= 0:
            return []
        
        rebalancing_orders = []
        
        # Calculate target values
        target_values = {symbol: weight * self.portfolio_value for symbol, weight in target_weights.items()}
        
        # Calculate current values
        current_values = {symbol: pos.market_value for symbol, pos in self.positions.items()}
        
        # Generate rebalancing orders
        for symbol in set(list(current_positions.keys()) + list(target_weights.keys())):
            current_value = current_values.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            
            rebalance_amount = target_value - current_value
            
            if abs(rebalance_amount) >= 100:  # Minimum $100 rebalancing
                current_price = self.positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0, 0)).current_price
                if current_price > 0:
                    quantity = int(rebalance_amount / current_price)
                    if quantity != 0:
                        side = 'buy' if quantity > 0 else 'sell'
                        rebalancing_orders.append({
                            'symbol': symbol,
                            'side': side,
                            'quantity': abs(quantity),
                            'order_type': 'market',
                            'reason': 'rebalancing'
                        })
        
        logger.info(f"Generated {len(rebalancing_orders)} rebalancing orders")
        return rebalancing_orders
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary.
        
        Returns:
            Dictionary with portfolio summary
        """
        metrics = self.calculate_portfolio_metrics()
        
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "net_position_value": metrics.net_position_value,
            "unrealized_pnl": metrics.unrealized_pnl,
            "realized_pnl": metrics.realized_pnl,
            "total_pnl": metrics.total_pnl,
            "daily_pnl": metrics.daily_pnl,
            "daily_pnl_pct": metrics.daily_pnl_pct,
            "positions_count": metrics.positions_count,
            "largest_position": metrics.largest_position,
            "largest_position_pct": metrics.largest_position_pct,
            "diversification_ratio": metrics.diversification_ratio,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all positions.
        
        Returns:
            Dictionary with position summaries
        """
        return {
            symbol: {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "weight": pos.market_value / self.portfolio_value if self.portfolio_value > 0 else 0,
                "last_updated": pos.last_updated.isoformat()
            }
            for symbol, pos in self.positions.items()
        }
    
    def should_update(self) -> bool:
        """Check if portfolio should be updated based on time interval."""
        if self.last_sync is None:
            return True
        
        return (datetime.now() - self.last_sync).total_seconds() > self.update_interval
    
    def update_daily_pnl(self, new_daily_pnl: float) -> None:
        """Update daily P&L tracking."""
        self.daily_pnl = new_daily_pnl
        logger.debug(f"Updated daily P&L: ${new_daily_pnl:.2f}")
    
    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (typically at market open)."""
        self.daily_pnl = 0.0
        logger.info("Reset daily P&L")
    
    def add_portfolio_snapshot(self) -> None:
        """Add current portfolio state to history."""
        snapshot = {
            "timestamp": datetime.now(),
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "net_position_value": sum(pos.market_value for pos in self.positions.values()),
            "positions_count": len(self.positions),
            "daily_pnl": self.daily_pnl
        }
        
        self.portfolio_history.append(snapshot)
        
        # Keep only last 30 days of history
        cutoff_date = datetime.now() - timedelta(days=30)
        self.portfolio_history = [
            entry for entry in self.portfolio_history
            if entry["timestamp"] >= cutoff_date
        ]
