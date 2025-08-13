"""
Portfolio State Management
Single source of truth for position tracking and PnL calculation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Individual position tracking with average price."""
    qty: float = 0.0
    avg_price: float = 0.0
    
    def __post_init__(self):
        if self.qty < 0:
            logger.warning(f"Negative quantity detected: {self.qty}, clamping to 0")
            self.qty = 0.0
            self.avg_price = 0.0


@dataclass
class PortfolioState:
    """Portfolio state with cash, positions, and mark-to-market functionality."""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    last_prices: Dict[str, float] = field(default_factory=dict)
    
    def mark_to_market(self) -> float:
        """Calculate total portfolio value including unrealized PnL."""
        pos_val = 0.0
        for sym, pos in self.positions.items():
            if pos.qty != 0:
                px = self.last_prices.get(sym, pos.avg_price)
                pos_val += pos.qty * px
        return self.cash + pos_val
    
    def get_position_value(self, symbol: str) -> float:
        """Get current value of a specific position."""
        pos = self.positions.get(symbol)
        if pos is None or pos.qty == 0:
            return 0.0
        px = self.last_prices.get(symbol, pos.avg_price)
        return pos.qty * px
    
    def get_unrealized_pnl(self, symbol: str) -> float:
        """Get unrealized PnL for a specific position."""
        pos = self.positions.get(symbol)
        if pos is None or pos.qty == 0:
            return 0.0
        px = self.last_prices.get(symbol, pos.avg_price)
        return pos.qty * (px - pos.avg_price)
    
    def apply_fill(self, symbol: str, side: str, qty: float, price: float, fee_bps: float) -> float:
        """
        Apply a trade fill to the portfolio.
        
        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            qty: Quantity (positive for both sides)
            price: Fill price
            fee_bps: Fee rate in basis points
            
        Returns:
            Realized PnL for this fill (0 for buys, calculated for sells)
        """
        fees = (abs(qty) * price) * (fee_bps / 10_000.0)
        
        if side.upper() == "SELL":
            # For sells, check if position exists first
            pos = self.positions.get(symbol)
            if pos is None:
                logger.warning(f"Cannot sell {qty} of {symbol} - no position exists")
                return 0.0
        else:
            # For buys, create position if it doesn't exist
            pos = self.positions.setdefault(symbol, Position())
        
        if side.upper() == "BUY":
            new_qty = pos.qty + qty
            if new_qty <= 0:  # should not happen with shorting disabled
                logger.warning(f"Buy would result in negative quantity: {new_qty}, clamping to 0")
                new_qty = 0
                pos.avg_price = 0.0
            else:
                # Update average price
                total_cost = (pos.qty * pos.avg_price) + (qty * price)
                pos.avg_price = total_cost / new_qty
            pos.qty = new_qty
            self.cash -= (qty * price + fees)
            return 0.0  # No realized PnL on buys
            
        elif side.upper() == "SELL":
            # Reduce-only: clamp to available quantity
            sell_qty = min(qty, pos.qty)
            if sell_qty <= 0:
                logger.warning(f"Cannot sell {qty} of {symbol} - only {pos.qty} available")
                return 0.0
            
            # Calculate realized PnL for this partial close
            realized = (price - pos.avg_price) * sell_qty
            
            # Update position
            pos.qty -= sell_qty
            if pos.qty == 0:
                pos.avg_price = 0.0
                # Remove position from portfolio if quantity is 0
                self.positions.pop(symbol, None)
            
            # Update cash
            self.cash += (sell_qty * price - fees)
            
            return realized
            
        else:
            logger.error(f"Invalid side: {side}")
            return 0.0
    
    def update_price(self, symbol: str, price: float):
        """Update the last known price for a symbol."""
        self.last_prices[symbol] = price
    
    def get_total_exposure(self) -> float:
        """Get total position exposure as percentage of portfolio."""
        total_value = self.mark_to_market()
        if total_value == 0:
            return 0.0
        
        pos_value = sum(self.get_position_value(sym) for sym in self.positions)
        return pos_value / total_value
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of all positions."""
        summary = {}
        for symbol, pos in self.positions.items():
            if pos.qty != 0:
                current_price = self.last_prices.get(symbol, pos.avg_price)
                unrealized_pnl = self.get_unrealized_pnl(symbol)
                summary[symbol] = {
                    "quantity": pos.qty,
                    "avg_price": pos.avg_price,
                    "current_price": current_price,
                    "market_value": pos.qty * current_price,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": (unrealized_pnl / (pos.qty * pos.avg_price)) if pos.qty * pos.avg_price != 0 else 0.0
                }
        return summary
