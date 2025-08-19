"""
Portfolio State Management
Single source of truth for position tracking and PnL calculation.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Individual position tracking with average price and realized PnL."""

    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        # Allow negative quantities for short positions
        # The validation will be handled at the portfolio level
        pass

    def apply_fill(self, side: str, qty: float, price: float, fee: float = 0.0) -> float:
        """
        Apply a trade fill to this position.

        Args:
            side: "BUY" or "SELL"
            qty: Quantity (positive for both sides)
            price: Fill price
            fee: Fee amount

        Returns:
            Realized PnL for this fill
        """
        if side.upper() == "BUY":
            # Determine if this increases or reduces exposure
            if self.qty >= 0:
                # Long position or flat - this increases exposure
                new_qty = self.qty + qty
                if new_qty == 0:
                    self.avg_price = 0.0
                else:
                    # Update VWAP
                    total_cost = (self.qty * self.avg_price) + (qty * price)
                    self.avg_price = total_cost / new_qty
                self.qty = new_qty
                return 0.0  # No realized PnL on position increase
            else:
                # Short position - this reduces exposure (covers short)
                if qty <= abs(self.qty):
                    # Partial cover
                    realized = (self.avg_price - price) * qty  # For shorts: profit when price falls
                    self.qty += qty  # qty is positive, so this reduces the negative
                    if abs(self.qty) < 1e-6:
                        self.avg_price = 0.0
                    return realized
                else:
                    # Full cover plus new long
                    cover_qty = abs(self.qty)
                    new_long_qty = qty - cover_qty

                    # Realized PnL on the cover portion
                    realized = (self.avg_price - price) * cover_qty

                    # Set up new long position
                    self.qty = new_long_qty
                    self.avg_price = price

                    return realized

        elif side.upper() == "SELL":
            # Determine if this increases or reduces exposure
            if self.qty <= 0:
                # Short position or flat - this increases short exposure
                new_qty = self.qty - qty
                if abs(new_qty) < 1e-6:
                    self.avg_price = 0.0
                else:
                    # Update VWAP for short
                    total_value = (self.qty * self.avg_price) - (qty * price)
                    self.avg_price = total_value / new_qty
                self.qty = new_qty
                return 0.0  # No realized PnL on position increase
            else:
                # Long position - this reduces exposure (sells long)
                if qty <= self.qty:
                    # Partial sell
                    realized = (price - self.avg_price) * qty
                    self.qty -= qty
                    if abs(self.qty) < 1e-6:
                        self.avg_price = 0.0
                    return realized
                else:
                    # Full sell plus new short
                    sell_qty = self.qty
                    new_short_qty = qty - sell_qty

                    # Realized PnL on the sell portion
                    realized = (price - self.avg_price) * sell_qty

                    # Set up new short position
                    self.qty = -new_short_qty
                    self.avg_price = price

                    return realized
        else:
            raise ValueError(f"Invalid side: {side}")

    def unrealized_pnl(self, price: float) -> float:
        """Calculate unrealized PnL at given price."""
        return self.qty * (price - self.avg_price)


@dataclass
class PortfolioState:
    """Portfolio state with cash, positions, and mark-to-market functionality."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    total_trades: int = 0
    trades: list[dict] = field(default_factory=list)
    ledger: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    shorting_enabled: bool = True

    def __post_init__(self):
        if not hasattr(self, "ledger") or self.ledger.empty:
            self.ledger = pd.DataFrame(
                columns=[
                    "date",
                    "equity",
                    "cash",
                    "gross_exposure",
                    "net_exposure",
                    "unrealized_pnl_total",
                    "realized_pnl_total",
                    "fees_paid",
                    "total_trades",
                ]
            )

    def value_at(self, prices_by_symbol: dict[str, float]) -> float:
        """Calculate portfolio value at given prices."""
        pos_val = 0.0
        for symbol, pos in self.positions.items():
            if pos.qty != 0:
                price = prices_by_symbol.get(symbol, pos.avg_price)
                pos_val += pos.qty * price
        return self.cash + pos_val

    def mark_to_market(self, date, prices_by_symbol):
        """Mark portfolio to market and record daily state."""
        unrealized_pnl_total = 0.0
        gross_exposure = 0.0
        net_exposure = 0.0

        # Calculate unrealized PnL for each position
        for symbol, position in self.positions.items():
            if symbol in prices_by_symbol:
                price = prices_by_symbol[symbol]
                unrealized = position.unrealized_pnl(price)
                unrealized_pnl_total += unrealized
                position_value = position.qty * price
                gross_exposure += abs(position_value)
                net_exposure += position_value

        # Calculate total equity
        equity = self.cash + net_exposure

        # Create MTM row
        mtm_row = {
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "unrealized_pnl_total": unrealized_pnl_total,
            "realized_pnl_total": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "total_trades": self.total_trades,
            "open_positions": self.get_open_positions_count(),
        }

        # Log MTM data
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime("%Y-%m-%d")

        logger.debug(
            f"MTM {date_str} equity=${equity:,.2f} cash=${self.cash:,.2f} unreal=${unrealized_pnl_total:,.2f} realized=${self.realized_pnl:,.2f} gross=${gross_exposure:,.0f} net=${net_exposure:,.0f}"
        )

        # Append to ledger
        self.ledger = pd.concat([self.ledger, pd.DataFrame([mtm_row])], ignore_index=True)

    def execute_order(
        self,
        symbol: str,
        target_qty: float,
        price: float,
        fee: float = 0.0,
        *,
        timestamp: pd.Timestamp | None = None,
        log_trade: bool = True,
    ) -> bool:
        """
        Execute an order to reach target quantity.

        Args:
            symbol: Trading symbol
            target_qty: Target quantity (negative for shorts)
            price: Execution price
            fee: Fee amount
            timestamp: Fill timestamp (timezone-naive)
            log_trade: Whether to append this fill to the trades log and count it

        Returns:
            True if trade was executed, False if blocked or no change
        """
        current_qty = self.positions.get(symbol, Position()).qty
        delta = target_qty - current_qty

        # Check if change is significant
        if abs(delta) < 1e-6:
            return False

        # Determine side and quantity
        if delta > 0:
            side = "BUY"
            qty = delta
        else:
            side = "SELL"
            qty = abs(delta)

        # Check guards for selling when flat or short (but allow short creation)
        if side == "SELL" and current_qty <= 0:
            # If we're trying to sell more than we have, and we're not creating a short position
            if abs(delta) > abs(current_qty) and (target_qty >= 0 or not self.shorting_enabled):
                logger.warning(
                    f"Cannot sell {abs(delta):.2f} of {symbol} - only {abs(current_qty):.2f} available"
                )
                return False
        elif side == "SELL" and current_qty > 0:
            # If we're trying to sell more than we have, and we're not creating a short position
            if abs(delta) > current_qty and (target_qty >= 0 or not self.shorting_enabled):
                logger.warning(
                    f"Cannot sell {abs(delta):.2f} of {symbol} - only {current_qty:.2f} available"
                )
                return False

        # Execute the trade
        pos = self.positions.setdefault(symbol, Position())
        before_qty = pos.qty

        # Apply fill to position
        realized_pnl = pos.apply_fill(side, qty, price, fee)

        # Update portfolio cash
        if side == "BUY":
            self.cash -= qty * price + fee
        else:
            self.cash += qty * price - fee

        # Update tracking
        self.realized_pnl += realized_pnl
        self.fees_paid += fee

        # Log trade if requested
        if log_trade:
            self.total_trades += 1
            trade_record = {
                "trade_id": self.total_trades,
                "timestamp": timestamp if timestamp is not None else pd.Timestamp.now(),
                "symbol": symbol,
                "side": side,
                "delta_qty": float(delta),
                "price": float(price),
                "fee": float(fee),
                "before_qty": float(before_qty),
                "after_qty": float(pos.qty),
                "realized_pnl": float(realized_pnl),
            }
            self.trades.append(trade_record)

        # Log debug
        logger.debug(
            f"TRADE fill {symbol} delta={delta:+.2f}@{price:.2f} before={before_qty:.2f} after={pos.qty:.2f} fee={fee:.2f}"
        )

        # Remove position if quantity is zero
        if abs(pos.qty) < 1e-6:
            self.positions.pop(symbol, None)

        return True

    def close_all_positions(
        self,
        prices_by_symbol: dict[str, float],
        fee: float = 0.0,
        *,
        timestamp: pd.Timestamp | None = None,
        log_trade: bool = True,
    ):
        """Close all positions at given prices."""
        for symbol, pos in list(self.positions.items()):
            if pos.qty != 0:
                if pos.qty > 0:
                    # Close long position
                    self.execute_order(
                        symbol,
                        0.0,
                        prices_by_symbol.get(symbol, pos.avg_price),
                        fee,
                        timestamp=timestamp,
                        log_trade=log_trade,
                    )
                else:
                    # Close short position
                    self.execute_order(
                        symbol,
                        0.0,
                        prices_by_symbol.get(symbol, pos.avg_price),
                        fee,
                        timestamp=timestamp,
                        log_trade=log_trade,
                    )

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        return self.positions.get(symbol)

    def get_open_positions_count(self) -> int:
        """Get count of open positions."""
        return len([pos for pos in self.positions.values() if abs(pos.qty) > 1e-6])

    def get_summary(self) -> dict:
        """Get portfolio summary."""
        return {
            "initial_capital": (
                self.ledger.iloc[0]["equity"] if not self.ledger.empty else self.cash
            ),
            "final_equity": (
                self.ledger.iloc[-1]["equity"] if not self.ledger.empty else self.cash
            ),
            "total_pnl": (
                (self.ledger.iloc[-1]["equity"] - self.ledger.iloc[0]["equity"])
                if not self.ledger.empty
                else 0.0
            ),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": (
                self.ledger.iloc[-1]["unrealized_pnl_total"] if not self.ledger.empty else 0.0
            ),
            "total_fees": self.fees_paid,
            "total_trades": self.total_trades,
            "open_positions": self.get_open_positions_count(),
        }
