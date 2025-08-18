"""
Trade Logger with Lifecycle Tracking
Handles trade records with partial closes and proper PnL calculation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Complete trade record with entry/exit tracking."""

    symbol: str
    entry_date: str
    entry_qty: float = 0.0
    entry_vwap: float = 0.0
    cum_entry_notional: float = 0.0
    cum_entry_qty: float = 0.0
    exit_date: Optional[str] = None
    exit_qty: float = 0.0
    exit_vwap: float = 0.0
    cum_exit_notional: float = 0.0
    cum_fees: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown_pct: float = 0.0  # while open
    is_closed: bool = False

    def __post_init__(self):
        if self.entry_qty < 0:
            logger.warning(f"Negative entry quantity: {self.entry_qty}")
            self.entry_qty = 0.0


class TradeBook:
    """Trade book managing open and closed trades."""

    def __init__(self):
        self.open: Dict[str, TradeRecord] = {}
        self.closed: List[TradeRecord] = []
        self.logger = logging.getLogger(__name__)

    def on_buy(self, date: str, symbol: str, qty: float, price: float, fees: float):
        """Record a buy order."""
        tr = self.open.get(symbol)
        if tr is None:
            tr = TradeRecord(symbol=symbol, entry_date=date, entry_qty=0, entry_vwap=0)
            self.open[symbol] = tr

        # Update cumulative entry data
        tr.cum_entry_notional += qty * price
        tr.cum_entry_qty += qty
        tr.entry_qty = tr.cum_entry_qty
        tr.entry_vwap = tr.cum_entry_notional / max(tr.cum_entry_qty, 1e-12)
        tr.cum_fees += fees

        self.logger.info(
            f"BUY: {symbol} {qty} @ ${price:.2f}, VWAP: ${tr.entry_vwap:.2f}"
        )

    def on_sell(
        self,
        date: str,
        symbol: str,
        qty: float,
        price: float,
        fees: float,
        remaining_qty_after_sell: float,
    ):
        """Record a sell order with partial close handling."""
        tr = self.open.get(symbol)
        if tr is None:
            self.logger.warning(f"No open position for {symbol} to sell")
            return

        real_qty = qty  # qty actually sold
        tr.cum_exit_notional += real_qty * price
        tr.exit_qty += real_qty
        tr.cum_fees += fees

        # Calculate realized PnL for this partial
        realized_partial = (price - tr.entry_vwap) * real_qty
        tr.realized_pnl += realized_partial

        self.logger.info(
            f"SELL: {symbol} {real_qty} @ ${price:.2f}, Realized: ${realized_partial:.2f}"
        )

        if remaining_qty_after_sell == 0.0:  # trade fully closed
            tr.exit_date = date
            tr.exit_vwap = tr.cum_exit_notional / max(tr.exit_qty, 1e-12)
            tr.is_closed = True

            # Move to closed trades
            self.closed.append(tr)
            self.open.pop(symbol, None)

            self.logger.info(
                f"CLOSED: {symbol} trade, Total PnL: ${tr.realized_pnl:.2f}"
            )

    def reset(self):
        """Reset the trade book (clear all trades)."""
        self.open.clear()
        self.closed.clear()
        self.logger.info("Trade book reset")

    def mark_drawdown(self, symbol: str, mtm_pnl_pct: float):
        """Track worst drawdown while position is open."""
        tr = self.open.get(symbol)
        if tr:
            tr.max_drawdown_pct = min(tr.max_drawdown_pct, mtm_pnl_pct)

    def get_open_positions(self) -> Dict[str, TradeRecord]:
        """Get all open positions."""
        return self.open.copy()

    def get_closed_trades(self) -> List[TradeRecord]:
        """Get all closed trades."""
        return self.closed.copy()

    def get_ledger(self) -> pd.DataFrame:
        """Get trade ledger as DataFrame."""
        trades_data = self.export_trades_csv()
        if not trades_data:
            return pd.DataFrame()
        return pd.DataFrame(trades_data)

    def get_trades(self) -> List[Dict]:
        """Get all trades as list of dictionaries."""
        return self.export_trades_csv()

    def get_trade_summary(self) -> Dict:
        """Get summary statistics of all trades."""
        total_trades = len(self.closed)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": "N/A",
                "total_pnl": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "open_positions": len(self.open),
            }

        pnls = [t.realized_pnl for t in self.closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        sum_pos = sum(wins)
        sum_neg = abs(sum(losses))

        if sum_neg == 0:
            profit_factor = "N/A"
        else:
            profit_factor = sum_pos / sum_neg

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": sum(pnls),
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
            "avg_win": (sum_pos / len(wins)) if wins else 0.0,
            "avg_loss": (-sum(losses) / len(losses)) if losses else 0.0,
            "open_positions": len(self.open),
        }

    def export_trades_csv(self) -> List[Dict]:
        """Export trades for CSV output."""
        trades_data = []

        # Add closed trades
        for tr in self.closed:
            trades_data.append(
                {
                    "date": tr.exit_date,
                    "symbol": tr.symbol,
                    "action": "CLOSE",
                    "qty": tr.exit_qty,
                    "price": tr.exit_vwap,
                    "fees": tr.cum_fees,
                    "entry_vwap": tr.entry_vwap,
                    "exit_vwap": tr.exit_vwap,
                    "realized_pnl": tr.realized_pnl,
                    "position_after": 0.0,
                    "trade_status": "CLOSED",
                }
            )

        # Add open positions
        for symbol, tr in self.open.items():
            trades_data.append(
                {
                    "date": tr.entry_date,
                    "symbol": tr.symbol,
                    "action": "OPEN",
                    "qty": tr.entry_qty,
                    "price": tr.entry_vwap,
                    "fees": tr.cum_fees,
                    "entry_vwap": tr.entry_vwap,
                    "exit_vwap": 0.0,
                    "realized_pnl": tr.realized_pnl,
                    "position_after": tr.entry_qty,
                    "trade_status": "OPEN",
                }
            )

        return trades_data
