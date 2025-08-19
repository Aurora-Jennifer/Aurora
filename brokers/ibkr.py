# brokers/ibkr.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

try:
    # Optional at runtime; tests will mock this out
    from ib_insync import IB, MarketOrder, Stock
except Exception:  # pragma: no cover
    IB = None  # type: ignore

# ---- Config ----


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 = paper, 7496 = live
    client_id: int = 123
    account: str | None = None
    route: str = "SMART"
    currency: str = "USD"
    allow_fractional: bool = False
    px_tick: float = 0.01
    qty_min: float = 1.0


# ---- Adapter ----


class IBKRBroker:
    """
    Minimal IBKR adapter using ib_insync (TWS / IB Gateway).
    Designed to satisfy your Broker Protocol (submit/cancel/positions/cash/now).
    """

    def __init__(self, cfg: IBKRConfig):
        self.cfg = cfg
        if IB is None:
            raise RuntimeError("ib_insync not installed; `pip install ib-insync` or mock in tests.")
        self.ib = IB()
        self._connected = False

    # --- lifecycle ---
    def connect(self):
        if not self._connected:
            self.ib.connect(
                self.cfg.host, self.cfg.port, clientId=self.cfg.client_id, readonly=False
            )
            self._connected = True
        return self

    def disconnect(self):
        if self._connected:
            self.ib.disconnect()
            self._connected = False

    # --- helpers ---
    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat()

    def _mk_contract(self, symbol: str) -> Any:
        # Contract by symbol; for production, prefer conid or set primaryExchange.
        return Stock(symbol, exchange=self.cfg.route, currency=self.cfg.currency)

    def _round_qty(self, qty: float) -> float:
        if self.cfg.allow_fractional:
            return float(qty)
        # Whole-share rounding
        return float(int(max(qty, self.cfg.qty_min)))

    # --- Broker API ---

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        client_order_id: str = None,
    ) -> dict[str, Any]:
        """
        Submit order to IBKR.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            qty: Quantity to trade
            order_type: MKT or LMT
            client_order_id: Optional client order ID for idempotency

        Returns:
            Dict with status, broker_order_id, ack_ms, etc.
        """
        self.connect()

        side = side.upper()
        qty = self._round_qty(float(qty))
        otype = order_type.upper()

        contract = self._mk_contract(symbol)
        if otype in ["MKT", "MARKET"]:
            ib_order = MarketOrder(action=side, totalQuantity=qty, tif="DAY")
        elif otype in ["LMT", "LIMIT"]:
            # For limit orders, we'd need limit_price parameter
            # For now, use market order
            ib_order = MarketOrder(action=side, totalQuantity=qty, tif="DAY")
        else:
            raise ValueError(f"Unsupported order type: {otype}")

        # Set orderRef for idempotency
        if client_order_id:
            ib_order.orderRef = client_order_id

        # Submit and time the ACK
        t0 = time.perf_counter()
        trade = self.ib.placeOrder(contract, ib_order)  # returns Trade
        # Wait (short) for orderStatus to arrive; bounded
        self.ib.sleep(0.05)
        ack_ms = int((time.perf_counter() - t0) * 1000)

        # Derive identifiers
        broker_order_id = getattr(trade.order, "orderId", None)
        perm_id = getattr(trade.order, "permId", None)
        actual_client_order_id = (
            getattr(trade.order, "orderRef", None) or client_order_id or str(broker_order_id)
        )

        return {
            "order_id": broker_order_id,
            "status": "ACK",
            "broker_order_id": broker_order_id,
            "perm_id": perm_id,
            "client_order_id": actual_client_order_id,
            "ack_ms": ack_ms,
            "timestamp": self._utc_now(),
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel order by broker order ID."""
        self.connect()
        # Find Trade by orderId
        trade = next(
            (t for t in self.ib.trades() if getattr(t.order, "orderId", None) == order_id), None
        )
        if not trade:
            return {
                "status": "NOT_FOUND",
                "broker_order_id": order_id,
                "timestamp": self._utc_now(),
            }
        self.ib.cancelOrder(trade.order)
        return {"status": "CANCEL_SENT", "broker_order_id": order_id, "timestamp": self._utc_now()}

    def get_positions(self) -> dict[str, float]:
        """Get current positions by symbol."""
        self.connect()
        pos = (
            self.ib.positions(account=self.cfg.account) if self.cfg.account else self.ib.positions()
        )
        out: dict[str, float] = {}
        for p in pos:
            sym = getattr(p.contract, "symbol", None) or getattr(p.contract, "localSymbol", None)
            out[sym] = out.get(sym, 0.0) + float(p.position)
        return out

    def get_cash(self) -> float:
        """Get available cash balance."""
        self.connect()
        # Account summary tags: https://interactivebrokers.github.io/tws-api/account_updates.html
        summary = (
            self.ib.accountSummary(self.cfg.account)
            if self.cfg.account
            else self.ib.accountSummary()
        )
        total_cash = 0.0
        for s in summary:
            if s.tag == "TotalCashValue" and s.currency == self.cfg.currency:
                total_cash = float(s.value)
                break
        return total_cash

    def get_fills(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """Get recent fills."""
        self.connect()
        fills = []
        for trade in self.ib.trades():
            if trade.orderStatus.status == "Filled":
                fill_time = getattr(trade.orderStatus, "updateTime", None)
                if fill_time:
                    try:
                        # Parse IBKR timestamp format
                        fill_dt = datetime.strptime(fill_time, "%Y%m%d %H:%M:%S")
                        if since and fill_dt < since:
                            continue
                    except ValueError:
                        pass

                fills.append(
                    {
                        "symbol": getattr(trade.contract, "symbol", ""),
                        "side": getattr(trade.order, "action", ""),
                        "qty": float(getattr(trade.orderStatus, "filled", 0)),
                        "price": float(getattr(trade.orderStatus, "avgFillPrice", 0)),
                        "timestamp": fill_time,
                        "order_id": getattr(trade.order, "orderId", ""),
                    }
                )
        return fills

    def now(self) -> datetime:
        """Get current broker time."""
        return datetime.now(UTC)
