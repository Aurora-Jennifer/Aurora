# utils/quotes_provider_ibkr.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict
import time

try:
    from ib_insync import IB, Stock  # pip install ib-insync
except ImportError:  # pragma: no cover
    IB = None
    Stock = None


@dataclass
class IBKRQuoteConfig:
    host: str = "127.0.0.1"
    port: int = 7497        # 7497 paper, 7496 live
    client_id: int = 321
    route: str = "SMART"
    currency: str = "USD"
    snap_ms: int = 150      # small wait for ticker to update


class IBKRQuoteProvider:
    def __init__(self, cfg: IBKRQuoteConfig):
        self.cfg = cfg
        if IB is None:
            raise RuntimeError("ib_insync not installed; `pip install ib-insync`")
        
        self.ib = IB()
        self.ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, readonly=True)
        self._contracts: Dict[str, Stock] = {}

    def _c(self, symbol: str) -> Stock:
        if symbol not in self._contracts:
            self._contracts[symbol] = Stock(symbol, exchange=self.cfg.route, currency=self.cfg.currency)
        return self._contracts[symbol]

    def quote(self, symbol: str) -> dict:
        """Get real-time quote for symbol."""
        c = self._c(symbol)
        t = self.ib.reqMktData(c, "", False, False)
        self.ib.sleep(self.cfg.snap_ms / 1000.0)
        
        bid = float(t.bid) if t.bid else float("nan")
        ask = float(t.ask) if t.ask else float("nan")
        mid = (bid + ask) / 2.0 if bid == bid and ask == ask else float("nan")
        
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def close(self):
        """Close connection."""
        try:
            self.ib.disconnect()
        except Exception:
            pass
