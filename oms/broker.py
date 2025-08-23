from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Order:
    id: str
    symbol: str
    side: str  # BUY/SELL
    qty: int
    type: str = "MKT"
    limit: float | None = None
    tif: str = "DAY"
    intended_ts: float | None = None
    idempotency_key: str | None = None
    meta: dict[str, Any] = None


@dataclass
class Fill:
    order_id: str
    qty: int
    price: float
    ts: float
    liquidity_flag: str = ""


class BrokerAdapter:
    def get_quote(self, symbol: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def place(self, order: Order) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def cancel(self, order_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def status(self, order_id: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def positions(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def cash(self) -> float:  # pragma: no cover
        raise NotImplementedError


