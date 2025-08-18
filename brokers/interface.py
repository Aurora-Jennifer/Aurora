from __future__ import annotations
from typing import Protocol, Dict, Any


class Broker(Protocol):
    def submit(self, order: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def cancel(self, order_id: str) -> bool:
        ...

    def positions(self) -> Dict[str, float]:
        ...

    def cash(self) -> float:
        ...

    def now(self):
        ...


def normalize_order(symbol: str, side: str, qty: float, px: float | None = None) -> Dict[str, Any]:
    return {
        "symbol": str(symbol),
        "side": str(side).upper(),
        "qty": float(qty),
        "px": (float(px) if px is not None else None),
    }


