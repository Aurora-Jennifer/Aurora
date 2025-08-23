from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .broker import BrokerAdapter, Order


class PaperAdapter(BrokerAdapter):
    def __init__(self, out_dir: str = "artifacts/paper") -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def get_quote(self, symbol: str) -> dict[str, Any]:  # toy mid
        return {"symbol": symbol, "mid": 100.0, "bid": 99.99, "ask": 100.01, "ts": time.time()}

    def place(self, order: Order) -> dict[str, Any]:
        # Immediate fill at mid with small slip
        q = self.get_quote(order.symbol)
        price = float(q["mid"]) * 1.0001
        fill = {
            "order_id": order.id,
            "qty": int(order.qty),
            "price": price,
            "ts": time.time(),
            "status": "FILLED",
        }
        with open(self.out / "fills.jsonl", "a") as f:
            f.write(json.dumps(fill) + "\n")
        return {"status": "FILLED", "fill": fill}

    def cancel(self, order_id: str) -> bool:
        return True

    def status(self, order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "status": "FILLED"}

    def positions(self) -> dict[str, Any]:
        return {}

    def cash(self) -> float:
        return 0.0


