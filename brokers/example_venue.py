"""
Example broker adapter skeleton for live trading.
Replace with your actual venue implementation.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import requests

from .interface import Broker


class ExampleVenueBroker(Broker):
    """Example broker adapter - replace with your venue."""

    def __init__(self):
        # Read credentials from environment (never commit)
        self.api_key = os.getenv("BROKER_KEY")
        self.api_secret = os.getenv("BROKER_SECRET")
        self.account_id = os.getenv("BROKER_ACCOUNT_ID")
        self.base_url = os.getenv("BROKER_URL", "https://api.example.com")

        if not all([self.api_key, self.api_secret, self.account_id]):
            raise ValueError("Missing broker credentials in environment")

        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )

    def submit_order(
        self, symbol: str, side: str, qty: float, order_type: str = "market"
    ) -> dict[str, Any]:
        """Submit order to broker."""
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": abs(qty),
            "type": order_type.upper(),
            "account_id": self.account_id,
        }

        try:
            response = self.session.post(f"{self.base_url}/orders", json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "order_id": data.get("order_id"),
                "status": data.get("status", "submitted"),
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": symbol,
                "side": side,
                "qty": qty,
            }
        except Exception as e:
            return {
                "order_id": None,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel order."""
        try:
            response = self.session.delete(f"{self.base_url}/orders/{order_id}", timeout=10)
            response.raise_for_status()

            return {
                "order_id": order_id,
                "status": "cancelled",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {
                "order_id": order_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def get_positions(self) -> dict[str, float]:
        """Get current positions."""
        try:
            response = self.session.get(
                f"{self.base_url}/accounts/{self.account_id}/positions", timeout=10
            )
            response.raise_for_status()
            data = response.json()

            positions = {}
            for pos in data.get("positions", []):
                symbol = pos.get("symbol")
                qty = float(pos.get("quantity", 0))
                if symbol and qty != 0:
                    positions[symbol] = qty

            return positions
        except Exception:
            return {}

    def get_cash(self) -> float:
        """Get available cash."""
        try:
            response = self.session.get(f"{self.base_url}/accounts/{self.account_id}", timeout=10)
            response.raise_for_status()
            data = response.json()

            return float(data.get("cash", 0.0))
        except Exception:
            return 0.0

    def get_fills(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """Get recent fills."""
        try:
            params = {}
            if since:
                params["since"] = since.isoformat()

            response = self.session.get(
                f"{self.base_url}/accounts/{self.account_id}/fills", params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            fills = []
            for fill in data.get("fills", []):
                fills.append(
                    {
                        "symbol": fill.get("symbol"),
                        "side": fill.get("side"),
                        "qty": float(fill.get("quantity", 0)),
                        "price": float(fill.get("price", 0)),
                        "timestamp": fill.get("timestamp"),
                        "order_id": fill.get("order_id"),
                    }
                )

            return fills
        except Exception:
            return []

    def now(self) -> datetime:
        """Get broker time."""
        try:
            response = self.session.get(f"{self.base_url}/time", timeout=5)
            response.raise_for_status()
            data = response.json()

            # Parse broker timestamp
            broker_time = datetime.fromisoformat(data.get("timestamp", "").replace("Z", "+00:00"))
            return broker_time
        except Exception:
            # Fallback to local time
            return datetime.now(UTC)
