"""
Local JSON store for order idempotency and deduplication.
"""

import json
import pathlib
import time
import uuid
from typing import Any

DB = pathlib.Path("reports/orders_kv.json")


def load() -> dict[str, Any]:
    """Load the orders KV store."""
    if DB.exists():
        return json.loads(DB.read_text())
    return {}


def save(data: dict[str, Any]) -> None:
    """Save the orders KV store."""
    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.write_text(json.dumps(data, indent=2))


def generate_client_order_id(symbol: str, run_id: str | None = None) -> str:
    """Generate a unique client order ID.

    Args:
        symbol: Trading symbol
        run_id: Optional run ID for additional uniqueness

    Returns:
        Client order ID in format: {timestamp}-{symbol}-{random}
    """
    timestamp = int(time.time() * 1e3)
    random_suffix = uuid.uuid4().hex[:6]

    if run_id:
        return f"{run_id}-{symbol}-{timestamp}-{random_suffix}"
    return f"{timestamp}-{symbol}-{random_suffix}"


def is_seen(client_order_id: str) -> bool:
    """Check if a client order ID has been seen before.

    Args:
        client_order_id: Client order ID to check

    Returns:
        True if the order ID exists in the store
    """
    return client_order_id in load()


def put_order(
    client_order_id: str, broker_order_id: str | None = None, status: str = "NEW"
) -> None:
    """Store an order in the KV store.

    Args:
        client_order_id: Client order ID
        broker_order_id: Broker's order ID (if available)
        status: Order status (NEW, SUBMITTED, FILLED, CANCELLED, etc.)
    """
    data = load()
    data[client_order_id] = {
        "broker_order_id": broker_order_id,
        "status": status,
        "timestamp": time.time(),
    }
    save(data)


def get_order(client_order_id: str) -> dict[str, Any] | None:
    """Get order details from the KV store.

    Args:
        client_order_id: Client order ID

    Returns:
        Order details or None if not found
    """
    data = load()
    return data.get(client_order_id)


def update_order_status(
    client_order_id: str, status: str, broker_order_id: str | None = None
) -> None:
    """Update order status in the KV store.

    Args:
        client_order_id: Client order ID
        status: New status
        broker_order_id: Broker order ID (if updating)
    """
    data = load()
    if client_order_id in data:
        data[client_order_id]["status"] = status
        if broker_order_id:
            data[client_order_id]["broker_order_id"] = broker_order_id
        data[client_order_id]["updated"] = time.time()
        save(data)


def cleanup_old_orders(max_age_hours: int = 24) -> int:
    """Clean up old orders from the KV store.

    Args:
        max_age_hours: Maximum age in hours before cleanup

    Returns:
        Number of orders cleaned up
    """
    data = load()
    cutoff_time = time.time() - (max_age_hours * 3600)

    cleaned = 0
    for client_order_id in list(data.keys()):
        order_time = data[client_order_id].get("timestamp", 0)
        if order_time < cutoff_time:
            del data[client_order_id]
            cleaned += 1

    if cleaned > 0:
        save(data)

    return cleaned
