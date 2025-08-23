#!/usr/bin/env python3
"""
Order reconciliation script.
Compares local order journal vs venue orders/positions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.orders_kv import load as load_orders_kv

if TYPE_CHECKING:
    from brokers.interface import Broker


def get_broker() -> Broker:
    """Get broker instance based on environment."""
    venue = os.getenv("BROKER", "example")

    if venue == "example":
        from brokers.example_venue import ExampleVenueBroker

        return ExampleVenueBroker()
    if venue == "ibkr":
        import yaml

        from brokers.ibkr import IBKRBroker, IBKRConfig

        # Load IBKR config from overlay
        ibkr_cfg = yaml.safe_load(Path("config/brokers/ibkr.yaml").read_text())

        # Merge configs
        config = IBKRConfig(
            host=ibkr_cfg.get("ibkr", {}).get("host", "127.0.0.1"),
            port=ibkr_cfg.get("ibkr", {}).get("port", 7497),
            client_id=ibkr_cfg.get("ibkr", {}).get("client_id", 123),
            account=ibkr_cfg.get("ibkr", {}).get("account"),
            route=ibkr_cfg.get("ibkr", {}).get("route", "SMART"),
            currency=ibkr_cfg.get("ibkr", {}).get("currency", "USD"),
            allow_fractional=ibkr_cfg.get("ibkr", {}).get("allow_fractional", False),
            px_tick=ibkr_cfg.get("ibkr", {}).get("px_tick", 0.01),
            qty_min=ibkr_cfg.get("ibkr", {}).get("qty_min", 1.0),
        )
        return IBKRBroker(config)
    raise ValueError(f"Unknown broker venue: {venue}")


def load_local_orders() -> dict[str, Any]:
    """Load local order journal from JSONL files."""
    local_orders = {}

    # Load from orders KV store
    kv_orders = load_orders_kv()
    for client_order_id, order_data in kv_orders.items():
        local_orders[client_order_id] = {
            "client_order_id": client_order_id,
            "broker_order_id": order_data.get("broker_order_id"),
            "status": order_data.get("status", "UNKNOWN"),
            "timestamp": order_data.get("timestamp"),
        }

    # Also load from trade logs if they exist
    trade_logs = list(Path("logs/trades").glob("*.jsonl"))
    for log_file in trade_logs[-3:]:  # Last 3 days
        for line in log_file.read_text().splitlines():
            if line.strip():
                try:
                    trade = json.loads(line)
                    client_order_id = trade.get("client_order_id")
                    if client_order_id and client_order_id not in local_orders:
                        local_orders[client_order_id] = {
                            "client_order_id": client_order_id,
                            "broker_order_id": trade.get("broker_order_id"),
                            "status": "FILLED",
                            "timestamp": trade.get("timestamp"),
                        }
                except json.JSONDecodeError:
                    continue

    return local_orders


def get_venue_orders(broker: Broker) -> dict[str, Any]:
    """Get open orders from venue."""
    try:
        # This would need to be implemented in your actual broker adapter
        # For now, return empty dict for example broker
        return {}
    except Exception as e:
        print(f"ERROR: Failed to get venue orders: {e}", file=sys.stderr)
        return {}


def get_venue_positions(broker: Broker) -> dict[str, float]:
    """Get current positions from venue."""
    try:
        return broker.get_positions()
    except Exception as e:
        print(f"ERROR: Failed to get venue positions: {e}", file=sys.stderr)
        return {}


def reconcile_orders(local_orders: dict[str, Any], venue_orders: dict[str, Any]) -> dict[str, Any]:
    """Reconcile local vs venue orders."""
    local_ids = set(local_orders.keys())
    venue_ids = set(venue_orders.keys())

    missing_local = list(venue_ids - local_ids)
    missing_venue = list(local_ids - venue_ids)

    # Check for status mismatches
    status_mismatches = []
    for order_id in local_ids & venue_ids:
        local_status = local_orders[order_id].get("status")
        venue_status = venue_orders[order_id].get("status")
        if local_status != venue_status:
            status_mismatches.append(
                {"order_id": order_id, "local_status": local_status, "venue_status": venue_status}
            )

    return {
        "missing_local": missing_local,
        "missing_venue": missing_venue,
        "status_mismatches": status_mismatches,
        "local_count": len(local_orders),
        "venue_count": len(venue_orders),
    }


def reconcile_positions(
    local_positions: dict[str, float], venue_positions: dict[str, float]
) -> dict[str, Any]:
    """Reconcile local vs venue positions."""
    all_symbols = set(local_positions.keys()) | set(venue_positions.keys())

    qty_diffs = {}
    for symbol in all_symbols:
        local_qty = local_positions.get(symbol, 0.0)
        venue_qty = venue_positions.get(symbol, 0.0)

        if abs(local_qty - venue_qty) > 0.001:  # Allow for small precision differences
            qty_diffs[symbol] = {
                "local": local_qty,
                "venue": venue_qty,
                "diff": venue_qty - local_qty,
            }

    return {
        "qty_diffs": qty_diffs,
        "local_positions": local_positions,
        "venue_positions": venue_positions,
    }


def main():
    """Main reconciliation script."""
    ap = argparse.ArgumentParser(description="Reconcile local orders vs venue")
    ap.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without executing"
    )
    ap.add_argument(
        "--output", help="Output file for results (default: reports/reconcile_YYYYMMDD_HHMMSS.json)"
    )
    args = ap.parse_args()

    try:
        broker = get_broker()
        print(f"Connected to broker: {broker.__class__.__name__}")

        # Load local orders
        local_orders = load_local_orders()
        print(f"Local orders: {len(local_orders)}")

        # Get venue orders (if supported)
        venue_orders = get_venue_orders(broker)
        print(f"Venue orders: {len(venue_orders)}")

        # Reconcile orders
        order_diffs = reconcile_orders(local_orders, venue_orders)

        # Get and reconcile positions
        venue_positions = get_venue_positions(broker)
        print(f"Venue positions: {venue_positions}")

        # For now, assume local positions are empty (would need to track these)
        local_positions = {}
        position_diffs = reconcile_positions(local_positions, venue_positions)

        # Combine results
        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "broker": broker.__class__.__name__,
            "order_reconciliation": order_diffs,
            "position_reconciliation": position_diffs,
            "has_differences": bool(
                order_diffs["missing_local"]
                or order_diffs["missing_venue"]
                or order_diffs["status_mismatches"]
                or position_diffs["qty_diffs"]
            ),
        }

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_path = Path("reports") / f"reconcile_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))

        print(f"Results saved to: {output_path}")

        # Check for differences
        if results["has_differences"]:
            print("❌ RECONCILIATION FAILED - Differences found:")
            print(json.dumps(results, indent=2))
            if not args.dry_run:
                return 1
        else:
            print("✅ Reconciliation passed - No differences found")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
