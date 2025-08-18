#!/usr/bin/env python3
"""
Emergency position flattening script.
Closes all open positions with market orders.
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from brokers.interface import Broker


def get_broker() -> Broker:
    """Get broker instance based on environment."""
    venue = os.getenv("BROKER", "example")
    
    if venue == "example":
        from brokers.example_venue import ExampleVenueBroker
        return ExampleVenueBroker()
    elif venue == "ibkr":
        from brokers.ibkr import IBKRBroker, IBKRConfig
        import yaml
        
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
            qty_min=ibkr_cfg.get("ibkr", {}).get("qty_min", 1.0)
        )
        return IBKRBroker(config)
    else:
        raise ValueError(f"Unknown broker venue: {venue}")


def flatten_all_positions(broker: Broker, dry_run: bool = False) -> Dict[str, Any]:
    """Flatten all open positions with market orders."""
    positions = broker.get_positions()
    
    if not positions:
        return {"status": "no_positions", "message": "No open positions to flatten"}
    
    results = {
        "status": "flattening",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "positions": positions,
        "orders": []
    }
    
    for symbol, qty in positions.items():
        if abs(qty) < 0.01:  # Skip tiny positions
            continue
            
        # Determine side: if long (positive), sell; if short (negative), buy
        side = "SELL" if qty > 0 else "BUY"
        order_qty = abs(qty)
        
        if dry_run:
            order_result = {
                "symbol": symbol,
                "side": side,
                "qty": order_qty,
                "status": "dry_run",
                "order_id": None
            }
        else:
            order_result = broker.submit_order(symbol, side, order_qty, "market")
        
        results["orders"].append(order_result)
        
        if not dry_run and order_result.get("status") == "error":
            print(f"ERROR: Failed to flatten {symbol}: {order_result.get('error')}", file=sys.stderr)
    
    return results


def main():
    """Main flatten script."""
    import argparse
    
    ap = argparse.ArgumentParser(description="Flatten all open positions")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    ap.add_argument("--output", help="Output file for results (default: reports/flatten_YYYYMMDD_HHMMSS.json)")
    args = ap.parse_args()
    
    try:
        broker = get_broker()
        print(f"Connected to broker: {broker.__class__.__name__}")
        
        # Check connectivity
        cash = broker.get_cash()
        print(f"Available cash: ${cash:,.2f}")
        
        positions = broker.get_positions()
        if positions:
            print(f"Open positions: {positions}")
        else:
            print("No open positions")
            return 0
        
        # Flatten positions
        results = flatten_all_positions(broker, dry_run=args.dry_run)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = Path("reports") / f"flatten_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        
        print(f"Results saved to: {output_path}")
        
        if results["status"] == "flattening":
            print("✅ Position flattening completed")
        else:
            print(f"ℹ️  {results['message']}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
