"""
Execution router for order submission with idempotency.
"""
from __future__ import annotations
import time
import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from brokers.interface import Broker
from utils.orders_kv import (
    generate_client_order_id, 
    is_seen, 
    put_order, 
    update_order_status
)


def backoff_with_jitter(attempt: int, max_delay: float = 5.0) -> None:
    """Exponential backoff with jitter for retries.
    
    Args:
        attempt: Retry attempt number (0-based)
        max_delay: Maximum delay in seconds
    """
    delay = min(2 ** attempt + random.random(), max_delay)
    time.sleep(delay)


def submit_order_with_idempotency(
    broker: Broker,
    symbol: str,
    side: str,
    qty: float,
    order_type: str = "market",
    run_id: Optional[str] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Submit order with idempotency and retry logic.
    
    Args:
        broker: Broker instance
        symbol: Trading symbol
        side: BUY or SELL
        qty: Quantity to trade
        order_type: Order type (market, limit, etc.)
        run_id: Optional run ID for client order ID generation
        max_retries: Maximum retry attempts
        
    Returns:
        Order result with client_order_id, broker_order_id, status, etc.
    """
    # Generate client order ID
    client_order_id = generate_client_order_id(symbol, run_id)
    
    # Check idempotency
    if is_seen(client_order_id):
        existing_order = put_order(client_order_id)
        return {
            "client_order_id": client_order_id,
            "broker_order_id": existing_order.get("broker_order_id"),
            "status": "DUPLICATE",
            "message": "Order already submitted",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Store order in KV store
    put_order(client_order_id, status="NEW")
    
    # Submit with retry logic
    for attempt in range(max_retries + 1):
        try:
            # Submit order to broker with client_order_id for idempotency
            result = broker.submit_order(symbol, side, qty, order_type, client_order_id)
            
            # Update KV store with broker response
            broker_order_id = result.get("order_id")
            perm_id = result.get("perm_id")
            status = result.get("status", "SUBMITTED")
            
            update_order_status(client_order_id, status, broker_order_id)
            
            # Add client_order_id to result
            result["client_order_id"] = client_order_id
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # Don't retry after venue acknowledgment
            if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                update_order_status(client_order_id, "DUPLICATE")
                return {
                    "client_order_id": client_order_id,
                    "status": "DUPLICATE",
                    "error": error_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Don't retry on auth errors
            if "unauthorized" in error_msg.lower() or "auth" in error_msg.lower():
                update_order_status(client_order_id, "AUTH_ERROR")
                return {
                    "client_order_id": client_order_id,
                    "status": "AUTH_ERROR",
                    "error": error_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Retry on rate limits and network errors
            if attempt < max_retries:
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    backoff_with_jitter(attempt)
                    continue
                elif "timeout" in error_msg.lower() or "network" in error_msg.lower():
                    backoff_with_jitter(attempt)
                    continue
                elif "5" in error_msg and "xx" in error_msg:  # 5xx errors
                    backoff_with_jitter(attempt)
                    continue
            
            # Final attempt failed
            update_order_status(client_order_id, "ERROR")
            return {
                "client_order_id": client_order_id,
                "status": "ERROR",
                "error": error_msg,
                "attempts": attempt + 1,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    # Should not reach here
    return {
        "client_order_id": client_order_id,
        "status": "ERROR",
        "error": "Max retries exceeded",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def cancel_order_with_idempotency(
    broker: Broker,
    client_order_id: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Cancel order with idempotency and retry logic.
    
    Args:
        broker: Broker instance
        client_order_id: Client order ID to cancel
        max_retries: Maximum retry attempts
        
    Returns:
        Cancel result
    """
    # Get broker order ID from KV store
    order_data = put_order(client_order_id)
    broker_order_id = order_data.get("broker_order_id")
    
    if not broker_order_id:
        return {
            "client_order_id": client_order_id,
            "status": "ERROR",
            "error": "No broker order ID found",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Cancel with retry logic
    for attempt in range(max_retries + 1):
        try:
            result = broker.cancel_order(broker_order_id)
            
            # Update KV store
            status = result.get("status", "CANCELLED")
            update_order_status(client_order_id, status)
            
            # Add client_order_id to result
            result["client_order_id"] = client_order_id
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # Don't retry on auth errors
            if "unauthorized" in error_msg.lower() or "auth" in error_msg.lower():
                return {
                    "client_order_id": client_order_id,
                    "status": "AUTH_ERROR",
                    "error": error_msg,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Retry on rate limits and network errors
            if attempt < max_retries:
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    backoff_with_jitter(attempt)
                    continue
                elif "timeout" in error_msg.lower() or "network" in error_msg.lower():
                    backoff_with_jitter(attempt)
                    continue
                elif "5" in error_msg and "xx" in error_msg:  # 5xx errors
                    backoff_with_jitter(attempt)
                    continue
            
            # Final attempt failed
            return {
                "client_order_id": client_order_id,
                "status": "ERROR",
                "error": error_msg,
                "attempts": attempt + 1,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    return {
        "client_order_id": client_order_id,
        "status": "ERROR",
        "error": "Max retries exceeded",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


