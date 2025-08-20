import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from utils.orders_kv import (
    cleanup_old_orders,
    generate_client_order_id,
    get_order,
    is_seen,
    load,
    put_order,
    save,
    update_order_status,
)


def test_orders_kv_basic_operations():
    """Test basic KV store operations."""
    with tempfile.TemporaryDirectory() as tmpdir, patch("utils.orders_kv.DB", Path(tmpdir) / "test_orders_kv.json"):
            # Test empty store
            assert load() == {}

            # Test save and load
            test_data = {"order1": {"status": "NEW", "timestamp": 123.0}}
            save(test_data)
            assert load() == test_data

            # Test generate_client_order_id
            cloid = generate_client_order_id("SPY")
            assert "SPY" in cloid
            assert len(cloid.split("-")) >= 3

            # Test with run_id
            cloid_with_run = generate_client_order_id("TSLA", "run_123")
            assert "run_123" in cloid_with_run
            assert "TSLA" in cloid_with_run


def test_orders_kv_idempotency():
    """Test idempotency operations."""
    with tempfile.TemporaryDirectory() as tmpdir, patch("utils.orders_kv.DB", Path(tmpdir) / "test_orders_kv.json"):
        # Test is_seen
        assert not is_seen("nonexistent_order")

        # Test put_order
        put_order("order1", "broker_123", "SUBMITTED")
        assert is_seen("order1")

        # Test get_order
        order_data = get_order("order1")
        assert order_data["broker_order_id"] == "broker_123"
        assert order_data["status"] == "SUBMITTED"

        # Test update_order_status
        update_order_status("order1", "FILLED", "broker_123")
        updated_order = get_order("order1")
        assert updated_order["status"] == "FILLED"
        assert "updated" in updated_order


def test_orders_kv_cleanup():
    """Test cleanup of old orders."""
    with tempfile.TemporaryDirectory() as tmpdir, patch("utils.orders_kv.DB", Path(tmpdir) / "test_orders_kv.json"):
        # Add some orders
        put_order("old_order", status="FILLED")
        put_order("new_order", status="NEW")

        # Manually set old timestamp for old_order
        data = load()
        data["old_order"]["timestamp"] = time.time() - 25 * 3600  # 25 hours ago
        save(data)

        # Clean up orders older than 24 hours
        cleaned = cleanup_old_orders(max_age_hours=24)
        assert cleaned == 1

        # Verify old order is gone, new order remains
        assert not is_seen("old_order")
        assert is_seen("new_order")


def test_orders_kv_persistence():
    """Test that data persists between operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_orders_kv.json"
        with patch("utils.orders_kv.DB", db_path):
            # Add order
            put_order("persistent_order", "broker_456", "SUBMITTED")

            # Verify file exists and contains data
            assert db_path.exists()

            # Read file directly
            with open(db_path) as f:
                data = json.load(f)
                assert "persistent_order" in data
                assert data["persistent_order"]["broker_order_id"] == "broker_456"
