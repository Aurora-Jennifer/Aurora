"""
Tests for realtime feed with execution trust validation.

Validates:
- Single-cycle per bar (no duplicates)
- Stale feed detection and kill switch
- Heartbeat monitoring
- Feature flag behavior
"""

import os
import pytest
import time
import pandas as pd

from brokers.realtime_feed import RealtimeFeed, is_realtime_enabled, check_trading_halted


class TestRealtimeFeed:
    """Test suite for RealtimeFeed execution trust features."""
    
    def test_init(self):
        """Test feed initialization."""
        feed = RealtimeFeed("BTCUSDT", "1m", testnet=True)
        assert feed.symbol == "BTCUSDT"
        assert feed.interval == "1m"
        assert feed.testnet is True
        assert feed.heartbeat_timeout == 5.0
        assert feed.trading_halted is False
        assert feed.last_msg_ts is None
    
    def test_feature_flags(self):
        """Test feature flag behavior."""
        # Default disabled
        os.environ.pop("FLAG_REALTIME", None)
        assert is_realtime_enabled() is False
        
        # Explicitly enabled
        os.environ["FLAG_REALTIME"] = "1"
        assert is_realtime_enabled() is True
        
        # Trading halt flag
        os.environ.pop("FLAG_TRADING_HALTED", None)
        assert check_trading_halted() is False
        
        os.environ["FLAG_TRADING_HALTED"] = "1"
        assert check_trading_halted() is True
        
        # Cleanup
        os.environ.pop("FLAG_REALTIME", None)
        os.environ.pop("FLAG_TRADING_HALTED", None)
    
    def test_staleness_detection(self):
        """Test stale feed detection."""
        feed = RealtimeFeed(heartbeat_timeout=2.0)
        
        # No messages yet - should be stale
        assert feed.is_stale() is True
        
        # Fresh message
        feed.last_msg_ts = time.time()
        assert feed.is_stale() is False
        
        # Old message (simulate 3s ago)
        feed.last_msg_ts = time.time() - 3.0
        assert feed.is_stale() is True
    
    def test_kill_switch(self):
        """Test trading halt/resume functionality."""
        feed = RealtimeFeed()
        halt_calls = []
        
        feed.on_halt = lambda reason: halt_calls.append(reason)
        
        # Initially not halted
        assert feed.trading_halted is False
        assert "FLAG_TRADING_HALTED" not in os.environ
        
        # Halt trading
        feed.halt_trading("Test halt")
        assert feed.trading_halted is True
        assert os.environ["FLAG_TRADING_HALTED"] == "1"
        assert len(halt_calls) == 1
        assert "Test halt" in halt_calls[0]
        
        # Resume trading
        feed.resume_trading()
        assert feed.trading_halted is False
        assert "FLAG_TRADING_HALTED" not in os.environ
    
    def test_kline_parsing(self):
        """Test Binance kline message parsing."""
        feed = RealtimeFeed("BTCUSDT")
        
        # Valid closed kline
        msg = {
            "k": {
                "t": 1699520400000,  # 2023-11-09 09:00:00 UTC
                "o": "35000.00",
                "h": "35100.00", 
                "l": "34900.00",
                "c": "35050.00",
                "v": "123.45",
                "x": True  # Closed bar
            }
        }
        
        bar = feed._parse_kline(msg)
        assert bar is not None
        assert bar["symbol"] == "BTCUSDT"
        assert bar["open"] == 35000.0
        assert bar["high"] == 35100.0
        assert bar["low"] == 34900.0
        assert bar["close"] == 35050.0
        assert bar["volume"] == 123.45
        assert isinstance(bar["timestamp"], pd.Timestamp)
        
        # Unclosed kline should be ignored
        msg["k"]["x"] = False
        bar = feed._parse_kline(msg)
        assert bar is None
    
    def test_duplicate_timestamp_skipping(self):
        """Test that duplicate timestamps are skipped."""
        feed = RealtimeFeed("BTCUSDT")
        
        msg1 = {
            "k": {
                "t": 1699520400000,  # Same timestamp
                "o": "35000.00", "h": "35100.00", "l": "34900.00", "c": "35050.00", "v": "123.45",
                "x": True
            }
        }
        
        msg2 = {
            "k": {
                "t": 1699520400000,  # Same timestamp (duplicate)
                "o": "35010.00", "h": "35110.00", "l": "34910.00", "c": "35060.00", "v": "124.45",
                "x": True
            }
        }
        
        # First bar should parse
        bar1 = feed._parse_kline(msg1)
        assert bar1 is not None
        
        # Duplicate timestamp should be skipped
        bar2 = feed._parse_kline(msg2)
        assert bar2 is None
    
    def test_stats_collection(self):
        """Test feed statistics collection."""
        feed = RealtimeFeed("BTCUSDT", "5m")
        feed.last_msg_ts = time.time()
        feed.last_bar_ts = pd.Timestamp.now(tz="UTC")
        
        stats = feed.get_stats()
        assert stats["symbol"] == "BTCUSDT"
        assert stats["interval"] == "5m"
        assert stats["last_msg_ts"] == feed.last_msg_ts
        assert stats["trading_halted"] is False
        assert stats["is_stale"] is False
        assert stats["staleness_sec"] < 1.0
    
    def test_heartbeat_monitor_logic(self):
        """Test heartbeat monitoring logic (without async)."""
        feed = RealtimeFeed(heartbeat_timeout=0.1)
        halt_calls = []
        
        feed.on_halt = lambda reason: halt_calls.append(reason)
        
        # Test initial state (no messages - should be stale)
        assert feed.is_stale() is True
        
        # Manually trigger halt logic
        if feed.is_stale() and not feed.trading_halted:
            feed.halt_trading("Feed stale: test")
        
        assert feed.trading_halted is True
        assert len(halt_calls) == 1
        assert "stale" in halt_calls[0].lower()
        
        # Send a fresh message
        feed.last_msg_ts = time.time()
        assert feed.is_stale() is False
        
        # Manually trigger resume logic 
        if not feed.is_stale() and feed.trading_halted:
            feed.resume_trading()
            
        assert feed.trading_halted is False


class TestIntegration:
    """Integration tests for realtime feed behavior."""
    
    def test_single_cycle_per_bar(self):
        """Test that each bar triggers exactly one decision cycle."""
        feed = RealtimeFeed("BTCUSDT")
        bar_calls = []
        
        feed.on_bar = lambda bar: bar_calls.append(bar)
        
        # Simulate 3 bars with different timestamps
        bars = [
            {"k": {"t": 1699520400000, "o": "35000", "h": "35100", "l": "34900", "c": "35050", "v": "123", "x": True}},
            {"k": {"t": 1699520460000, "o": "35050", "h": "35150", "l": "34950", "c": "35100", "v": "124", "x": True}},
            {"k": {"t": 1699520520000, "o": "35100", "h": "35200", "l": "35000", "c": "35150", "v": "125", "x": True}}
        ]
        
        for msg in bars:
            bar = feed._parse_kline(msg)
            if bar and feed.on_bar:
                feed.on_bar(bar)
        
        # Should have exactly 3 bar calls (one per unique timestamp)
        assert len(bar_calls) == 3
        
        # Verify ascending timestamps
        timestamps = [bar["timestamp"] for bar in bar_calls]
        assert timestamps == sorted(timestamps)
    
    def test_golden_snapshot_compatibility(self):
        """Test that static mode still works (no breaking changes)."""
        # This test ensures existing functionality isn't broken
        # by realtime infrastructure additions
        
        # Feature flag should be off by default
        os.environ.pop("FLAG_REALTIME", None)
        assert is_realtime_enabled() is False
        
        # No trading halt by default
        os.environ.pop("FLAG_TRADING_HALTED", None)
        assert check_trading_halted() is False
        
        # Feed creation shouldn't change environment
        feed = RealtimeFeed("BTCUSDT")
        assert is_realtime_enabled() is False
        assert check_trading_halted() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
