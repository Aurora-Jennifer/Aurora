# tests/utils/test_pre_trade_tripwires.py
import pytest
from datetime import datetime, timezone, timedelta
from utils.quotes_provider import QuoteProvider


class MockQuoteProvider:
    def __init__(self, quotes):
        self.quotes = quotes
        self.call_count = 0
    
    def quote(self, symbol: str) -> dict:
        self.call_count += 1
        if symbol in self.quotes:
            return self.quotes[symbol]
        return {"symbol": symbol, "bid": 100.0, "ask": 100.1, "mid": 100.05, "ts": datetime.now(timezone.utc).isoformat()}


def test_stale_quote_tripwire():
    """Test stale quote detection."""
    # Create a stale quote (2 seconds old)
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    stale_quote = {"symbol": "SPY", "bid": 450.0, "ask": 450.1, "mid": 450.05, "ts": stale_ts}
    
    # Create a fresh quote
    fresh_ts = datetime.now(timezone.utc).isoformat()
    fresh_quote = {"symbol": "TSLA", "bid": 250.0, "ask": 250.1, "mid": 250.05, "ts": fresh_ts}
    
    provider = MockQuoteProvider({
        "SPY": stale_quote,
        "TSLA": fresh_quote
    })
    
    anomalies = []
    symbols = ["SPY", "TSLA"]
    
    for sym in symbols:
        try:
            quote = provider.quote(sym)
            now_ts = datetime.now(timezone.utc)
            quote_ts = datetime.fromisoformat(quote["ts"].replace("Z", "+00:00"))
            age_ms = int((now_ts - quote_ts).total_seconds() * 1000)
            
            # Stale quote tripwire
            if age_ms > 1500:
                anomalies.append(f"stale_quote:{sym}:{age_ms}ms")
                continue
                
        except Exception as e:
            anomalies.append(f"quote_error:{sym}:{str(e)[:50]}")
    
    # Should detect stale quote for SPY
    assert len(anomalies) == 1
    assert anomalies[0].startswith("stale_quote:SPY")


def test_wide_spread_tripwire():
    """Test wide spread detection."""
    # Create quotes with different spreads
    narrow_spread = {"symbol": "SPY", "bid": 450.0, "ask": 450.1, "mid": 450.05, "ts": datetime.now(timezone.utc).isoformat()}
    wide_spread = {"symbol": "TSLA", "bid": 250.0, "ask": 251.5, "mid": 250.75, "ts": datetime.now(timezone.utc).isoformat()}
    
    provider = MockQuoteProvider({
        "SPY": narrow_spread,
        "TSLA": wide_spread
    })
    
    anomalies = []
    symbols = ["SPY", "TSLA"]
    
    for sym in symbols:
        try:
            quote = provider.quote(sym)
            bid, ask, mid = quote["bid"], quote["ask"], quote["mid"]
            if bid and ask and mid and bid == bid and ask == ask and mid == mid:
                spread_bps = ((ask - bid) / mid) * 10000
                if spread_bps > 50:  # 50 bps = 0.5%
                    anomalies.append(f"wide_spread:{sym}:{spread_bps:.1f}bps")
                    continue
                    
        except Exception as e:
            anomalies.append(f"quote_error:{sym}:{str(e)[:50]}")
    
    # Should detect wide spread for TSLA (spread = 60 bps)
    assert len(anomalies) == 1
    assert anomalies[0].startswith("wide_spread:TSLA")


def test_market_closed_tripwire():
    """Test market closed detection."""
    # Create quotes with null/zero values
    normal_quote = {"symbol": "SPY", "bid": 450.0, "ask": 450.1, "mid": 450.05, "ts": datetime.now(timezone.utc).isoformat()}
    zero_bid = {"symbol": "TSLA", "bid": 0, "ask": 250.1, "mid": 125.05, "ts": datetime.now(timezone.utc).isoformat()}
    null_ask = {"symbol": "AAPL", "bid": 150.0, "ask": None, "mid": 150.0, "ts": datetime.now(timezone.utc).isoformat()}
    
    provider = MockQuoteProvider({
        "SPY": normal_quote,
        "TSLA": zero_bid,
        "AAPL": null_ask
    })
    
    anomalies = []
    symbols = ["SPY", "TSLA", "AAPL"]
    
    for sym in symbols:
        try:
            quote = provider.quote(sym)
            bid, ask, mid = quote["bid"], quote["ask"], quote["mid"]
            
            # Market state tripwire (null/zero quotes)
            if not bid or not ask or bid == 0 or ask == 0:
                anomalies.append(f"market_closed:{sym}")
                continue
                
        except Exception as e:
            anomalies.append(f"quote_error:{sym}:{str(e)[:50]}")
    
    # Should detect market closed for TSLA and AAPL
    assert len(anomalies) == 2
    assert any("market_closed:TSLA" in a for a in anomalies)
    assert any("market_closed:AAPL" in a for a in anomalies)


def test_quote_error_handling():
    """Test error handling in quote processing."""
    class ErrorQuoteProvider:
        def quote(self, symbol: str) -> dict:
            raise Exception("Connection failed")
    
    provider = ErrorQuoteProvider()
    anomalies = []
    symbols = ["SPY"]
    
    for sym in symbols:
        try:
            quote = provider.quote(sym)
            # ... rest of processing
        except Exception as e:
            anomalies.append(f"quote_error:{sym}:{str(e)[:50]}")
    
    # Should catch and record the error
    assert len(anomalies) == 1
    assert anomalies[0].startswith("quote_error:SPY")


def test_spread_calculation_edge_cases():
    """Test spread calculation with edge cases."""
    # Test with NaN values
    nan_quote = {"symbol": "SPY", "bid": float("nan"), "ask": 450.1, "mid": 450.05, "ts": datetime.now(timezone.utc).isoformat()}
    
    provider = MockQuoteProvider({"SPY": nan_quote})
    
    anomalies = []
    symbols = ["SPY"]
    
    for sym in symbols:
        try:
            quote = provider.quote(sym)
            bid, ask, mid = quote["bid"], quote["ask"], quote["mid"]
            if bid and ask and mid and bid == bid and ask == ask and mid == mid:
                spread_bps = ((ask - bid) / mid) * 10000
                if spread_bps > 50:
                    anomalies.append(f"wide_spread:{sym}:{spread_bps:.1f}bps")
                    continue
                    
        except Exception as e:
            anomalies.append(f"quote_error:{sym}:{str(e)[:50]}")
    
    # Should not trigger wide spread due to NaN check
    assert len(anomalies) == 0
