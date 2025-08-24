"""
Tests for adaptive price guard system.

Covers:
1. Historical SPY 2000 validation (should pass)
2. Split-day exemptions for large jumps
3. Vendor spike detection and blocking
4. Warmup period handling
5. Regime band adaptation
"""
import pytest
import pandas as pd
from math import log
from core.guards.price import (
    price_sane, 
    split_aware_jump_ok, 
    regime_band_ok,
    extract_rolling_prices
)


class TestSplitAwareJumpCheck:
    """Test jump detection with corporate action awareness."""
    
    def test_normal_jump_within_limit(self):
        """Normal 10% jump should pass."""
        p_tm1, p_t = 100.0, 110.0  # 10% jump
        result = split_aware_jump_ok(p_t, p_tm1, had_corporate_action_today=False)
        assert result is True
    
    def test_large_jump_without_corporate_action_fails(self):
        """50% jump without corporate action should fail."""
        p_tm1, p_t = 100.0, 150.0  # 50% jump
        result = split_aware_jump_ok(p_t, p_tm1, had_corporate_action_today=False)
        assert result is False
    
    def test_large_jump_with_corporate_action_passes(self):
        """50% jump with corporate action (split) should pass."""
        p_tm1, p_t = 200.0, 100.0  # 2:1 split (50% drop)
        result = split_aware_jump_ok(p_t, p_tm1, had_corporate_action_today=True)
        assert result is True
    
    def test_extreme_jump_with_corporate_action_passes(self):
        """Even 5:1 split should pass with corporate action flag."""
        p_tm1, p_t = 500.0, 100.0  # 5:1 split (80% drop)
        result = split_aware_jump_ok(p_t, p_tm1, had_corporate_action_today=True)
        assert result is True
    
    def test_invalid_previous_price_fails(self):
        """Zero or negative previous price should fail."""
        assert split_aware_jump_ok(100.0, 0.0, False) is False
        assert split_aware_jump_ok(100.0, -10.0, False) is False


class TestRegimeBandCheck:
    """Test adaptive regime band validation."""
    
    def test_insufficient_history_passes(self):
        """With insufficient history, should pass (no band check)."""
        rolling_prices = [100, 105, 98]  # Only 3 prices, min_bars=30
        result = regime_band_ok(102.0, rolling_prices, band_frac=0.80, min_bars=30)
        assert result is True
    
    def test_price_within_band_passes(self):
        """Price within ±80% of median should pass."""
        # Median of 95-105 range = 100
        rolling_prices = list(range(95, 106))  # 95, 96, ..., 105
        median = 100.0  # median of this range
        # Band: [20, 180] with 80% band
        
        assert regime_band_ok(80.0, rolling_prices, band_frac=0.80, min_bars=10) is True   # Within band
        assert regime_band_ok(120.0, rolling_prices, band_frac=0.80, min_bars=10) is True  # Within band
    
    def test_price_outside_band_fails(self):
        """Price outside ±80% of median should fail."""
        rolling_prices = list(range(95, 106))  # Median = 100
        # Band with 80%: [20, 180]
        
        assert regime_band_ok(15.0, rolling_prices, band_frac=0.80, min_bars=10) is False  # Too low
        assert regime_band_ok(200.0, rolling_prices, band_frac=0.80, min_bars=10) is False # Too high
    
    def test_different_band_fractions(self):
        """Test different band fractions work correctly."""
        rolling_prices = [100] * 50  # All prices = 100, so median = 100
        
        # Tight band (±10%): [90, 110]
        assert regime_band_ok(95.0, rolling_prices, band_frac=0.10) is True
        assert regime_band_ok(85.0, rolling_prices, band_frac=0.10) is False
        
        # Wide band (±50%): [50, 150]  
        assert regime_band_ok(140.0, rolling_prices, band_frac=0.50) is True
        assert regime_band_ok(45.0, rolling_prices, band_frac=0.50) is False


class TestComprehensivePriceSanity:
    """Test the main price_sane function."""
    
    def test_historical_spy_2000_passes(self):
        """Historical SPY ~$135 from year 2000 should pass."""
        # Simulate 90 days of SPY prices around $130-140 (year 2000 range)
        rolling_prices = []
        for i in range(90):
            rolling_prices.append(130 + 10 * (i % 10) / 10)  # 130-140 range
        
        # Current price: $134.75 (typical 2000-era SPY)
        is_sane, reason = price_sane(
            symbol="SPY",
            p_t=134.75,
            p_tm1=133.20,  # Previous day
            rolling_prices=rolling_prices,
            had_corp_action_today=False
        )
        
        assert is_sane is True
        assert reason == "ok"
    
    def test_vendor_spike_blocked(self):
        """Absurd vendor spike should be blocked."""
        rolling_prices = [100] * 90
        
        is_sane, reason = price_sane(
            symbol="SPY",
            p_t=1_500_000,  # Absurd spike
            p_tm1=100.0,
            rolling_prices=rolling_prices,
            had_corp_action_today=False
        )
        
        assert is_sane is False
        assert "absurd" in reason
    
    def test_split_day_large_move_allowed(self):
        """Large price move on split day should be allowed."""
        rolling_prices = [200] * 90  # Stock was trading around $200
        
        is_sane, reason = price_sane(
            symbol="AAPL",
            p_t=100.0,      # After 2:1 split
            p_tm1=200.0,    # Before split
            rolling_prices=rolling_prices,
            had_corp_action_today=True  # Split occurred
        )
        
        assert is_sane is True
        assert reason == "ok"
    
    def test_warmup_period_skips_band_check(self):
        """During warmup period, regime band check should be skipped."""
        rolling_prices = [100] * 20  # Only 20 bars, less than warmup_bars=30
        
        # Price way outside historical range, but should pass due to warmup
        is_sane, reason = price_sane(
            symbol="TEST",
            p_t=500.0,      # 5x the historical price
            p_tm1=100.0,
            rolling_prices=rolling_prices,
            had_corp_action_today=False,
            config={"warmup_bars": 30, "jump_limit_frac": 0.30, "band_frac": 0.80, "absurd_max": 1000000}
        )
        
        assert is_sane is False  # Should still fail on jump check (500/100 = 5x > 30% limit)
        assert "jump_exceeds_limit" in reason
    
    def test_fatal_checks_always_apply(self):
        """Fatal checks should always apply regardless of other conditions."""
        rolling_prices = [100] * 90
        
        # Negative price
        is_sane, reason = price_sane("TEST", -10.0, 100.0, rolling_prices, False)
        assert is_sane is False
        assert "fatal" in reason
        
        # Zero price  
        is_sane, reason = price_sane("TEST", 0.0, 100.0, rolling_prices, False)
        assert is_sane is False
        assert "fatal" in reason
        
        # NaN price
        is_sane, reason = price_sane("TEST", float('nan'), 100.0, rolling_prices, False)
        assert is_sane is False
        assert "fatal" in reason


class TestExtractRollingPrices:
    """Test helper function for extracting rolling prices from DataFrame."""
    
    def test_extract_from_dataframe(self):
        """Should extract Close prices excluding current bar."""
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        rolling_prices = extract_rolling_prices(df, lookback=10)
        
        # Should exclude the last price (104) and return [100, 101, 102, 103]
        assert rolling_prices == [100, 101, 102, 103]
    
    def test_empty_dataframe(self):
        """Empty DataFrame should return empty list."""
        df = pd.DataFrame({'Close': []})
        rolling_prices = extract_rolling_prices(df)
        assert rolling_prices == []
    
    def test_lookback_limit(self):
        """Should limit to lookback window size."""
        df = pd.DataFrame({'Close': list(range(200))})  # 0, 1, 2, ..., 199
        
        rolling_prices = extract_rolling_prices(df, lookback=50)
        
        # Should return last 50 prices excluding current (199)
        # So: [149, 150, ..., 198] (50 prices)
        assert len(rolling_prices) == 50
        assert rolling_prices[0] == 149
        assert rolling_prices[-1] == 198


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
