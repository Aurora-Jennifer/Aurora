"""
Split-aware, regime-aware price sanity guards.

Replaces brittle hard-coded price bands with adaptive checks:
1. Fatal checks: price <= 0, not finite, absurd spikes
2. Jump guard: Day-over-day % changes with split-awareness  
3. Regime band: Rolling median-based bands that adapt to data
"""
from math import log, isfinite
import pandas as pd


def split_aware_jump_ok(p_t: float, p_tm1: float, had_corporate_action_today: bool, jump_limit: float = 0.30) -> bool:
    """Check if day-over-day price jump is acceptable given corporate actions."""
    if p_tm1 <= 0 or not isfinite(p_tm1) or not isfinite(p_t):
        return False
    
    r = abs(log(p_t / p_tm1))
    
    if had_corporate_action_today:
        return True  # allow split-induced jumps
        
    return r <= log(1.0 + jump_limit)


def regime_band_ok(p_t: float, rolling_prices: list[float], band_frac: float = 0.80, min_bars: int = 30) -> bool:
    """Check if price is within rolling regime band."""
    if len(rolling_prices) < min_bars:
        return True  # not enough history to form a band
    
    # median as robust center
    sorted_win = sorted(rolling_prices)
    mid = len(sorted_win) // 2
    p_ref = (sorted_win[mid] if len(sorted_win) % 2 else 0.5 * (sorted_win[mid-1] + sorted_win[mid]))
    
    lo, hi = p_ref * (1 - band_frac), p_ref * (1 + band_frac)
    return (p_t >= lo) and (p_t <= hi)


def price_sane(
    symbol: str, 
    p_t: float, 
    p_tm1: float | None, 
    rolling_prices: list[float], 
    had_corp_action_today: bool,
    config: dict = None
) -> tuple[bool, str]:
    """
    Comprehensive price sanity check with split-awareness and regime adaptation.
    
    Args:
        symbol: Trading symbol
        p_t: Current price
        p_tm1: Previous price (None if first bar)
        rolling_prices: Historical prices for regime band
        had_corp_action_today: Whether corporate action occurred
        config: Guard configuration
        
    Returns:
        (is_sane, reason) tuple
    """
    if config is None:
        config = {
            "jump_limit_frac": 0.30,
            "band_frac": 0.80,
            "warmup_bars": 30,
            "absurd_max": 1_000_000
        }
    
    # A) Fatal checks (always on)
    if (p_t is None) or (not isfinite(p_t)) or (p_t <= 0):
        return False, "fatal_price_invalid"
        
    if p_t > config["absurd_max"]:
        return False, "fatal_price_absurd_spike"
    
    # B) Jump guard (skip if no previous price)
    if p_tm1 is not None:
        if not split_aware_jump_ok(p_t, p_tm1, had_corp_action_today, config["jump_limit_frac"]):
            return False, "jump_exceeds_limit_no_corporate_action"
    
    # C) Regime band
    if not regime_band_ok(p_t, rolling_prices, config["band_frac"], config["warmup_bars"]):
        return False, "regime_band_violation"
    
    # D) Recent trading range check (Clearframe enhancement)
    if len(rolling_prices) >= 390:  # ~1 trading day for 1m bars
        recent_window = rolling_prices[-390:]  # Last day
        pmin = min(recent_window)
        pmax = max(recent_window)
        if not (0.5 * pmin <= p_t <= 1.5 * pmax):
            return False, f"price_outside_recent_range: {p_t:.2f} not in [{0.5*pmin:.2f}, {1.5*pmax:.2f}]"
    
    # E) Mid/Last deviation check (for live data)
    if len(rolling_prices) >= 2:
        # Use recent OHLC to estimate "middle price" 
        recent_close = rolling_prices[-1]
        if recent_close > 0 and abs(p_t - recent_close) / recent_close > 0.01:  # 1% deviation
            return False, f"mid_last_deviation: current={p_t:.2f} vs recent={recent_close:.2f}"
    
    return True, "ok"


def extract_rolling_prices(bars: pd.DataFrame, lookback: int = 90) -> list[float]:
    """Extract rolling prices from bars DataFrame."""
    if len(bars) == 0:
        return []
    
    # Get the last `lookback` close prices, excluding current bar
    close_col = 'Close' if 'Close' in bars.columns else 'close'
    prices = bars[close_col].iloc[:-1].tolist()  # exclude current bar
    
    return prices[-lookback:] if len(prices) > lookback else prices


def has_corporate_action_today(symbol: str, timestamp: pd.Timestamp) -> bool:
    """
    Check if corporate action occurred on given date.
    
    TODO: Integrate with corporate actions data when available.
    For now, return False (conservative approach).
    """
    # Placeholder - integrate with corporate_actions.py when available
    return False
