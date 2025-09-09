#!/usr/bin/env python3
"""
Tests for Risk V2 - Institutional-Grade Risk Management
======================================================

Test per-trade risk budgeting, ATR-based stops, and portfolio position caps.
"""

import pandas as pd
from risk.v2 import apply, RiskV2Config


def _mkbars(prices):
    """Create minimal OHLC DataFrame for testing."""
    df = pd.DataFrame({"Close": prices})
    df["High"] = df["Close"] * 1.001
    df["Low"] = df["Close"] * 0.999
    return df


CFG = RiskV2Config(
    per_trade_risk_bps=10, atr_window=5, atr_multiplier=2.0,
    max_position_pct=10, portfolio_max_gross_pct=100, min_trade_notional=1
)


def test_caps_and_weight_sign():
    """Test that position weights are properly capped and signed."""
    bars = _mkbars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    d = apply("SPY", bars, desired_weight=+0.5, equity=10_000,
              open_positions={}, portfolio_gross_weight=0.0, cfg=CFG)
    assert not d["veto"]
    assert d["weight"] <= 0.10 + 1e-9
    assert d["weight"] >= 0.0


def test_portfolio_gross_veto():
    """Test that portfolio gross cap prevents oversizing."""
    bars = _mkbars(list(range(100, 120)))
    d = apply("SPY", bars, desired_weight=+0.5, equity=10_000,
              open_positions={}, portfolio_gross_weight=1.0, cfg=CFG)
    assert d["veto"]
    assert d["reason"] == "portfolio_gross_cap"


def test_stop_exit_long():
    """Test that long positions exit when price drops below stop."""
    prices = [100] * 10 + [98]  # drop below stop
    bars = _mkbars(prices)
    pos = {"SPY": {"entry_price": 100.0, "stop": 99.0, "weight": 0.05}}
    d = apply("SPY", bars, desired_weight=+0.05, equity=10_000,
              open_positions=pos, portfolio_gross_weight=0.05, cfg=CFG)
    assert d["action"].startswith("EXIT_STOP")
    assert d["weight"] == 0.0


def test_stop_exit_short():
    """Test that short positions exit when price rises above stop."""
    prices = [100] * 10 + [102]  # rise above stop
    bars = _mkbars(prices)
    pos = {"SPY": {"entry_price": 100.0, "stop": 101.0, "weight": -0.05}}
    d = apply("SPY", bars, desired_weight=-0.05, equity=10_000,
              open_positions=pos, portfolio_gross_weight=0.05, cfg=CFG)
    assert d["action"].startswith("EXIT_STOP")
    assert d["weight"] == 0.0


def test_insufficient_history():
    """Test that insufficient data results in veto."""
    bars = _mkbars([100, 101, 102])  # less than atr_window + 1
    d = apply("SPY", bars, desired_weight=+0.1, equity=10_000,
              open_positions={}, portfolio_gross_weight=0.0, cfg=CFG)
    assert d["veto"]
    assert d["reason"] == "insufficient_history"


def test_below_min_notional():
    """Test that trades below minimum notional are vetoed."""
    cfg = RiskV2Config(
        per_trade_risk_bps=1, atr_window=5, atr_multiplier=2.0,
        max_position_pct=10, portfolio_max_gross_pct=100, min_trade_notional=1000
    )
    bars = _mkbars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    d = apply("SPY", bars, desired_weight=+0.1, equity=10_000,
              open_positions={}, portfolio_gross_weight=0.0, cfg=cfg)
    assert d["veto"]
    assert d["reason"] == "below_min_notional"


def test_atr_calculation():
    """Test that ATR is calculated correctly."""
    bars = _mkbars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    d = apply("SPY", bars, desired_weight=+0.1, equity=10_000,
              open_positions={}, portfolio_gross_weight=0.0, cfg=CFG)
    assert d["atr"] is not None
    assert d["atr"] > 0


def test_risk_dollars_calculation():
    """Test that risk dollars are calculated correctly."""
    bars = _mkbars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    d = apply("SPY", bars, desired_weight=+0.1, equity=10_000,
              open_positions={}, portfolio_gross_weight=0.0, cfg=CFG)
    expected_risk = 10_000 * (10 / 1e4)  # equity * (per_trade_risk_bps / 1e4)
    assert d["risk_dollars"] == expected_risk


def test_existing_position_maintenance():
    """Test that existing positions maintain their stops."""
    bars = _mkbars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    pos = {"SPY": {"entry_price": 100.0, "stop": 99.0, "weight": 0.05}}
    d = apply("SPY", bars, desired_weight=+0.05, equity=10_000,
              open_positions=pos, portfolio_gross_weight=0.05, cfg=CFG)
    assert d["stop"] == 99.0
    assert d["weight"] == 0.05  # maintain position


def test_new_position_stop_calculation():
    """Test that new positions get proper stops calculated."""
    bars = _mkbars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    d = apply("SPY", bars, desired_weight=+0.1, equity=10_000,
              open_positions={}, portfolio_gross_weight=0.0, cfg=CFG)
    assert d["stop"] is not None
    assert d["stop"] < 109  # stop should be below current price for long


def test_negative_weight_handling():
    """Test that negative weights (short positions) are handled correctly."""
    bars = _mkbars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    d = apply("SPY", bars, desired_weight=-0.1, equity=10_000,
              open_positions={}, portfolio_gross_weight=0.0, cfg=CFG)
    assert d["weight"] <= 0
    assert d["weight"] >= -0.10  # capped at max_position_pct


def test_zero_weight_handling():
    """Test that zero weights result in HOLD action."""
    bars = _mkbars([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    d = apply("SPY", bars, desired_weight=0.0, equity=10_000,
              open_positions={}, portfolio_gross_weight=0.0, cfg=CFG)
    assert d["weight"] == 0.0
    assert d["action"] == "HOLD"


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_caps_and_weight_sign,
        test_portfolio_gross_veto,
        test_stop_exit_long,
        test_stop_exit_short,
        test_insufficient_history,
        test_below_min_notional,
        test_atr_calculation,
        test_risk_dollars_calculation,
        test_existing_position_maintenance,
        test_new_position_stop_calculation,
        test_negative_weight_handling,
        test_zero_weight_handling
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            raise
