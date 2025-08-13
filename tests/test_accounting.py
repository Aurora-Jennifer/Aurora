"""
Tests for portfolio accounting functionality.
"""

import pandas as pd

from core.portfolio import PortfolioState


def test_unrealized_short_mark_to_market():
    """Test unrealized PnL calculation for short positions."""
    # short 10 @ 500, price goes to 480; profit = +200
    p = PortfolioState(cash=100000)
    price0 = 500.0
    p.execute_order("SPY", target_qty=-10.0, price=price0, fee=0.0)  # open short
    p.mark_to_market(pd.Timestamp("2024-01-02"), {"SPY": 480.0})
    row = p.ledger.iloc[-1]
    # After shorting 10 @ 500: cash = 100000 + 5000 = 105000
    # Position value = -10 * 480 = -4800
    # Equity = 105000 - 4800 = 100200
    expected_equity = (
        100000 + (10 * 500) + (-10 * 480)
    )  # Initial cash + short proceeds + position value
    assert abs(row["equity"] - expected_equity) < 1e-6
    # unrealized = qty*(P-avg) = -10*(480-500) = +200
    assert abs(row["unrealized_pnl_total"] - 200.0) < 1e-6


def test_realized_from_closing_short():
    """Test realized PnL calculation when closing short positions."""
    p = PortfolioState(cash=100000)
    p.execute_order("SPY", target_qty=-10.0, price=500.0, fee=0.0)  # open short
    # cover 6 @ 480 → realized = (-6)*(480-500) = +120
    p.execute_order("SPY", target_qty=-4.0, price=480.0, fee=0.0)
    # remaining qty = -4, avg unchanged at 500 for the remainder
    assert abs(p.realized_pnl - 120.0) < 1e-6
    p.mark_to_market(pd.Timestamp("2024-01-02"), {"SPY": 495.0})
    # unrealized on remaining = -4*(495-500) = +20
    assert abs(p.ledger.iloc[-1]["unrealized_pnl_total"] - 20.0) < 1e-6


def test_blocked_orders_do_not_count_as_trades():
    """Test that blocked orders don't count as trades."""
    p = PortfolioState(cash=100000, shorting_enabled=False)
    # Try to sell more than available (should be blocked)
    p.execute_order("SPY", target_qty=10.0, price=100.0, fee=0.0)  # buy 10 first
    p.execute_order(
        "SPY", target_qty=-15.0, price=100.0, fee=0.0
    )  # try to sell 15 (should be blocked)
    assert p.total_trades == 1  # Only the buy should count
    assert p.positions["SPY"].qty == 10.0  # Should still have 10 shares


def test_long_position_pnl():
    """Test PnL calculation for long positions."""
    p = PortfolioState(cash=100000)
    # Buy 10 @ 100, price goes to 110; unrealized profit = +100
    p.execute_order("SPY", target_qty=10.0, price=100.0, fee=0.0)
    p.mark_to_market(pd.Timestamp("2024-01-02"), {"SPY": 110.0})
    row = p.ledger.iloc[-1]
    assert abs(row["unrealized_pnl_total"] - 100.0) < 1e-6


def test_position_flipping():
    """Test position flipping from long to short."""
    p = PortfolioState(cash=100000)
    # Buy 10 @ 100
    p.execute_order("SPY", target_qty=10.0, price=100.0, fee=0.0)
    # Sell 15 @ 110 (should flip to short 5)
    p.execute_order("SPY", target_qty=-5.0, price=110.0, fee=0.0)

    # Check realized PnL from the sell
    assert abs(p.realized_pnl - 100.0) < 1e-6  # (110-100) * 10

    # Check remaining position
    assert abs(p.positions["SPY"].qty - (-5.0)) < 1e-6  # Short 5 shares
    assert abs(p.positions["SPY"].avg_price - 110.0) < 1e-6  # New avg price for short


def test_fee_calculation():
    """Test that fees are properly calculated and tracked."""
    p = PortfolioState(cash=100000)
    # Buy with 1% fee
    p.execute_order("SPY", target_qty=10.0, price=100.0, fee=10.0)
    assert p.fees_paid == 10.0
    assert p.cash == 100000 - (10 * 100) - 10  # Cash reduced by trade value + fees


def test_zero_quantity_cleanup():
    """Test that positions with zero quantity are removed."""
    p = PortfolioState(cash=100000)
    # Buy 10 shares
    p.execute_order("SPY", target_qty=10.0, price=100.0, fee=0.0)
    assert "SPY" in p.positions

    # Sell all 10 shares
    p.execute_order("SPY", target_qty=0.0, price=110.0, fee=0.0)
    assert "SPY" not in p.positions  # Position should be removed


def test_ledger_consistency():
    """Test that ledger entries are consistent."""
    p = PortfolioState(cash=100000)
    p.execute_order("SPY", target_qty=10.0, price=100.0, fee=5.0)
    p.mark_to_market(pd.Timestamp("2024-01-01"), {"SPY": 100.0})

    row = p.ledger.iloc[-1]
    assert row["cash"] == 100000 - (10 * 100) - 5  # Initial cash - trade value - fees
    assert row["realized_pnl_total"] == 0.0  # No realized PnL yet
    assert row["fees_paid"] == 5.0
    assert row["total_trades"] == 1
