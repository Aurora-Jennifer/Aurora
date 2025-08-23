import builtins
import contextlib
import json
import tempfile
from pathlib import Path

import pytest

from scripts.paper_broker import HoldingsLedger, calculate_position_size


def test_holdings_ledger_initialization():
    """Test HoldingsLedger initialization"""
    ledger = HoldingsLedger(initial_cash=100000.0)
    assert ledger.cash == 100000.0
    assert len(ledger.positions) == 0
    assert len(ledger.trades) == 0


def test_holdings_ledger_buy_trade():
    """Test buying a position"""
    ledger = HoldingsLedger(initial_cash=10000.0)

    # Buy 10 shares at $100 each
    result = ledger.execute_trade("SPY", 10.0, 100.0)

    assert result["status"] == "filled"
    expected_cash = 10000.0 - 1000.0 - (1000.0 * 0.3 / 10000.0) - (1000.0 * 0.7 / 10000.0)  # price - commission - slippage
    assert abs(ledger.cash - expected_cash) < 0.01  # Allow for floating point precision
    assert ledger.positions["SPY"] == 10.0
    assert ledger.avg_prices["SPY"] == 100.0
    assert len(ledger.trades) == 1


def test_holdings_ledger_sell_trade():
    """Test selling a position"""
    ledger = HoldingsLedger(initial_cash=10000.0)

    # Buy first
    ledger.execute_trade("SPY", 10.0, 100.0)
    initial_cash = ledger.cash

    # Then sell
    result = ledger.execute_trade("SPY", -10.0, 110.0)

    assert result["status"] == "filled"
    assert ledger.cash > initial_cash  # Should have more cash after selling
    assert "SPY" not in ledger.positions  # Position should be closed
    assert len(ledger.trades) == 2


def test_holdings_ledger_insufficient_cash():
    """Test rejection when insufficient cash"""
    ledger = HoldingsLedger(initial_cash=100.0)

    # Try to buy more than we can afford
    result = ledger.execute_trade("SPY", 10.0, 100.0)

    assert result["status"] == "rejected"
    assert result["reason"] == "insufficient_cash"
    assert ledger.cash == 100.0  # Cash unchanged
    assert len(ledger.trades) == 0


def test_calculate_position_size():
    """Test position size calculation"""
    # Test with rank
    size1 = calculate_position_size(score=0.5, rank=0.8, max_position_pct=0.15, base_position_size=1000.0)
    assert size1 == 150.0  # min(1000 * 0.8, 1000 * 0.15) = min(800, 150) = 150

    # Test with score (no rank)
    size2 = calculate_position_size(score=0.5, rank=None, max_position_pct=0.15, base_position_size=1000.0)
    assert size2 == 150.0  # min(1000 * 0.75, 1000 * 0.15) = min(750, 150) = 150


def test_paper_broker_integration():
    """Test the full paper broker integration"""
    import sys

    from scripts.paper_broker import main

    # Create test signals
    signals = [
        {"symbol": "SPY", "score": 0.8, "rank": 0.9, "price": 450.0},
        {"symbol": "QQQ", "score": -0.3, "rank": 0.2, "price": 380.0}
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for signal in signals:
            f.write(json.dumps(signal) + '\n')
        signals_path = f.name

    fills_path = tempfile.mktemp(suffix='.jsonl')
    ledger_path = tempfile.mktemp(suffix='.json')

    try:
        # Run paper broker
        sys.argv = [
            'paper_broker.py',
            '--signals', signals_path,
            '--fills', fills_path,
            '--ledger', ledger_path,
            '--initial-cash', '100000.0'
        ]

        result = main()
        assert result == 0

        # Check fills were created
        assert Path(fills_path).exists()
        with open(fills_path) as f:
            fills = [json.loads(line) for line in f]
        assert len(fills) == 2

        # Check ledger was created
        assert Path(ledger_path).exists()
        with open(ledger_path) as f:
            ledger_data = json.load(f)
        assert "final_summary" in ledger_data
        assert "portfolio_value" in ledger_data
        assert "total_return_pct" in ledger_data

    finally:
        # Cleanup
        for path in [signals_path, fills_path, ledger_path]:
            with contextlib.suppress(builtins.BaseException):
                Path(path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
