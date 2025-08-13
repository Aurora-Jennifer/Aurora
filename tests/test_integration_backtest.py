"""
Integration tests for backtest functionality.
"""

from backtest import BacktestEngine


def test_backtest_includes_unrealized_in_equity(tmp_path):
    """Test that backtest includes unrealized PnL in equity calculation."""
    engine = BacktestEngine("config/enhanced_paper_trading_config.json")
    # short window to keep test fast
    engine.run_backtest("2024-01-01", "2024-01-05", ["SPY"])
    summary = engine.get_last_summary()

    assert "Initial Capital" in summary
    assert "Final Equity" in summary
    assert "Total PnL" in summary
    assert "Total Trades" in summary

    # Equity must equal cash + mtm at final day
    assert (
        abs(
            summary["Final Equity"]
            - (summary["Initial Capital"] + summary["Total PnL"])
        )
        < 1e-6
    )


def test_backtest_generates_ledger():
    """Test that backtest generates a ledger with proper columns."""
    engine = BacktestEngine("config/enhanced_paper_trading_config.json")
    engine.run_backtest("2024-01-01", "2024-01-05", ["SPY"])

    # Check that ledger was created
    assert not engine.portfolio.ledger.empty

    # Check required columns
    required_columns = [
        "date",
        "equity",
        "cash",
        "gross_exposure",
        "net_exposure",
        "unrealized_pnl_total",
        "realized_pnl_total",
        "fees_paid",
    ]
    for col in required_columns:
        assert col in engine.portfolio.ledger.columns


def test_backtest_trade_tracking():
    """Test that backtest properly tracks trades."""
    engine = BacktestEngine("config/enhanced_paper_trading_config.json")
    engine.run_backtest("2024-01-01", "2024-01-05", ["SPY"])

    # Check that trades list exists
    assert hasattr(engine.portfolio, "trades")

    # Check that total_trades is tracked
    assert hasattr(engine.portfolio, "total_trades")
    assert engine.portfolio.total_trades >= 0


def test_backtest_position_closing():
    """Test that backtest can close positions at end when configured."""
    # Create a config with position closing enabled
    config = {
        "symbols": ["SPY"],
        "initial_capital": 100000,
        "use_ibkr": False,
        "shorting_enabled": True,
        "close_positions_at_end": True,
    }

    # Save temporary config
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_file = f.name

    try:
        engine = BacktestEngine(config_file)
        engine.run_backtest("2024-01-01", "2024-01-05", ["SPY"])

        # Check that positions were closed
        assert engine.portfolio.get_open_positions_count() == 0
    finally:
        import os

        os.unlink(config_file)


def test_backtest_summary_format():
    """Test that backtest summary has the expected format."""
    engine = BacktestEngine("config/enhanced_paper_trading_config.json")
    engine.run_backtest("2024-01-01", "2024-01-05", ["SPY"])

    summary = engine.get_last_summary()

    # Check required fields
    required_fields = ["Initial Capital", "Final Equity", "Total PnL", "Total Trades"]
    for field in required_fields:
        assert field in summary
        assert isinstance(summary[field], (int, float))


def test_backtest_file_output():
    """Test that backtest generates the expected output files."""
    engine = BacktestEngine("config/enhanced_paper_trading_config.json")
    engine.run_backtest("2024-01-01", "2024-01-05", ["SPY"])

    # Check that files were created
    import pathlib

    results_dir = pathlib.Path("results/backtest")

    assert results_dir.exists()
    assert (results_dir / "ledger.csv").exists()
    assert (results_dir / "results.json").exists()
    assert (results_dir / "summary.txt").exists()
