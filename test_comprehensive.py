#!/usr/bin/env python3
"""
Comprehensive Test Suite
Tests all components of the trading system end-to-end.
"""

import logging
import sys
from datetime import date

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)s - %(name)s - %(message)s"
)


def test_portfolio_system():
    """Test portfolio system functionality."""
    print("\n🧪 Testing Portfolio System")
    print("=" * 40)

    from core.portfolio import PortfolioState

    # Initialize portfolio
    portfolio = PortfolioState(cash=10000.0)
    print(f"✅ Initial portfolio: ${portfolio.cash:,.2f}")

    # Test buy
    realized = portfolio.apply_fill("AAPL", "BUY", 10, 150.0, 10)
    print(f"✅ Buy 10 AAPL @ $150: PnL=${realized:.2f}, Cash=${portfolio.cash:.2f}")

    # Test sell
    realized = portfolio.apply_fill("AAPL", "SELL", 5, 160.0, 10)
    print(f"✅ Sell 5 AAPL @ $160: PnL=${realized:.2f}, Cash=${portfolio.cash:.2f}")

    # Test mark to market
    portfolio.update_price("AAPL", 155.0)
    mtm = portfolio.mark_to_market()
    print(f"✅ Mark to market: ${mtm:,.2f}")

    return True


def test_trade_logging():
    """Test trade logging system."""
    print("\n🧪 Testing Trade Logging")
    print("=" * 40)

    from core.trade_logger import TradeBook

    trade_book = TradeBook()

    # Test buy
    trade_book.on_buy("2024-01-01", "AAPL", 10, 150.0, 1.0)
    print(f"✅ Recorded buy: {len(trade_book.get_open_positions())} open positions")

    # Test sell
    trade_book.on_sell("2024-01-02", "AAPL", 10, 160.0, 1.0, 0.0)
    print(f"✅ Recorded sell: {len(trade_book.get_closed_trades())} closed trades")

    # Test metrics
    metrics = trade_book.get_trade_summary()
    print(
        f"✅ Trade metrics: {metrics['total_trades']} trades, PnL=${metrics['total_pnl']:.2f}"
    )

    return True


def test_performance_metrics():
    """Test performance metrics calculation."""
    print("\n🧪 Testing Performance Metrics")
    print("=" * 40)

    from core.performance import calculate_portfolio_metrics, calculate_trade_metrics
    from core.trade_logger import TradeRecord

    # Test trade metrics
    trades = [
        TradeRecord("AAPL", "2024-01-01", realized_pnl=100.0),
        TradeRecord("AAPL", "2024-01-02", realized_pnl=-50.0),
    ]

    trade_metrics = calculate_trade_metrics(trades)
    print(
        f"✅ Trade metrics: Win rate={trade_metrics['win_rate']:.1%}, PF={trade_metrics['profit_factor']}"
    )

    # Test portfolio metrics
    equity_curve = [
        {"equity": 10000.0},
        {"equity": 10100.0},
        {"equity": 10050.0},
    ]

    portfolio_metrics = calculate_portfolio_metrics(equity_curve)
    print(f"✅ Portfolio metrics: Return={portfolio_metrics['total_return']:.1%}")

    return True


def test_paper_trading():
    """Test paper trading system."""
    print("\n🧪 Testing Paper Trading System")
    print("=" * 40)

    from enhanced_paper_trading import EnhancedPaperTradingSystem

    # Initialize system
    system = EnhancedPaperTradingSystem(
        "config/enhanced_paper_trading_config.json", "config/live_profile.json"
    )

    print(f"✅ Initialized with ${system.capital:,.2f} capital")
    print(f"✅ Symbols: {system.config['symbols']}")

    # Test kill switches
    kill_ok = system.check_kill_switches()
    print(f"✅ Kill switches: {'PASS' if kill_ok else 'FAIL'}")

    return True


def test_backtest_engine():
    """Test backtest engine."""
    print("\n🧪 Testing Backtest Engine")
    print("=" * 40)

    from backtest import BacktestEngine

    # Initialize backtest engine
    engine = BacktestEngine(
        "config/enhanced_paper_trading_config.json", "config/live_profile.json"
    )

    print("✅ Initialized backtest engine")
    print(f"✅ Initial capital: ${engine.initial_capital:,.2f}")
    print(f"✅ Portfolio cash: ${engine.portfolio.cash:,.2f}")

    # Test data loading
    start_date = date(2024, 7, 1)
    end_date = date(2024, 7, 5)

    data = engine._load_historical_data(start_date, end_date)
    if data is not None and not data.empty:
        print(f"✅ Data loading: {len(data)} points loaded")

        # Test date extraction
        trading_dates = engine._get_trading_dates_from_data(data)
        print(f"✅ Date extraction: {len(trading_dates)} trading dates")

        # Test price update
        if trading_dates:
            engine._update_portfolio_prices(trading_dates[0], data)
            print(
                f"✅ Price update: Portfolio has {len(engine.portfolio.last_prices)} prices"
            )
    else:
        print("❌ Data loading failed")
        return False

    return True


def test_strategy_signals():
    """Test strategy signal generation."""
    print("\n🧪 Testing Strategy Signals")
    print("=" * 40)

    import numpy as np
    import pandas as pd

    from enhanced_paper_trading import EnhancedPaperTradingSystem

    # Create mock data
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    mock_data = pd.DataFrame(
        {
            "Open": np.random.uniform(100, 200, 100),
            "High": np.random.uniform(100, 200, 100),
            "Low": np.random.uniform(100, 200, 100),
            "Close": np.random.uniform(100, 200, 100),
            "Volume": np.random.uniform(1000000, 5000000, 100),
        },
        index=dates,
    )

    # Initialize system
    system = EnhancedPaperTradingSystem(
        "config/enhanced_paper_trading_config.json", "config/live_profile.json"
    )

    try:
        # Test regime detection
        regime_name, confidence, regime_params = system.regime_detector.detect_regime(
            mock_data
        )
        print(f"✅ Regime detection: {regime_name} (confidence: {confidence:.2f})")

        # Test signal generation
        signals = system._generate_regime_aware_signals(
            mock_data, regime_name, regime_params
        )
        print(f"✅ Signal generation: {signals}")

        return True
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False


def main():
    """Run comprehensive tests."""
    print("🚀 Comprehensive Trading System Test Suite")
    print("=" * 60)

    tests = [
        ("Portfolio System", test_portfolio_system),
        ("Trade Logging", test_trade_logging),
        ("Performance Metrics", test_performance_metrics),
        ("Paper Trading", test_paper_trading),
        ("Backtest Engine", test_backtest_engine),
        ("Strategy Signals", test_strategy_signals),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! System is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
