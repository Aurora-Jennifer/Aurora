#!/usr/bin/env python3
"""
Enhanced Trading System Test
Tests the 1% daily growth target system and market-adaptive strategy selection
"""

import logging
import sys
from datetime import date
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine.paper import PaperTradingEngine
from core.performance import GrowthTargetCalculator
from core.strategy_selector import StrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_growth_target_calculator():
    """Test the growth target calculator functionality."""
    print("\n=== Testing Growth Target Calculator ===")

    config = {
        "initial_capital": 100000,
        "objective": {
            "type": "log_utility",
            "kelly_cap_fraction": 0.25,
            "volatility_adjustment": True,
            "performance_lookback_days": 30,
        },
        "risk_params": {
            "max_position_size": 0.15,
            "volatility_target": 0.20,
            "kelly_fraction": 0.25,
            "position_sizing_method": "kelly_optimal",
        },
    }

    calculator = GrowthTargetCalculator(config)

    # Test position sizing calculation
    position_size = calculator.calculate_dynamic_position_size(
        signal_strength=0.8,
        current_capital=100000,
        symbol_volatility=0.02,
        portfolio_volatility=0.015,
    )

    print(
        f"Position size for strong signal: {position_size:.3f} ({position_size*100:.1f}%)"
    )

    # Test with some performance data
    for i in range(10):
        daily_return = 0.01 + (i * 0.001)  # Simulate improving performance
        calculator.update_performance(daily_return, 100000 * (1.01 ** (i + 1)))

    metrics = calculator.get_growth_metrics()
    print("Growth metrics after 10 days:")
    print(f"  Average daily return: {metrics['avg_daily_return']:.4f}")
    print(f"  Volatility: {metrics['volatility']:.4f}")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Objective score: {metrics['objective_score']:.4f}")

    should_adjust, reason = calculator.should_adjust_target()
    print(f"Should adjust target: {should_adjust} - {reason}")

    return True


def test_strategy_selector():
    """Test the strategy selector functionality."""
    print("\n=== Testing Strategy Selector ===")

    config = {"symbols": ["SPY", "AAPL", "NVDA"], "initial_capital": 100000}

    selector = StrategySelector(config)

    # Test strategy summary
    summary = selector.get_strategy_summary()
    print(f"Available strategies: {summary['available_strategies']}")

    # Create mock market data
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

    market_data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.randn(100) * 0.001),
            "High": prices * (1 + abs(np.random.randn(100)) * 0.002),
            "Low": prices * (1 - abs(np.random.randn(100)) * 0.002),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 100),
        }
    )

    # Test strategy selection
    strategy_name, strategy_params, expected_sharpe = selector.select_best_strategy(
        market_data
    )

    print(f"Selected strategy: {strategy_name}")
    print(f"Strategy parameters: {strategy_params}")
    print(f"Expected Sharpe ratio: {expected_sharpe:.3f}")

    # Test performance update
    performance_metrics = {
        "sharpe_ratio": 0.8,
        "total_return": 0.15,
        "max_drawdown": -0.05,
    }

    selector.update_performance_data(
        "regime_aware_ensemble", "trend", performance_metrics
    )
    print("Updated performance data for regime-aware ensemble in trend regime")

    return True


def test_paper_trading_engine():
    """Test the enhanced paper trading engine."""
    print("\n=== Testing Enhanced Paper Trading Engine ===")

    # Use the enhanced configuration
    config_file = "config/enhanced_paper_trading_config.json"

    if not Path(config_file).exists():
        print(f"Configuration file {config_file} not found. Skipping engine test.")
        return False

    try:
        # Initialize the engine
        engine = PaperTradingEngine(config_file)

        print(f"Initial capital: ${engine.capital:,.2f}")
        # Objective-based sizing; no fixed daily target
        print(
            f"Available strategies: {list(engine.strategy_selector.strategies.keys())}"
        )

        # Test a single trading cycle
        test_date = date.today()
        result = engine.run_trading_cycle(test_date)

        print(f"Trading cycle result: {result['status']}")
        if result["status"] == "success":
            print(f"  Regime: {result['regime']}")
            print(f"  Strategy: {result['strategy']}")
            print(f"  Expected Sharpe: {result['expected_sharpe']:.3f}")
            print(f"  Trades executed: {result['trades_executed']}")
            print(f"  Performance metrics: {result['performance_metrics']}")

        # Test growth metrics
        growth_metrics = engine.growth_calculator.get_growth_metrics()
        print("\nGrowth metrics:")
        print(f"  Days tracked: {growth_metrics['days_tracked']}")
        print(f"  Objective score: {growth_metrics.get('objective_score', 0.0):.4f}")

        return True

    except Exception as e:
        print(f"Error testing paper trading engine: {e}")
        return False


def test_risk_management():
    """Test risk management features."""
    print("\n=== Testing Risk Management ===")

    config = {
        "risk_params": {
            "max_gross_exposure_pct": 50.0,
            "max_daily_loss_pct": 2.0,
            "max_position_size": 0.15,
            "max_drawdown_pct": 15.0,
            "stop_loss_pct": 3.0,
            "take_profit_pct": 6.0,
            "volatility_target": 0.20,
            "max_correlation": 0.7,
            "position_sizing_method": "kelly_optimal",
            "kelly_fraction": 0.25,
            "dynamic_position_sizing": True,
        }
    }

    from core.risk.guardrails import RiskGuardrails

    guardrails = RiskGuardrails(config)

    # Test position size validation
    is_valid = guardrails.validate_position_size("AAPL", 100, 150.0, 100000)
    print(f"Position size validation (AAPL 100 shares @ $150): {is_valid}")

    # Test portfolio risk check
    positions = {"AAPL": 100, "SPY": 50}
    prices = {"AAPL": 150.0, "SPY": 400.0}
    risk_metrics = guardrails.check_portfolio_risk(positions, prices, 100000)

    print("Portfolio risk metrics:")
    print(f"  Portfolio value: ${risk_metrics['portfolio_value']:,.2f}")
    print(f"  Leverage: {risk_metrics['leverage']:.2f}")
    print(f"  Max concentration: {risk_metrics['max_concentration']:.1f}%")
    print(f"  Position count: {risk_metrics['position_count']}")

    return True


def main():
    """Run all tests."""
    print("Enhanced Trading System Test Suite")
    print("=" * 50)

    tests = [
        ("Growth Target Calculator", test_growth_target_calculator),
        ("Strategy Selector", test_strategy_selector),
        ("Risk Management", test_risk_management),
        ("Paper Trading Engine", test_paper_trading_engine),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed! Enhanced trading system is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
