#!/usr/bin/env python3
"""
Meaningful Validation Test
Tests that the trading system produces meaningful results, not just that it doesn't crash.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from test_paper_engine import TestPaperTradingEngine

from core.objectives import build_objective
from core.strategy_selector import StrategySelector

# Configure logging to capture errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeaningfulValidationTest:
    """Test class that validates meaningful trading results."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.test_results = {}

    def log_error(self, message: str):
        """Log an error and add to error list."""
        logger.error(f"❌ ERROR: {message}")
        self.errors.append(message)

    def log_warning(self, message: str):
        """Log a warning and add to warning list."""
        logger.warning(f"⚠️  WARNING: {message}")
        self.warnings.append(message)

    def assert_true(self, condition: bool, message: str):
        """Assert condition is true, log error if not."""
        if not condition:
            self.log_error(message)
        return condition

    def assert_greater(self, value: float, threshold: float, message: str):
        """Assert value is greater than threshold."""
        if value <= threshold:
            self.log_error(f"{message} (got {value:.6f}, expected > {threshold:.6f})")
        return value > threshold

    def assert_greater_equal(self, value: float, threshold: float, message: str):
        """Assert value is greater than or equal to threshold."""
        if value < threshold:
            self.log_error(f"{message} (got {value:.6f}, expected >= {threshold:.6f})")
        return value >= threshold

    def assert_not_equal(self, value1, value2, message: str):
        """Assert two values are not equal."""
        if value1 == value2:
            self.log_error(f"{message} (got {value1}, expected != {value2})")
        return value1 != value2


def load_test_data(symbol: str = "SPY") -> pd.DataFrame:
    """Load test data for validation with DataSanity validation."""
    data_file = Path(f"data/ibkr/{symbol}_300_D_1_day.pkl")

    if not data_file.exists():
        # Create synthetic data if file doesn't exist
        logger.info(f"Creating synthetic data for {symbol}")
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)  # For reproducible results

        # Generate realistic price data
        returns = np.random.normal(
            0.0005, 0.015, len(dates)
        )  # 0.05% daily return, 1.5% volatility
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": prices * (1 + np.random.normal(0, 0.002, len(dates))),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                "Close": prices,
                "Volume": np.random.lognormal(10, 0.5, len(dates)),
            }
        )

        # Ensure OHLC relationships are valid
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        # Apply DataSanity validation to synthetic data
        from core.data_sanity import validate_market_data

        return validate_market_data(data, symbol)
    else:
        # Use DataSanity wrapper for loading and validation
        from core.data_sanity import get_data_sanity_wrapper

        wrapper = get_data_sanity_wrapper()
        return wrapper.load_and_validate(str(data_file), symbol)


def test_strategy_selector_meaningful(test: MeaningfulValidationTest) -> bool:
    """Test that strategy selector produces meaningful results."""
    logger.info("🔍 Testing Strategy Selector Meaningful Results...")

    # Load test data
    data = load_test_data("SPY")

    # Create strategy selector with ML enabled
    config = {"ml_selector": {"enabled": True, "epsilon": 0.1}}

    selector = StrategySelector(config)

    # Test multiple selections
    selections = []
    for i in range(10):
        # Use different slices of data to simulate different market conditions
        start_idx = i * 20
        end_idx = start_idx + 100
        if end_idx > len(data):
            break

        slice_data = data.iloc[start_idx:end_idx].copy()
        if len(slice_data) < 50:  # Need minimum data
            continue

        try:
            strategy_name, params, expected_sharpe = selector.select_best_strategy(
                slice_data
            )
            selections.append(
                {
                    "strategy": strategy_name,
                    "expected_sharpe": expected_sharpe,
                    "data_length": len(slice_data),
                }
            )
        except Exception as e:
            test.log_error(f"Strategy selection failed: {e}")
            return False

    # Validate meaningful results
    if not selections:
        test.log_error("No strategy selections made")
        return False

    # Check that we got different strategies (not always fallback)
    strategies_used = set(s["strategy"] for s in selections)
    test.assert_greater(len(strategies_used), 1, "Should use multiple strategies")

    # Check that expected Sharpe ratios are reasonable
    avg_sharpe = np.mean([s["expected_sharpe"] for s in selections])
    test.assert_greater(avg_sharpe, 0.0, "Expected Sharpe should be positive")
    test.assert_greater(avg_sharpe, 0.1, "Expected Sharpe should be meaningful")

    # Check that we have sufficient data for regime detection
    avg_data_length = np.mean([s["data_length"] for s in selections])
    test.assert_greater_equal(
        avg_data_length, 50, "Should have sufficient data for analysis"
    )

    logger.info(
        f"✅ Strategy Selector: {len(selections)} selections, {len(strategies_used)} strategies, avg Sharpe: {avg_sharpe:.3f}"
    )
    return len(test.errors) == 0


def test_paper_trading_meaningful(test: MeaningfulValidationTest) -> bool:
    """Test that paper trading produces meaningful results."""
    logger.info("🔍 Testing Paper Trading Meaningful Results...")

    # Load test data
    data = load_test_data("SPY")

    # Create paper trading engine
    config = {
        "initial_capital": 100000,
        "commission": 0.001,
        "slippage": 0.0005,
        "objective": {
            "type": "expected_log_utility",
            "params": {"risk_aversion": 2.0, "max_position_size": 0.15},
        },
        "ml_selector": {"enabled": True, "epsilon": 0.1},
    }

    engine = TestPaperTradingEngine(config)

    # Run trading simulation
    results = []
    for i in range(20):  # 20 trading cycles
        start_idx = i * 10
        end_idx = start_idx + 50
        if end_idx > len(data):
            break

        slice_data = data.iloc[start_idx:end_idx].copy()
        if len(slice_data) < 30:
            continue

        try:
            # Run trading cycle
            cycle_result = engine.run_trading_cycle(slice_data)
            results.append(cycle_result)
        except Exception as e:
            test.log_error(f"Trading cycle failed: {e}")
            return False

    if not results:
        test.log_error("No trading cycles completed")
        return False

    # Validate meaningful trading results
    total_trades = sum(r.get("trades_executed", 0) for r in results)
    test.assert_greater(total_trades, 0, "Should execute some trades")

    # Check portfolio growth
    final_equity = results[-1].get("portfolio_value", 100000)
    initial_equity = 100000
    total_return = (final_equity - initial_equity) / initial_equity
    test.assert_not_equal(total_return, 0.0, "Should have some portfolio change")

    # Check that we have trading activity
    active_cycles = sum(1 for r in results if r.get("trades_executed", 0) > 0)
    test.assert_greater(active_cycles, 0, "Should have active trading cycles")

    # Check strategy usage
    strategies_used = set()
    for r in results:
        strategy = r.get("selected_strategy", "unknown")
        strategies_used.add(strategy)

    test.assert_greater(len(strategies_used), 0, "Should use strategies")

    logger.info(
        f"✅ Paper Trading: {len(results)} cycles, {total_trades} trades, {total_return:.2%} return"
    )
    return len(test.errors) == 0


def test_walkforward_meaningful(test: MeaningfulValidationTest) -> bool:
    """Test that walk-forward analysis produces meaningful results."""
    logger.info("🔍 Testing Walk-Forward Analysis Meaningful Results...")

    # Load test data
    data = load_test_data("SPY")

    # Create walk-forward analysis
    train_size = 120
    test_size = 40
    step_size = 20

    results = []
    for i in range(0, len(data) - train_size - test_size, step_size):
        train_start = i
        train_end = i + train_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > len(data):
            break

        train_data = data.iloc[train_start:train_end].copy()
        test_data = data.iloc[test_start:test_end].copy()

        try:
            # Create engine for this fold
            config = {
                "initial_capital": 100000,
                "commission": 0.001,
                "slippage": 0.0005,
                "objective": {
                    "type": "mean_variance",
                    "params": {"risk_aversion": 1.5, "max_position_size": 0.15},
                },
                "ml_selector": {"enabled": True, "epsilon": 0.1},
            }

            engine = TestPaperTradingEngine(config)

            # Run test period
            fold_result = engine.run_trading_cycle(test_data)
            fold_result["fold"] = len(results) + 1
            # Handle date formatting safely
            try:
                train_start = str(train_data.iloc[0]["Date"])
                train_end = str(train_data.iloc[-1]["Date"])
                test_start = str(test_data.iloc[0]["Date"])
                test_end = str(test_data.iloc[-1]["Date"])
                fold_result["train_period"] = f"{train_start} - {train_end}"
                fold_result["test_period"] = f"{test_start} - {test_end}"
            except Exception:
                fold_result["train_period"] = f"fold_{len(results) + 1}_train"
                fold_result["test_period"] = f"fold_{len(results) + 1}_test"

            results.append(fold_result)

        except Exception as e:
            test.log_error(f"Walk-forward fold failed: {e}")
            return False

    if not results:
        test.log_error("No walk-forward folds completed")
        return False

    # Validate meaningful walk-forward results
    total_folds = len(results)
    test.assert_greater(total_folds, 5, "Should have multiple folds")

    # Check that we have trading activity across folds
    total_trades = sum(r.get("trades_executed", 0) for r in results)
    test.assert_greater(total_trades, 0, "Should execute trades across folds")

    # Check portfolio performance across folds
    final_values = [r.get("portfolio_value", 100000) for r in results]
    returns = [(v - 100000) / 100000 for v in final_values]

    # Should have some variability in returns
    return_std = np.std(returns)
    test.assert_greater(
        return_std, 0.001, "Should have variability in returns across folds"
    )

    # Should have some positive returns
    positive_returns = sum(1 for r in returns if r > 0)
    test.assert_greater(positive_returns, 0, "Should have some positive returns")

    logger.info(
        f"✅ Walk-Forward: {total_folds} folds, {total_trades} trades, return std: {return_std:.4f}"
    )
    return len(test.errors) == 0


def test_objective_functions_meaningful(test: MeaningfulValidationTest) -> bool:
    """Test that objective functions produce meaningful results."""
    logger.info("🔍 Testing Objective Functions Meaningful Results...")

    # Test different objective types with more realistic parameters
    objectives = [
        (
            "expected_log_utility",
            {"kelly_cap_fraction": 0.25, "risk_aversion_lambda": 2.0},
        ),
        ("mean_variance", {"risk_aversion_lambda": 1.5}),
        ("sortino_utility", {"downside_lambda": 1.0}),
    ]

    for obj_type, params in objectives:
        try:
            # Create config dict for build_objective
            obj_config = {"objective": {"type": obj_type, **params}}
            objective = build_objective(obj_config)

            # Test position sizing with more realistic data
            portfolio_value = 100000
            # Create a series of returns to test with
            returns_series = pd.Series(
                [0.001, 0.002, -0.001, 0.003, 0.001]
            )  # 0.1% to 0.3% returns
            equity_series = pd.Series(
                [portfolio_value * (1 + r) for r in returns_series]
            )

            # Calculate position size using objective's risk budget
            risk_budget, pos_mult = objective.derive_risk_budget(
                returns_series, equity_series, {}
            )
            position_size = risk_budget * pos_mult

            # Validate position sizing
            test.assert_greater_equal(
                position_size, 0.0, f"{obj_type}: Position size should be non-negative"
            )
            test.assert_greater_equal(
                position_size, 0.001, f"{obj_type}: Position size should be meaningful"
            )

            # Test risk budget calculation
            risk_budget, pos_mult = objective.derive_risk_budget(
                returns_series, equity_series, {}
            )
            test.assert_greater(
                risk_budget, 0.0, f"{obj_type}: Risk budget should be positive"
            )

        except Exception as e:
            test.log_error(f"Objective function {obj_type} failed: {e}")
            return False

    logger.info("✅ Objective Functions: All types working correctly")
    return len(test.errors) == 0


def main():
    """Run all meaningful validation tests."""
    print("🎯 MEANINGFUL VALIDATION TEST SUITE")
    print("=" * 50)

    test = MeaningfulValidationTest()

    # Run all tests
    tests = [
        ("Strategy Selector", test_strategy_selector_meaningful),
        ("Paper Trading", test_paper_trading_meaningful),
        ("Walk-Forward Analysis", test_walkforward_meaningful),
        ("Objective Functions", test_objective_functions_meaningful),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            if test_func(test):
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            test.log_error(f"{test_name} test crashed: {e}")
            print(f"💥 {test_name}: CRASHED")

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if test.errors:
        print(f"\n❌ ERRORS ({len(test.errors)}):")
        for error in test.errors:
            print(f"  - {error}")

    if test.warnings:
        print(f"\n⚠️  WARNINGS ({len(test.warnings)}):")
        for warning in test.warnings:
            print(f"  - {warning}")

    # Save detailed results
    results = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "total": total,
        "success_rate": passed / total * 100,
        "errors": test.errors,
        "warnings": test.warnings,
    }

    with open("results/meaningful_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if passed == total and len(test.errors) == 0:
        print("\n🎉 ALL TESTS PASSED - SYSTEM PRODUCES MEANINGFUL RESULTS!")
        return 0
    else:
        print(f"\n❌ {total - passed} TESTS FAILED - SYSTEM NEEDS FIXES")
        return 1


if __name__ == "__main__":
    sys.exit(main())
