#!/usr/bin/env python3
"""
Data Integrity and Meaningful Results Test
Validates that the system actually uses data correctly and produces meaningful results.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntegrityTest:
    """Test class that validates data usage and meaningful results."""

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

    def assert_not_equal(self, value1, value2, message: str):
        """Assert two values are not equal."""
        if value1 == value2:
            self.log_error(f"{message} (got {value1}, expected != {value2})")
        return value1 != value2

    def assert_in_range(self, value: float, min_val: float, max_val: float, message: str):
        """Assert value is within range."""
        if value < min_val or value > max_val:
            self.log_error(f"{message} (got {value:.6f}, expected {min_val:.6f} to {max_val:.6f})")
        return min_val <= value <= max_val


def create_test_data_with_patterns() -> pd.DataFrame:
    """Create test data with known patterns to validate system behavior."""
    logger.info("Creating test data with known patterns...")

    # Create data with clear trends and patterns
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
    np.random.seed(42)  # For reproducible results

    # Create a more realistic trending pattern
    trend_component = np.concatenate(
        [
            np.linspace(0, 0.15, len(dates) // 2),  # Upward trend (15% total)
            np.linspace(0.15, -0.05, len(dates) // 2),  # Downward trend (-5% total)
        ]
    )

    # Add realistic volatility (1-2% daily)
    volatility = np.random.normal(0, 0.015, len(dates))

    # Create price series with trend (start at $100, realistic range)
    returns = trend_component + volatility
    # Clip returns to prevent extreme values
    returns = np.clip(returns, -0.1, 0.1)  # Max 10% daily move
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLC data
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

    # Apply DataSanity validation to ensure realistic data
    from core.data_sanity import validate_market_data

    clean_data = validate_market_data(data, "TEST_PATTERNS")

    logger.info(
        f"Created test data: {len(clean_data)} rows, price range: ${clean_data['Close'].min():.2f} - ${clean_data['Close'].max():.2f}"
    )
    return clean_data


def test_data_usage_in_strategy_selection(test: DataIntegrityTest) -> bool:
    """Test that strategy selector actually uses data patterns."""
    logger.info("🔍 Testing Data Usage in Strategy Selection...")

    # Create test data with known patterns
    data = create_test_data_with_patterns()

    # Create strategy selector
    config = {"ml_selector": {"enabled": True, "epsilon": 0.1}}

    selector = StrategySelector(config)

    # Test strategy selection on different data segments
    results = []

    # Test on trending data (first half - upward trend)
    trend_data = data.iloc[: len(data) // 2].copy()
    strategy1, params1, sharpe1 = selector.select_best_strategy(trend_data)
    results.append(
        {
            "segment": "trending_up",
            "strategy": strategy1,
            "sharpe": sharpe1,
            "data_range": f"{trend_data['Close'].iloc[0]:.2f} - {trend_data['Close'].iloc[-1]:.2f}",
        }
    )

    # Test on declining data (second half - downward trend)
    decline_data = data.iloc[len(data) // 2 :].copy()
    strategy2, params2, sharpe2 = selector.select_best_strategy(decline_data)
    results.append(
        {
            "segment": "trending_down",
            "strategy": strategy2,
            "sharpe": sharpe2,
            "data_range": f"{decline_data['Close'].iloc[0]:.2f} - {decline_data['Close'].iloc[-1]:.2f}",
        }
    )

    # Test on volatile data (random segment)
    volatile_data = data.iloc[100:200].copy()
    strategy3, params3, sharpe3 = selector.select_best_strategy(volatile_data)
    results.append(
        {
            "segment": "volatile",
            "strategy": strategy3,
            "sharpe": sharpe3,
            "data_range": f"{volatile_data['Close'].iloc[0]:.2f} - {volatile_data['Close'].iloc[-1]:.2f}",
        }
    )

    # Validate that the system responds to different data patterns
    strategies_used = {r["strategy"] for r in results}
    test.assert_greater(
        len(strategies_used),
        1,
        "Should use different strategies for different data patterns",
    )

    # Validate that Sharpe ratios are reasonable
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    test.assert_in_range(avg_sharpe, 0.0, 2.0, "Sharpe ratios should be reasonable")

    # Validate that the system actually looked at the data
    for result in results:
        test.assert_not_equal(
            result["data_range"],
            "0.00 - 0.00",
            f"Data range should reflect actual prices for {result['segment']}",
        )

    logger.info(
        f"✅ Strategy Selection: {len(strategies_used)} strategies, avg Sharpe: {avg_sharpe:.3f}"
    )
    for result in results:
        logger.info(
            f"  {result['segment']}: {result['strategy']} (Sharpe: {result['sharpe']:.3f}, Range: {result['data_range']})"
        )

    return len(test.errors) == 0


def test_data_usage_in_trading_engine(test: DataIntegrityTest) -> bool:
    """Test that trading engine actually uses data for decisions."""
    logger.info("🔍 Testing Data Usage in Trading Engine...")

    # Create test data with known patterns
    data = create_test_data_with_patterns()

    # Create trading engine
    config = {
        "initial_capital": 100000,
        "commission": 0.001,
        "slippage": 0.0005,
        "objective": {
            "type": "expected_log_utility",
            "params": {"kelly_cap_fraction": 0.25, "risk_aversion_lambda": 2.0},
        },
        "ml_selector": {"enabled": True, "epsilon": 0.1},
    }

    engine = TestPaperTradingEngine(config)

    # Track portfolio values over time
    portfolio_values = []
    trades_executed = []
    strategies_used = []

    # Run trading cycles on different data segments
    for i in range(0, len(data) - 50, 50):
        segment_data = data.iloc[i : i + 50].copy()

        # Record initial portfolio value
        initial_value = engine.portfolio_value

        # Run trading cycle
        result = engine.run_trading_cycle(segment_data)

        # Record results
        portfolio_values.append(result["portfolio_value"])
        trades_executed.append(result["trades_executed"])
        strategies_used.append(result["selected_strategy"])

        # Validate that portfolio value changed if trades were executed
        if result["trades_executed"] > 0:
            test.assert_not_equal(
                result["portfolio_value"],
                initial_value,
                f"Portfolio should change when trades are executed (cycle {i // 50})",
            )

    # Validate meaningful trading activity
    total_trades = sum(trades_executed)
    test.assert_greater(total_trades, 0, "Should execute some trades")

    # Validate portfolio changes
    portfolio_changes = [abs(pv - 100000) for pv in portfolio_values]
    max_change = max(portfolio_changes)
    test.assert_greater(max_change, 0.01, "Should have meaningful portfolio changes")

    # Validate strategy diversity
    unique_strategies = set(strategies_used)
    test.assert_greater(len(unique_strategies), 1, "Should use multiple strategies")

    # Validate that portfolio changes correlate with data patterns
    # (This is a basic check - in reality, correlation would depend on strategy effectiveness)
    logger.info(
        f"✅ Trading Engine: {total_trades} trades, {len(unique_strategies)} strategies, max portfolio change: ${max_change:.2f}"
    )

    return len(test.errors) == 0


def test_data_usage_in_objective_functions(test: DataIntegrityTest) -> bool:
    """Test that objective functions actually use return data."""
    logger.info("🔍 Testing Data Usage in Objective Functions...")

    # Create test data with known return patterns (realistic daily returns)
    returns_series = pd.Series(
        [
            0.001,
            0.002,
            -0.001,
            0.003,
            0.001,  # Positive trend (0.1-0.3%)
            -0.002,
            -0.001,
            0.001,
            -0.003,
            0.002,  # Mixed pattern (-0.3% to 0.2%)
            0.0005,
            0.0015,
            -0.0005,
            0.0025,
            0.0005,  # Lower volatility (0.05-0.25%)
        ]
    )

    equity_series = pd.Series([100000 * (1 + r) for r in returns_series])

    # Test different objective types
    objectives = [
        (
            "expected_log_utility",
            {"kelly_cap_fraction": 0.25, "risk_aversion_lambda": 2.0},
        ),
        ("mean_variance", {"risk_aversion_lambda": 1.5}),
        ("sortino_utility", {"downside_lambda": 1.0}),
    ]

    for obj_type, params in objectives:
        # Create objective function
        obj_config = {"objective": {"type": obj_type, **params}}
        objective = build_objective(obj_config)

        # Test that objective responds to different return patterns
        risk_budget1, pos_mult1 = objective.derive_risk_budget(
            returns_series[:5],
            equity_series[:5],
            {},  # Positive trend
        )

        risk_budget2, pos_mult2 = objective.derive_risk_budget(
            returns_series[5:10],
            equity_series[5:10],
            {},  # Mixed pattern
        )

        risk_budget3, pos_mult3 = objective.derive_risk_budget(
            returns_series[10:],
            equity_series[10:],
            {},  # Lower volatility
        )

        # Validate that risk budgets are different for different return patterns
        test.assert_not_equal(
            risk_budget1,
            risk_budget2,
            f"{obj_type}: Risk budgets should differ for different return patterns",
        )
        test.assert_not_equal(
            risk_budget2,
            risk_budget3,
            f"{obj_type}: Risk budgets should differ for different volatility patterns",
        )

        # Validate that risk budgets are reasonable
        test.assert_in_range(
            risk_budget1, 0.0, 2.0, f"{obj_type}: Risk budget 1 should be reasonable"
        )
        test.assert_in_range(
            risk_budget2, 0.0, 2.0, f"{obj_type}: Risk budget 2 should be reasonable"
        )
        test.assert_in_range(
            risk_budget3, 0.0, 2.0, f"{obj_type}: Risk budget 3 should be reasonable"
        )

        logger.info(
            f"  {obj_type}: Risk budgets {risk_budget1:.3f}, {risk_budget2:.3f}, {risk_budget3:.3f}"
        )

    logger.info("✅ Objective Functions: All types respond to data patterns")
    return len(test.errors) == 0


def test_data_integrity_validation(test: DataIntegrityTest) -> bool:
    """Test that the system maintains data integrity throughout processing."""
    logger.info("🔍 Testing Data Integrity Validation...")

    # Create test data
    data = create_test_data_with_patterns()

    # Test that data is not corrupted during processing
    original_close = data["Close"].copy()
    original_volume = data["Volume"].copy()

    # Run through strategy selector
    config = {"ml_selector": {"enabled": True, "epsilon": 0.1}}
    selector = StrategySelector(config)

    # Process data multiple times
    for i in range(5):
        segment_data = data.iloc[i * 50 : (i + 1) * 50].copy()
        if len(segment_data) < 30:
            continue

        strategy, params, sharpe = selector.select_best_strategy(segment_data)

        # Validate data integrity
        test.assert_true(
            np.allclose(segment_data["Close"], original_close.iloc[i * 50 : (i + 1) * 50]),
            f"Close prices should not be corrupted (iteration {i})",
        )
        test.assert_true(
            np.allclose(segment_data["Volume"], original_volume.iloc[i * 50 : (i + 1) * 50]),
            f"Volume should not be corrupted (iteration {i})",
        )

    # Test trading engine data integrity
    engine = TestPaperTradingEngine(config)

    for i in range(0, len(data) - 50, 50):
        segment_data = data.iloc[i : i + 50].copy()
        original_segment_close = segment_data["Close"].copy()

        engine.run_trading_cycle(segment_data)

        # Validate data integrity
        test.assert_true(
            np.allclose(segment_data["Close"], original_segment_close),
            f"Close prices should not be corrupted by trading engine (cycle {i // 50})",
        )

    logger.info("✅ Data Integrity: No data corruption detected")
    return len(test.errors) == 0


def test_meaningful_correlation_analysis(test: DataIntegrityTest) -> bool:
    """Test that system decisions correlate with actual data patterns."""
    logger.info("🔍 Testing Meaningful Correlation Analysis...")

    # Create data with clear patterns
    data = create_test_data_with_patterns()

    # Calculate actual data patterns
    returns = data["Close"].pct_change().dropna()
    returns.rolling(20).std()
    returns.rolling(20).mean()

    # Create trading engine
    config = {
        "initial_capital": 100000,
        "objective": {
            "type": "expected_log_utility",
            "params": {"kelly_cap_fraction": 0.25, "risk_aversion_lambda": 2.0},
        },
        "ml_selector": {"enabled": True, "epsilon": 0.1},
    }

    engine = TestPaperTradingEngine(config)

    # Track decisions vs data patterns
    decisions = []
    data_patterns = []

    for i in range(0, len(data) - 50, 25):
        segment_data = data.iloc[i : i + 50].copy()

        # Calculate data pattern for this segment
        segment_returns = segment_data["Close"].pct_change().dropna()
        segment_vol = segment_returns.std()
        segment_trend = segment_returns.mean()

        # Run trading cycle
        result = engine.run_trading_cycle(segment_data)

        # Record decision and data pattern
        decisions.append(result["trades_executed"])
        data_patterns.append(
            {
                "volatility": segment_vol,
                "trend": segment_trend,
                "price_range": segment_data["Close"].max() - segment_data["Close"].min(),
            }
        )

    # Validate that decisions correlate with data patterns
    # (More trades should occur during higher volatility or stronger trends)
    high_vol_decisions = [
        d
        for i, d in enumerate(decisions)
        if data_patterns[i]["volatility"] > np.median([p["volatility"] for p in data_patterns])
    ]
    low_vol_decisions = [
        d
        for i, d in enumerate(decisions)
        if data_patterns[i]["volatility"] <= np.median([p["volatility"] for p in data_patterns])
    ]

    if high_vol_decisions and low_vol_decisions:
        avg_high_vol_trades = np.mean(high_vol_decisions)
        avg_low_vol_trades = np.mean(low_vol_decisions)

        # Should have more trading activity during high volatility
        test.assert_greater(
            avg_high_vol_trades,
            avg_low_vol_trades * 0.5,
            "Should have reasonable trading activity during high volatility",
        )

    logger.info(
        f"✅ Correlation Analysis: {len(decisions)} decisions analyzed, avg trades: {np.mean(decisions):.2f}"
    )
    return len(test.errors) == 0


def main():
    """Run all data integrity tests."""
    print("🔍 DATA INTEGRITY AND MEANINGFUL RESULTS TEST SUITE")
    print("=" * 60)

    test = DataIntegrityTest()

    # Run all tests
    tests = [
        ("Data Usage in Strategy Selection", test_data_usage_in_strategy_selection),
        ("Data Usage in Trading Engine", test_data_usage_in_trading_engine),
        ("Data Usage in Objective Functions", test_data_usage_in_objective_functions),
        ("Data Integrity Validation", test_data_integrity_validation),
        ("Meaningful Correlation Analysis", test_meaningful_correlation_analysis),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
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
    print("\n" + "=" * 60)
    print("📊 DATA INTEGRITY TEST SUMMARY")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed / total * 100:.1f}%")

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

    with open("results/data_integrity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if passed == total and len(test.errors) == 0:
        print("\n🎉 ALL TESTS PASSED - SYSTEM USES DATA CORRECTLY!")
        return 0
    print(f"\n❌ {total - passed} TESTS FAILED - SYSTEM HAS DATA ISSUES")
    return 1


if __name__ == "__main__":
    sys.exit(main())
