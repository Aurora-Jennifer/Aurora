#!/usr/bin/env python3
"""
Component-Level Validation Tests
Comprehensive tests for each component of the enhanced trading system
"""

import json
import logging
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine.paper import PaperTradingEngine
from core.performance import GrowthTargetCalculator
from core.risk.guardrails import RiskGuardrails
from core.strategy_selector import StrategySelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGrowthTargetCalculator(unittest.TestCase):
    """Test Growth Target Calculator functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            "initial_capital": 100000,
            "growth_target": {
                "daily_target_pct": 1.0,
                "compound_growth": True,
                "volatility_adjustment": True,
                "performance_lookback_days": 30,
                "target_adjustment_factor": 0.8,
            },
            "risk_params": {
                "max_position_size": 0.15,
                "volatility_target": 0.20,
                "kelly_fraction": 0.25,
                "position_sizing_method": "kelly_optimal",
            },
        }
        self.calculator = GrowthTargetCalculator(self.config)

    def test_initialization(self):
        """Test calculator initialization."""
        # Test objective-based initialization
        self.assertIsNotNone(self.calculator.objective)
        self.assertEqual(self.calculator.max_position_size, 0.15)
        self.assertEqual(self.calculator.max_position_size, 0.15)
        self.assertEqual(self.calculator.volatility_target, 0.20)
        self.assertEqual(self.calculator.kelly_fraction, 0.25)

    def test_kelly_criterion(self):
        """Test Kelly criterion calculation."""
        # Add some performance data
        for i in range(20):
            daily_return = 0.01 + (i * 0.001)  # Simulate improving performance
            self.calculator.update_performance(daily_return, 100000 * (1.01 ** (i + 1)))

        # Test Kelly position calculation
        kelly_position = self.calculator._calculate_kelly_position_size(0.8)
        self.assertGreater(kelly_position, 0.0)
        self.assertLessEqual(kelly_position, self.calculator.max_position_size)

    def test_volatility_adjustment(self):
        """Test volatility adjustment."""
        position_size = self.calculator.calculate_dynamic_position_size(
            signal_strength=0.8,
            current_capital=100000,
            symbol_volatility=0.02,
            portfolio_volatility=0.03,  # Higher than target
        )

        # Should be reduced due to high volatility
        self.assertLess(position_size, 0.8 * self.calculator.max_position_size)

    def test_performance_adjustment(self):
        """Test performance-based adjustment."""
        # Add underperforming data
        for i in range(10):
            self.calculator.update_performance(0.005, 100000 * (1.005 ** (i + 1)))

        adjustment = self.calculator._calculate_performance_adjustment()
        # Performance adjustment is deprecated; should return 1.0
        self.assertEqual(adjustment, 1.0)

    def test_growth_metrics(self):
        """Test growth metrics calculation."""
        # Add performance data
        for i in range(30):
            daily_return = 0.01 + (np.random.randn() * 0.005)
            self.calculator.update_performance(daily_return, 100000 * (1.01 ** (i + 1)))

        metrics = self.calculator.get_growth_metrics()

        self.assertIn("avg_daily_return", metrics)
        self.assertIn("volatility", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("objective_score", metrics)
        self.assertIn("days_tracked", metrics)
        self.assertEqual(metrics["days_tracked"], 30)


class TestStrategySelector(unittest.TestCase):
    """Test Strategy Selector functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {"symbols": ["SPY", "AAPL", "NVDA"], "initial_capital": 100000}
        self.selector = StrategySelector(self.config)

    def test_initialization(self):
        """Test selector initialization."""
        self.assertIn("regime_aware_ensemble", self.selector.strategies)
        self.assertIn("momentum", self.selector.strategies)
        self.assertIn("mean_reversion", self.selector.strategies)
        self.assertIn("sma_crossover", self.selector.strategies)
        self.assertIn("ensemble_basic", self.selector.strategies)

    def test_strategy_selection(self):
        """Test strategy selection logic."""
        # Create mock market data
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

        (
            strategy_name,
            strategy_params,
            expected_sharpe,
        ) = self.selector.select_best_strategy(market_data)

        self.assertIn(strategy_name, self.selector.strategies.keys())
        self.assertIsInstance(strategy_params, dict)
        self.assertIsInstance(expected_sharpe, float)
        self.assertGreater(expected_sharpe, 0.0)

    def test_parameter_optimization(self):
        """Test parameter optimization."""
        # Test different regimes
        for regime in ["trend", "chop", "volatile"]:
            params = self.selector._get_optimized_params(
                "regime_aware_ensemble", regime, 0.02
            )
            self.assertIn("confidence_threshold", params)
            self.assertIn("trend_following_weight", params)
            self.assertIn("mean_reversion_weight", params)

    def test_performance_update(self):
        """Test performance data update."""
        performance_metrics = {
            "sharpe_ratio": 0.8,
            "total_return": 0.15,
            "max_drawdown": -0.05,
        }

        self.selector.update_performance_data(
            "regime_aware_ensemble", "trend", performance_metrics
        )

        # Check if performance file was created
        performance_file = Path("results/strategy_performance.json")
        if performance_file.exists():
            with open(performance_file) as f:
                data = json.load(f)
            self.assertIn("trend", data)
            self.assertIn("regime_aware_ensemble", data["trend"])


class TestRiskManagement(unittest.TestCase):
    """Test Risk Management functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
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
            },
            "kill_switches": {
                "enabled": True,
                "max_daily_loss_pct": 2.0,
                "max_daily_loss_dollars": 2000,
                "max_drawdown_pct": 10.0,
                "max_position_size_pct": 20.0,
                "max_sector_exposure_pct": 30.0,
            },
        }
        self.guardrails = RiskGuardrails(self.config)

    def test_initialization(self):
        """Test guardrails initialization."""
        self.assertTrue(self.guardrails.kill_switches["enabled"])
        self.assertEqual(self.guardrails.risk_limits["max_position_size"], 0.15)
        self.assertEqual(self.guardrails.risk_limits["stop_loss_pct"], 3.0)

    def test_position_validation(self):
        """Test position size validation."""
        # Valid position
        is_valid = self.guardrails.validate_position_size("AAPL", 100, 150.0, 100000)
        self.assertTrue(is_valid)

        # Invalid position (too large)
        is_valid = self.guardrails.validate_position_size("AAPL", 10000, 150.0, 100000)
        self.assertFalse(is_valid)

    def test_portfolio_risk_check(self):
        """Test portfolio risk monitoring."""
        positions = {"AAPL": 100, "SPY": 50}
        prices = {"AAPL": 150.0, "SPY": 400.0}
        capital = 100000

        risk_metrics = self.guardrails.check_portfolio_risk(positions, prices, capital)

        self.assertIn("portfolio_value", risk_metrics)
        self.assertIn("leverage", risk_metrics)
        self.assertIn("max_concentration", risk_metrics)
        self.assertIn("position_count", risk_metrics)

        self.assertGreater(risk_metrics["portfolio_value"], capital)
        self.assertGreater(risk_metrics["leverage"], 1.0)
        self.assertGreater(risk_metrics["max_concentration"], 0.0)

    def test_kill_switches(self):
        """Test kill switch functionality."""
        # Test with normal returns
        daily_returns = [{"pnl": 100, "capital": 100000}]
        capital = 100000

        triggered = self.guardrails.check_kill_switches(daily_returns, capital)
        self.assertFalse(triggered)

        # Test with large loss
        daily_returns = [{"pnl": -3000, "capital": 97000}]
        triggered = self.guardrails.check_kill_switches(daily_returns, capital)
        self.assertTrue(triggered)


class TestPaperTradingEngine(unittest.TestCase):
    """Test Paper Trading Engine functionality."""

    def setUp(self):
        """Set up test configuration."""
        self.config_file = "config/enhanced_paper_trading_config.json"
        if Path(self.config_file).exists():
            self.engine = PaperTradingEngine(self.config_file)
        else:
            self.skipTest("Enhanced config file not found")

    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine.growth_calculator)
        self.assertIsNotNone(self.engine.strategy_selector)
        self.assertIsNotNone(self.engine.regime_detector)
        self.assertIsNotNone(self.engine.risk_guardrails)
        self.assertEqual(self.engine.capital, 100000)

    def test_trading_cycle(self):
        """Test complete trading cycle."""
        test_date = date.today()
        result = self.engine.run_trading_cycle(test_date)

        self.assertIn("status", result)
        self.assertIn("date", result)
        self.assertIn("regime", result)
        self.assertIn("strategy", result)
        self.assertIn("expected_sharpe", result)
        self.assertIn("trades_executed", result)
        self.assertIn("performance_metrics", result)

        # Should not crash
        self.assertIn(result["status"], ["success", "no_data", "error"])

    def test_growth_metrics(self):
        """Test growth metrics tracking."""
        metrics = self.engine.growth_calculator.get_growth_metrics()

        self.assertIn("avg_daily_return", metrics)
        self.assertIn("volatility", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("objective_score", metrics)
        self.assertIn("days_tracked", metrics)

    def test_strategy_selection(self):
        """Test strategy selection integration."""
        summary = self.engine.strategy_selector.get_strategy_summary()

        self.assertIn("available_strategies", summary)
        self.assertIn("strategy_details", summary)
        self.assertIn("last_update", summary)

        self.assertGreater(len(summary["available_strategies"]), 0)


def run_component_tests():
    """Run all component tests and return results."""
    print("🧪 Running Component-Level Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestGrowthTargetCalculator)
    )
    test_suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestStrategySelector)
    )
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRiskManagement))
    test_suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestPaperTradingEngine)
    )

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 50)
    print("COMPONENT TEST SUMMARY")
    print("=" * 50)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)

    print(f"Total Tests: {total_tests}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(
        f"Success Rate: {((total_tests - failures - errors) / total_tests * 100):.1f}%"
    )

    if failures > 0:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if errors > 0:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")

    success = failures == 0 and errors == 0
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")

    return success


if __name__ == "__main__":
    success = run_component_tests()
    sys.exit(0 if success else 1)
