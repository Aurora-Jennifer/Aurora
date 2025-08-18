#!/usr/bin/env python3
"""
Live Paper Trading Test Script

This script tests live paper trading using signal templates generated
from walk-forward analysis.
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from core.engine.paper import PaperTradingEngine
from core.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class LivePaperTradingTester:
    """Test live paper trading with signal templates."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the live paper trading tester.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.symbols = config.get("symbols", ["SPY"])
        self.initial_capital = config.get("initial_capital", 100000)
        self.test_duration_days = config.get("test_duration_days", 30)
        self.templates_dir = config.get("templates_dir", "results/signal_templates")
        self.results_dir = config.get("results_dir", "results/live_paper_trading")

        # Initialize results tracking
        self.test_results = {
            "test_start": datetime.now().isoformat(),
            "symbols": self.symbols,
            "initial_capital": self.initial_capital,
            "daily_results": [],
            "trades": [],
            "performance_metrics": {},
        }

    def load_signal_templates(self) -> Dict[str, Any]:
        """
        Load signal templates from the templates directory.

        Returns:
            Dictionary of signal templates by symbol
        """
        templates = {}
        templates_path = Path(self.templates_dir)

        if not templates_path.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return templates

        # Look for the most recent templates
        template_files = list(
            templates_path.glob("individual_templates/*_template_*.json")
        )

        if not template_files:
            logger.warning("No template files found")
            return templates

        # Sort by modification time and get the most recent
        latest_templates = {}
        for file_path in template_files:
            symbol = file_path.stem.split("_template_")[0]
            if (
                symbol not in latest_templates
                or file_path.stat().st_mtime > latest_templates[symbol].stat().st_mtime
            ):
                latest_templates[symbol] = file_path

        # Load templates
        for symbol, file_path in latest_templates.items():
            try:
                with open(file_path) as f:
                    template_data = json.load(f)
                    templates[symbol] = template_data
                    logger.info(f"Loaded template for {symbol}")
            except Exception as e:
                logger.error(f"Error loading template for {symbol}: {e}")

        return templates

    def create_paper_trading_config(self, templates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create paper trading configuration from signal templates.

        Args:
            templates: Signal templates dictionary

        Returns:
            Paper trading configuration
        """
        config = {
            "symbols": self.symbols,
            "initial_capital": self.initial_capital,
            "use_ibkr": False,  # Use yfinance for testing
            "strategy": "regime_aware_ensemble",
            "strategy_params": {
                "regime_aware_ensemble": {
                    "confidence_threshold": 0.3,
                    "regime_lookback": 252,
                    "trend_following_weight": 0.6,
                    "mean_reversion_weight": 0.4,
                }
            },
            "risk_params": {"max_position_size_pct": 10.0, "max_daily_loss_pct": 2.0},
            "execution_params": {"slippage_bps": 2, "commission_bps": 5},
            "notifications": {"discord_enabled": False},
            "performance_tracking": {"save_results": True},
            "signal_templates": templates,
        }

        # Override with template-specific parameters if available
        for symbol in self.symbols:
            if symbol in templates:
                template = templates[symbol].get("signal_template", {})

                # Update strategy parameters based on template
                if "regime_weights" in template:
                    config["strategy_params"]["regime_aware_ensemble"].update(
                        {
                            "trend_following_weight": template["regime_weights"].get(
                                "trend_following_weight", 0.6
                            ),
                            "mean_reversion_weight": template["regime_weights"].get(
                                "mean_reversion_weight", 0.4
                            ),
                        }
                    )

                # Update risk parameters based on template
                if "position_sizing" in template:
                    max_size = template["position_sizing"].get(
                        "max_position_size", 1000
                    )
                    config["risk_params"]["max_position_size_pct"] = min(
                        10.0, max_size / self.initial_capital * 100
                    )

        return config

    def run_live_paper_trading_test(self) -> Dict[str, Any]:
        """
        Run live paper trading test.

        Returns:
            Test results dictionary
        """
        logger.info("Starting live paper trading test")

        # Load signal templates
        templates = self.load_signal_templates()

        if not templates:
            logger.warning("No templates loaded, using default configuration")

        # Create paper trading configuration
        paper_config = self.create_paper_trading_config(templates)

        # Save configuration
        config_file = "test_live_paper_trading_config.json"
        with open(config_file, "w") as f:
            json.dump(paper_config, f, indent=2)

        # Initialize paper trading engine
        try:
            engine = PaperTradingEngine(config_file)
            logger.info("Paper trading engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing paper trading engine: {e}")
            return self.test_results

        # Run trading cycles for the test duration
        start_date = date.today()
        current_date = start_date

        logger.info(
            f"Running trading cycles from {start_date} for {self.test_duration_days} days"
        )

        for day in range(self.test_duration_days):
            try:
                # Run trading cycle
                cycle_result = engine.run_trading_cycle(current_date)

                # Record daily results
                daily_result = {
                    "date": current_date.isoformat(),
                    "cycle_result": cycle_result,
                    "portfolio_value": engine.capital,  # Use capital directly
                    "positions": engine.get_positions(),
                    "cash": engine.capital,  # Use capital as cash
                }

                self.test_results["daily_results"].append(daily_result)

                # Log progress
                if day % 5 == 0:
                    portfolio_value = engine.capital  # Use capital directly
                    pnl = portfolio_value - self.initial_capital
                    pnl_pct = (pnl / self.initial_capital) * 100
                    logger.info(
                        f"Day {day+1}/{self.test_duration_days}: Portfolio ${portfolio_value:,.2f} (PnL: {pnl:+,.2f}, {pnl_pct:+.2f}%)"
                    )

                # Move to next trading day
                current_date += timedelta(days=1)

                # Small delay to simulate real-time processing
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in trading cycle for {current_date}: {e}")
                continue

        # Get final results
        final_portfolio_value = engine.capital  # Use capital directly
        final_positions = engine.get_positions()
        trade_history = engine.get_trade_history()

        # Calculate performance metrics
        total_pnl = final_portfolio_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(self.test_results["daily_results"])):
            prev_value = self.test_results["daily_results"][i - 1]["portfolio_value"]
            curr_value = self.test_results["daily_results"][i]["portfolio_value"]
            daily_return = (
                (curr_value - prev_value) / prev_value if prev_value > 0 else 0
            )
            daily_returns.append(daily_return)

        # Calculate metrics
        if daily_returns:
            avg_daily_return = sum(daily_returns) / len(daily_returns)
            volatility = (
                sum((r - avg_daily_return) ** 2 for r in daily_returns)
                / len(daily_returns)
            ) ** 0.5
            sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
        else:
            avg_daily_return = volatility = sharpe_ratio = 0

        # Update test results
        self.test_results.update(
            {
                "test_end": datetime.now().isoformat(),
                "final_portfolio_value": final_portfolio_value,
                "final_positions": final_positions,
                "trade_history": trade_history,
                "performance_metrics": {
                    "total_pnl": total_pnl,
                    "total_pnl_pct": total_pnl_pct,
                    "avg_daily_return": avg_daily_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "total_trades": len(trade_history),
                    "test_duration_days": self.test_duration_days,
                },
            }
        )

        # Shutdown engine
        engine.shutdown()

        logger.info("Live paper trading test completed")
        return self.test_results

    def save_test_results(self, results: Dict[str, Any]):
        """
        Save test results to files.

        Args:
            results: Test results dictionary
        """
        results_path = Path(self.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        results_file = results_path / f"live_paper_trading_test_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save performance summary
        summary = {
            "test_period": {
                "start": results.get("test_start"),
                "end": results.get("test_end"),
            },
            "symbols": results.get("symbols", []),
            "initial_capital": results.get("initial_capital", 0),
            "final_portfolio_value": results.get("final_portfolio_value", 0),
            "performance_metrics": results.get("performance_metrics", {}),
        }

        summary_file = results_path / f"performance_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save daily results as CSV
        daily_results = results.get("daily_results", [])
        if daily_results:
            df = pd.DataFrame(daily_results)
            csv_file = results_path / f"daily_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)

        logger.info(f"Test results saved to {results_path}")
        logger.info(f"Full results: {results_file}")
        logger.info(f"Performance summary: {summary_file}")
        if daily_results:
            logger.info(f"Daily results: {csv_file}")

    def print_test_summary(self, results: Dict[str, Any]):
        """
        Print test summary to console.

        Args:
            results: Test results dictionary
        """
        metrics = results.get("performance_metrics", {})

        print("\n" + "=" * 60)
        print("📊 LIVE PAPER TRADING TEST SUMMARY")
        print("=" * 60)
        print(
            f"Test Period: {results.get('test_start', 'N/A')} to {results.get('test_end', 'N/A')}"
        )
        print(f"Symbols: {', '.join(results.get('symbols', []))}")
        print(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}")
        print(f"Final Portfolio Value: ${results.get('final_portfolio_value', 0):,.2f}")
        print()

        print("📈 PERFORMANCE METRICS")
        print("-" * 30)
        print(f"Total PnL: ${metrics.get('total_pnl', 0):+,.2f}")
        print(f"Total PnL %: {metrics.get('total_pnl_pct', 0):+.2f}%")
        print(f"Average Daily Return: {metrics.get('avg_daily_return', 0):+.4f}%")
        print(f"Volatility: {metrics.get('volatility', 0):.4f}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Test Duration: {metrics.get('test_duration_days', 0)} days")
        print()

        # Final positions
        final_positions = results.get("final_positions", {})
        if final_positions:
            print("💰 FINAL POSITIONS")
            print("-" * 30)
            for symbol, position in final_positions.items():
                if position.get("qty", 0) != 0:
                    print(
                        f"{symbol}: {position.get('qty', 0):.2f} shares @ ${position.get('avg_price', 0):.2f}"
                    )

        print("=" * 60)


def main():
    """Main function to run live paper trading test."""
    # Configuration
    config = {
        "symbols": ["SPY", "QQQ"],
        "initial_capital": 100000,
        "test_duration_days": 30,
        "templates_dir": "results/signal_templates",
        "results_dir": "results/live_paper_trading",
    }

    print("🚀 Starting Live Paper Trading Test")
    print("=" * 50)

    # Initialize tester
    tester = LivePaperTradingTester(config)

    # Run test
    results = tester.run_live_paper_trading_test()

    # Save results
    tester.save_test_results(results)

    # Print summary
    tester.print_test_summary(results)

    print("\n✅ Live paper trading test completed!")
    print(f"📁 Results saved to: {config['results_dir']}/")


if __name__ == "__main__":
    main()
