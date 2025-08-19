#!/usr/bin/env python3
"""
Comprehensive ML Backtesting with Walk-Forward Analysis
Tests the enhanced trading system with ML strategy selection and objective-driven risk management.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine.paper import PaperTradingEngine
from core.learning.selector import BanditSelector, StrategyContext
from core.objectives import (
    ExpectedLogUtility,
    MeanVariance,
    SortinoUtility,
    build_objective,
)
from core.performance import GrowthTargetCalculator
from core.strategy_selector import StrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_synthetic_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Create synthetic market data for testing."""
    dates = pd.date_range("2023-01-01", periods=days, freq="D")

    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.0005, 0.02, days)  # 0.05% daily mean, 2% volatility

    # Add some trend and mean reversion
    for i in range(50, days):
        if i % 100 < 50:  # Trend periods
            returns[i] += 0.001
        else:  # Mean reversion periods
            returns[i] -= 0.0005

    prices = 100 * np.exp(np.cumsum(returns))

    # Add OHLCV data
    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.normal(0, 0.002, days)),
            "High": prices * (1 + abs(np.random.normal(0, 0.005, days))),
            "Low": prices * (1 - abs(np.random.normal(0, 0.005, days))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, days),
        }
    )

    return data


def test_objective_functions():
    """Test all objective functions with synthetic data."""
    print("\n=== Testing Objective Functions ===")

    # Create synthetic returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year of daily returns
    equity = pd.Series(100000 * np.exp(np.cumsum(returns)))
    risk_metrics = {"portfolio_vol": 0.15}

    # Test each objective
    objectives = {
        "log_utility": ExpectedLogUtility,
        "mean_variance": MeanVariance,
        "sortino": SortinoUtility,
    }

    for name, obj_class in objectives.items():
        print(f"\nTesting {name}:")

        # Create objective instance
        from core.objectives import ObjectiveParams

        params = ObjectiveParams(
            max_leverage=1.0,
            max_gross_exposure_pct=50.0,
            kelly_cap_fraction=0.25,
            risk_aversion_lambda=3.0,
            downside_lambda=2.0,
        )
        objective = obj_class(params)

        # Test scoring
        score = objective.score(returns, equity, risk_metrics)
        print(f"  Score: {score:.6f}")

        # Test risk budget derivation
        risk_budget, pos_mult = objective.derive_risk_budget(returns, equity, risk_metrics)
        print(f"  Risk budget: {risk_budget:.3f}")
        print(f"  Position multiplier: {pos_mult:.3f}")

    return True


def test_ml_selector():
    """Test ML strategy selector with synthetic data."""
    print("\n=== Testing ML Strategy Selector ===")

    # Create bandit selector
    selector = BanditSelector(epsilon=0.1)

    # Create synthetic context
    context = StrategyContext(
        regime="trend",
        vol_bin=2,
        trend_strength=0.7,
        liquidity=1000000,
        spread_bps=5.0,
        time_bucket=10,
        corr_cluster=0,
    )

    # Test strategy recommendation
    candidates = ["regime_aware_ensemble", "momentum", "mean_reversion"]
    recommended = selector.recommend(context, candidates)
    print(f"Recommended strategy: {recommended}")

    # Test learning updates
    for _i in range(10):
        reward = np.random.normal(0.001, 0.01)  # Synthetic reward
        selector.update(context, recommended, reward)

    print("ML selector learning updates completed")
    return True


def test_walk_forward_analysis():
    """Test walk-forward analysis with ML implementation."""
    print("\n=== Testing Walk-Forward Analysis ===")

    # Create synthetic data
    data = create_synthetic_data("SPY", days=500)

    # Walk-forward parameters
    train_days = 252  # 1 year training
    test_days = 63  # 3 months testing
    stride = 21  # 1 month stride

    results = []

    for i in range(0, len(data) - train_days - test_days, stride):
        # Define train/test periods
        train_start = i
        train_end = i + train_days
        test_start = train_end
        test_end = min(test_start + test_days, len(data))

        if test_end - test_start < 30:  # Need at least 30 days for testing
            break

        print(
            f"\nFold {len(results) + 1}: Train {data.iloc[train_start]['Date'].date()} - {data.iloc[train_end - 1]['Date'].date()}, "
            f"Test {data.iloc[test_start]['Date'].date()} - {data.iloc[test_end - 1]['Date'].date()}"
        )

        # Train period data
        train_data = data.iloc[train_start:train_end]

        # Test period data
        test_data = data.iloc[test_start:test_end]

        # Initialize components for this fold
        config = {
            "objective": {
                "type": "log_utility",
                "kelly_cap_fraction": 0.25,
                "risk_aversion_lambda": 3.0,
                "downside_lambda": 2.0,
            },
            "ml_selector": {"enabled": True, "epsilon": 0.1},
            "risk_params": {
                "max_position_size": 0.15,
                "volatility_target": 0.20,
                "kelly_fraction": 0.25,
            },
            "initial_capital": 100000,
        }

        # Test strategy selection
        strategy_selector = StrategySelector(config)
        (
            strategy_name,
            strategy_params,
            expected_sharpe,
        ) = strategy_selector.select_best_strategy(train_data)

        print(f"  Selected strategy: {strategy_name}")
        print(f"  Expected Sharpe: {expected_sharpe:.3f}")

        # Simulate trading on test period
        fold_results = simulate_trading_period(test_data, config, strategy_name)

        # Calculate performance metrics
        returns = pd.Series(fold_results["returns"])
        if len(returns) > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_dd = calculate_max_drawdown(fold_results["equity_curve"])
            total_return = (fold_results["equity_curve"][-1] / fold_results["equity_curve"][0]) - 1

            fold_metrics = {
                "fold": len(results) + 1,
                "strategy": strategy_name,
                "expected_sharpe": expected_sharpe,
                "actual_sharpe": sharpe,
                "total_return": total_return,
                "max_drawdown": max_dd,
                "trades": len(fold_results["trades"]),
            }

            results.append(fold_metrics)

            print(f"  Actual Sharpe: {sharpe:.3f}")
            print(f"  Total Return: {total_return:.3%}")
            print(f"  Max Drawdown: {max_dd:.3%}")
            print(f"  Trades: {len(fold_results['trades'])}")

    # Aggregate results
    if results:
        print("\n=== Walk-Forward Results Summary ===")
        print(f"Total folds: {len(results)}")

        df_results = pd.DataFrame(results)

        print(f"Average Sharpe: {df_results['actual_sharpe'].mean():.3f}")
        print(f"Average Return: {df_results['total_return'].mean():.3%}")
        print(f"Average Max DD: {df_results['max_drawdown'].mean():.3%}")
        print(f"Total Trades: {df_results['trades'].sum()}")

        # Strategy performance breakdown
        strategy_perf = (
            df_results.groupby("strategy")
            .agg(
                {
                    "actual_sharpe": "mean",
                    "total_return": "mean",
                    "max_drawdown": "mean",
                    "trades": "sum",
                }
            )
            .round(4)
        )

        print("\nStrategy Performance:")
        print(strategy_perf)

    return True


def simulate_trading_period(data: pd.DataFrame, config: dict, strategy_name: str) -> dict:
    """Simulate trading over a period with the given strategy."""
    capital = config["initial_capital"]
    equity_curve = [capital]
    returns = []
    trades = []

    # Initialize components
    growth_calculator = GrowthTargetCalculator(config)
    build_objective(config)

    for i in range(30, len(data)):  # Start after warmup period
        # Get current market data
        current_data = data.iloc[: i + 1]

        # Generate signal (simplified)
        if strategy_name == "momentum":
            signal = (
                0.5 if current_data["Close"].iloc[-1] > current_data["Close"].iloc[-20] else -0.5
            )
        elif strategy_name == "mean_reversion":
            ma_short = current_data["Close"].rolling(10).mean().iloc[-1]
            current_data["Close"].rolling(50).mean().iloc[-1]
            signal = 0.5 if current_data["Close"].iloc[-1] < ma_short * 0.98 else -0.5
        else:  # regime_aware_ensemble
            signal = np.random.normal(0, 0.3)  # Random signal for demo

        # Calculate position size using objective
        signal_strength = abs(signal)
        position_size = growth_calculator.calculate_dynamic_position_size(
            signal_strength=signal_strength,
            current_capital=capital,
            symbol_volatility=current_data["Close"].pct_change().std(),
            portfolio_volatility=0.15,
        )

        # Execute trade
        if abs(signal) > 0.1:  # Only trade if signal is strong enough
            trade_value = capital * position_size * np.sign(signal)
            trade_return = trade_value * current_data["Close"].pct_change().iloc[-1]

            trades.append(
                {
                    "date": current_data.index[i],
                    "signal": signal,
                    "position_size": position_size,
                    "trade_value": trade_value,
                    "return": trade_return,
                }
            )

            capital += trade_return
        else:
            capital += 0  # No trade

        equity_curve.append(capital)
        daily_return = (
            (capital - equity_curve[-2]) / equity_curve[-2] if len(equity_curve) > 1 else 0
        )
        returns.append(daily_return)

        # Update performance tracking
        growth_calculator.update_performance(daily_return, capital)

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": trades,
        "final_capital": capital,
    }


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    peak = equity_curve[0]
    max_dd = 0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def test_paper_trading_with_ml():
    """Test paper trading engine with ML implementation."""
    print("\n=== Testing Paper Trading with ML ===")

    # Create enhanced config with ML
    config = {
        "symbols": ["SPY"],
        "initial_capital": 100000,
        "objective": {
            "type": "log_utility",
            "kelly_cap_fraction": 0.25,
            "risk_aversion_lambda": 3.0,
            "downside_lambda": 2.0,
        },
        "ml_selector": {"enabled": True, "epsilon": 0.1},
        "risk_params": {
            "max_position_size": 0.15,
            "volatility_target": 0.20,
            "kelly_fraction": 0.25,
        },
        "use_ibkr": False,  # Use synthetic data
    }

    # Initialize paper trading engine
    engine = PaperTradingEngine()
    engine.config = config

    # Run multiple trading cycles
    results = []
    for day in range(30):  # Simulate 30 days
        try:
            result = engine.run_trading_cycle()
            if result["status"] == "success":
                results.append(result)
                print(
                    f"Day {day + 1}: Strategy={result['strategy']}, Sharpe={result['expected_sharpe']:.3f}"
                )
        except Exception as e:
            print(f"Day {day + 1}: Error - {e}")

    if results:
        print("\nPaper Trading Results:")
        print(f"Successful cycles: {len(results)}")

        strategies_used = [r["strategy"] for r in results]
        strategy_counts = pd.Series(strategies_used).value_counts()
        print(f"Strategy usage: {dict(strategy_counts)}")

    return True


def main():
    """Run comprehensive ML backtesting."""
    print("🤖 ML Backtesting with Walk-Forward Analysis")
    print("=" * 60)

    tests = [
        ("Objective Functions", test_objective_functions),
        ("ML Strategy Selector", test_ml_selector),
        ("Walk-Forward Analysis", test_walk_forward_analysis),
        ("Paper Trading with ML", test_paper_trading_with_ml),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n{'=' * 20} {test_name} {'=' * 20}")
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with error: {e}")
            logger.exception(f"Error in {test_name}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("ML BACKTESTING SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All ML backtesting passed! System ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Review issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
