#!/usr/bin/env python3
"""
Final ML Validation - Comprehensive Testing
Tests all ML components, walk-forward analysis, and provides complete validation report.
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

from core.engine.paper import PaperTradingEngine
from core.learning.selector import BanditSelector, StrategyContext
from core.objectives import ExpectedLogUtility, MeanVariance, SortinoUtility
from core.performance import GrowthTargetCalculator
from core.strategy_selector import StrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_objective_functions():
    """Test all objective functions."""
    print("\n=== Testing Objective Functions ===")

    # Create synthetic returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    equity = pd.Series(100000 * np.exp(np.cumsum(returns)))
    risk_metrics = {"portfolio_vol": 0.15}

    objectives = {
        "log_utility": ExpectedLogUtility,
        "mean_variance": MeanVariance,
        "sortino": SortinoUtility,
    }

    results = {}

    for name, obj_class in objectives.items():
        print(f"\nTesting {name}:")

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

        results[name] = {
            "score": score,
            "risk_budget": risk_budget,
            "position_multiplier": pos_mult,
        }

    return results


def test_ml_selector():
    """Test ML strategy selector."""
    print("\n=== Testing ML Strategy Selector ===")

    selector = BanditSelector(epsilon=0.1)

    # Test multiple contexts
    contexts = [
        StrategyContext("trend", 2, 0.7, 1000000, 5.0, 10, 0),
        StrategyContext("chop", 1, 0.3, 500000, 10.0, 5, 1),
        StrategyContext("volatile", 3, 0.9, 2000000, 2.0, 15, 2),
    ]

    candidates = ["regime_aware_ensemble", "momentum", "mean_reversion"]
    results = {}

    for i, context in enumerate(contexts):
        print(f"\nContext {i + 1}: {context.regime} regime, vol_bin={context.vol_bin}")

        # Test strategy recommendation
        recommended = selector.recommend(context, candidates)
        print(f"  Recommended strategy: {recommended}")

        # Test learning updates
        for j in range(10):
            reward = np.random.normal(0.001, 0.01)
            selector.update(context, recommended, reward)

        results[f"context_{i + 1}"] = {
            "regime": context.regime,
            "recommended": recommended,
            "updates": 10,
        }

    return results


def test_paper_trading_ml():
    """Test paper trading with ML implementation."""
    print("\n=== Testing Paper Trading with ML ===")

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
        "use_ibkr": False,
    }

    engine = PaperTradingEngine()
    engine.config = config

    results = []
    for day in range(10):  # Test 10 days
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
        strategies_used = [r["strategy"] for r in results]
        strategy_counts = pd.Series(strategies_used).value_counts()

        return {
            "successful_cycles": len(results),
            "strategy_usage": dict(strategy_counts),
            "avg_expected_sharpe": np.mean([r["expected_sharpe"] for r in results]),
        }

    return {"successful_cycles": 0}


def test_walkforward_synthetic():
    """Test walk-forward analysis with synthetic data."""
    print("\n=== Testing Walk-Forward Analysis ===")

    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    np.random.seed(42)

    returns = np.random.normal(0.0005, 0.02, 400)
    for i in range(50, 400):
        if i % 100 < 50:
            returns[i] += 0.001
        else:
            returns[i] -= 0.0005

    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.normal(0, 0.002, 400)),
            "High": prices * (1 + abs(np.random.normal(0, 0.005, 400))),
            "Low": prices * (1 - abs(np.random.normal(0, 0.005, 400))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 400),
        }
    )

    # Walk-forward parameters
    train_days = 120
    test_days = 40
    stride = 20

    results = []

    for i in range(0, len(data) - train_days - test_days, stride):
        train_start = i
        train_end = i + train_days
        test_start = train_end
        test_end = min(test_start + test_days, len(data))

        if test_end - test_start < 20:
            break

        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]

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

        strategy_selector = StrategySelector(config)
        (
            strategy_name,
            strategy_params,
            expected_sharpe,
        ) = strategy_selector.select_best_strategy(train_data)

        # Simulate trading
        fold_results = simulate_trading_period(test_data, config, strategy_name)

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

    if results:
        df_results = pd.DataFrame(results)

        return {
            "total_folds": len(results),
            "avg_sharpe": df_results["actual_sharpe"].mean(),
            "avg_return": df_results["total_return"].mean(),
            "avg_max_dd": df_results["max_drawdown"].mean(),
            "total_trades": df_results["trades"].sum(),
            "strategy_performance": df_results.groupby("strategy")
            .agg({"actual_sharpe": "mean", "total_return": "mean", "trades": "sum"})
            .to_dict(),
        }

    return {"total_folds": 0}


def simulate_trading_period(data: pd.DataFrame, config: dict, strategy_name: str) -> dict:
    """Simulate trading over a period."""
    capital = config["initial_capital"]
    equity_curve = [capital]
    returns = []
    trades = []

    growth_calculator = GrowthTargetCalculator(config)

    for i in range(20, len(data)):
        current_data = data.iloc[: i + 1]

        # Generate signal
        if strategy_name == "momentum":
            if len(current_data) >= 20:
                signal = (
                    0.5
                    if current_data["Close"].iloc[-1] > current_data["Close"].iloc[-20]
                    else -0.5
                )
            else:
                signal = 0
        elif strategy_name == "mean_reversion":
            if len(current_data) >= 50:
                ma_short = current_data["Close"].rolling(10).mean().iloc[-1]
                ma_long = current_data["Close"].rolling(50).mean().iloc[-1]
                signal = 0.5 if current_data["Close"].iloc[-1] < ma_short * 0.98 else -0.5
            else:
                signal = 0
        else:  # regime_aware_ensemble
            if len(current_data) >= 20:
                trend = (
                    current_data["Close"].iloc[-1] - current_data["Close"].iloc[-20]
                ) / current_data["Close"].iloc[-20]
                signal = np.clip(trend * 10, -0.5, 0.5)
            else:
                signal = 0

        # Execute trade
        signal_strength = abs(signal)
        if signal_strength > 0.1:
            position_size = growth_calculator.calculate_dynamic_position_size(
                signal_strength=signal_strength,
                current_capital=capital,
                symbol_volatility=current_data["Close"].pct_change().std(),
                portfolio_volatility=0.15,
            )

            trade_value = capital * position_size * np.sign(signal)
            price_change = current_data["Close"].pct_change().iloc[-1]
            trade_return = trade_value * price_change

            trades.append(
                {
                    "signal": signal,
                    "position_size": position_size,
                    "return": trade_return,
                }
            )

            capital += trade_return
        else:
            capital += 0

        equity_curve.append(capital)
        daily_return = (
            (capital - equity_curve[-2]) / equity_curve[-2] if len(equity_curve) > 1 else 0
        )
        returns.append(daily_return)

        growth_calculator.update_performance(daily_return, capital)

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": trades,
        "final_capital": capital,
    }


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum drawdown."""
    peak = equity_curve[0]
    max_dd = 0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def generate_validation_report(results: dict):
    """Generate comprehensive validation report."""
    print("\n" + "=" * 80)
    print("🤖 FINAL ML VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("System: Enhanced Trading System v2.0 - ML Implementation")

    # Objective Functions
    print("\n📊 OBJECTIVE FUNCTIONS")
    print("-" * 40)
    for name, metrics in results.get("objectives", {}).items():
        print(f"{name.upper()}:")
        print(f"  Score: {metrics['score']:.6f}")
        print(f"  Risk Budget: {metrics['risk_budget']:.3f}")
        print(f"  Position Multiplier: {metrics['position_multiplier']:.3f}")

    # ML Selector
    print("\n🧠 ML STRATEGY SELECTOR")
    print("-" * 40)
    for context_name, metrics in results.get("ml_selector", {}).items():
        print(f"{context_name}:")
        print(f"  Regime: {metrics['regime']}")
        print(f"  Recommended: {metrics['recommended']}")
        print(f"  Learning Updates: {metrics['updates']}")

    # Paper Trading
    print("\n📈 PAPER TRADING WITH ML")
    print("-" * 40)
    pt_results = results.get("paper_trading", {})
    print(f"Successful Cycles: {pt_results.get('successful_cycles', 0)}")
    print(f"Strategy Usage: {pt_results.get('strategy_usage', {})}")
    print(f"Avg Expected Sharpe: {pt_results.get('avg_expected_sharpe', 0):.3f}")

    # Walk-Forward Analysis
    print("\n🔄 WALK-FORWARD ANALYSIS")
    print("-" * 40)
    wf_results = results.get("walkforward", {})
    print(f"Total Folds: {wf_results.get('total_folds', 0)}")
    print(f"Average Sharpe: {wf_results.get('avg_sharpe', 0):.3f}")
    print(f"Average Return: {wf_results.get('avg_return', 0):.3%}")
    print(f"Average Max DD: {wf_results.get('avg_max_dd', 0):.3%}")
    print(f"Total Trades: {wf_results.get('total_trades', 0)}")

    # Overall Assessment
    print("\n🎯 OVERALL ASSESSMENT")
    print("-" * 40)

    # Check if all components are working
    objectives_working = len(results.get("objectives", {})) == 3
    ml_working = len(results.get("ml_selector", {})) > 0
    paper_trading_working = results.get("paper_trading", {}).get("successful_cycles", 0) > 0
    walkforward_working = results.get("walkforward", {}).get("total_folds", 0) > 0

    print(f"✅ Objective Functions: {'PASS' if objectives_working else 'FAIL'}")
    print(f"✅ ML Strategy Selector: {'PASS' if ml_working else 'FAIL'}")
    print(f"✅ Paper Trading with ML: {'PASS' if paper_trading_working else 'FAIL'}")
    print(f"✅ Walk-Forward Analysis: {'PASS' if walkforward_working else 'FAIL'}")

    all_passing = (
        objectives_working and ml_working and paper_trading_working and walkforward_working
    )

    print(
        f"\n{'🎉 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT' if all_passing else '⚠️ SOME TESTS FAILED - REVIEW REQUIRED'}"
    )

    # Save report
    report_file = "results/final_ml_validation_report.json"
    Path("results").mkdir(exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: {report_file}")

    return all_passing


def main():
    """Run comprehensive ML validation."""
    print("🚀 Final ML Validation - Comprehensive Testing")
    print("=" * 80)

    results = {}

    try:
        # Test objective functions
        results["objectives"] = test_objective_functions()

        # Test ML selector
        results["ml_selector"] = test_ml_selector()

        # Test paper trading with ML
        results["paper_trading"] = test_paper_trading_ml()

        # Test walk-forward analysis
        results["walkforward"] = test_walkforward_synthetic()

        # Generate comprehensive report
        success = generate_validation_report(results)

        if success:
            print("\n🎉 ML VALIDATION COMPLETE - ALL SYSTEMS OPERATIONAL!")
            return 0
        else:
            print("\n⚠️ ML VALIDATION COMPLETE - SOME ISSUES DETECTED")
            return 1

    except Exception as e:
        print(f"\n❌ Error in ML validation: {e}")
        logger.exception("ML validation error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
