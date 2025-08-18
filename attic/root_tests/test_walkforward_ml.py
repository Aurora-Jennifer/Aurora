#!/usr/bin/env python3
"""
Walk-Forward Analysis with ML Implementation
Tests the enhanced trading system with real data files and ML strategy selection.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.objectives import build_objective
from core.performance import GrowthTargetCalculator
from core.strategy_selector import StrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data_file(filepath: str) -> pd.DataFrame:
    """Load data from pickle file with DataSanity validation."""
    try:
        from core.data_sanity import get_data_sanity_wrapper

        # Use DataSanity wrapper for loading and validation
        wrapper = get_data_sanity_wrapper()
        symbol = Path(filepath).stem

        # Load and validate data
        df = wrapper.load_and_validate(filepath, symbol)

        # Ensure we have the required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            # Try to infer from index or other columns
            if "close" in df.columns:
                df["Close"] = df["close"]
            if "open" in df.columns:
                df["Open"] = df["open"]
            if "high" in df.columns:
                df["High"] = df["high"]
            if "low" in df.columns:
                df["Low"] = df["low"]
            if "volume" in df.columns:
                df["Volume"] = df["volume"]

        # Add Date column if missing
        if "Date" not in df.columns:
            df["Date"] = df.index

        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def test_walkforward_with_real_data():
    """Test walk-forward analysis with real data files."""
    print("\n=== Testing Walk-Forward with Real Data ===")

    # Find data files
    data_dir = Path("data/ibkr")
    data_files = list(data_dir.glob("*_300_D_1_day.pkl"))

    if not data_files:
        print("No data files found. Creating synthetic data...")
        return test_walkforward_synthetic()

    print(f"Found {len(data_files)} data files")

    # Use SPY data for testing
    spy_file = data_dir / "SPY_300_D_1_day.pkl"
    if not spy_file.exists():
        spy_file = data_files[0]  # Use first available file

    print(f"Using data file: {spy_file}")

    # Load data
    data = load_data_file(str(spy_file))
    if data is None:
        print("Failed to load data. Using synthetic data...")
        return test_walkforward_synthetic()

    print(f"Loaded data: {len(data)} rows, {len(data.columns)} columns")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

    # Walk-forward parameters
    train_days = 180  # 6 months training
    test_days = 60  # 2 months testing
    stride = 30  # 1 month stride

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
            f"\nFold {len(results) + 1}: Train {data.iloc[train_start]['Date']} - {data.iloc[train_end-1]['Date']}, "
            f"Test {data.iloc[test_start]['Date']} - {data.iloc[test_end-1]['Date']}"
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
        fold_results = simulate_trading_period_real(test_data, config, strategy_name)

        # Calculate performance metrics
        returns = pd.Series(fold_results["returns"])
        if len(returns) > 0:
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0
            )
            max_dd = calculate_max_drawdown(fold_results["equity_curve"])
            total_return = (
                fold_results["equity_curve"][-1] / fold_results["equity_curve"][0]
            ) - 1

            fold_metrics = {
                "fold": len(results) + 1,
                "strategy": strategy_name,
                "expected_sharpe": expected_sharpe,
                "actual_sharpe": sharpe,
                "total_return": total_return,
                "max_drawdown": max_dd,
                "trades": len(fold_results["trades"]),
                "avg_trade_return": np.mean(
                    [t["return"] for t in fold_results["trades"]]
                )
                if fold_results["trades"]
                else 0,
            }

            results.append(fold_metrics)

            print(f"  Actual Sharpe: {sharpe:.3f}")
            print(f"  Total Return: {total_return:.3%}")
            print(f"  Max Drawdown: {max_dd:.3%}")
            print(f"  Trades: {len(fold_results['trades'])}")
            print(f"  Avg Trade Return: {fold_metrics['avg_trade_return']:.3%}")

    # Aggregate results
    if results:
        print("\n=== Walk-Forward Results Summary ===")
        print(f"Total folds: {len(results)}")

        df_results = pd.DataFrame(results)

        print(f"Average Sharpe: {df_results['actual_sharpe'].mean():.3f}")
        print(f"Average Return: {df_results['total_return'].mean():.3%}")
        print(f"Average Max DD: {df_results['max_drawdown'].mean():.3%}")
        print(f"Total Trades: {df_results['trades'].sum()}")
        print(f"Average Trade Return: {df_results['avg_trade_return'].mean():.3%}")

        # Strategy performance breakdown
        strategy_perf = (
            df_results.groupby("strategy")
            .agg(
                {
                    "actual_sharpe": "mean",
                    "total_return": "mean",
                    "max_drawdown": "mean",
                    "trades": "sum",
                    "avg_trade_return": "mean",
                }
            )
            .round(4)
        )

        print("\nStrategy Performance:")
        print(strategy_perf)

        # Save results
        results_file = "results/walkforward_ml_results.json"
        Path("results").mkdir(exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

    return True


def test_walkforward_synthetic():
    """Test walk-forward analysis with synthetic data."""
    print("\n=== Testing Walk-Forward with Synthetic Data ===")

    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    np.random.seed(42)

    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.02, 500)

    # Add some trend and mean reversion
    for i in range(50, 500):
        if i % 100 < 50:  # Trend periods
            returns[i] += 0.001
        else:  # Mean reversion periods
            returns[i] -= 0.0005

    prices = 100 * np.exp(np.cumsum(returns))

    # Create DataFrame
    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.normal(0, 0.002, 500)),
            "High": prices * (1 + abs(np.random.normal(0, 0.005, 500))),
            "Low": prices * (1 - abs(np.random.normal(0, 0.005, 500))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 500),
        }
    )

    print(f"Created synthetic data: {len(data)} rows")

    # Walk-forward parameters
    train_days = 180
    test_days = 60
    stride = 30

    results = []

    for i in range(0, len(data) - train_days - test_days, stride):
        train_start = i
        train_end = i + train_days
        test_start = train_end
        test_end = min(test_start + test_days, len(data))

        if test_end - test_start < 30:
            break

        print(
            f"\nFold {len(results) + 1}: Train {data.iloc[train_start]['Date']} - {data.iloc[train_end-1]['Date']}, "
            f"Test {data.iloc[test_start]['Date']} - {data.iloc[test_end-1]['Date']}"
        )

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

        print(f"  Selected strategy: {strategy_name}")
        print(f"  Expected Sharpe: {expected_sharpe:.3f}")

        fold_results = simulate_trading_period_real(test_data, config, strategy_name)

        returns = pd.Series(fold_results["returns"])
        if len(returns) > 0:
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0
            )
            max_dd = calculate_max_drawdown(fold_results["equity_curve"])
            total_return = (
                fold_results["equity_curve"][-1] / fold_results["equity_curve"][0]
            ) - 1

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

    if results:
        print("\n=== Synthetic Walk-Forward Results ===")
        print(f"Total folds: {len(results)}")

        df_results = pd.DataFrame(results)
        print(f"Average Sharpe: {df_results['actual_sharpe'].mean():.3f}")
        print(f"Average Return: {df_results['total_return'].mean():.3%}")
        print(f"Average Max DD: {df_results['max_drawdown'].mean():.3%}")
        print(f"Total Trades: {df_results['trades'].sum()}")

    return True


def simulate_trading_period_real(
    data: pd.DataFrame, config: Dict, strategy_name: str
) -> Dict:
    """Simulate trading over a period with real data."""
    capital = config["initial_capital"]
    equity_curve = [capital]
    returns = []
    trades = []

    # Initialize components
    growth_calculator = GrowthTargetCalculator(config)
    objective = build_objective(config)

    for i in range(30, len(data)):  # Start after warmup period
        current_data = data.iloc[: i + 1]

        # Generate signal based on strategy
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
                signal = (
                    0.5 if current_data["Close"].iloc[-1] < ma_short * 0.98 else -0.5
                )
            else:
                signal = 0
        else:  # regime_aware_ensemble
            # Simple trend following
            if len(current_data) >= 20:
                trend = (
                    current_data["Close"].iloc[-1] - current_data["Close"].iloc[-20]
                ) / current_data["Close"].iloc[-20]
                signal = np.clip(trend * 10, -0.5, 0.5)  # Scale and clip
            else:
                signal = 0

        # Calculate position size using objective
        signal_strength = abs(signal)
        if signal_strength > 0.1:  # Only trade if signal is strong enough
            position_size = growth_calculator.calculate_dynamic_position_size(
                signal_strength=signal_strength,
                current_capital=capital,
                symbol_volatility=current_data["Close"].pct_change().std(),
                portfolio_volatility=0.15,
            )

            # Execute trade
            trade_value = capital * position_size * np.sign(signal)
            price_change = current_data["Close"].pct_change().iloc[-1]
            trade_return = trade_value * price_change

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
            (capital - equity_curve[-2]) / equity_curve[-2]
            if len(equity_curve) > 1
            else 0
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


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    peak = equity_curve[0]
    max_dd = 0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        max_dd = max(max_dd, dd)

    return max_dd


def main():
    """Run walk-forward analysis with ML implementation."""
    print("🚀 Walk-Forward Analysis with ML Implementation")
    print("=" * 60)

    try:
        success = test_walkforward_with_real_data()

        if success:
            print("\n✅ Walk-forward analysis completed successfully!")
            print("📊 Check results/walkforward_ml_results.json for detailed results")
            return 0
        else:
            print("\n❌ Walk-forward analysis failed")
            return 1

    except Exception as e:
        print(f"\n❌ Error in walk-forward analysis: {e}")
        logger.exception("Walk-forward analysis error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
