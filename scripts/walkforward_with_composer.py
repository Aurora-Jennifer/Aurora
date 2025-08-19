#!/usr/bin/env python3
"""
Walk-forward framework with composer integration.
Features: leakage-proof, warm-start models, composer-based strategy selection.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import composer integration
from core.engine.composer_integration import ComposerIntegration

# Import centralized logging setup
from core.utils import setup_logging

# Configure logging
logger = setup_logging("logs/walkforward_composer.log", logging.INFO)

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from core.data_sanity import DataSanityValidator

    DATASANITY_AVAILABLE = True
except ImportError:
    DATASANITY_AVAILABLE = False
    DataSanityValidator = None

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@dataclass
class Fold:
    """Represents a single walk-forward fold."""

    train_lo: int
    train_hi: int  # inclusive
    test_lo: int
    test_hi: int  # inclusive
    fold_id: int

    def __post_init__(self):
        assert self.train_lo <= self.train_hi
        assert self.test_lo <= self.test_hi
        assert self.train_hi < self.test_lo  # no overlap between train/test


def gen_walkforward(
    n: int,
    train_len: int,
    test_len: int,
    stride: int,
    warmup: int = 0,
    anchored: bool = False,
    validate_boundaries: bool = True,
) -> Iterator[Fold]:
    """
    Generate walk-forward folds.
    """
    fold_id = 0
    t0 = warmup + (0 if anchored else train_len)

    while True:
        if anchored:
            train_lo = warmup
            train_hi = t0 - 1
        else:
            train_hi = t0 - 1
            train_lo = train_hi - train_len + 1

        test_lo = t0
        test_hi = min(test_lo + test_len - 1, n - 1)

        if test_lo >= n or train_lo < warmup or train_hi < train_lo:
            break

        if validate_boundaries:
            test_window_size = test_hi - test_lo + 1
            if test_window_size < test_len // 2:
                logger.warning(
                    f"Fold {fold_id}: test window too small ({test_window_size} < {test_len // 2})"
                )
                break
            # Additional safety: skip if test window is smaller than stride
            if test_window_size < stride:
                logger.warning(
                    f"Fold {fold_id}: test window smaller than stride ({test_window_size} < {stride})"
                )
                break

        yield Fold(train_lo, train_hi, test_lo, test_hi, fold_id)
        fold_id += 1
        t0 += stride


def simulate_orders_python(
    predictions: np.ndarray, prices: np.ndarray
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Simulate trading orders using pure Python (fallback for numba issues).
    """
    logger.debug(
        f"Starting simulation with {len(predictions)} predictions and {len(prices)} prices"
    )
    n = len(predictions)
    pnl_series = np.zeros(n)
    position = 0.0
    entry_price = 0.0  # Initialize entry_price
    trades_count = 0
    wins = 0
    losses = 0
    hold_times = []
    current_hold = 0

    for i in range(n):
        signal = predictions[i]
        price = prices[i]

        # Update hold time for current position
        if position != 0:
            current_hold += 1

        # Trading logic
        if signal > 0.1 and position <= 0:  # Buy signal
            if position < 0:  # Close short position
                pnl = position * (price - entry_price)
                pnl_series[i] = pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                hold_times.append(current_hold)
                current_hold = 0
                trades_count += 1

            # Open long position
            position = 1.0
            entry_price = price

        elif signal < -0.1 and position >= 0:  # Sell signal
            if position > 0:  # Close long position
                pnl = position * (price - entry_price)
                pnl_series[i] = pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                hold_times.append(current_hold)
                current_hold = 0
                trades_count += 1

            # Open short position
            position = -1.0
            entry_price = price

        # Update PnL for current position
        if position != 0:
            pnl_series[i] = position * (price - entry_price)

    # Close final position
    if position != 0:
        final_price = prices[-1]
        pnl = position * (final_price - entry_price)
        pnl_series[-1] = pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        hold_times.append(current_hold)
        trades_count += 1

    median_hold = int(np.median(hold_times)) if hold_times else 0
    logger.debug(f"Simulation completed: {trades_count} trades, {wins} wins, {losses} losses")
    return pnl_series, trades_count, wins, losses, median_hold


def compute_metrics_from_pnl(pnl_series: np.ndarray, trades: list[dict]) -> dict[str, float]:
    """Compute performance metrics from PnL series."""
    # Handle empty or NaN equity curves
    if len(pnl_series) == 0 or np.all(np.isnan(pnl_series)):
        return {
            "total_return": 0.0,
            "sharpe_nw": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "trade_count": 0,
            "reason": "no_trades",
        }

    # Handle zero trades case
    total_trades = 0
    if trades and len(trades) > 0:
        trade = trades[0]  # Simplified - assume single trade summary
        wins = trade.get("wins", 0)
        losses = trade.get("losses", 0)
        total_trades = wins + losses

    if total_trades == 0:
        return {
            "total_return": 0.0,
            "sharpe_nw": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "avg_trade": 0.0,
            "trade_count": 0,
            "reason": "no_trades",
        }

    # Cumulative returns
    cumulative = np.cumsum(pnl_series)
    total_return = cumulative[-1] if len(cumulative) > 0 else 0.0

    # Sharpe ratio (no risk-free rate)
    returns = np.diff(cumulative)
    sharpe_nw = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0.0

    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Trade statistics
    if trades and len(trades) > 0:
        trade = trades[0]  # Simplified - assume single trade summary
        wins = trade.get("wins", 0)
        losses = trade.get("losses", 0)
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        avg_trade = total_return / total_trades if total_trades > 0 else 0.0
        trade_count = total_trades
    else:
        win_rate = 0.0
        avg_trade = 0.0
        trade_count = 0

    return {
        "total_return": total_return,
        "sharpe_nw": sharpe_nw,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "trade_count": trade_count,
    }


def run_walkforward_with_composer(
    data: pd.DataFrame,
    symbol: str,
    config: dict[str, Any],
    train_len: int = 252,
    test_len: int = 126,
    stride: int = 63,
    performance_mode: str = "RELAXED",
    validate_data: bool = True,
) -> list[tuple[int, dict[str, float], list[dict]]]:
    """
    Run walk-forward analysis with composer integration.

    Args:
        data: OHLCV DataFrame
        symbol: Trading symbol
        config: Configuration dictionary with composer settings
        train_len: Training window length
        test_len: Test window length
        stride: Stride between folds
        performance_mode: Performance mode (STRICT/RELAXED)
        validate_data: Whether to validate data

    Returns:
        List of (fold_id, metrics, trades) tuples
    """
    logger.info(f"Starting walk-forward analysis for {symbol}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Composer enabled: {config.get('composer', {}).get('use_composer', False)}")

    # Data validation
    if validate_data and DATASANITY_AVAILABLE:
        validator = DataSanityValidator(profile="walkforward")
        try:
            data = validator.validate_dataframe(data, symbol)
            logger.info("Data validation completed")
        except Exception as e:
            logger.warning(f"Data validation failed: {e}, continuing without validation")
    else:
        logger.info("Data validation skipped")

    # Initialize composer integration
    composer_integration = ComposerIntegration(config)

    # Prepare data
    prices = data["Close"].values
    n = len(prices)

    # Generate folds
    folds = list(gen_walkforward(n, train_len, test_len, stride, warmup=252))
    logger.info(f"Generated {len(folds)} folds")

    if not folds:
        logger.error("No valid folds generated")
        return []

    # Performance monitoring
    start_time = time.time()
    fold_times = []
    results = []

    # Process each fold
    for fold in folds:
        fold_start_time = time.time()

        # Extract train/test data
        tr = slice(fold.train_lo, fold.train_hi + 1)
        te = slice(fold.test_lo, fold.test_hi + 1)

        data.iloc[tr]
        test_data = data.iloc[te]

        # Generate predictions using composer with fold-level logging
        predictions = []
        fold_stats = {
            "insufficient_history": 0,
            "empty_data": 0,
            "insufficient_data_for_regime": 0,
            "composer_exception": 0,
            "composer_used": 0,
        }

        for i in range(len(test_data)):
            current_idx = fold.test_lo + i
            # Get composer decision (composer_integration handles min_history_bars internally)
            try:
                signal, metadata = composer_integration.get_composer_decision(
                    data,
                    symbol,
                    current_idx,  # Pass full data, not sliced
                )
                if metadata.get("composer_used", False):
                    fold_stats["composer_used"] += 1
                else:
                    reason = metadata.get("reason", "unknown")
                    if reason in fold_stats:
                        fold_stats[reason] += 1
            except Exception as e:
                fold_stats["composer_exception"] += 1
                signal = 0.0
                metadata = {"composer_used": False, "error": str(e)}
            predictions.append(signal)

        # Log fold-level summary instead of per-bar errors
        logger.info(
            f"Fold {fold.fold_id}: {len(test_data)} bars processed; "
            f"composer_used={fold_stats['composer_used']}, "
            f"holds={sum(v for k, v in fold_stats.items() if k != 'composer_used')}, "
            f"causes={dict((k, v) for k, v in fold_stats.items() if v > 0 and k != 'composer_used')})"
        )

        predictions = np.array(predictions)

        # Simulate orders using the core simulation function
        test_prices = prices[te]
        try:
            (
                pnl_series,
                trades_count,
                wins,
                losses,
                median_hold,
            ) = simulate_orders_python(predictions, test_prices)
        except Exception as e:
            logger.error(f"Simulation failed for fold {fold.fold_id}: {e}")
            # Return default values on failure
            pnl_series = np.zeros(len(predictions))
            trades_count = 0
            wins = 0
            losses = 0
            median_hold = 0

        # Create simplified trade summary for compatibility
        trades = [
            {
                "count": int(trades_count),
                "wins": int(wins),
                "losses": int(losses),
                "median_hold": int(median_hold),
            }
        ]

        # Compute metrics from PnL series
        metrics = compute_metrics_from_pnl(pnl_series, trades)

        # Add composer-specific metrics
        if composer_integration.use_composer:
            asset_class = composer_integration.get_asset_class(symbol)
            evaluation = composer_integration.evaluate_strategy_performance(
                metrics, symbol, asset_class
            )
            metrics["composite_score"] = evaluation.get("composite_score", 0.0)
            metrics["composer_used"] = True
        else:
            metrics["composer_used"] = False

        # Performance monitoring
        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)

        # Log fold completion with performance info
        logger.debug(
            f"Fold {fold.fold_id} completed in {fold_time:.3f}s - "
            f"Sharpe: {metrics['sharpe_nw']:.3f}, "
            f"Composite: {metrics.get('composite_score', 0.0):.3f}"
        )

        results.append((fold.fold_id, metrics, trades))

    # Performance summary
    total_time = time.time() - start_time
    avg_fold_time = np.mean(fold_times) if fold_times else 0

    logger.info(f"Walk-forward completed in {total_time:.2f}s")
    logger.info(f"Average fold time: {avg_fold_time:.3f}s")
    logger.info(f"Performance mode: {performance_mode}")
    logger.info(f"Composer used: {composer_integration.use_composer}")

    # Performance validation
    if performance_mode == "STRICT" and avg_fold_time > 0.6:  # 10k rows baseline
        logger.warning(f"Fold time {avg_fold_time:.3f}s exceeds STRICT threshold of 0.6s")

    return results


def save_results(results: list[tuple[int, dict[str, float], list[dict]]], symbol: str):
    """Save walk-forward results to JSON."""
    # Convert results to serializable format
    serializable_results = []
    for fold_id, metrics, trades in results:
        serializable_results.append(
            {
                "fold_id": fold_id,
                "metrics": {
                    k: float(v) if isinstance(v, np.integer | np.floating) else v
                    for k, v in metrics.items()
                },
                "trades": trades,
            }
        )

    # Calculate aggregate metrics
    if results:
        all_metrics = [metrics for _, metrics, _ in results]
        aggregate = {
            "mean_total_return": np.mean([m["total_return"] for m in all_metrics]),
            "mean_sharpe": np.mean([m["sharpe_nw"] for m in all_metrics]),
            "mean_max_dd": np.mean([m["max_dd"] for m in all_metrics]),
            "mean_win_rate": np.mean([m["win_rate"] for m in all_metrics]),
            "mean_composite_score": np.mean([m.get("composite_score", 0.0) for m in all_metrics]),
            "std_total_return": np.std([m["total_return"] for m in all_metrics]),
            "std_sharpe": np.std([m["sharpe_nw"] for m in all_metrics]),
            "total_trades": sum(m["trade_count"] for m in all_metrics),
            "composer_usage_rate": np.mean([m.get("composer_used", False) for m in all_metrics]),
        }
    else:
        aggregate = {}

    # Save to file
    output = {
        "symbol": symbol,
        "timestamp": pd.Timestamp.now().isoformat(),
        "fold_results": serializable_results,
        "aggregate": aggregate,
    }

    # Ensure directory exists
    results_dir = Path("results/walkforward")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_file = results_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    return output


def main():
    """Main function for walk-forward analysis with composer."""
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward analysis with composer integration")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--train-len", type=int, default=252, help="Training window length")
    parser.add_argument("--test-len", type=int, default=126, help="Test window length")
    parser.add_argument("--stride", type=int, default=63, help="Stride between folds")
    parser.add_argument(
        "--config", default="config/composer_config.json", help="Configuration file"
    )
    parser.add_argument(
        "--perf-mode",
        default="RELAXED",
        choices=["STRICT", "RELAXED"],
        help="Performance mode",
    )
    parser.add_argument(
        "--validate-data", action="store_true", help="Validate data with DataSanity"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = json.load(f)

    # Load data (simplified - you would use your actual data loading)
    # For demo purposes, create synthetic data
    dates = pd.date_range(args.start_date, args.end_date, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, len(dates)))

    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * 0.999,
            "High": prices * 1.002,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": np.random.lognormal(10, 0.5, len(dates)),
        }
    )

    # Run walk-forward analysis
    results = run_walkforward_with_composer(
        data=data,
        symbol=args.symbol,
        config=config,
        train_len=args.train_len,
        test_len=args.test_len,
        stride=args.stride,
        performance_mode=args.perf_mode,
        validate_data=args.validate_data,
    )

    # Save results
    output = save_results(results, args.symbol)

    # Print summary
    if output["aggregate"]:
        agg = output["aggregate"]
        print(f"\n=== Walk-Forward Results for {args.symbol} ===")
        print(f"Mean Sharpe: {agg['mean_sharpe']:.3f}")
        print(f"Mean Total Return: {agg['mean_total_return']:.3f}")
        print(f"Mean Max DD: {agg['mean_max_dd']:.3f}")
        print(f"Mean Composite Score: {agg['mean_composite_score']:.3f}")
        print(f"Composer Usage Rate: {agg['composer_usage_rate']:.1%}")
        print(f"Total Trades: {agg['total_trades']}")


if __name__ == "__main__":
    main()
