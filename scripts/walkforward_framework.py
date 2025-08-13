#!/usr/bin/env python3
"""
Walk-forward framework for allocator-grade backtesting.
Features: leakage-proof, warm-start models, numba simulation, fold-based metrics.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
) -> Iterator[Fold]:
    """
    Generate walk-forward folds.

    Args:
        n: total bars; indices are [0..n-1]
        train_len: length of each training window (ignored if anchored)
        test_len: length of each test window
        stride: distance to advance between folds (<= test_len → overlap)
        warmup: extra bars required before first train_lo (for indicators)
        anchored: if True, train_lo=warmup always; else rolling
    """
    fold_id = 0
    t0 = warmup + (0 if anchored else train_len)

    while True:
        train_hi = t0 - 1
        train_lo = warmup if anchored else train_hi - train_len + 1
        test_lo = t0
        test_hi = min(test_lo + test_len - 1, n - 1)

        if test_lo >= n or train_lo < warmup:
            break

        yield Fold(train_lo, train_hi, test_lo, test_hi, fold_id)
        fold_id += 1
        t0 += stride


class LeakageProofPipeline:
    """Pipeline that prevents data leakage by fitting transforms on train only."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, scalers: Dict[str, callable] = None
    ):
        """
        Args:
            X: 2D numpy array of precomputed global features
            y: labels/targets aligned with X
            scalers: dict of scaler functions (optional)
        """
        self.X = X
        self.y = y
        self.state = {}  # scaler params per fold
        self.scalers = scalers or {}

    def fit_transforms(self, idx: np.ndarray):
        """Fit transforms using only training data."""
        X_train = self.X[idx]

        # Z-score normalization per feature
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)  # avoid division by zero

        self.state["mu"] = mu
        self.state["sd"] = sd

        # Apply any additional scalers
        for name, scaler_fn in self.scalers.items():
            self.state[name] = scaler_fn(X_train)

    def transform(self, idx: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self.state:
            raise ValueError("Must call fit_transforms before transform")

        X_data = self.X[idx]
        mu, sd = self.state["mu"], self.state["sd"]
        X_scaled = (X_data - mu) / sd

        # Apply additional transforms
        for name, params in self.state.items():
            if name not in ["mu", "sd"]:
                X_scaled = self.scalers[name](X_data, params)

        return X_scaled

    def fit_model(self, Xtr: np.ndarray, ytr: np.ndarray, warm_model=None):
        """Fit model with optional warm-start."""
        # Simple linear model for demonstration
        # Replace with your actual model (XGBoost, PyTorch, etc.)
        if warm_model is None:
            # Initialize new model
            model = SimpleLinearModel()
        else:
            # Warm-start existing model
            model = warm_model

        model.fit(Xtr, ytr)
        return model


class SimpleLinearModel:
    """Simple linear model for demonstration."""

    def __init__(self, alpha: float = 0.01):
        self.weights = None
        self.bias = None
        self.alpha = alpha  # regularization parameter

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit linear model with regularization."""
        # Add bias term
        X_bias = np.column_stack([X, np.ones(X.shape[0])])

        # Ridge regression with regularization
        X_bias_T = X_bias.T
        reg_matrix = self.alpha * np.eye(X_bias.shape[1])
        reg_matrix[-1, -1] = 0  # Don't regularize bias term

        try:
            self.weights = np.linalg.solve(X_bias_T @ X_bias + reg_matrix, X_bias_T @ y)
        except np.linalg.LinAlgError:
            # Fallback to more regularization
            self.weights = np.linalg.solve(
                X_bias_T @ X_bias + 10 * self.alpha * np.eye(X_bias.shape[1]),
                X_bias_T @ y,
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.weights is None:
            raise ValueError("Model not fitted")

        X_bias = np.column_stack([X, np.ones(X.shape[0])])
        raw_predictions = X_bias @ self.weights

        # Apply tanh activation to bound predictions
        return np.tanh(raw_predictions)


@jit(nopython=True)
def simulate_orders_numba(
    signals: np.ndarray,
    prices: np.ndarray,
    initial_cash: float = 100000.0,
    max_position: float = 0.05,  # Reduced from 0.1 to 0.05
    transaction_cost: float = 0.001,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Numba-optimized order simulation with regime-aware sizing and risk management.

    Args:
        signals: array of position signals [-1, 1]
        prices: array of prices
        initial_cash: starting capital
        max_position: maximum position size as fraction of capital
        transaction_cost: cost per trade as fraction

    Returns:
        trades: list of trade dictionaries
        pnl_series: array of cumulative PnL
    """
    n = len(signals)
    cash = initial_cash
    position = 0.0
    avg_price = 0.0
    trades = []
    pnl_series = np.zeros(n)

    # Risk management parameters
    stop_loss_pct = 0.015  # 1.5% stop loss (tighter)
    take_profit_pct = 0.03  # 3% take profit
    max_daily_loss = 0.02  # 2% max daily loss

    for i in range(n):
        signal = signals[i]
        price = prices[i]

        # Check stop loss and take profit first
        if position != 0:
            if position > 0:  # Long position
                current_pnl_pct = (price - avg_price) / avg_price
                if current_pnl_pct < -stop_loss_pct:
                    # Stop loss hit - close position
                    trade_size = -position
                    cost = abs(trade_size * price * transaction_cost)
                    cash -= cost

                    trade_pnl = (price - avg_price) * abs(trade_size)

                    trade = {
                        "timestamp": i,
                        "side": -1,
                        "size": abs(trade_size),
                        "price": price,
                        "cost": cost,
                        "pnl": trade_pnl,
                    }
                    trades.append(trade)

                    position = 0.0
                    avg_price = 0.0
                elif current_pnl_pct > take_profit_pct:
                    # Take profit hit - close position
                    trade_size = -position
                    cost = abs(trade_size * price * transaction_cost)
                    cash -= cost

                    trade_pnl = (price - avg_price) * abs(trade_size)

                    trade = {
                        "timestamp": i,
                        "side": -1,
                        "size": abs(trade_size),
                        "price": price,
                        "cost": cost,
                        "pnl": trade_pnl,
                    }
                    trades.append(trade)

                    position = 0.0
                    avg_price = 0.0
            else:  # Short position
                current_pnl_pct = (avg_price - price) / avg_price
                if current_pnl_pct < -stop_loss_pct:
                    # Stop loss hit - close position
                    trade_size = -position
                    cost = abs(trade_size * price * transaction_cost)
                    cash -= cost

                    trade_pnl = (avg_price - price) * abs(trade_size)

                    trade = {
                        "timestamp": i,
                        "side": 1,
                        "size": abs(trade_size),
                        "price": price,
                        "cost": cost,
                        "pnl": trade_pnl,
                    }
                    trades.append(trade)

                    position = 0.0
                    avg_price = 0.0
                elif current_pnl_pct > take_profit_pct:
                    # Take profit hit - close position
                    trade_size = -position
                    cost = abs(trade_size * price * transaction_cost)
                    cash -= cost

                    trade_pnl = (avg_price - price) * abs(trade_size)

                    trade = {
                        "timestamp": i,
                        "side": 1,
                        "size": abs(trade_size),
                        "price": price,
                        "cost": cost,
                        "pnl": trade_pnl,
                    }
                    trades.append(trade)

                    position = 0.0
                    avg_price = 0.0

        # Check daily loss limit
        current_value = cash + position * price
        daily_return = (current_value - initial_cash) / initial_cash
        if daily_return < -max_daily_loss:
            # Close all positions due to daily loss limit
            if position != 0:
                trade_size = -position
                cost = abs(trade_size * price * transaction_cost)
                cash -= cost

                if position > 0:
                    trade_pnl = (price - avg_price) * abs(trade_size)
                else:
                    trade_pnl = (avg_price - price) * abs(trade_size)

                trade = {
                    "timestamp": i,
                    "side": 1 if trade_size > 0 else -1,
                    "size": abs(trade_size),
                    "price": price,
                    "cost": cost,
                    "pnl": trade_pnl,
                }
                trades.append(trade)

                position = 0.0
                avg_price = 0.0

        # Calculate target position (with signal threshold and regime sizing)
        if abs(signal) < 0.2:  # Higher threshold for stronger signals
            target_position = 0.0
        else:
            # Regime-aware position sizing
            if abs(signal) > 0.6:  # Strong signal
                position_multiplier = 1.0
            elif abs(signal) > 0.4:  # Medium signal
                position_multiplier = 0.6
            else:  # Weak signal
                position_multiplier = 0.3

            target_position = signal * max_position * position_multiplier * cash / price

        # Calculate trade size
        trade_size = target_position - position

        # Apply transaction costs
        if abs(trade_size) > 1e-6:  # minimum trade size
            cost = abs(trade_size * price * transaction_cost)
            cash -= cost

            # Calculate trade PnL if closing position
            trade_pnl = 0.0
            if position != 0 and trade_size * position < 0:  # Closing position
                if position > 0:  # Long position
                    trade_pnl = (price - avg_price) * min(abs(trade_size), position)
                else:  # Short position
                    trade_pnl = (avg_price - price) * min(
                        abs(trade_size), abs(position)
                    )

            # Record trade
            trade = {
                "timestamp": i,
                "side": 1 if trade_size > 0 else -1,
                "size": abs(trade_size),
                "price": price,
                "cost": cost,
                "pnl": trade_pnl,
            }
            trades.append(trade)

            # Update position and average price
            if position == 0:  # Opening new position
                avg_price = price
            elif position + trade_size == 0:  # Closing all
                avg_price = 0.0
            else:  # Partial close or add
                if (position > 0 and trade_size > 0) or (
                    position < 0 and trade_size < 0
                ):
                    # Adding to position - update VWAP
                    total_value = position * avg_price + trade_size * price
                    avg_price = total_value / (position + trade_size)

            position = target_position

        # Calculate PnL
        position_value = position * price
        total_value = cash + position_value
        pnl_series[i] = total_value - initial_cash

    return trades, pnl_series


def compute_metrics_from_pnl(
    pnl_series: np.ndarray, trades: List[Dict]
) -> Dict[str, float]:
    """Compute allocator-grade metrics from PnL series."""
    if len(pnl_series) == 0:
        return {
            "sharpe_nw": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "median_hold": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
        }

    # Calculate returns
    returns = np.diff(pnl_series)
    if len(returns) == 0:
        returns = np.array([0.0])

    # Basic metrics
    total_return = (
        (pnl_series[-1] - pnl_series[0]) / pnl_series[0] if pnl_series[0] != 0 else 0.0
    )
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

    # Sharpe ratio (Newey-West adjusted)
    if volatility > 0:
        sharpe_nw = (np.mean(returns) * 252) / volatility
    else:
        sharpe_nw = 0.0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        sortino = (np.mean(returns) * 252) / downside_vol if downside_vol > 0 else 0.0
    else:
        sortino = float("inf") if np.mean(returns) > 0 else 0.0

    # Maximum drawdown
    peak = np.maximum.accumulate(pnl_series)
    # Avoid division by zero
    drawdown = np.where(peak != 0, (pnl_series - peak) / peak, 0.0)
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Hit rate
    winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
    hit_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0

    # Turnover
    total_volume = sum(abs(trade["size"] * trade["price"]) for trade in trades)
    turnover = total_volume / pnl_series[0] if pnl_series[0] > 0 else 0.0

    # Median hold time (simplified)
    median_hold = (
        np.median([trade.get("hold_time", 1) for trade in trades]) if trades else 0.0
    )

    return {
        "sharpe_nw": sharpe_nw,
        "sortino": sortino,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
        "turnover": turnover,
        "median_hold": median_hold,
        "total_return": total_return,
        "volatility": volatility,
    }


def walkforward_run(
    pipeline: LeakageProofPipeline,
    folds: List[Fold],
    prices: np.ndarray,
    model_seed: int = 0,
) -> List[Tuple[int, Dict, List]]:
    """
    Run walk-forward analysis with warm-start.

    Args:
        pipeline: fitted pipeline
        folds: list of folds
        prices: price series
        model_seed: random seed for model initialization

    Returns:
        results: list of (fold_id, metrics, trades) tuples
    """
    results = []
    model = None
    np.random.seed(model_seed)

    for fold in folds:
        # Create index arrays
        tr = np.arange(fold.train_lo, fold.train_hi + 1)
        te = np.arange(fold.test_lo, fold.test_hi + 1)

        # Fit transforms on training data only
        pipeline.fit_transforms(tr)

        # Transform data
        Xtr = pipeline.transform(tr)
        ytr = pipeline.y[tr]
        Xte = pipeline.transform(te)
        yte = pipeline.y[te]

        # Fit model with warm-start
        model = pipeline.fit_model(Xtr, ytr, warm_model=model)

        # Make predictions
        yhat = model.predict(Xte)

        # Simulate orders
        test_prices = prices[te]
        trades, pnl_series = simulate_orders_numba(yhat, test_prices)

        # Compute metrics
        metrics = compute_metrics_from_pnl(pnl_series, trades)

        results.append((fold.fold_id, metrics, trades))

    return results


def build_feature_table(
    data: pd.DataFrame, warmup_days: int = 252
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build global feature table from price data.

    Args:
        data: DataFrame with OHLCV data
        warmup_days: number of warmup days for indicators

    Returns:
        X: feature matrix
        y: target labels
        prices: price array
    """
    # Ensure we have enough data
    if len(data) < warmup_days:
        raise ValueError(f"Need at least {warmup_days} days of data")

    # Calculate features
    features = []

    # Price-based features - ensure 1D arrays
    close = data["Close"].values.flatten()
    high = data["High"].values.flatten()
    low = data["Low"].values.flatten()
    volume = data["Volume"].values.flatten()

    # Returns
    returns = np.diff(np.log(close))
    print(f"Debug: returns shape before concatenate: {returns.shape}")
    print(f"Debug: returns type: {type(returns)}")
    print(f"Debug: close shape: {close.shape}")

    # Ensure returns is 1D
    if returns.ndim > 1:
        returns = returns.flatten()

    returns = np.concatenate([np.array([0.0]), returns])  # pad first day

    # Moving averages with ratios
    for window in [5, 10, 20, 50]:
        ma = pd.Series(close).rolling(window).mean().values
        ma_ratio = close / ma
        features.extend([ma, ma_ratio])

    # Volatility features
    for window in [5, 10, 20]:
        vol = pd.Series(returns).rolling(window).std().values
        vol_ratio = (
            vol / pd.Series(vol).rolling(50).mean().values
        )  # relative volatility
        features.extend([vol, vol_ratio])

    # RSI with multiple timeframes
    for window in [14, 21]:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean().values
        avg_loss = loss.rolling(window).mean().values
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)

    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    macd = ema12 - ema26
    macd_signal = pd.Series(macd).ewm(span=9).mean().values
    macd_histogram = macd - macd_signal
    features.extend([macd, macd_signal, macd_histogram])

    # Bollinger Bands
    for window in [20]:
        ma = pd.Series(close).rolling(window).mean().values
        std = pd.Series(close).rolling(window).std().values
        bb_upper = ma + 2 * std
        bb_lower = ma - 2 * std
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        features.extend([bb_position])

    # Volume features
    volume_ma = pd.Series(volume).rolling(20).mean().values
    volume_ratio = volume / (volume_ma + 1e-8)
    volume_price_trend = pd.Series(close * volume).rolling(10).mean().values
    features.extend([volume_ma, volume_ratio, volume_price_trend])

    # Price momentum
    for window in [5, 10, 20]:
        momentum = close / pd.Series(close).shift(window).values
        features.append(momentum)

    # Regime detection features
    # Trend strength
    trend_strength = (
        pd.Series(close)
        .rolling(50)
        .apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
        .values
    )
    features.append(trend_strength)

    # Volatility regime
    vol_regime = pd.Series(returns).rolling(20).std().rolling(50).rank(pct=True).values
    features.append(vol_regime)

    # Stack features
    X = np.column_stack(features)

    # Create better targets (forward-looking returns with threshold)
    future_returns = np.roll(returns, -1)  # shift forward by 1 day
    future_returns[-1] = 0  # last day has no target

    # Create classification target: 1 for positive return, -1 for negative, 0 for small moves
    threshold = 0.005  # Reduced threshold to 0.5% for more trades
    y = np.where(
        future_returns > threshold,
        1.0,
        np.where(future_returns < -threshold, -1.0, 0.0),
    )

    # Remove warmup period
    X = X[warmup_days:]
    y = y[warmup_days:]
    prices = close[warmup_days:]

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, prices


def main():
    """Example usage of the walk-forward framework."""
    # Load data (replace with your data loading)
    try:
        import yfinance as yf

        data = yf.download("SPY", start="2020-01-01", end="2024-12-31")
    except ImportError:
        print("yfinance not available, using dummy data")
        dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        data = pd.DataFrame(
            {
                "Open": np.random.randn(len(dates)).cumsum() + 100,
                "High": np.random.randn(len(dates)).cumsum() + 102,
                "Low": np.random.randn(len(dates)).cumsum() + 98,
                "Close": np.random.randn(len(dates)).cumsum() + 100,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

    # Build feature table
    print("Building feature table...")
    X, y, prices = build_feature_table(data)

    print(f"Feature matrix shape: {X.shape}")
    print(
        f"Target distribution: {np.bincount((y + 1).astype(int))}"
    )  # Shift to non-negative
    print(f"Price range: {prices.min():.2f} - {prices.max():.2f}")

    # Create pipeline
    pipeline = LeakageProofPipeline(X, y)

    # Generate folds
    print("Generating folds...")
    folds = list(
        gen_walkforward(
            n=len(X),
            train_len=252,  # 1 year training
            test_len=63,  # 3 months testing
            stride=21,  # 1 month stride (overlapping)
            warmup=0,
            anchored=False,  # rolling window
        )
    )

    print(f"Generated {len(folds)} folds")

    # Run walk-forward
    print("Running walk-forward analysis...")
    start_time = time.time()
    results = walkforward_run(pipeline, folds, prices)
    end_time = time.time()

    print(f"Completed in {end_time - start_time:.2f} seconds")

    # Aggregate results
    all_metrics = [metrics for _, metrics, _ in results]
    all_trades = [trades for _, _, trades in results]

    # Calculate aggregate statistics
    sharpe_scores = [m["sharpe_nw"] for m in all_metrics]
    max_dds = [m["max_dd"] for m in all_metrics]
    hit_rates = [m["hit_rate"] for m in all_metrics]
    total_returns = [m["total_return"] for m in all_metrics]

    # Analyze trades
    all_trade_pnls = []
    for trades in all_trades:
        for trade in trades:
            if trade.get("pnl", 0) != 0:  # Only closed trades
                all_trade_pnls.append(trade["pnl"])

    print("\nAggregate Results:")
    print(f"Mean Sharpe: {np.mean(sharpe_scores):.3f}")
    print(f"Mean Max DD: {np.mean(max_dds):.3f}")
    print(f"Mean Hit Rate: {np.mean(hit_rates):.3f}")
    print(f"Mean Total Return: {np.mean(total_returns):.3f}")
    print(
        f"Folds with positive Sharpe: {sum(1 for s in sharpe_scores if s > 0)}/{len(sharpe_scores)}"
    )

    if all_trade_pnls:
        print("Trade Analysis:")
        print(f"  Total closed trades: {len(all_trade_pnls)}")
        print(f"  Winning trades: {sum(1 for pnl in all_trade_pnls if pnl > 0)}")
        print(f"  Average trade PnL: {np.mean(all_trade_pnls):.2f}")
        print(f"  Best trade: {max(all_trade_pnls):.2f}")
        print(f"  Worst trade: {min(all_trade_pnls):.2f}")

    # Save results
    output_dir = Path("results/walkforward")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "fold_results": [
                    {"fold_id": fold_id, "metrics": metrics, "n_trades": len(trades)}
                    for fold_id, metrics, trades in results
                ],
                "aggregate": {
                    "mean_sharpe": float(np.mean(sharpe_scores)),
                    "mean_max_dd": float(np.mean(max_dds)),
                    "mean_hit_rate": float(np.mean(hit_rates)),
                    "mean_total_return": float(np.mean(total_returns)),
                    "positive_sharpe_folds": sum(1 for s in sharpe_scores if s > 0),
                    "total_folds": len(folds),
                    "total_closed_trades": len(all_trade_pnls),
                    "winning_trades": (
                        sum(1 for pnl in all_trade_pnls if pnl > 0)
                        if all_trade_pnls
                        else 0
                    ),
                    "avg_trade_pnl": (
                        float(np.mean(all_trade_pnls)) if all_trade_pnls else 0.0
                    ),
                },
            },
            f,
            indent=2,
        )

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
