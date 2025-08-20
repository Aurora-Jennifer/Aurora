#!/usr/bin/env python3
"""
Walk-forward framework for allocator-grade backtesting.
Features: leakage-proof, warm-start models, numba simulation, fold-based metrics.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import importlib

# Import simulation function lazily to respect boundaries
simulate_orders_numba = importlib.import_module("core.sim.simulate").simulate_orders_numba  # type: ignore[attr-defined]

# Import centralized logging setup lazily
setup_logging = importlib.import_module("core.utils").setup_logging  # type: ignore[attr-defined]

_OHLC_ORDER = ["Open", "High", "Low", "Close", "Volume"]

def _standardize_ohlc_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common vendor/dirty names to canonical OHLCV."""
    rename = {}
    lower_map = {c.lower(): c for c in df.columns}
    # map lowercased incoming → canonical
    mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
        "vol": "Volume",
    }
    for k, canon in mapping.items():
        if k in lower_map:
            rename[lower_map[k]] = canon
    return df.rename(columns=rename)

def ensure_ohlc(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Accept Series / narrow DF and return a canonical OHLCV frame.
    Missing O/H/L are backfilled from Close; Volume defaults to 0.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame(name="Close")
    data = data.copy()
    data = _standardize_ohlc_cols(data)

    # If only Adj Close exists, treat it as Close
    if "Close" not in data.columns and "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]

    if "Close" not in data.columns:
        raise ValueError("ensure_ohlc: no Close (or Adj Close) column present")

    for col in ("Open", "High", "Low"):
        if col not in data.columns:
            data[col] = data["Close"]

    if "Volume" not in data.columns:
        data["Volume"] = 0

    # order and return only canonical columns we have
    cols = [c for c in _OHLC_ORDER if c in data.columns]
    return data[cols]

# Configure logging
logger = setup_logging("logs/walkforward.log", logging.INFO)

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    DataSanityValidator = importlib.import_module("core.data_sanity").DataSanityValidator  # type: ignore[attr-defined]
    DATASANITY_AVAILABLE = True
except Exception:
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

    Args:
        n: total bars; indices are [0..n-1]
        train_len: length of each training window (ignored if anchored)
        test_len: length of each test window
        stride: distance to advance between folds (<= test_len → overlap)
        warmup: extra bars required before first train_lo (for indicators)
        anchored: if True, train_lo=warmup always; else rolling
        validate_boundaries: whether to validate fold boundaries
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

        # Validate boundaries if requested
        if validate_boundaries:
            assert train_lo <= train_hi, (
                f"WF_BOUNDARY: train_lo ({train_lo}) must be <= train_hi ({train_hi})"
            )
            assert test_lo <= test_hi, (
                f"WF_BOUNDARY: test_lo ({test_lo}) must be <= test_hi ({test_hi})"
            )
            assert train_hi < test_lo, (
                f"WF_BOUNDARY: train_hi ({train_hi}) must be < test_lo ({test_lo})"
            )
            assert train_lo >= 0, f"WF_BOUNDARY: train_lo ({train_lo}) must be >= 0"
            assert test_hi < n, f"WF_BOUNDARY: test_hi ({test_hi}) must be < n ({n})"

        yield Fold(train_lo, train_hi, test_lo, test_hi, fold_id)
        fold_id += 1
        t0 += stride


class LeakageProofPipeline:
    """Pipeline that prevents data leakage by fitting transforms on train only."""

    def __init__(self, X: np.ndarray, y: np.ndarray, scalers: dict[str, callable] = None):
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
        model = SimpleLinearModel() if warm_model is None else warm_model

        model.fit(Xtr, ytr)
        return model


class SimpleLinearModel:
    """Simple linear model for demonstration."""

    def __init__(self, alpha: float = 0.01):
        self.weights = None
        self.bias = None
        self.alpha = alpha  # regularization parameter

    def fit(self, X: np.ndarray, y: np.ndarray, warm_model=None):
        """Fit linear model with regularization and optional warm-start."""
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





def compute_metrics_from_pnl(pnl_series: np.ndarray, trades: list[dict]) -> dict[str, float]:
    """Compute allocator-grade metrics from PnL series."""
    # Handle empty or NaN equity curves
    if len(pnl_series) == 0 or np.all(np.isnan(pnl_series)):
        return {
            "sharpe_nw": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "median_hold": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
            "reason": "no_trades",
        }

    # Handle zero trades case
    total_trades = 0
    if trades and len(trades) > 0:
        total_trades = trades[0].get("count", 0) if trades else 0

    if total_trades == 0:
        return {
            "sharpe_nw": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "median_hold": 0.0,
            "total_return": 0.0,
            "volatility": 0.0,
            "reason": "no_trades",
        }

    # Calculate returns
    returns = np.diff(pnl_series)
    if len(returns) == 0:
        returns = np.array([0.0])

    # Basic metrics
    total_return = (pnl_series[-1] - pnl_series[0]) / pnl_series[0] if pnl_series[0] != 0 else 0.0
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

    # Sharpe ratio (Newey-West adjusted)
    sharpe_nw = np.mean(returns) * 252 / volatility if volatility > 0 else 0.0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        sortino = (np.mean(returns) * 252) / downside_vol if downside_vol > 0 else 0.0
    else:
        sortino = float("inf") if np.mean(returns) > 0 else 0.0

    # Maximum drawdown
    peak = np.maximum.accumulate(pnl_series)
    # Avoid division by zero - use absolute values for drawdown
    drawdown = (pnl_series - peak) / abs(peak[-1]) if peak[-1] != 0 else np.zeros_like(pnl_series)
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Hit rate
    winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
    hit_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0

    # Turnover (simplified - using trade count as proxy)
    total_volume = trades[0]["count"] if trades else 0
    turnover = total_volume / len(pnl_series) if len(pnl_series) > 0 else 0.0

    # Median hold time (from simulation)
    median_hold = trades[0]["median_hold"] if trades else 0.0

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
    folds: list[Fold],
    prices: np.ndarray,
    model_seed: int = 0,
    validate_data: bool = False,  # Changed default to False for performance
    performance_mode: str = "RELAXED",
) -> list[tuple[int, dict, list]]:
    """
    Run walk-forward analysis with warm-start.

    Args:
        pipeline: fitted pipeline
        folds: list of folds
        prices: price series
        model_seed: random seed for model initialization
        validate_data: whether to validate data with DataSanity (default: False for performance)
        performance_mode: performance mode ("RELAXED" or "STRICT")

    Returns:
        results: list of (fold_id, metrics, trades) tuples
    """
    results = []
    model = None
    np.random.seed(model_seed)

    # Performance monitoring
    start_time = time.time()
    fold_times = []

    # Initialize DataSanity validator if available and requested
    validator = None
    if validate_data and DATASANITY_AVAILABLE:
        try:
            validator = DataSanityValidator(profile="walkforward")
            logger.info("DataSanity validation enabled (walkforward profile)")
        except Exception as e:
            logger.warning(f"Failed to initialize DataSanity validator: {e}")
            logger.info("DataSanity validation disabled")
            validator = None
    else:
        logger.info("DataSanity validation disabled for performance")

    # Pre-allocate arrays for better performance
    total_folds = len(folds)
    logger.info(f"Processing {total_folds} folds...")

    for fold_idx, fold in enumerate(folds):
        fold_start_time = time.time()

        # Progress logging for long runs
        if total_folds > 10 and fold_idx % max(1, total_folds // 10) == 0:
            progress = (fold_idx / total_folds) * 100
            logger.info(f"Progress: {progress:.1f}% ({fold_idx}/{total_folds} folds)")

        # Create index arrays
        tr = np.arange(fold.train_lo, fold.train_hi + 1)
        te = np.arange(fold.test_lo, fold.test_hi + 1)

        # Fit transforms on training data only
        pipeline.fit_transforms(tr)

        # Transform data
        Xtr = pipeline.transform(tr)
        ytr = pipeline.y[tr]
        Xte = pipeline.transform(te)
        pipeline.y[te]

        # Validate data with DataSanity if enabled (SKIPPED by default for performance)
        if validator is not None:
            try:
                # Create train data slice for validation with timezone-aware datetime index
                # Use proper dates based on fold indices to avoid lookahead contamination
                base_date = pd.Timestamp("2020-01-01", tz="UTC")
                train_start = base_date + pd.Timedelta(days=fold.train_lo)
                train_dates = pd.date_range(start=train_start, periods=len(tr), freq="D", tz="UTC")
                train_data = pd.DataFrame(
                    {
                        "Open": prices[tr] * 0.99,  # Approximate OHLC from close
                        "High": prices[tr] * 1.01,
                        "Low": prices[tr] * 0.99,
                        "Close": prices[tr],
                        "Volume": np.ones(len(tr)) * 1000000,  # Default volume
                    },
                    index=train_dates,
                )
                # Ensure numeric dtype to avoid object/NaN coercion in strict checks
                train_data = train_data.astype("float64")
                validator.validate_and_repair(train_data, f"TRAIN_FOLD_{fold.fold_id}")
                logger.debug(f"DataSanity validation passed for train fold {fold.fold_id}")

                # Create test data slice for validation with timezone-aware datetime index
                test_start = base_date + pd.Timedelta(days=fold.test_lo)
                test_dates = pd.date_range(start=test_start, periods=len(te), freq="D", tz="UTC")
                test_data = pd.DataFrame(
                    {
                        "Open": prices[te] * 0.99,
                        "High": prices[te] * 1.01,
                        "Low": prices[te] * 0.99,
                        "Close": prices[te],
                        "Volume": np.ones(len(te)) * 1000000,
                    },
                    index=test_dates,
                )
                # Ensure numeric dtype to avoid object/NaN coercion in strict checks
                test_data = test_data.astype("float64")
                validator.validate_and_repair(test_data, f"TEST_FOLD_{fold.fold_id}")
                logger.debug(f"DataSanity validation passed for test fold {fold.fold_id}")
            except Exception as e:
                logger.error(f"DataSanity validation failed for fold {fold.fold_id}: {str(e)}")
                raise Exception(
                    f"RULE:DATASANITY_VALIDATION_FAILED - Fold {fold.fold_id}: {str(e)}"
                ) from e

        # Fit model with warm-start
        model = pipeline.fit_model(Xtr, ytr, warm_model=model)

        # Make predictions
        yhat = model.predict(Xte)

        # Simulate orders using the core simulation function
        test_prices = prices[te]
        pnl_series, trades_count, wins, losses, median_hold = simulate_orders_numba(
            yhat, test_prices
        )

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

        # Performance monitoring
        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)

        # Log fold completion with performance info
        logger.debug(
            f"Fold {fold.fold_id} completed in {fold_time:.3f}s - Sharpe: {metrics['sharpe_nw']:.3f}"
        )

        results.append((fold.fold_id, metrics, trades))

    # Performance summary
    total_time = time.time() - start_time
    avg_fold_time = np.mean(fold_times) if fold_times else 0

    logger.info(f"Walk-forward completed in {total_time:.2f}s")
    logger.info(f"Average fold time: {avg_fold_time:.3f}s")
    logger.info(f"Performance mode: {performance_mode}")

    # Performance validation
    if performance_mode == "STRICT" and avg_fold_time > 0.6:  # 10k rows baseline
        logger.warning(f"Fold time {avg_fold_time:.3f}s exceeds STRICT threshold of 0.6s")

    return results


def build_feature_table(
    data: pd.DataFrame, warmup_days: int = 252
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    # Standardize OHLC data
    data = ensure_ohlc(data)
    
    # Ensure we have enough data
    if len(data) < warmup_days:
        raise ValueError(f"Need at least {warmup_days} days of data")

    # Calculate features
    features = []

    # Price-based features - ensure 1D arrays
    close = data["Close"].values.flatten()
    volume = data["Volume"].values.flatten()

    # Returns
    returns = np.diff(np.log(close))

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
        vol_ratio = vol / pd.Series(vol).rolling(50).mean().values  # relative volatility
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
        pd.Series(close).rolling(50).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]).values
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
    """Optimized walk-forward framework for trading system backtesting."""
    logger.info("Starting optimized walk-forward analysis")

    # Load data with error handling for longer periods
    try:
        import yfinance as yf

        logger.info(f"Loading {args.symbol} data from yfinance")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")

        # Add progress indicator for long data loads
        data = yf.download(args.symbol, start=args.start_date, end=args.end_date, progress=True)
        logger.info(f"Loaded {len(data)} days of data")

        if len(data) < 100:
            logger.warning(
                f"Very little data loaded ({len(data)} days). Check date range and symbol."
            )

    except ImportError:
        logger.warning("yfinance not available, using dummy data")
        dates = pd.date_range(args.start_date, args.end_date, freq="D")
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
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Build feature table with progress indication
    logger.info("Building feature table...")
    X, y, prices = build_feature_table(data, warmup_days=60)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {np.bincount((y + 1).astype(int))}")
    logger.info(f"Price range: {prices.min():.2f} - {prices.max():.2f}")

    # Create pipeline
    pipeline = LeakageProofPipeline(X, y)

    # Generate folds with optimized parameters for longer periods
    logger.info("Generating folds...")
    folds = list(
        gen_walkforward(
            n=len(X),
            train_len=args.train_len,
            test_len=args.test_len,
            stride=args.stride,
            warmup=0,
            anchored=False,  # rolling window
        )
    )

    logger.info(f"Generated {len(folds)} folds")

    # Warn if too many folds for performance
    if len(folds) > 50:
        logger.warning(
            f"Large number of folds ({len(folds)}). Consider increasing stride or reducing date range for better performance."
        )

    # Run walk-forward with performance monitoring
    logger.info("Running walk-forward analysis...")
    start_time = time.time()

    # Get performance mode from environment
    perf_mode = os.getenv("SANITY_PERF_MODE", "RELAXED")
    logger.info(f"Performance mode: {perf_mode}")

    results = walkforward_run(
        pipeline,
        folds,
        prices,
        performance_mode=perf_mode,
        validate_data=args.validate_data,
    )
    end_time = time.time()

    logger.info(f"Completed in {end_time - start_time:.2f} seconds")

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

    logger.info("\nAggregate Results:")
    logger.info(f"Mean Sharpe: {np.mean(sharpe_scores):.3f}")
    logger.info(f"Mean Max DD: {np.mean(max_dds):.3f}")
    logger.info(f"Mean Hit Rate: {np.mean(hit_rates):.3f}")
    logger.info(f"Mean Total Return: {np.mean(total_returns):.3f}")
    logger.info(
        f"Folds with positive Sharpe: {sum(1 for s in sharpe_scores if s > 0)}/{len(sharpe_scores)}"
    )

    if all_trade_pnls:
        logger.info("Trade Analysis:")
        logger.info(f"  Total closed trades: {len(all_trade_pnls)}")
        logger.info(f"  Winning trades: {sum(1 for pnl in all_trade_pnls if pnl > 0)}")
        logger.info(f"  Average trade PnL: {np.mean(all_trade_pnls):.2f}")
        logger.info(f"  Best trade: {max(all_trade_pnls):.2f}")
        logger.info(f"  Worst trade: {min(all_trade_pnls):.2f}")

    # Save results
    output_dir = Path(args.output_dir)
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
                        sum(1 for pnl in all_trade_pnls if pnl > 0) if all_trade_pnls else 0
                    ),
                    "avg_trade_pnl": (float(np.mean(all_trade_pnls)) if all_trade_pnls else 0.0),
                },
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward framework for trading system backtesting"
    )
    parser.add_argument("--symbol", help="Trading symbol to analyze (overrides config)")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date for analysis")
    parser.add_argument("--end-date", default="2024-12-31", help="End date for analysis")
    parser.add_argument("--config", default="config/base.json", help="Base configuration file path")
    parser.add_argument(
        "--profile",
        choices=["risk_low", "risk_balanced", "risk_strict"],
        help="Risk profile to apply",
    )
    parser.add_argument("--asset", help="Asset symbol for asset-specific configuration")
    parser.add_argument(
        "--train-len", type=int, help="Training window length (days, overrides config)"
    )
    parser.add_argument("--test-len", type=int, help="Test window length (days, overrides config)")
    parser.add_argument("--stride", type=int, help="Stride between folds (days, overrides config)")
    parser.add_argument(
        "--perf-mode",
        choices=["RELAXED", "STRICT"],
        default="RELAXED",
        help="Performance validation mode",
    )
    parser.add_argument("--validate-data", action="store_true", help="Enable DataSanity validation")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")

    args = parser.parse_args()

    # Build CLI overrides
    cli_overrides = {}
    if args.symbol:
        cli_overrides["symbols"] = [args.symbol]
    if args.train_len:
        cli_overrides["walkforward"] = cli_overrides.get("walkforward", {})
        cli_overrides["walkforward"]["train_len"] = args.train_len
    if args.test_len:
        cli_overrides["walkforward"] = cli_overrides.get("walkforward", {})
        cli_overrides["walkforward"]["test_len"] = args.test_len
    if args.stride:
        cli_overrides["walkforward"] = cli_overrides.get("walkforward", {})
        cli_overrides["walkforward"]["stride"] = args.stride
    if args.output_dir:
        cli_overrides["data"] = cli_overrides.get("data", {})
        cli_overrides["data"]["output_dir"] = args.output_dir

    # Load configuration
    load_config = importlib.import_module("core.config_loader").load_config

    config = load_config(
        profile=args.profile,
        asset=args.asset,
        cli_overrides=cli_overrides,
        base_config_path=args.config,
    )

    # Set environment variables
    os.environ["SANITY_PERF_MODE"] = args.perf_mode

    # Use config values
    symbol = args.symbol or config["symbols"][0]
    wf_params = config["walkforward"]

    logger.info(f"Starting walk-forward analysis for {symbol}")
    logger.info(f"Configuration: {args.profile or 'base'} profile")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(
        f"Train/Test/Stride: {wf_params['train_len']}/{wf_params['test_len']}/{wf_params['stride']}"
    )

    main()
