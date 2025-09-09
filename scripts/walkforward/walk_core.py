#!/usr/bin/env python3
from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Lazy imports from core to avoid hard deps at import time
simulate_orders_numba = importlib.import_module("core.sim.simulate").simulate_orders_numba  # type: ignore[attr-defined]
setup_logging = importlib.import_module("core.utils").setup_logging  # type: ignore[attr-defined]


logger = setup_logging("logs/walkforward.log", logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class Fold:
    train_lo: int
    train_hi: int  # inclusive
    test_lo: int
    test_hi: int  # inclusive
    fold_id: int

    def __post_init__(self) -> None:
        assert self.train_lo <= self.train_hi
        assert self.test_lo <= self.test_hi
        assert self.train_hi < self.test_lo


def gen_walkforward(
    n: int,
    train_len: int,
    test_len: int,
    stride: int,
    warmup: int = 0,
    anchored: bool = False,
    validate_boundaries: bool = True,
) -> Iterator[Fold]:
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
            assert train_lo <= train_hi
            assert test_lo <= test_hi
            assert train_hi < test_lo
            assert train_lo >= 0
            assert test_hi < n
        yield Fold(train_lo, train_hi, test_lo, test_hi, fold_id)
        fold_id += 1
        t0 += stride


class LeakageProofPipeline:
    def __init__(self, X: np.ndarray, y: np.ndarray, scalers: dict[str, callable] | None = None):
        self.X = X
        self.y = y
        self.state: dict[str, np.ndarray] = {}
        self.scalers = scalers or {}

    def fit_transforms(self, idx: np.ndarray) -> None:
        X_train = self.X[idx]
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        self.state["mu"], self.state["sd"] = mu, sd
        for name, scaler_fn in self.scalers.items():
            self.state[name] = scaler_fn(X_train)

    def transform(self, idx: np.ndarray) -> np.ndarray:
        if not self.state:
            raise ValueError("Must call fit_transforms before transform")
        X_data = self.X[idx]
        mu, sd = self.state["mu"], self.state["sd"]
        X_scaled = (X_data - mu) / sd
        for name, params in self.state.items():
            if name not in ["mu", "sd"]:
                X_scaled = self.scalers[name](X_data, params)
        return X_scaled

    def fit_model(self, Xtr: np.ndarray, ytr: np.ndarray, warm_model=None):
        model = SimpleLinearModel() if warm_model is None else warm_model
        model.fit(Xtr, ytr)
        return model


class SimpleLinearModel:
    def __init__(self, alpha: float = 0.01):
        self.weights: np.ndarray | None = None
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray, warm_model=None) -> None:  # noqa: ARG002
        X_bias = np.column_stack([X, np.ones(X.shape[0])])
        XT = X_bias.T
        reg = self.alpha * np.eye(X_bias.shape[1])
        reg[-1, -1] = 0
        try:
            self.weights = np.linalg.solve(XT @ X_bias + reg, XT @ y)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.solve(XT @ X_bias + 10 * reg, XT @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not fitted")
        X_bias = np.column_stack([X, np.ones(X.shape[0])])
        raw = X_bias @ self.weights
        return np.tanh(raw)


def compute_metrics_from_pnl(pnl_series: np.ndarray, trades: list[dict]) -> dict[str, float]:
    if len(pnl_series) == 0 or np.all(np.isnan(pnl_series)):
        return dict.fromkeys(["sharpe_nw", "sortino", "max_dd", "hit_rate", "turnover", "median_hold", "total_return", "volatility"], 0.0)
    returns = np.diff(pnl_series)
    if len(returns) == 0:
        returns = np.array([0.0])
    total_return = (pnl_series[-1] - pnl_series[0]) / pnl_series[0] if pnl_series[0] != 0 else 0.0
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
    sharpe_nw = np.mean(returns) * 252 / volatility if volatility > 0 else 0.0
    downside = returns[returns < 0]
    if len(downside) > 0:
        dvol = np.std(downside) * np.sqrt(252)
        sortino = (np.mean(returns) * 252) / dvol if dvol > 0 else 0.0
    else:
        sortino = float("inf") if np.mean(returns) > 0 else 0.0
    peak = np.maximum.accumulate(pnl_series)
    drawdown = (pnl_series - peak) / abs(peak[-1]) if peak[-1] != 0 else np.zeros_like(pnl_series)
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
    hit_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0
    turnover = (trades[0]["count"] if trades else 0) / len(pnl_series) if len(pnl_series) > 0 else 0.0
    median_hold = trades[0].get("median_hold", 0.0) if trades else 0.0
    return {
        "sharpe_nw": float(sharpe_nw),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "hit_rate": float(hit_rate),
        "turnover": float(turnover),
        "median_hold": float(median_hold),
        "total_return": float(total_return),
        "volatility": float(volatility),
    }


def build_feature_table(data: pd.DataFrame, warmup_days: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Standardize OHLC
    df = data.copy()
    if "Close" not in df.columns:
        raise ValueError("Close column required")
    close = df["Close"].to_numpy().reshape(-1)
    volume = (df.get("Volume", pd.Series(np.zeros(len(df))))).to_numpy().reshape(-1)
    if len(df) < warmup_days:
        raise ValueError(f"Need at least {warmup_days} rows")
    features: list[np.ndarray] = []
    returns = np.diff(np.log(close))
    returns = np.concatenate([np.array([0.0]), returns])
    for window in [5, 10, 20, 50]:
        ma = pd.Series(close).rolling(window).mean().to_numpy()
        ma_ratio = close / (ma + 1e-12)
        features.extend([ma, ma_ratio])
    for window in [5, 10, 20]:
        vol = pd.Series(returns).rolling(window).std().to_numpy()
        vol_ratio = vol / (pd.Series(vol).rolling(50).mean().to_numpy() + 1e-12)
        features.extend([vol, vol_ratio])
    delta = pd.Series(close).diff()
    for window in [14, 21]:
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window).mean().to_numpy()
        avg_loss = loss.rolling(window).mean().to_numpy()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)
    ema12 = pd.Series(close).ewm(span=12).mean().to_numpy()
    ema26 = pd.Series(close).ewm(span=26).mean().to_numpy()
    macd = ema12 - ema26
    macd_signal = pd.Series(macd).ewm(span=9).mean().to_numpy()
    macd_hist = macd - macd_signal
    features.extend([macd, macd_signal, macd_hist])
    volume_ma = pd.Series(volume).rolling(20).mean().to_numpy()
    volume_ratio = volume / (volume_ma + 1e-8)
    features.extend([volume_ma, volume_ratio])
    X = np.column_stack(features)
    future_returns = np.roll(returns, -1)
    future_returns[-1] = 0
    threshold = 0.005
    y = np.where(future_returns > threshold, 1.0, np.where(future_returns < -threshold, -1.0, 0.0))
    X = X[warmup_days:]
    y = y[warmup_days:]
    prices = close[warmup_days:]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, prices


def walkforward_run(
    pipeline: LeakageProofPipeline,
    folds: list[Fold],
    prices: np.ndarray,
    model_seed: int = 0,
    validate_data: bool = False,
    datasanity_profile: str = "walkforward",
    performance_mode: str = "RELAXED",
) -> list[tuple[int, dict, list]]:
    results: list[tuple[int, dict, list]] = []
    model = None
    np.random.seed(model_seed)
    fold_times: list[float] = []
    logger.info(f"Processing {len(folds)} folds...")
    for fold in folds:
        t0 = time.time()
        tr = np.arange(fold.train_lo, fold.train_hi + 1)
        te = np.arange(fold.test_lo, fold.test_hi + 1)
        pipeline.fit_transforms(tr)
        Xtr = pipeline.transform(tr)
        ytr = pipeline.y[tr]
        Xte = pipeline.transform(te)
        model = pipeline.fit_model(Xtr, ytr, warm_model=model)
        yhat = model.predict(Xte)
        test_prices = prices[te]
        pnl_series, trades_count, wins, losses, median_hold = simulate_orders_numba(yhat, test_prices)
        trades = [{"count": int(trades_count), "wins": int(wins), "losses": int(losses), "median_hold": int(median_hold)}]
        metrics = compute_metrics_from_pnl(pnl_series, trades)
        fold_times.append(time.time() - t0)
        results.append((fold.fold_id, metrics, trades))
    logger.info(f"Average fold time: {np.mean(fold_times) if fold_times else 0:.3f}s; mode={performance_mode}")
    return results


