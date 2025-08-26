from __future__ import annotations

import contextlib
import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .model_interface import Model


def set_seeds(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def build_features(prices: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    df = prices.copy()
    close = df["Close"].astype(float)
    ret_1d = close.pct_change()
    feats: dict[str, pd.Series] = {}

    # Legacy minimal features (keep support)
    if "ret_1d" in feature_list:
        feats["ret_1d"] = ret_1d
    if "ret_5d" in feature_list:
        feats["ret_5d"] = close.pct_change(5)
    if "vol_10d" in feature_list:
        feats["vol_10d"] = ret_1d.rolling(10, min_periods=10).std()

    # Current 10-feature linear_v2_10f set
    if "ret_1d_lag1" in feature_list:
        feats["ret_1d_lag1"] = ret_1d.shift(1)
    if "sma_10" in feature_list:
        feats["sma_10"] = close.rolling(10, min_periods=10).mean().shift(1)
    if "sma_20" in feature_list:
        feats["sma_20"] = close.rolling(20, min_periods=20).mean().shift(1)
    if "vol_10" in feature_list:
        feats["vol_10"] = ret_1d.rolling(10, min_periods=10).std().shift(1)

    # RSI(14) replicated simply for runtime parity (no lookahead)
    if "rsi_14" in feature_list:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rsi = pd.Series(index=close.index, dtype=float)
        std_mask = (avg_gain > 0) & (avg_loss > 0)
        rs = avg_gain[std_mask] / avg_loss[std_mask]
        rsi.loc[std_mask] = 100.0 - (100.0 / (1.0 + rs))
        rsi.loc[(avg_loss == 0) & (avg_gain > 0)] = 100.0
        rsi.loc[(avg_gain == 0) & (avg_loss > 0)] = 0.0
        rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 50.0
        feats["rsi_14"] = rsi.shift(1)

    # Momentum suite
    if "momentum_3d" in feature_list:
        feats["momentum_3d"] = close.pct_change(3).shift(1)
    if "momentum_5d" in feature_list:
        feats["momentum_5d"] = close.pct_change(5).shift(1)
    if "momentum_10d" in feature_list:
        feats["momentum_10d"] = close.pct_change(10).shift(1)
    if "momentum_20d" in feature_list:
        feats["momentum_20d"] = close.pct_change(20).shift(1)
    if "momentum_strength" in feature_list:
        # safe compute using components if present; else derive minimally
        m5 = feats.get("momentum_5d", close.pct_change(5).shift(1))
        m20 = feats.get("momentum_20d", close.pct_change(20).shift(1))
        r1 = feats.get("ret_1d_lag1", ret_1d.shift(1))
        feats["momentum_strength"] = (r1 * m5 * m20) ** (1 / 3)

    feat_df = pd.DataFrame(feats, index=df.index)
    # Align and drop rows with NaNs across requested columns only
    cols = [c for c in feature_list if c in feat_df.columns]
    return feat_df[cols].dropna()


def _map_scores_to_weights(scores: np.ndarray, map_name: str, max_abs: float) -> np.ndarray:
    if map_name == "linear":
        w = scores
    elif map_name == "softmax":
        ex = np.exp(scores - scores.max())
        w = ex / max(ex.sum(), 1e-9)
    else:  # tanh
        w = np.tanh(scores)
    return np.clip(w, -max_abs, max_abs)


def infer_weights(
    model: Model,
    feat_df: pd.DataFrame,
    feature_order: list[str],
    map_name: str,
    max_abs: float,
    min_bars: int,
) -> dict[str, float] | dict:
    if len(feat_df) < min_bars:
        return {"status": "HOLD", "reason": "insufficient_history"}
    # Feature alignment; require columns to exist
    for col in feature_order:
        if col not in feat_df.columns:
            return {"status": "FAIL", "reason": "feature_mismatch"}
    X = feat_df[feature_order].to_numpy(dtype="float64")
    scores = model.predict(X[-1:].copy())
    arr = np.asarray(scores, dtype="float64").reshape(-1)
    if not np.isfinite(arr).all():
        return {"status": "HOLD", "reason": "nan_in_scores"}
    w = _map_scores_to_weights(arr, map_name, max_abs)
    if not np.isfinite(w).all():
        return {"status": "HOLD", "reason": "nan_in_scores"}
    return {i: float(w[i]) for i in range(len(w))}


def detect_weight_spikes(
    previous: dict[str, float] | None, current: dict[str, float], max_delta: float
) -> list[str]:
    if not previous:
        return []
    spikes: list[str] = []
    for symbol, weight in current.items():
        prev = float(previous.get(symbol, 0.0))
        if abs(weight - prev) > max_delta:
            spikes.append(symbol)
    return spikes


def compute_turnover(previous: dict[str, float] | None, current: dict[str, float]) -> float:
    if not previous:
        return float(sum(abs(float(w)) for w in current.values()))
    all_symbols = set(previous.keys()) | set(current.keys())
    t = 0.0
    for s in all_symbols:
        t += abs(float(current.get(s, 0.0)) - float(previous.get(s, 0.0)))
    return float(t)


def apply_risk_layer(decision: dict, context: dict, cfg: dict) -> dict:
    """Apply risk v2 rules if enabled."""
    if not cfg.get("flags", {}).get("enable_risk_v2", False):
        return decision  # legacy path
    
    try:
        from risk.v2 import apply as risk_v2_apply, RiskV2Config
        
        rcfg = RiskV2Config(**cfg["risk"]["v2"])
        sym = decision.get("symbol", "UNKNOWN")
        bar_df = context.get("bars", {}).get(sym)
        equity = context.get("account", {}).get("equity", 100000.0)
        open_positions = context.get("positions", {})
        gross = context.get("portfolio", {}).get("gross_weight", 0.0)
        
        if bar_df is None:
            decision["risk"] = {"veto": True, "reason": "no_bar_data"}
            return decision
        
        # Get desired weight from decision
        desired_weight = decision.get("weight", 0.0)
        
        out = risk_v2_apply(sym, bar_df, desired_weight, equity, open_positions, gross, rcfg)
        
        decision["risk"] = out  # persist telemetry
        if out["veto"]:
            decision["weight"] = 0.0
            decision["reason"] = f"risk_v2_veto::{out['reason']}"
        else:
            decision["weight"] = out["weight"]
            if out.get("stop"):
                decision["stop"] = out["stop"]
        return decision
        
    except Exception as e:
        # Fallback to legacy if risk v2 fails
        decision["risk"] = {"veto": False, "reason": f"risk_v2_error:{str(e)}"}
        return decision
