from __future__ import annotations
import math, random
from typing import Dict, List
import numpy as np
import pandas as pd
from .model_interface import Model


def set_seeds(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def build_features(prices: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    df = prices.copy()
    feats = {}
    if "ret_1d" in feature_list:
        feats["ret_1d"] = df["Close"].pct_change()
    if "ret_5d" in feature_list:
        feats["ret_5d"] = df["Close"].pct_change(5)
    if "vol_10d" in feature_list:
        feats["vol_10d"] = df["Close"].pct_change().rolling(10).std()
    F = pd.DataFrame(feats, index=df.index).dropna()
    return F


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
    feature_order: List[str],
    map_name: str,
    max_abs: float,
    min_bars: int,
) -> Dict[str, float] | dict:
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


