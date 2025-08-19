from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _finite(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(x)


def stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-feature mean/std/missing_pct. Returns finite values or NaN."""
    result: dict[str, dict[str, float]] = {}
    for name in df.columns:
        col = pd.to_numeric(df[name], errors="coerce")
        miss = float(col.isna().mean())
        mu = float(col.mean(skipna=True))
        sd = float(col.std(skipna=True, ddof=0))
        result[name] = {"mean": _finite(mu), "std": _finite(sd), "missing_pct": _finite(miss)}
    return result


def psi_from_edges(current: pd.Series, ref_edges: np.ndarray, ref_probs: np.ndarray) -> float:
    """Population Stability Index for a feature using reference bin edges + probabilities."""
    eps = 1e-12
    values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return float("nan")
    cur_counts, _ = np.histogram(values, bins=ref_edges)
    cur_probs = cur_counts.astype("float64")
    tot = cur_probs.sum()
    cur_probs = (cur_probs / max(tot, 1.0)).clip(eps, 1.0)
    ref_probs = np.asarray(ref_probs, dtype="float64").clip(eps, 1.0)
    # Align lengths defensively
    m = min(cur_probs.shape[0], ref_probs.shape[0])
    cur_probs = cur_probs[:m]
    ref_probs = ref_probs[:m]
    return float(np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs)))


def psi(current_df: pd.DataFrame, ref: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in current_df.columns:
        r = ref.get(name, {}) or {}
        edges = np.asarray(r.get("bin_edges", []), dtype="float64")
        probs = np.asarray(r.get("ref_probs", []), dtype="float64")
        if edges.size == 0 or probs.size == 0:
            out[name] = float("nan")
            continue
        out[name] = psi_from_edges(current_df[name], edges, probs)
    vals = [v for v in out.values() if np.isfinite(v)]
    out["__global__"] = float(np.mean(vals)) if vals else float("nan")
    return out
