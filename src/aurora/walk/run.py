from datetime import datetime, timedelta
from typing import Any

import numpy as np
from core.metrics.stats import (
    daily_turnover,
    max_drawdown,
    psr,
    sharpe_newey_west,
    sortino,
    win_rate,
)
from core.sim.simulate import simulate_safe

from .folds import Fold
from .pipeline import Pipeline


def run_fold(
    p: Pipeline, close: np.ndarray, X: np.ndarray, y: np.ndarray, f: Fold
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    tr = np.arange(f.train_lo, f.train_hi + 1)
    te = np.arange(f.test_lo, f.test_hi + 1)

    p.fit_transforms(tr)
    Xtr = p.transform(tr)
    ytr = y[tr]
    Xte = p.transform(te)

    p.fit_model(Xtr, ytr, warm=None)
    signal = p.predict(Xte)

    pnl, ntrades, wins, losses, med_hold = simulate_safe(close[te], signal.astype(np.int8))
    ret = np.diff(pnl)

    metrics = {
        "fold_id": f.fold_id,
        "n_bars": te.size,
        "n_trades": int(ntrades),
        "win_rate": float(win_rate(wins, losses)),
        "median_hold_bars": int(med_hold),
        "sharpe_nw": float(sharpe_newey_west(ret)),
        "sortino": float(sortino(ret)),
        "max_dd": float(max_drawdown(pnl)),
        "turnover": float(daily_turnover(signal.astype(np.float32))),
        "psr": float(psr(sharpe_ann=sharpe_newey_west(ret), T=max(5, ntrades))),
        "regime": getattr(p, "current_regime", "unknown"),  # Add regime label
    }
    return metrics, pnl, signal


def gate_fold(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    ok = True
    reasons = []
    if metrics["n_bars"] < gates.get("min_days", 30):
        ok = False
        reasons.append("short_fold")
    if metrics["n_trades"] < gates.get("min_trades", 10):
        ok = False
        reasons.append("few_trades")
    if not (metrics["psr"] >= gates.get("psr_min", 0.95)):
        ok = False
        reasons.append("psr_low")
    if metrics["max_dd"] < -gates.get("max_dd", 0.2):  # max_dd is negative (e.g., -0.2)
        ok = False
        reasons.append("dd_high")
    metrics["trusted"] = ok
    metrics["gate_reasons"] = reasons
    return metrics


def stitch_equity(pnl_per_fold: list[np.ndarray]) -> np.ndarray:
    """Stitch fold PnL series into a continuous equity curve."""
    if not pnl_per_fold:
        return np.array([])

    eq = [p.copy() for p in pnl_per_fold]
    base = 0.0
    out = []

    for p in eq:
        out.append(p + base)
        base = out[-1][-1]

    return np.concatenate(out)


def fold_weight(m: dict[str, Any], tau: float = 0.6) -> float:
    """Calculate fold weight based on composite score with stable softmax."""
    # Composite score from multiple metrics
    s = np.nan_to_num(m["sharpe_nw"], nan=0.0)
    so = np.nan_to_num(m["sortino"], nan=0.0)
    dd = -m["max_dd"]  # smaller (more negative) is worse

    # Weighted score with bounds to prevent overflow
    score = 0.6 * s + 0.3 * so + 0.1 * dd

    # Use stable softmax approach
    score = np.clip(score, -20.0, 20.0)  # Prevent exp overflow
    z = score / max(tau, 1e-6)
    z -= z.max()  # Subtract max for numerical stability
    w = np.exp(z)
    return w / (w.sum() if w.sum() > 0 else 1.0)


def reweight(metrics_list: list[dict[str, Any]]) -> np.ndarray:
    """Normalize fold weights using stable softmax."""
    if not metrics_list:
        return np.array([])

    # Calculate individual weights
    weights = [fold_weight(m) for m in metrics_list]
    weights = np.array(weights)

    # Apply stable softmax normalization
    weights = np.clip(weights, 1e-20, 1e20)  # Prevent log issues
    log_weights = np.log(weights)
    log_weights -= log_weights.max()  # Subtract max for numerical stability
    exp_weights = np.exp(log_weights)

    # Normalize
    total = exp_weights.sum()
    return exp_weights / total if total > 0 else np.ones_like(exp_weights) / len(exp_weights)



def contiguous_trusted_span(
    fold_meta: list[dict[str, Any]],
    fold_dates: list[tuple[datetime, datetime]],
    min_months: int = 6,
) -> tuple[bool, timedelta]:
    """Calculate longest contiguous trusted span."""
    span = timedelta(0)
    best = timedelta(0)

    for _i, (meta, (start, end)) in enumerate(zip(fold_meta, fold_dates, strict=False)):
        if meta["trusted"]:
            span += end - start
            best = max(best, span)
        else:
            span = timedelta(0)

    return best >= timedelta(days=30 * min_months), best


def calculate_weighted_metrics(
    metrics_list: list[dict[str, Any]], pnl_per_fold: list[np.ndarray]
) -> dict[str, float]:
    """Calculate weighted aggregate metrics with payoff analysis."""
    if not metrics_list or not pnl_per_fold:
        return {}

    # Calculate weights
    weights = reweight(metrics_list)

    # Weighted metrics
    weighted_sharpe = np.average([m["sharpe_nw"] for m in metrics_list], weights=weights)
    weighted_sortino = np.average([m["sortino"] for m in metrics_list], weights=weights)
    weighted_max_dd = np.average([m["max_dd"] for m in metrics_list], weights=weights)
    weighted_win_rate = np.average([m["win_rate"] for m in metrics_list], weights=weights)
    weighted_turnover = np.average([m["turnover"] for m in metrics_list], weights=weights)

    # Stitched equity metrics
    stitched_equity_curve = stitch_equity(pnl_per_fold)
    if len(stitched_equity_curve) > 1:
        stitched_returns = np.diff(stitched_equity_curve, prepend=stitched_equity_curve[0])
        stitched_sharpe = sharpe_newey_west(stitched_returns)
        stitched_max_dd = max_drawdown(stitched_equity_curve)
    else:
        stitched_sharpe = stitched_max_dd = 0.0

    # Payoff ratio analysis for high win rate anomalies
    payoff_analysis = []
    for i, m in enumerate(metrics_list):
        if m["win_rate"] >= 0.9 and m["sharpe_nw"] < 0:
            payoff_analysis.append(
                {
                    "fold_id": m["fold_id"],
                    "win_rate": m["win_rate"],
                    "sharpe_nw": m["sharpe_nw"],
                    "n_trades": m["n_trades"],
                    "turnover": m["turnover"],
                    "weight": weights[i] if i < len(weights) else 0,
                }
            )

    return {
        "weighted_sharpe": float(weighted_sharpe),
        "weighted_sortino": float(weighted_sortino),
        "weighted_max_dd": float(weighted_max_dd),
        "weighted_win_rate": float(weighted_win_rate),
        "weighted_turnover": float(weighted_turnover),
        "stitched_sharpe": float(stitched_sharpe),
        "stitched_max_dd": float(stitched_max_dd),
        "total_folds": len(metrics_list),
        "trusted_folds": sum(1 for m in metrics_list if m.get("trusted", False)),
        "avg_weight": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "payoff_anomalies": payoff_analysis,
    }
