"""
Improved Lookahead Detection
Detects structural leakage while ignoring legitimate zero-return runs.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class LookaheadResult:
    """Result of lookahead detection analysis."""
    suspicious_match_rate: float
    n_suspicious: int
    n_total: int
    indices_sample: list[Any]
    passed: bool
    stable_runs_count: int
    stable_runs_total: int


def detect_lookahead(returns: pd.Series, eps: float = 0.0, min_run: int = 2) -> LookaheadResult:
    """
    Detect structural leakage by testing whether ret_t predicts ret_{t+1} 
    beyond what price-stasis would imply.
    
    Args:
        returns: Return series
        eps: Epsilon for considering returns "zero" (default: 0.0)
        min_run: Minimum consecutive zero returns to consider a "stable run"
    
    Returns:
        LookaheadResult with detection metrics
    """
    if len(returns) < 2:
        return LookaheadResult(
            suspicious_match_rate=0.0,
            n_suspicious=0,
            n_total=len(returns),
            indices_sample=[],
            passed=True,
            stable_runs_count=0,
            stable_runs_total=0
        )

    # Mask stable-price runs (no price movement → legit identical returns)
    stable = returns.abs().le(eps)
    stable_runs = (stable.groupby((stable != stable.shift()).cumsum())
                          .transform('size') >= min_run) & stable

    # Count stable runs
    stable_runs_count = stable_runs.sum()
    stable_runs_total = len(returns.dropna())

    # Naive future match (suspicious if many returns equal their own future)
    future_eq = (returns == returns.shift(-1)) & returns.notna() & returns.shift(-1).notna()

    # Only count suspicious matches **outside** stable runs
    suspicious = future_eq & (~stable_runs)

    # Rate thresholds
    total = len(returns.dropna())
    rate = suspicious.sum() / max(total, 1)

    return LookaheadResult(
        suspicious_match_rate=rate,
        n_suspicious=int(suspicious.sum()),
        n_total=int(total),
        indices_sample=suspicious[suspicious].index.tolist()[:10],
        passed=rate < 0.001,  # flag threshold (configurable)
        stable_runs_count=int(stable_runs_count),
        stable_runs_total=int(stable_runs_total)
    )


def detect_lookahead_with_context(returns: pd.Series, close_prices: pd.Series = None,
                                 eps: float = 0.0, min_run: int = 2) -> dict[str, Any]:
    """
    Enhanced lookahead detection with additional context.
    
    Args:
        returns: Return series
        close_prices: Close price series (optional, for additional context)
        eps: Epsilon for considering returns "zero"
        min_run: Minimum consecutive zero returns to consider a "stable run"
    
    Returns:
        Dictionary with detailed detection results
    """
    result = detect_lookahead(returns, eps, min_run)

    # Additional context if close prices available
    context = {}
    if close_prices is not None and len(close_prices) == len(returns):
        # Check if suspicious matches correspond to identical close prices
        if result.n_suspicious > 0 and len(result.indices_sample) > 0:
            sample_indices = result.indices_sample[:5]  # Check first 5
            identical_closes = []

            for idx in sample_indices:
                if idx in close_prices.index:
                    next_idx = close_prices.index[close_prices.index.get_loc(idx) + 1] if close_prices.index.get_loc(idx) + 1 < len(close_prices) else None
                    if next_idx:
                        close_diff = abs(close_prices.loc[idx] - close_prices.loc[next_idx])
                        identical_closes.append(close_diff < 1e-12)

            context["sample_identical_closes"] = identical_closes
            context["likely_legitimate"] = all(identical_closes) if identical_closes else False

    return {
        "suspicious_match_rate": result.suspicious_match_rate,
        "n_suspicious": result.n_suspicious,
        "n_total": result.n_total,
        "indices_sample": result.indices_sample,
        "passed": result.passed,
        "stable_runs_count": result.stable_runs_count,
        "stable_runs_total": result.stable_runs_total,
        "context": context
    }
