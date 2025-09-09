"""
Composite Scoring System
Combines multiple performance metrics into a single weighted score.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CompositeWeights:
    """Weights for composite scoring."""

    alpha: float = 0.4  # CAGR weight
    beta: float = 0.3  # Sharpe weight
    gamma: float = 0.2  # Win rate weight
    delta: float = 0.1  # Avg trade return weight


@dataclass
class CompositePenalties:
    """Penalty thresholds for composite scoring."""

    max_dd_cap: float = 0.25  # Maximum allowed drawdown
    min_trades: int = 200  # Minimum required trades
    dd_penalty_factor: float = 2.0  # Multiplier for drawdown penalty
    trade_penalty_factor: float = 1.5  # Multiplier for trade count penalty


def normalize_cagr(cagr: float) -> float:
    """Normalize CAGR to 0-1 range."""
    # Assume reasonable CAGR range: -50% to +100%
    return np.clip((cagr + 0.5) / 1.5, 0, 1)


def normalize_sharpe(sharpe: float) -> float:
    """Normalize Sharpe ratio to 0-1 range."""
    # Assume reasonable Sharpe range: -2 to +3
    return np.clip((sharpe + 2) / 5, 0, 1)


def normalize_win_rate(win_rate: float) -> float:
    """Normalize win rate to 0-1 range."""
    # Win rate is already 0-1, just ensure bounds
    return np.clip(win_rate, 0, 1)


def normalize_avg_trade_return(avg_return: float) -> float:
    """Normalize average trade return to 0-1 range."""
    # Assume reasonable range: -1% to +1%
    return np.clip((avg_return + 0.01) / 0.02, 0, 1)


def composite_score(
    metrics: dict[str, float | int],
    weights: CompositeWeights | None = None,
    penalties: CompositePenalties | None = None,
) -> float:
    """
    Calculate composite performance score.

    Args:
        metrics: Dictionary containing performance metrics
            - cagr: Compound annual growth rate (as decimal)
            - sharpe: Sharpe ratio
            - win_rate: Win rate (0-1)
            - avg_trade_return: Average trade return (as decimal)
            - max_dd: Maximum drawdown (as decimal)
            - trade_count: Number of trades
        weights: Optional custom weights
        penalties: Optional custom penalty thresholds

    Returns:
        Composite score (0-1, higher is better)
    """
    if weights is None:
        weights = CompositeWeights()

    if penalties is None:
        penalties = CompositePenalties()

    # Extract metrics with defaults
    cagr = metrics.get("cagr", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    avg_trade_return = metrics.get("avg_trade_return", 0.0)
    max_dd = abs(metrics.get("max_dd", 0.0))  # Use absolute value
    trade_count = metrics.get("trade_count", 0)

    # Normalize metrics
    norm_cagr = normalize_cagr(cagr * 100)  # Convert to percentage
    norm_sharpe = normalize_sharpe(sharpe)
    norm_win_rate = normalize_win_rate(win_rate)
    norm_avg_return = normalize_avg_trade_return(avg_trade_return * 100)  # Convert to percentage

    # Calculate weighted sum
    weighted_sum = (
        weights.alpha * norm_cagr
        + weights.beta * norm_sharpe
        + weights.gamma * norm_win_rate
        + weights.delta * norm_avg_return
    )

    # Apply penalties
    penalty = 0.0

    # Drawdown penalty
    if max_dd > penalties.max_dd_cap:
        dd_excess = (max_dd - penalties.max_dd_cap) / penalties.max_dd_cap
        penalty += penalties.dd_penalty_factor * dd_excess

    # Trade count penalty
    if trade_count < penalties.min_trades:
        trade_deficit = (penalties.min_trades - trade_count) / penalties.min_trades
        penalty += penalties.trade_penalty_factor * trade_deficit

    # Apply penalty (reduce score)
    final_score = max(0, weighted_sum - penalty)

    return np.clip(final_score, 0, 1)


def evaluate_strategy_performance(
    results: dict[str, float | int],
    weights: CompositeWeights | None = None,
    penalties: CompositePenalties | None = None,
) -> dict[str, float]:
    """
    Evaluate strategy performance with detailed breakdown.

    Args:
        results: Strategy results dictionary
        weights: Optional custom weights
        penalties: Optional custom penalty thresholds

    Returns:
        Dictionary with composite score and component breakdown
    """
    if weights is None:
        weights = CompositeWeights()

    if penalties is None:
        penalties = CompositePenalties()

    # Extract and normalize metrics
    cagr = results.get("cagr", 0.0)
    sharpe = results.get("sharpe", 0.0)
    win_rate = results.get("win_rate", 0.0)
    avg_trade_return = results.get("avg_trade_return", 0.0)
    max_dd = abs(results.get("max_dd", 0.0))
    trade_count = results.get("trade_count", 0)

    norm_cagr = normalize_cagr(cagr * 100)
    norm_sharpe = normalize_sharpe(sharpe)
    norm_win_rate = normalize_win_rate(win_rate)
    norm_avg_return = normalize_avg_trade_return(avg_trade_return * 100)

    # Calculate component scores
    cagr_score = weights.alpha * norm_cagr
    sharpe_score = weights.beta * norm_sharpe
    win_rate_score = weights.gamma * norm_win_rate
    avg_return_score = weights.delta * norm_avg_return

    # Calculate penalties
    dd_penalty = 0.0
    trade_penalty = 0.0

    if max_dd > penalties.max_dd_cap:
        dd_excess = (max_dd - penalties.max_dd_cap) / penalties.max_dd_cap
        dd_penalty = penalties.dd_penalty_factor * dd_excess

    if trade_count < penalties.min_trades:
        trade_deficit = (penalties.min_trades - trade_count) / penalties.min_trades
        trade_penalty = penalties.trade_penalty_factor * trade_deficit

    total_penalty = dd_penalty + trade_penalty
    weighted_sum = cagr_score + sharpe_score + win_rate_score + avg_return_score
    final_score = max(0, weighted_sum - total_penalty)

    return {
        "composite_score": np.clip(final_score, 0, 1),
        "weighted_sum": weighted_sum,
        "total_penalty": total_penalty,
        "components": {
            "cagr_score": cagr_score,
            "sharpe_score": sharpe_score,
            "win_rate_score": win_rate_score,
            "avg_return_score": avg_return_score,
        },
        "normalized_metrics": {
            "norm_cagr": norm_cagr,
            "norm_sharpe": norm_sharpe,
            "norm_win_rate": norm_win_rate,
            "norm_avg_return": norm_avg_return,
        },
        "penalties": {"dd_penalty": dd_penalty, "trade_penalty": trade_penalty},
    }


def load_composite_config(config: dict) -> tuple[CompositeWeights, CompositePenalties]:
    """
    Load composite scoring configuration from config dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (weights, penalties)
    """
    # Extract weights
    weights_dict = config.get("metric_weights", {})
    weights = CompositeWeights(
        alpha=weights_dict.get("alpha", 0.4),
        beta=weights_dict.get("beta", 0.3),
        gamma=weights_dict.get("gamma", 0.2),
        delta=weights_dict.get("delta", 0.1),
    )

    # Extract penalties
    penalties = CompositePenalties(
        max_dd_cap=config.get("metric_weight_max_dd_cap", 0.25),
        min_trades=config.get("metric_weight_min_trades", 200),
        dd_penalty_factor=config.get("metric_weight_dd_penalty_factor", 2.0),
        trade_penalty_factor=config.get("metric_weight_trade_penalty_factor", 1.5),
    )

    return weights, penalties
