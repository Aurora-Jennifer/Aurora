"""
Weight Tuner System
Optimizes composite scoring weights through walkforward analysis.
"""

import json
import logging
import os
import random
import subprocess
from pathlib import Path

import numpy as np

from .composite import CompositePenalties, CompositeWeights, composite_score

logger = logging.getLogger(__name__)


def softmax_normalize(weights: list[float]) -> list[float]:
    """Normalize weights using softmax function."""
    exp_weights = np.exp(weights)
    return (exp_weights / np.sum(exp_weights)).tolist()


def generate_weight_candidate() -> CompositeWeights:
    """Generate a random weight candidate."""
    # Generate random weights
    raw_weights = [random.uniform(-2, 2) for _ in range(4)]
    normalized = softmax_normalize(raw_weights)

    return CompositeWeights(
        alpha=normalized[0],
        beta=normalized[1],
        gamma=normalized[2],
        delta=normalized[3],
    )


def run_walkforward_analysis(
    symbol: str, start_date: str, end_date: str, config_file: str
) -> dict | None:
    """
    Run walkforward analysis and extract metrics.

    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        config_file: Configuration file path

    Returns:
        Dictionary with performance metrics or None if failed
    """
    try:
        cmd = [
            "python",
            "scripts/walkforward_framework.py",
            "--symbol",
            symbol,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--train-len",
            "252",
            "--test-len",
            "126",
            "--stride",
            "63",
            "--perf-mode",
            "RELAXED",
            "--validate-data",
        ]

        logger.info(f"Running walkforward for {symbol}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Load results from saved file
            results_file = "results/walkforward/results.json"
            if os.path.exists(results_file):
                with open(results_file) as f:
                    data = json.load(f)

                # Extract metrics
                aggregate = data.get("aggregate", {})
                fold_results = data.get("fold_results", [])

                # Calculate additional metrics
                total_trades = sum(fold.get("n_trades", 0) for fold in fold_results)

                # Estimate CAGR (simplified calculation)
                total_return = aggregate.get("mean_total_return", 0)
                years = 5  # Approximate for 2019-2023
                cagr = ((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -0.5

                # Estimate win rate (simplified)
                win_rate = aggregate.get("mean_hit_rate", 0.5)

                # Estimate average trade return
                avg_trade_return = (
                    aggregate.get("avg_trade_pnl", 0.0) / 100000
                )  # Normalize to portfolio size

                return {
                    "cagr": cagr,
                    "sharpe": aggregate.get("mean_sharpe", 0.0),
                    "win_rate": win_rate,
                    "avg_trade_return": avg_trade_return,
                    "max_dd": abs(aggregate.get("mean_max_dd", 0.0)),
                    "trade_count": total_trades,
                }

        logger.error(f"Walkforward failed for {symbol}: {result.stderr}")
        return None

    except Exception as e:
        logger.error(f"Exception in walkforward for {symbol}: {e}")
        return None


def evaluate_weights(
    weights: CompositeWeights,
    symbols: list[str],
    start_date: str,
    end_date: str,
    config_file: str,
    penalties: CompositePenalties,
) -> float:
    """
    Evaluate a set of weights across multiple symbols.

    Args:
        weights: Weight configuration to test
        symbols: List of symbols to test
        start_date: Start date
        end_date: End date
        config_file: Configuration file
        penalties: Penalty configuration

    Returns:
        Average composite score across symbols
    """
    scores = []

    for symbol in symbols:
        metrics = run_walkforward_analysis(symbol, start_date, end_date, config_file)

        if metrics:
            score = composite_score(metrics, weights, penalties)
            scores.append(score)
            logger.info(f"{symbol}: score={score:.4f}, weights={weights}")
        else:
            logger.warning(f"Failed to get metrics for {symbol}")

    if not scores:
        return 0.0

    return np.mean(scores)


def tune_weights(
    config_path: str,
    trials: int = 50,
    save_to: str = "config/metrics_weights.json",
    symbols: list[str] | None = None,
    start_date: str = "2019-01-01",
    end_date: str = "2023-12-31",
) -> tuple[CompositeWeights, float]:
    """
    Tune metric weights through walkforward analysis.

    Args:
        config_path: Path to configuration file
        trials: Number of optimization trials
        save_to: Path to save best weights
        symbols: List of symbols to test (default: SPY, TSLA, BTC-USD)
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        Tuple of (best_weights, best_score)
    """
    if symbols is None:
        symbols = ["SPY", "TSLA", "BTC-USD"]

    # Load configuration
    with open(config_path) as f:
        config = json.load(f)

    # Extract penalty configuration
    penalties = CompositePenalties(
        max_dd_cap=config.get("metric_weight_max_dd_cap", 0.25),
        min_trades=config.get("metric_weight_min_trades", 200),
        dd_penalty_factor=config.get("metric_weight_dd_penalty_factor", 2.0),
        trade_penalty_factor=config.get("metric_weight_trade_penalty_factor", 1.5),
    )

    logger.info(f"Starting weight tuning with {trials} trials")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Penalties: {penalties}")

    best_weights = None
    best_score = 0.0
    no_improve_count = 0

    for trial in range(trials):
        logger.info(f"Trial {trial + 1}/{trials}")

        # Generate candidate weights
        candidate_weights = generate_weight_candidate()

        # Evaluate candidate
        score = evaluate_weights(
            candidate_weights, symbols, start_date, end_date, config_path, penalties
        )

        logger.info(f"Trial {trial + 1}: score={score:.4f}")

        # Update best if improved
        if score > best_score:
            best_score = score
            best_weights = candidate_weights
            no_improve_count = 0
            logger.info(f"New best score: {best_score:.4f}")
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= 10:
            logger.info(f"Early stopping after {trial + 1} trials (no improvement for 10 trials)")
            break

    # Save best weights
    if best_weights:
        weights_dict = {
            "alpha": best_weights.alpha,
            "beta": best_weights.beta,
            "gamma": best_weights.gamma,
            "delta": best_weights.delta,
            "metadata": {
                "best_score": best_score,
                "trials_run": trial + 1,
                "symbols_tested": symbols,
                "date_range": f"{start_date} to {end_date}",
            },
        }

        # Ensure directory exists
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_to, "w") as f:
            json.dump(weights_dict, f, indent=2)

        logger.info(f"Best weights saved to {save_to}")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best weights: {best_weights}")

    return best_weights, best_score


def load_tuned_weights(weights_path: str) -> CompositeWeights:
    """
    Load tuned weights from file.

    Args:
        weights_path: Path to weights file

    Returns:
        CompositeWeights object
    """
    with open(weights_path) as f:
        data = json.load(f)

    return CompositeWeights(
        alpha=data["alpha"], beta=data["beta"], gamma=data["gamma"], delta=data["delta"]
    )


def validate_weights(weights: CompositeWeights) -> bool:
    """
    Validate that weights sum to approximately 1.0.

    Args:
        weights: Weights to validate

    Returns:
        True if valid, False otherwise
    """
    total = weights.alpha + weights.beta + weights.gamma + weights.delta
    return abs(total - 1.0) < 0.01


if __name__ == "__main__":
    # Example usage
    from core.utils import setup_logging
    setup_logging("logs/weight_tuner.log", logging.INFO)

    # Test weight tuning
    best_weights, best_score = tune_weights(
        config_path="config/risk_balanced.json",
        trials=10,  # Small number for testing
        symbols=["SPY", "TSLA"],
    )

    print(f"Best score: {best_score:.4f}")
    print(f"Best weights: {best_weights}")
