# ml/eval/alpha_eval.py
"""
Alpha evaluation with walkforward testing and cost-aware metrics.
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def load_feature_config() -> dict:
    """Load feature configuration."""
    config_path = Path("config/features.yaml")
    return yaml.safe_load(config_path.read_text())


def load_model_config() -> dict:
    """Load model configuration."""
    config_path = Path("config/models.yaml")
    return yaml.safe_load(config_path.read_text())


def load_trading_config() -> dict:
    """Load trading configuration for costs."""
    config_path = Path("config/base.yaml")
    return yaml.safe_load(config_path.read_text())


def calculate_ic_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Spearman correlation (IC)."""
    if len(predictions) < 2:
        return 0.0
    try:
        correlation, _ = spearmanr(predictions, targets)
        return correlation if not np.isnan(correlation) else 0.0
    except Exception:
        return 0.0


def calculate_hit_rate(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate directional accuracy (hit rate)."""
    if len(predictions) < 1:
        return 0.0
    correct_direction = np.sign(predictions) == np.sign(targets)
    return np.mean(correct_direction)


def calculate_turnover(positions: np.ndarray) -> float:
    """Calculate portfolio turnover."""
    if len(positions) < 2:
        return 0.0
    position_changes = np.abs(np.diff(positions))
    return np.mean(position_changes)


def calculate_return_with_costs(
    predictions: np.ndarray, targets: np.ndarray, slippage_bps: float = 5.0, fee_bps: float = 1.0
) -> float:
    """
    Calculate return with trading costs applied.

    Args:
        predictions: Model predictions
        targets: Actual returns
        slippage_bps: Slippage in basis points
        fee_bps: Commission in basis points

    Returns:
        Net return after costs
    """
    if len(predictions) < 1:
        return 0.0

    # Convert predictions to positions (long/short)
    positions = np.sign(predictions)

    # Calculate gross returns
    gross_returns = positions * targets

    # Calculate costs
    # Slippage: proportional to position size
    slippage_costs = np.abs(positions) * (slippage_bps / 10000)
    # Fees: proportional to position changes
    position_changes = np.abs(np.diff(positions, prepend=0))
    fee_costs = position_changes * (fee_bps / 10000)

    # Net returns
    net_returns = gross_returns - slippage_costs - fee_costs

    return np.sum(net_returns)


def create_walkforward_folds(
    df: pd.DataFrame, n_folds: int = 5, min_train_size: int = 252
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walkforward folds for time series validation.

    Args:
        df: Combined feature DataFrame
        n_folds: Number of folds
        min_train_size: Minimum training size in days

    Returns:
        List of (train_df, test_df) tuples
    """
    df = df.sort_index()
    total_size = len(df)

    if total_size < min_train_size * 2:
        raise ValueError(f"Insufficient data: {total_size} < {min_train_size * 2}")

    folds = []
    fold_size = (total_size - min_train_size) // n_folds

    for i in range(n_folds):
        train_end = min_train_size + i * fold_size
        test_end = min(total_size, train_end + fold_size)

        if test_end <= train_end:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        folds.append((train_df, test_df))

    return folds


def evaluate_fold(
    train_df: pd.DataFrame, test_df: pd.DataFrame, model_path: str, config: dict
) -> dict:
    """
    Evaluate a single fold.

    Args:
        train_df: Training data
        test_df: Test data
        model_path: Path to trained model
        config: Configuration dict

    Returns:
        Fold evaluation results
    """
    import pickle

    # Load model
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Prepare features and target
    feature_cols = list(config["features"].keys())
    target_col = "ret_fwd_1d"

    # Prepare test data
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Remove NaN values
    mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    if len(X_test) == 0:
        return {
            "ic_spearman": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "return_with_costs": 0.0,
            "n_predictions": 0,
        }

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Calculate metrics
    ic = calculate_ic_spearman(predictions, y_test)
    hit_rate = calculate_hit_rate(predictions, y_test)
    turnover = calculate_turnover(predictions)

    # Load trading config for costs
    trading_config = load_trading_config()
    risk_config = trading_config.get("risk", {})
    slippage_bps = risk_config.get("slippage_bps", 5.0)
    fee_bps = risk_config.get("fee_bps", 1.0)

    return_with_costs = calculate_return_with_costs(predictions, y_test, slippage_bps, fee_bps)

    return {
        "ic_spearman": ic,
        "hit_rate": hit_rate,
        "turnover": turnover,
        "return_with_costs": return_with_costs,
        "n_predictions": len(predictions),
    }


def evaluate_alpha_model(
    symbols: list[str],
    model_path: str = "artifacts/models/linear_v1.pkl",
    n_folds: int = 5,
    output_path: str = "reports/alpha_eval.json",
) -> dict:
    """
    Evaluate alpha model with walkforward testing.

    Args:
        symbols: List of symbols to evaluate
        model_path: Path to trained model
        n_folds: Number of walkforward folds
        output_path: Path to save evaluation results

    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Evaluating alpha model on {symbols}")

    # Load configurations
    feature_config = load_feature_config()
    load_model_config()

    # Load feature data
    from ml.trainers.train_linear import load_feature_data

    df = load_feature_data(symbols)

    # Create walkforward folds
    folds = create_walkforward_folds(df, n_folds)

    if len(folds) == 0:
        raise ValueError("No valid folds created")

    logger.info(f"Created {len(folds)} walkforward folds")

    # Evaluate each fold
    fold_summaries = []
    all_ics = []
    all_hit_rates = []
    all_turnovers = []
    all_returns = []

    for i, (train_df, test_df) in enumerate(folds):
        logger.info(f"Evaluating fold {i + 1}/{len(folds)}")

        # Evaluate fold
        fold_results = evaluate_fold(train_df, test_df, model_path, feature_config)

        # Create fold summary
        fold_summary = {
            "fold": i + 1,
            "train_start": train_df.index[0].strftime("%Y-%m-%d"),
            "train_end": train_df.index[-1].strftime("%Y-%m-%d"),
            "test_start": test_df.index[0].strftime("%Y-%m-%d"),
            "test_end": test_df.index[-1].strftime("%Y-%m-%d"),
            **fold_results,
        }

        fold_summaries.append(fold_summary)

        # Collect metrics for overall summary
        all_ics.append(fold_results["ic_spearman"])
        all_hit_rates.append(fold_results["hit_rate"])
        all_turnovers.append(fold_results["turnover"])
        all_returns.append(fold_results["return_with_costs"])

    # Calculate overall metrics
    overall_metrics = {
        "mean_ic": np.mean(all_ics),
        "mean_hit_rate": np.mean(all_hit_rates),
        "mean_turnover": np.mean(all_turnovers),
        "mean_return_with_costs": np.mean(all_returns),
        "std_ic": np.std(all_ics),
        "std_hit_rate": np.std(all_hit_rates),
        "min_ic": np.min(all_ics),
        "max_ic": np.max(all_ics),
        "min_hit_rate": np.min(all_hit_rates),
        "max_hit_rate": np.max(all_hit_rates),
    }

    # Create evaluation results
    results = {
        "model_id": "linear_v1",
        "symbols": symbols,
        "fold_summaries": fold_summaries,
        "overall_metrics": overall_metrics,
        "metadata": {
            "evaluation_date": datetime.now(UTC).isoformat(),
            "model_path": model_path,
            "feature_config": feature_config,
            "training_samples": len(folds[0][0]) if folds else 0,
            "test_samples": sum(len(fold[1]) for fold in folds),
            "n_folds": len(folds),
        },
    }

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {output_file}")

    # Log summary
    logger.info(f"Overall IC: {overall_metrics['mean_ic']:.4f} ± {overall_metrics['std_ic']:.4f}")
    logger.info(
        f"Overall Hit Rate: {overall_metrics['mean_hit_rate']:.4f} ± {overall_metrics['std_hit_rate']:.4f}"
    )
    logger.info(f"Overall Turnover: {overall_metrics['mean_turnover']:.4f}")
    logger.info(f"Overall Return (with costs): {overall_metrics['mean_return_with_costs']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="SPY,TSLA", help="Comma-separated symbols")
    parser.add_argument("--model-path", default="artifacts/models/linear_v1.pkl", help="Model path")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of walkforward folds")
    parser.add_argument("--output-path", default="reports/alpha_eval.json", help="Output path")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    results = evaluate_alpha_model(symbols, args.model_path, args.n_folds, args.output_path)

    print(f"Evaluation complete. Results saved to {args.output_path}")
