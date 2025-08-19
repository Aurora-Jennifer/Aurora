#!/usr/bin/env python3
"""
Train Composer System to Maximize Returns
Uses walkforward analysis to optimize composer parameters and strategy weights.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import load_config, save_config
from core.metrics.composite import CompositeWeights, composite_score
from scripts.walkforward_with_composer import run_walkforward_with_composer


def setup_logging():
    """Setup logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/composer_training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def generate_composer_trials() -> list[dict[str, Any]]:
    """Generate different composer parameter combinations to test."""
    trials = []

    # Base composer parameters
    base_params = {
        "temperature": 1.0,
        "trend_bias": 1.2,
        "chop_bias": 1.1,
        "min_confidence": 0.1,
    }

    # Temperature variations
    for temp in [0.5, 0.8, 1.0, 1.2, 1.5]:
        params = base_params.copy()
        params["temperature"] = temp
        trials.append({"name": f"temp_{temp}", "composer_params": params})

    # Trend bias variations
    for trend_bias in [0.8, 1.0, 1.2, 1.4, 1.6]:
        params = base_params.copy()
        params["trend_bias"] = trend_bias
        trials.append({"name": f"trend_{trend_bias}", "composer_params": params})

    # Chop bias variations
    for chop_bias in [0.8, 1.0, 1.1, 1.3, 1.5]:
        params = base_params.copy()
        params["chop_bias"] = chop_bias
        trials.append({"name": f"chop_{chop_bias}", "composer_params": params})

    # Strategy combinations
    strategy_combinations = [
        ["momentum", "mean_reversion"],
        ["momentum", "breakout"],
        ["mean_reversion", "breakout"],
        ["momentum", "mean_reversion", "breakout"],
    ]

    for i, strategies in enumerate(strategy_combinations):
        trials.append(
            {
                "name": f"strategies_{i + 1}",
                "eligible_strategies": strategies,
                "composer_params": base_params,
            }
        )

    # Regime extractor variations
    for extractor in ["basic_kpis", "advanced_kpis"]:
        trials.append(
            {
                "name": f"extractor_{extractor}",
                "regime_extractor": extractor,
                "composer_params": base_params,
            }
        )

    return trials


def generate_weight_trials() -> list[dict[str, Any]]:
    """Generate different metric weight combinations to test."""
    trials = []

    # CAGR-focused weights
    for alpha in [0.5, 0.6, 0.7, 0.8]:
        weights = CompositeWeights(alpha=alpha, beta=0.3, gamma=0.2, delta=1.0 - alpha - 0.3 - 0.2)
        trials.append(
            {
                "name": f"cagr_focused_{alpha}",
                "metric_weights": {
                    "alpha": weights.alpha,
                    "beta": weights.beta,
                    "gamma": weights.gamma,
                    "delta": weights.delta,
                },
            }
        )

    # Sharpe-focused weights
    for beta in [0.5, 0.6, 0.7, 0.8]:
        weights = CompositeWeights(alpha=0.2, beta=beta, gamma=0.2, delta=1.0 - 0.2 - beta - 0.2)
        trials.append(
            {
                "name": f"sharpe_focused_{beta}",
                "metric_weights": {
                    "alpha": weights.alpha,
                    "beta": weights.beta,
                    "gamma": weights.gamma,
                    "delta": weights.delta,
                },
            }
        )

    # Balanced weights
    balanced_weights = [
        (0.25, 0.25, 0.25, 0.25),
        (0.3, 0.3, 0.2, 0.2),
        (0.4, 0.3, 0.2, 0.1),
        (0.2, 0.4, 0.3, 0.1),
    ]

    for i, (alpha, beta, gamma, delta) in enumerate(balanced_weights):
        trials.append(
            {
                "name": f"balanced_{i + 1}",
                "metric_weights": {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": delta,
                },
            }
        )

    return trials


def evaluate_trial(
    trial_config: dict[str, Any],
    symbols: list[str],
    start_date: str,
    end_date: str,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a single trial configuration."""
    logger = logging.getLogger(__name__)

    # Create trial-specific config (deep copy)
    import copy

    config = copy.deepcopy(base_config)

    # Apply composer overrides
    if "composer_params" in trial_config:
        config["composer"]["composer_params"] = trial_config["composer_params"]

    if "eligible_strategies" in trial_config:
        config["composer"]["eligible_strategies"] = trial_config["eligible_strategies"]

    if "regime_extractor" in trial_config:
        config["composer"]["regime_extractor"] = trial_config["regime_extractor"]

    # Apply metric weight overrides
    if "metric_weights" in trial_config:
        config["optimization"]["metric_weights"] = trial_config["metric_weights"]

    # Enable composer
    config["composer"]["use_composer"] = True
    logger.info(f"Composer enabled: {config['composer']['use_composer']}")
    logger.info(f"Composer config: {config['composer']}")

    try:
        # Run walkforward analysis for each symbol
        all_results = []

        for symbol in symbols:
            logger.info(f"Running trial {trial_config['name']} for {symbol}")

            # Load data for the symbol
            import yfinance as yf

            # Get auto_adjust setting from config
            auto_adjust = config.get("data", {}).get("auto_adjust", False)
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=auto_adjust,
            )

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                continue

            # Add required columns
            data["Returns"] = data["Close"].pct_change()
            data["Symbol"] = symbol

            results = run_walkforward_with_composer(data=data, symbol=symbol, config=config)

            if results:
                all_results.append(results)

        if not all_results:
            logger.warning(f"Trial {trial_config['name']} produced no results")
            return {
                "trial_name": trial_config["name"],
                "composite_score": 0.0,
                "metrics": {},
                "error": "No results produced",
            }

        # Aggregate results across symbols
        aggregate_metrics = aggregate_results(all_results)

        # Calculate composite score
        composite = composite_score(aggregate_metrics)

        logger.info(f"Trial {trial_config['name']} - Composite Score: {composite:.4f}")

        return {
            "trial_name": trial_config["name"],
            "composite_score": composite,
            "metrics": aggregate_metrics,
            "config": trial_config,
        }

    except Exception as e:
        logger.error(f"Error in trial {trial_config['name']}: {e}")
        return {
            "trial_name": trial_config["name"],
            "composite_score": 0.0,
            "metrics": {},
            "error": str(e),
        }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results across multiple symbols."""
    if not results:
        return {}

    # Extract metrics from each result
    all_metrics = []
    for result in results:
        if "aggregate" in result:
            all_metrics.append(result["aggregate"])

    if not all_metrics:
        return {}

    # Calculate averages
    avg_metrics = {}
    for key in all_metrics[0]:
        if key in ["mean_sharpe", "mean_total_return", "mean_hit_rate", "mean_max_dd"]:
            values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)

    # Convert to expected format for composite_score
    return {
        "cagr": avg_metrics.get("mean_total_return", 0.0),
        "sharpe": avg_metrics.get("mean_sharpe", 0.0),
        "win_rate": avg_metrics.get("mean_hit_rate", 0.0),
        "avg_trade_return": avg_metrics.get("mean_total_return", 0.0) / 100,  # Rough estimate
        "max_dd": abs(avg_metrics.get("mean_max_dd", 0.0)),
        "trade_count": sum(m.get("total_closed_trades", 0) for m in all_metrics),
    }


def train_composer_system(
    symbols: list[str],
    start_date: str,
    end_date: str,
    base_config: dict[str, Any],
    max_trials: int = 20,
) -> dict[str, Any]:
    """Train the composer system to maximize returns."""
    logger = logging.getLogger(__name__)

    logger.info("🎯 Starting Composer System Training")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Max trials: {max_trials}")

    # Generate trials
    composer_trials = generate_composer_trials()
    weight_trials = generate_weight_trials()

    # Combine trials
    all_trials = []

    # Ensure we have at least one trial from each category
    composer_count = max(1, max_trials // 2)
    weight_count = max(1, max_trials // 2)

    # Test composer parameters with default weights
    for trial in composer_trials[:composer_count]:
        all_trials.append(trial)

    # Test weight combinations with default composer
    for trial in weight_trials[:weight_count]:
        all_trials.append(trial)

    logger.info(f"Generated {len(all_trials)} trials to evaluate")

    # Evaluate each trial
    results = []
    for i, trial in enumerate(all_trials):
        logger.info(f"Evaluating trial {i + 1}/{len(all_trials)}: {trial['name']}")

        result = evaluate_trial(
            trial_config=trial,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            base_config=base_config,
        )

        results.append(result)

        # Log progress
        if result["composite_score"] > 0:
            logger.info(f"  Score: {result['composite_score']:.4f}")
        else:
            logger.info(f"  Failed: {result.get('error', 'Unknown error')}")

    # Find best configuration
    valid_results = [r for r in results if r["composite_score"] > 0]

    if not valid_results:
        logger.error("No valid trials completed!")
        return {}

    best_result = max(valid_results, key=lambda x: x["composite_score"])

    logger.info("\n🏆 Best Configuration Found:")
    logger.info(f"Trial: {best_result['trial_name']}")
    logger.info(f"Composite Score: {best_result['composite_score']:.4f}")
    logger.info(f"Metrics: {best_result['metrics']}")

    # Save best configuration
    best_config = base_config.copy()
    best_trial_config = best_result["config"]

    # Apply best composer settings
    if "composer_params" in best_trial_config:
        best_config["composer"]["composer_params"] = best_trial_config["composer_params"]

    if "eligible_strategies" in best_trial_config:
        best_config["composer"]["eligible_strategies"] = best_trial_config["eligible_strategies"]

    if "regime_extractor" in best_trial_config:
        best_config["composer"]["regime_extractor"] = best_trial_config["regime_extractor"]

    # Apply best metric weights
    if "metric_weights" in best_trial_config:
        best_config["optimization"]["metric_weights"] = best_trial_config["metric_weights"]

    # Save optimized config
    save_config(best_config, "config/optimized_composer.json")

    # Save training results
    training_results = {
        "best_config": best_result,
        "all_results": results,
        "training_params": {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "max_trials": max_trials,
        },
    }

    save_config(training_results, "results/composer_training_results.json")

    logger.info("✅ Training completed! Best config saved to config/optimized_composer.json")

    return best_result


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Composer System to Maximize Returns")
    parser.add_argument("--symbols", nargs="+", help="Symbols to train on (overrides config)")
    parser.add_argument("--start-date", default="2020-01-01", help="Training start date")
    parser.add_argument("--end-date", default="2023-12-31", help="Training end date")
    parser.add_argument("--config", default="config/base.json", help="Base configuration file")
    parser.add_argument(
        "--profile",
        choices=["risk_low", "risk_balanced", "risk_strict"],
        help="Risk profile to apply",
    )
    parser.add_argument("--max-trials", type=int, default=20, help="Maximum trials to run")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Build CLI overrides
    cli_overrides = {}
    if args.symbols:
        cli_overrides["symbols"] = args.symbols

    # Load configuration
    config = load_config(
        profile=args.profile, cli_overrides=cli_overrides, base_config_path=args.config
    )

    # Use config symbols if not specified
    symbols = args.symbols or config["symbols"][:3]  # Use first 3 symbols for training

    # Train the composer system
    best_result = train_composer_system(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        base_config=config,
        max_trials=args.max_trials,
    )

    if best_result:
        print("\n🎉 Training completed successfully!")
        print(f"Best trial: {best_result['trial_name']}")
        print(f"Best score: {best_result['composite_score']:.4f}")
        print("Optimized config saved to: config/optimized_composer.json")
    else:
        print("❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
