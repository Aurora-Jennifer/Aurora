#!/usr/bin/env python3
"""
End-to-end Alpha v1 training pipeline.
Builds features, trains Ridge model, and evaluates with walkforward testing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(description="Train Alpha v1 model end-to-end")
    parser.add_argument("--symbols", default="SPY,TSLA", help="Comma-separated symbols")
    parser.add_argument(
        "--feature-dir", default="artifacts/feature_store", help="Feature directory"
    )
    parser.add_argument(
        "--model-path", default="artifacts/models/linear_v1.pkl", help="Model output path"
    )
    parser.add_argument(
        "--eval-output", default="reports/alpha_eval.json", help="Evaluation output path"
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of walkforward folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    symbols = [s.strip() for s in args.symbols.split(",")]
    logger.info(f"Starting Alpha v1 training pipeline for {symbols}")

    try:
        # Step 1: Build features
        logger.info("Step 1: Building features...")
        from ml.features.build_daily import build_feature_store

        feature_results = build_feature_store(symbols, args.feature_dir)

        if not feature_results:
            raise ValueError("No features built successfully")

        logger.info(f"Built features for {len(feature_results)} symbols")

        # Step 2: Train model
        logger.info("Step 2: Training Ridge model...")
        from ml.trainers.train_linear import train_linear_model

        pipeline = train_linear_model(symbols, args.feature_dir, args.model_path, args.random_state)

        logger.info("Model training complete")

        # Step 3: Evaluate model
        logger.info("Step 3: Evaluating model with walkforward testing...")
        from ml.eval.alpha_eval import evaluate_alpha_model

        eval_results = evaluate_alpha_model(
            symbols, args.model_path, args.n_folds, args.eval_output
        )

        logger.info("Evaluation complete")

        # Step 4: Print summary
        overall_metrics = eval_results["overall_metrics"]
        logger.info("=" * 50)
        logger.info("ALPHA V1 TRAINING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Evaluation: {args.eval_output}")
        logger.info("")
        logger.info("OVERALL METRICS:")
        logger.info(
            f"  IC (Spearman): {overall_metrics['mean_ic']:.4f} ± {overall_metrics['std_ic']:.4f}"
        )
        logger.info(
            f"  Hit Rate: {overall_metrics['mean_hit_rate']:.4f} ± {overall_metrics['std_hit_rate']:.4f}"
        )
        logger.info(f"  Turnover: {overall_metrics['mean_turnover']:.4f}")
        logger.info(f"  Return (with costs): {overall_metrics['mean_return_with_costs']:.4f}")
        logger.info("")
        logger.info("PROMOTION GATES:")
        logger.info(f"  IC ≥ 0.02: {'✅' if overall_metrics['mean_ic'] >= 0.02 else '❌'}")
        logger.info(
            f"  Hit Rate ≥ 0.52: {'✅' if overall_metrics['mean_hit_rate'] >= 0.52 else '❌'}"
        )

        # Check if model meets promotion criteria
        ic_ok = overall_metrics["mean_ic"] >= 0.02
        hit_rate_ok = overall_metrics["mean_hit_rate"] >= 0.52

        if ic_ok and hit_rate_ok:
            logger.info("")
            logger.info("🎉 MODEL MEETS PROMOTION CRITERIA!")
            logger.info("Next steps:")
            logger.info("  1. python tools/validate_alpha.py reports/alpha_eval.json")
            logger.info("  2. python tools/bless_model_inference.py")
            logger.info("  3. make smoke")
            logger.info("  4. Update config/models.yaml to select 'linear_v1'")
        else:
            logger.info("")
            logger.info("⚠️  MODEL DOES NOT MEET PROMOTION CRITERIA")
            logger.info("Consider:")
            logger.info("  - Adding more features")
            logger.info("  - Adjusting hyperparameters")
            logger.info("  - Expanding universe")
            logger.info("  - Longer training period")

        return 0

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
