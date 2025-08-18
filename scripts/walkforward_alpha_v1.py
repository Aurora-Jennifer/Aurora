#!/usr/bin/env python3
"""
Walkforward testing with Alpha v1 ML model.
Uses the trained Ridge regression model for predictions.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.walk.ml_pipeline import create_ml_pipeline
from core.walk.folds import gen_walkforward, Fold
from core.sim.simulate import simulate_safe
from core.metrics.stats import (
    daily_turnover, max_drawdown, psr, 
    sharpe_newey_west, sortino, win_rate
)
from ml.features.build_daily import build_features_for_symbol
from ml.trainers.train_linear import load_feature_data
from core.utils import setup_logging

logger = setup_logging("logs/walkforward_alpha_v1.log", logging.INFO)


def load_alpha_v1_features(symbols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Alpha v1 features and prepare for walkforward testing.
    
    Args:
        symbols: List of symbols to load
        
    Returns:
        Tuple of (features, targets, prices)
    """
    logger.info(f"Loading Alpha v1 features for {symbols}")
    
    # Load feature data
    df = load_feature_data(symbols, "artifacts/feature_store")
    
    # Load feature config
    from ml.features.build_daily import load_feature_config
    config = load_feature_config()
    feature_cols = list(config['features'].keys())
    target_col = 'ret_fwd_1d'
    
    # Prepare features and targets
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Get price data for simulation
    # We need to reconstruct prices from returns
    returns = df['ret_1d'].values
    prices = np.cumprod(1 + returns) * 100  # Start at 100
    
    # Remove any NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    prices = prices[mask]
    
    logger.info(f"Loaded {len(X)} samples with {len(feature_cols)} features")
    return X, y, prices


def run_alpha_v1_fold(pipeline, X: np.ndarray, y: np.ndarray, prices: np.ndarray, fold: Fold) -> dict:
    """
    Run a single fold with Alpha v1 model.
    
    Args:
        pipeline: ML pipeline with Alpha v1 model
        X: Feature matrix
        y: Target values
        prices: Price series
        fold: Fold object
        
    Returns:
        Dictionary with fold metrics
    """
    # Get train/test indices
    train_idx = np.arange(fold.train_lo, fold.train_hi + 1)
    test_idx = np.arange(fold.test_lo, fold.test_hi + 1)
    
    # Prepare train/test data
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    prices_test = prices[test_idx]
    
    # Fit pipeline (model is already trained, just store training data)
    pipeline.fit_transforms(train_idx)
    pipeline.fit_model(X_train, y_train)
    
    # Generate predictions
    signals = pipeline.predict(X_test)
    
    # Simulate trading
    pnl, ntrades, wins, losses, med_hold = simulate_safe(
        prices_test, signals.astype(np.int8)
    )
    ret = np.diff(pnl)
    
    # Calculate metrics
    metrics = {
        "fold_id": fold.fold_id,
        "n_bars": len(test_idx),
        "n_trades": int(ntrades),
        "win_rate": float(win_rate(wins, losses)),
        "median_hold_bars": int(med_hold),
        "sharpe_nw": float(sharpe_newey_west(ret)),
        "sortino": float(sortino(ret)),
        "max_dd": float(max_drawdown(pnl)),
        "turnover": float(daily_turnover(signals.astype(np.float32))),
        "psr": float(psr(sharpe_ann=sharpe_newey_west(ret), T=max(5, ntrades))),
        "regime": "alpha_v1_ml",
        "model_type": "ridge_regression"
    }
    
    return metrics, pnl, signals


def main():
    parser = argparse.ArgumentParser(description="Walkforward testing with Alpha v1 ML model")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "TSLA"], 
                       help="Symbols to test")
    parser.add_argument("--model-path", default="artifacts/models/linear_v1.pkl",
                       help="Path to Alpha v1 model")
    parser.add_argument("--train-len", type=int, default=252,
                       help="Training window length")
    parser.add_argument("--test-len", type=int, default=63,
                       help="Test window length")
    parser.add_argument("--stride", type=int, default=21,
                       help="Stride between folds")
    parser.add_argument("--warmup", type=int, default=60,
                       help="Warmup period")
    parser.add_argument("--output", default="reports/alpha_v1_walkforward.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    logger.info("Starting Alpha v1 walkforward testing")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Model: {args.model_path}")
    
    # Check if model exists
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        logger.info("Please train the Alpha v1 model first:")
        logger.info(f"python tools/train_alpha_v1.py --symbols {','.join(args.symbols)}")
        sys.exit(1)
    
    # Load features and data
    X, y, prices = load_alpha_v1_features(args.symbols)
    
    # Create ML pipeline
    pipeline = create_ml_pipeline(args.model_path)
    
    # Generate folds
    folds = list(gen_walkforward(
        n=len(X),
        train_len=args.train_len,
        test_len=args.test_len,
        stride=args.stride,
        warmup=args.warmup
    ))
    
    logger.info(f"Generated {len(folds)} folds")
    
    # Run folds
    results = []
    all_pnl = []
    
    for fold in folds:
        logger.info(f"Running fold {fold.fold_id + 1}/{len(folds)}")
        
        try:
            metrics, pnl, signals = run_alpha_v1_fold(pipeline, X, y, prices, fold)
            results.append(metrics)
            all_pnl.append(pnl)
            
            logger.info(f"Fold {fold.fold_id + 1}: Sharpe={metrics['sharpe_nw']:.3f}, "
                       f"WinRate={metrics['win_rate']:.3f}, Trades={metrics['n_trades']}")
            
        except Exception as e:
            logger.error(f"Fold {fold.fold_id + 1} failed: {e}")
            continue
    
    if not results:
        logger.error("No folds completed successfully")
        sys.exit(1)
    
    # Calculate aggregate metrics
    avg_sharpe = np.mean([r['sharpe_nw'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results])
    avg_turnover = np.mean([r['turnover'] for r in results])
    total_trades = sum([r['n_trades'] for r in results])
    
    # Combine PnL series
    if all_pnl:
        combined_pnl = np.concatenate(all_pnl)
        combined_ret = np.diff(combined_pnl)
        overall_sharpe = sharpe_newey_west(combined_ret)
        overall_max_dd = max_drawdown(combined_pnl)
    else:
        overall_sharpe = 0.0
        overall_max_dd = 0.0
    
    # Create summary
    summary = {
        "model": "alpha_v1_ridge",
        "symbols": args.symbols,
        "n_folds": len(results),
        "train_len": args.train_len,
        "test_len": args.test_len,
        "stride": args.stride,
        "warmup": args.warmup,
        "fold_results": results,
        "aggregate_metrics": {
            "avg_sharpe": float(avg_sharpe),
            "avg_win_rate": float(avg_win_rate),
            "avg_turnover": float(avg_turnover),
            "total_trades": total_trades,
            "overall_sharpe": float(overall_sharpe),
            "overall_max_dd": float(overall_max_dd)
        }
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("Alpha v1 walkforward testing complete")
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Average Sharpe: {avg_sharpe:.3f}")
    logger.info(f"Average Win Rate: {avg_win_rate:.3f}")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Overall Sharpe: {overall_sharpe:.3f}")
    logger.info(f"Overall Max DD: {overall_max_dd:.3f}")


if __name__ == "__main__":
    main()
