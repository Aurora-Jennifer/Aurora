#!/usr/bin/env python3
"""
Compare old regime-based walkforward with new Alpha v1 ML walkforward.
"""
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.walkforward_framework import main as run_regime_walkforward
from scripts.walkforward_alpha_v1 import main as run_alpha_v1_walkforward
from core.utils import setup_logging

logger = setup_logging("logs/compare_walkforward.log", logging.INFO)


def compare_results(regime_file: str, alpha_v1_file: str):
    """Compare results from both approaches."""
    
    import json
    
    # Load results
    with open(regime_file, 'r') as f:
        regime_results = json.load(f)
    
    with open(alpha_v1_file, 'r') as f:
        alpha_v1_results = json.load(f)
    
    print("\n" + "="*80)
    print("WALKFORWARD COMPARISON: Regime vs Alpha v1 ML")
    print("="*80)
    
    # Compare aggregate metrics
    regime_metrics = regime_results.get('aggregate_metrics', {})
    alpha_v1_metrics = alpha_v1_results.get('aggregate_metrics', {})
    
    print(f"\n📊 AGGREGATE METRICS COMPARISON:")
    print(f"{'Metric':<20} {'Regime':<15} {'Alpha v1 ML':<15} {'Difference':<15}")
    print("-" * 65)
    
    metrics_to_compare = [
        ('avg_sharpe', 'Avg Sharpe'),
        ('avg_win_rate', 'Avg Win Rate'),
        ('avg_turnover', 'Avg Turnover'),
        ('total_trades', 'Total Trades'),
        ('overall_sharpe', 'Overall Sharpe'),
        ('overall_max_dd', 'Overall Max DD')
    ]
    
    for metric, label in metrics_to_compare:
        regime_val = regime_metrics.get(metric, 0.0)
        alpha_v1_val = alpha_v1_metrics.get(metric, 0.0)
        diff = alpha_v1_val - regime_val
        
        print(f"{label:<20} {regime_val:<15.3f} {alpha_v1_val:<15.3f} {diff:<15.3f}")
    
    # Compare fold-by-fold
    print(f"\n📈 FOLD-BY-FOLD COMPARISON:")
    print(f"{'Fold':<8} {'Regime Sharpe':<15} {'Alpha v1 Sharpe':<15} {'Regime WR':<10} {'Alpha v1 WR':<10}")
    print("-" * 70)
    
    regime_folds = regime_results.get('fold_results', [])
    alpha_v1_folds = alpha_v1_results.get('fold_results', [])
    
    for i in range(min(len(regime_folds), len(alpha_v1_folds))):
        regime_fold = regime_folds[i]
        alpha_v1_fold = alpha_v1_folds[i]
        
        regime_sharpe = regime_fold.get('sharpe_nw', 0.0)
        alpha_v1_sharpe = alpha_v1_fold.get('sharpe_nw', 0.0)
        regime_wr = regime_fold.get('win_rate', 0.0)
        alpha_v1_wr = alpha_v1_fold.get('win_rate', 0.0)
        
        print(f"Fold {i+1:<3} {regime_sharpe:<15.3f} {alpha_v1_sharpe:<15.3f} {regime_wr:<10.3f} {alpha_v1_wr:<10.3f}")
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    print(f"Regime-based approach: {len(regime_folds)} folds")
    print(f"Alpha v1 ML approach: {len(alpha_v1_folds)} folds")
    
    # Model type comparison
    print(f"\n🤖 MODEL COMPARISON:")
    print(f"Regime-based: Simple ensemble strategy with regime detection")
    print(f"Alpha v1 ML: Ridge regression with 8 technical features")
    print(f"Alpha v1 ML: Leakage guards, cost-aware evaluation, promotion gates")


def main():
    parser = argparse.ArgumentParser(description="Compare regime vs Alpha v1 walkforward")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "TSLA"], 
                       help="Symbols to test")
    parser.add_argument("--train-len", type=int, default=100,
                       help="Training window length")
    parser.add_argument("--test-len", type=int, default=30,
                       help="Test window length")
    parser.add_argument("--stride", type=int, default=15,
                       help="Stride between folds")
    parser.add_argument("--warmup", type=int, default=20,
                       help="Warmup period")
    
    args = parser.parse_args()
    
    logger.info("Starting walkforward comparison")
    logger.info(f"Symbols: {args.symbols}")
    
    # Check if Alpha v1 model exists
    model_path = "artifacts/models/linear_v1.pkl"
    if not Path(model_path).exists():
        logger.error(f"Alpha v1 model not found: {model_path}")
        logger.info("Please train the Alpha v1 model first:")
        logger.info(f"python tools/train_alpha_v1.py --symbols {','.join(args.symbols)}")
        sys.exit(1)
    
    # Run regime-based walkforward
    logger.info("Running regime-based walkforward...")
    try:
        # Import and run regime walkforward
        from scripts.walkforward_framework import main as run_regime
        import sys
        old_argv = sys.argv
        
        # Set up arguments for regime walkforward
        sys.argv = [
            'walkforward_framework.py',
            '--symbol', args.symbols[0],  # Regime walkforward only supports single symbol
            '--train-len', str(args.train_len),
            '--test-len', str(args.test_len),
            '--stride', str(args.stride),
            '--warmup', str(args.warmup)
        ]
        
        run_regime()
        
        # Restore original argv
        sys.argv = old_argv
        
    except Exception as e:
        logger.error(f"Regime walkforward failed: {e}")
        logger.info("Skipping regime comparison")
        return
    
    # Run Alpha v1 walkforward
    logger.info("Running Alpha v1 walkforward...")
    try:
        # Import and run Alpha v1 walkforward
        from scripts.walkforward_alpha_v1 import main as run_alpha_v1
        import sys
        old_argv = sys.argv
        
        # Set up arguments for Alpha v1 walkforward
        sys.argv = [
            'walkforward_alpha_v1.py',
            '--symbols'] + args.symbols + [
            '--train-len', str(args.train_len),
            '--test-len', str(args.test_len),
            '--stride', str(args.stride),
            '--warmup', str(args.warmup)
        ]
        
        run_alpha_v1()
        
        # Restore original argv
        sys.argv = old_argv
        
    except Exception as e:
        logger.error(f"Alpha v1 walkforward failed: {e}")
        return
    
    # Compare results
    regime_file = "reports/walkforward_results.json"
    alpha_v1_file = "reports/alpha_v1_walkforward.json"
    
    if Path(regime_file).exists() and Path(alpha_v1_file).exists():
        compare_results(regime_file, alpha_v1_file)
    else:
        logger.error("Results files not found for comparison")


if __name__ == "__main__":
    main()
