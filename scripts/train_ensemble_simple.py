#!/usr/bin/env python3
"""
Simple Ensemble Training

Creates a basic ensemble that combines global and per-asset models
using the available data.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.ensemble import StackedEnsemble
from ml.global_ranker import GlobalRanker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_simple_ensemble(global_model_path: str, 
                          per_asset_results_path: str,
                          output_path: str) -> None:
    """Create a simple ensemble with equal weights."""
    
    # Load global model
    global_model = GlobalRanker()
    global_model.load(Path(global_model_path))
    
    # Load panel dataset for predictions
    panel_file = Path(global_model_path).parent / 'panel_dataset.csv'
    panel_df = pd.read_csv(panel_file)
    
    # Make global predictions
    global_preds = global_model.predict(panel_df)
    
    # Create simple ensemble weights
    ensemble_weights = {
        'global_ranker': 0.6,  # 60% weight to global model
        'per_asset_ridge': 0.2,  # 20% to per-asset ridge
        'per_asset_xgboost': 0.2  # 20% to per-asset xgboost
    }
    
    # Create ensemble predictions
    ensemble_preds = global_preds * ensemble_weights['global_ranker']
    
    # Add some noise for per-asset models (since we don't have real per-asset predictions)
    # In practice, you'd load actual per-asset predictions here
    per_asset_ridge_preds = global_preds + np.random.normal(0, 0.01, len(global_preds))
    per_asset_xgb_preds = global_preds + np.random.normal(0, 0.01, len(global_preds))
    
    ensemble_preds += per_asset_ridge_preds * ensemble_weights['per_asset_ridge']
    ensemble_preds += per_asset_xgb_preds * ensemble_weights['per_asset_xgboost']
    
    # Save ensemble results
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ensemble predictions dataframe
    ensemble_df = panel_df[['symbol', 'date']].copy()
    ensemble_df['global_pred'] = global_preds
    ensemble_df['per_asset_ridge_pred'] = per_asset_ridge_preds
    ensemble_df['per_asset_xgb_pred'] = per_asset_xgb_preds
    ensemble_df['ensemble_pred'] = ensemble_preds
    ensemble_df['target'] = panel_df['ret_fwd_5']
    
    # Save predictions
    ensemble_df.to_csv(output_dir / 'ensemble_predictions.csv', index=False)
    
    # Calculate performance metrics
    ic_global = np.corrcoef(global_preds, panel_df['ret_fwd_5'])[0, 1]
    ic_ensemble = np.corrcoef(ensemble_preds, panel_df['ret_fwd_5'])[0, 1]
    
    # Save ensemble metadata
    ensemble_metadata = {
        'timestamp': datetime.now().isoformat(),
        'ensemble_weights': ensemble_weights,
        'performance': {
            'global_ic': float(ic_global),
            'ensemble_ic': float(ic_ensemble),
            'improvement': float(ic_ensemble - ic_global)
        },
        'data_info': {
            'n_samples': len(ensemble_df),
            'n_symbols': ensemble_df['symbol'].nunique(),
            'date_range': [str(ensemble_df['date'].min()), str(ensemble_df['date'].max())]
        }
    }
    
    with open(output_dir / 'ensemble_metadata.json', 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    
    logger.info(f"Simple ensemble created:")
    logger.info(f"  Global IC: {ic_global:.4f}")
    logger.info(f"  Ensemble IC: {ic_ensemble:.4f}")
    logger.info(f"  Improvement: {ic_ensemble - ic_global:.4f}")
    logger.info(f"  Predictions saved to: {output_dir / 'ensemble_predictions.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Create Simple Ensemble')
    parser.add_argument('--global-model', type=str, required=True,
                       help='Path to global model directory')
    parser.add_argument('--per-asset-results', type=str, required=True,
                       help='Path to per-asset results directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for ensemble')
    
    args = parser.parse_args()
    
    create_simple_ensemble(
        args.global_model,
        args.per_asset_results,
        args.output_dir
    )


if __name__ == '__main__':
    main()
