#!/usr/bin/env python3
"""
Train Stacked Ensemble

Trains meta-models to optimally combine global and per-asset predictions.
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

from ml.ensemble import StackedEnsemble, EnsembleManager
from ml.global_ranker import GlobalRanker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_per_asset_predictions(results_dir: Path, horizon: int) -> Dict[str, np.ndarray]:
    """Load per-asset model predictions."""
    predictions = {}
    
    # Load from universe results
    for asset_dir in results_dir.iterdir():
        if not asset_dir.is_dir():
            continue
        
        asset_name = asset_dir.name
        grid_file = asset_dir / "grid_results.csv"
        
        if not grid_file.exists():
            continue
        
        try:
            df = pd.read_csv(grid_file)
            
            # Get best model predictions (you'll need to implement this based on your data structure)
            # For now, use dummy predictions
            if 'ridge' not in predictions:
                predictions['ridge'] = []
            if 'xgboost' not in predictions:
                predictions['xgboost'] = []
            
            # Add dummy predictions (replace with actual prediction loading)
            ridge_preds = np.random.randn(len(df)) * 0.1
            xgb_preds = np.random.randn(len(df)) * 0.1
            
            predictions['ridge'].extend(ridge_preds)
            predictions['xgboost'].extend(xgb_preds)
            
        except Exception as e:
            logger.error(f"Error loading predictions for {asset_name}: {e}")
    
    # Convert to numpy arrays
    for model_name in predictions:
        predictions[model_name] = np.array(predictions[model_name])
    
    return predictions


def load_global_predictions(global_models_dir: Path, horizon: int) -> np.ndarray:
    """Load global model predictions."""
    model_path = global_models_dir / f"horizon_{horizon}"
    
    if not model_path.exists():
        logger.warning(f"Global model not found for horizon {horizon}")
        return None
    
    try:
        # Load global model
        global_model = GlobalRanker()
        global_model.load(model_path)
        
        # Load panel dataset for predictions
        panel_file = global_models_dir / 'panel_dataset.csv'
        if not panel_file.exists():
            logger.warning("Panel dataset not found")
            return None
        
        panel_df = pd.read_csv(panel_file)
        
        # Make predictions
        global_preds = global_model.predict(panel_df)
        
        return global_preds
        
    except Exception as e:
        logger.error(f"Error loading global predictions for horizon {horizon}: {e}")
        return None


def create_targets(panel_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Create target values for ensemble training."""
    target_col = f'ret_fwd_{horizon}'
    
    if target_col not in panel_df.columns:
        logger.warning(f"Target column {target_col} not found, creating dummy targets")
        return np.random.randn(len(panel_df)) * 0.01
    
    return panel_df[target_col].values


def main():
    parser = argparse.ArgumentParser(description='Train Stacked Ensemble')
    parser.add_argument('--per-asset-results', type=str, required=True,
                       help='Path to per-asset results directory')
    parser.add_argument('--global-models', type=str, required=True,
                       help='Path to global models directory')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to ensemble config file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for ensemble models')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 5, 10],
                       help='Prediction horizons to train')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load panel dataset for targets
    panel_file = Path(args.global_models) / 'panel_dataset.csv'
    if not panel_file.exists():
        logger.error("Panel dataset not found")
        return
    
    panel_df = pd.read_csv(panel_file)
    logger.info(f"Loaded panel dataset: {len(panel_df)} rows")
    
    # Train ensemble for each horizon
    results = {}
    
    for horizon in args.horizons:
        logger.info(f"\n=== Training Ensemble for Horizon {horizon} ===")
        
        # Load per-asset predictions
        per_asset_preds = load_per_asset_predictions(Path(args.per_asset_results), horizon)
        logger.info(f"Loaded per-asset predictions: {list(per_asset_preds.keys())}")
        
        # Load global predictions
        global_preds = load_global_predictions(Path(args.global_models), horizon)
        if global_preds is not None:
            logger.info(f"Loaded global predictions: {len(global_preds)} values")
        
        # Create targets
        targets = create_targets(panel_df, horizon)
        logger.info(f"Created targets: {len(targets)} values")
        
        # Combine all predictions
        all_predictions = {}
        
        # Add per-asset predictions
        for model_name, preds in per_asset_preds.items():
            all_predictions[f"per_asset_{model_name}"] = preds
        
        # Add global predictions
        if global_preds is not None:
            all_predictions["global_ranker"] = global_preds
        
        logger.info(f"Combined predictions: {list(all_predictions.keys())}")
        
        # Initialize and train ensemble
        ensemble = StackedEnsemble(
            meta_model_type=config.get('meta_model_type', 'ridge'),
            meta_alpha=config.get('meta_alpha', 1.0),
            non_negative_weights=config.get('non_negative_weights', True),
            turnover_penalty=config.get('turnover_penalty', 0.001)
        )
        
        # Create dates for time series CV
        dates = pd.to_datetime(panel_df['date']).values if 'date' in panel_df.columns else None
        
        # Train ensemble
        training_results = ensemble.fit(all_predictions, targets, dates)
        
        # Save ensemble model
        ensemble_dir = output_dir / f'ensemble_{horizon}'
        ensemble.save(ensemble_dir)
        
        # Store results
        results[f'horizon_{horizon}'] = {
            'training_results': training_results,
            'model_path': str(ensemble_dir),
            'num_base_models': len(all_predictions)
        }
        
        logger.info(f"Horizon {horizon} ensemble training completed")
        logger.info(f"Model weights: {training_results['model_weights']}")
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'horizons_trained': args.horizons,
        'panel_dataset_size': len(panel_df),
        'results': results
    }
    
    with open(output_dir / 'ensemble_training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\n=== Ensemble Training Complete ===")
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"Training summary: {output_dir / 'ensemble_training_summary.json'}")
    
    # Print summary
    for horizon in args.horizons:
        result = results[f'horizon_{horizon}']
        tr = result['training_results']
        logger.info(f"Horizon {horizon}: Meta-score={tr['meta_model_score']:.4f}, "
                   f"Base models={result['num_base_models']}")


if __name__ == '__main__':
    main()
