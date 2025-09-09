#!/usr/bin/env python3
"""
Train Global Cross-Sectional Model

Trains a single XGBRanker model across all symbols simultaneously,
optimizing for cross-sectional ranking within each date.
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.global_ranker import GlobalRanker, build_panel_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_time_splits(df: pd.DataFrame, 
                      train_start: str = '2020-01-01',
                      train_end: str = '2022-12-31',
                      valid_start: str = '2023-01-01',
                      valid_end: str = '2023-12-31') -> tuple:
    """Create time-based train/validation splits."""
    df['date'] = pd.to_datetime(df['date'])
    
    train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
    valid_mask = (df['date'] >= valid_start) & (df['date'] <= valid_end)
    
    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()
    
    logger.info(f"Train period: {train_start} to {train_end} ({len(train_df)} rows)")
    logger.info(f"Valid period: {valid_start} to {valid_end} ({len(valid_df)} rows)")
    
    return train_df, valid_df


def main():
    parser = argparse.ArgumentParser(description='Train Global Cross-Sectional Model')
    parser.add_argument('--universe-config', type=str, required=True,
                       help='Path to universe configuration file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to global model config file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for trained models and results')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 5, 10],
                       help='Prediction horizons to train')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build panel dataset
    feature_cols = config.get('features', [])
    
    logger.info("Building panel dataset...")
    panel_df = build_panel_dataset(
        args.universe_config, 
        str(output_dir / 'panel_dataset.csv'),
        config.get('train_start', '2020-01-01'),
        config.get('valid_end', '2024-01-01'),
        args.horizons
    )
    
    # Save panel dataset
    panel_df.to_csv(output_dir / 'panel_dataset.csv', index=False)
    logger.info(f"Panel dataset saved to {output_dir / 'panel_dataset.csv'}")
    
    # Train models for each horizon
    results = {}
    
    for horizon in args.horizons:
        logger.info(f"\n=== Training Global Ranker for Horizon {horizon} ===")
        
        # Create time splits
        train_df, valid_df = create_time_splits(
            panel_df,
            train_start=config.get('train_start', '2020-01-01'),
            train_end=config.get('train_end', '2022-12-31'),
            valid_start=config.get('valid_start', '2023-01-01'),
            valid_end=config.get('valid_end', '2023-12-31')
        )
        
        # Initialize and train model
        ranker = GlobalRanker(
            horizon=horizon,
            **config.get('model_params', {})
        )
        
        target_col = f'ret_fwd_{horizon}'
        training_results = ranker.fit(train_df, valid_df, feature_cols, target_col)
        
        # Save model
        model_dir = output_dir / f'horizon_{horizon}'
        ranker.save(model_dir)
        
        # Store results
        results[f'horizon_{horizon}'] = {
            'training_results': training_results,
            'model_path': str(model_dir),
            'target_column': target_col
        }
        
        logger.info(f"Horizon {horizon} training completed")
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'horizons_trained': args.horizons,
        'panel_dataset_size': len(panel_df),
        'results': results
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\n=== Global Training Complete ===")
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"Training summary: {output_dir / 'training_summary.json'}")
    
    # Print summary
    for horizon in args.horizons:
        result = results[f'horizon_{horizon}']
        tr = result['training_results']
        logger.info(f"Horizon {horizon}: IC={tr['validation_ic']:.4f}, "
                   f"Time={tr['fit_time_seconds']:.1f}s, "
                   f"Features={tr['n_features']}")


if __name__ == '__main__':
    main()
