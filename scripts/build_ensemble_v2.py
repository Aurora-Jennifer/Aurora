#!/usr/bin/env python3
"""
Build Ensemble v2 - Advanced Meta-Learning System

Trains the complete Ensemble v2 system with:
- Purged OOF meta-weights (NNLS)
- Residual MLP for what trees miss
- Regime-aware gating
- Uncertainty quantification
- Factor neutralization
- Turnover-penalized portfolio optimization
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

from ml.ensemble.blender import EnsembleBlender
from ml.models.residual_mlp import ResidualMLP
from ml.ensemble.gate import RegimeGate, compute_regime_features
from ml.uncertainty.conformal import ConformalPredictor, UncertaintyAwareSizer
from ml.utils.neutralize import FactorNeutralizer
from ml.portfolio.optimizer import PortfolioOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_oof_predictions(results_dir: Path, models: List[str]) -> Dict[str, np.ndarray]:
    """Load out-of-fold predictions from model results."""
    oof_preds = {}
    
    for model in models:
        oof_file = results_dir / f"oof_{model}.npy"
        if oof_file.exists():
            oof_preds[model] = np.load(oof_file)
            logger.info(f"Loaded OOF predictions for {model}: {len(oof_preds[model])} samples")
        else:
            logger.warning(f"OOF predictions not found for {model}")
    
    return oof_preds


def load_oos_predictions(results_dir: Path, models: List[str]) -> Dict[str, np.ndarray]:
    """Load out-of-sample predictions from model results."""
    oos_preds = {}
    
    for model in models:
        oos_file = results_dir / f"oos_{model}.npy"
        if oos_file.exists():
            oos_preds[model] = np.load(oos_file)
            logger.info(f"Loaded OOS predictions for {model}: {len(oos_preds[model])} samples")
        else:
            logger.warning(f"OOS predictions not found for {model}")
    
    return oos_preds


def load_panel_data(panel_file: Path) -> pd.DataFrame:
    """Load panel dataset."""
    if not panel_file.exists():
        raise FileNotFoundError(f"Panel dataset not found: {panel_file}")
    
    df = pd.read_csv(panel_file)
    logger.info(f"Loaded panel dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def create_ensemble_v2(config: Dict, 
                      results_dir: Path,
                      output_dir: Path) -> Dict[str, any]:
    """
    Create Ensemble v2 system.
    
    Args:
        config: Configuration dictionary
        results_dir: Directory with model results
        output_dir: Output directory for ensemble
    
    Returns:
        Dict of ensemble components
    """
    logger.info("=== Building Ensemble v2 ===")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    models = config.get('models', ['global_ranker', 'ridge', 'xgboost'])
    panel_file = results_dir / config.get('panel_file', 'panel_dataset.csv')
    
    # Load data
    panel_df = load_panel_data(panel_file)
    
    # Load OOF predictions
    oof_preds = load_oof_predictions(results_dir, models)
    if not oof_preds:
        raise ValueError("No OOF predictions found")
    
    # Load OOS predictions
    oos_preds = load_oos_predictions(results_dir, models)
    if not oos_preds:
        raise ValueError("No OOS predictions found")
    
    # Get targets (assuming ret_fwd_5 is available)
    target_col = config.get('target_col', 'ret_fwd_5')
    if target_col not in panel_df.columns:
        raise ValueError(f"Target column {target_col} not found in panel data")
    
    y_oof = panel_df[target_col].values
    y_oos = panel_df[target_col].values  # Same targets for OOS
    
    # Ensure predictions and targets are aligned
    min_length = min(len(y_oof), min(len(pred) for pred in oof_preds.values()))
    y_oof = y_oof[:min_length]
    y_oos = y_oos[:min_length]
    
    for model in oof_preds:
        oof_preds[model] = oof_preds[model][:min_length]
        oos_preds[model] = oos_preds[model][:min_length]
    
    logger.info(f"Aligned data: {min_length} samples")
    
    # 1. Meta-learning with NNLS
    logger.info("1. Training meta-weights with NNLS...")
    blender = EnsembleBlender(method='nnls')
    blender.fit(oof_preds, y_oof)
    
    # Save meta-weights
    blender.save(output_dir / 'meta_weights')
    
    # 2. Residual MLP
    logger.info("2. Training residual MLP...")
    residual_models = {}
    
    # Use global ranker as base for residual learning
    base_model_name = 'global_ranker'
    if base_model_name in oof_preds:
        # Prepare features (use a subset for simplicity)
        feature_cols = ['ret1', 'ret5', 'ret10', 'vol_5', 'vol_20', 'rsi_14', 'bb_position']
        available_features = [col for col in feature_cols if col in panel_df.columns]
        
        if available_features:
            X_features = panel_df[available_features].values[:min_length]
            
            # Train residual MLP
            residual_mlp = ResidualMLP(hidden_layers=(64, 32))
            residual_mlp.fit(X_features, y_oof, oof_preds[base_model_name])
            
            # Save residual model
            residual_mlp.save(output_dir / 'residual_mlp')
            residual_models[base_model_name] = residual_mlp
            
            logger.info(f"Residual MLP trained on {len(available_features)} features")
        else:
            logger.warning("No suitable features found for residual MLP")
    
    # 3. Regime-aware gating
    logger.info("3. Training regime-aware gating...")
    regime_gate = None
    
    # Create regime features
    regime_features = compute_regime_features(panel_df)
    if not regime_features.empty:
        X_regime = regime_features.values[:min_length]
        
        # Train regime gate
        regime_gate = RegimeGate(regime_threshold=0.7)
        regime_gate.fit(X_regime, oof_preds, y_oof)
        
        # Save regime gate
        regime_gate.save(output_dir / 'regime_gate')
        
        logger.info(f"Regime gate trained on {X_regime.shape[1]} regime features")
    else:
        logger.warning("No regime features available")
    
    # 4. Uncertainty quantification
    logger.info("4. Training uncertainty quantification...")
    
    # Get ensemble predictions for calibration
    ensemble_oof = blender.predict(oof_preds)
    
    # Train conformal predictor
    conformal_predictor = ConformalPredictor(alpha=0.1)
    conformal_predictor.fit(y_oof, ensemble_oof)
    
    # Save conformal predictor
    conformal_predictor.save(output_dir / 'conformal')
    
    # Train uncertainty-aware sizer
    sizer = UncertaintyAwareSizer(max_weight=0.02, min_weight=0.001)
    sizer.fit(y_oof, ensemble_oof, alpha=0.1)
    
    # Save sizer
    sizer.save(output_dir / 'sizer')
    
    # 5. Factor neutralization
    logger.info("5. Setting up factor neutralization...")
    
    # Create sector map (simplified)
    sector_map = {
        'AAPL': 'tech', 'NVDA': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 
        'META': 'tech', 'TSLA': 'tech',
        'JPM': 'financial', 'BAC': 'financial', 'WFC': 'financial', 'GS': 'financial',
        'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare',
        'AMZN': 'consumer', 'WMT': 'consumer', 'KO': 'consumer', 'PG': 'consumer',
        'XOM': 'energy', 'CVX': 'energy',
        'COIN': 'crypto'
    }
    
    # Initialize neutralizer
    symbols = panel_df['symbol'].unique().tolist()
    neutralizer = FactorNeutralizer(
        neutralize_market=True,
        neutralize_sectors=True,
        neutralize_size=False,
        neutralize_momentum=False
    )
    neutralizer.fit(symbols, sector_map)
    
    # Save neutralizer metadata
    neutralizer_metadata = {
        'symbols': symbols,
        'sector_map': sector_map,
        'neutralize_market': True,
        'neutralize_sectors': True,
        'neutralize_size': False,
        'neutralize_momentum': False
    }
    
    with open(output_dir / 'neutralizer_metadata.json', 'w') as f:
        json.dump(neutralizer_metadata, f, indent=2)
    
    # 6. Portfolio optimizer
    logger.info("6. Setting up portfolio optimizer...")
    
    optimizer = PortfolioOptimizer(
        lambda_risk=5.0,
        lambda_turnover=10.0,
        weight_cap=0.03,
        long_only=False
    )
    optimizer.initialize(symbols)
    
    # Save optimizer metadata
    optimizer_metadata = {
        'lambda_risk': 5.0,
        'lambda_turnover': 10.0,
        'weight_cap': 0.03,
        'long_only': False,
        'cov_method': 'ledoit_wolf',
        'cov_lookback': 60
    }
    
    with open(output_dir / 'optimizer_metadata.json', 'w') as f:
        json.dump(optimizer_metadata, f, indent=2)
    
    # 7. Evaluate ensemble performance
    logger.info("7. Evaluating ensemble performance...")
    
    # OOF performance
    ensemble_oof = blender.predict(oof_preds)
    oof_ic = np.corrcoef(ensemble_oof, y_oof)[0, 1]
    
    # OOS performance
    ensemble_oos = blender.predict(oos_preds)
    oos_ic = np.corrcoef(ensemble_oos, y_oos)[0, 1]
    
    # Individual model performance
    individual_ics = {}
    for model, pred in oof_preds.items():
        ic = np.corrcoef(pred, y_oof)[0, 1]
        individual_ics[model] = float(ic)
    
    # Create performance summary
    performance_summary = {
        'timestamp': datetime.now().isoformat(),
        'ensemble_oof_ic': float(oof_ic),
        'ensemble_oos_ic': float(oos_ic),
        'individual_ics': individual_ics,
        'improvement_vs_best': float(oof_ic - max(individual_ics.values())),
        'meta_weights': blender.weights,
        'meta_metrics': blender.metrics,
        'n_samples': min_length,
        'n_models': len(models)
    }
    
    # Save performance summary
    with open(output_dir / 'performance_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    logger.info("=== Ensemble v2 Complete ===")
    logger.info(f"OOF IC: {oof_ic:.4f}")
    logger.info(f"OOS IC: {oos_ic:.4f}")
    logger.info(f"Improvement: {oof_ic - max(individual_ics.values()):.4f}")
    
    # Return ensemble components
    ensemble_components = {
        'blender': blender,
        'residual_models': residual_models,
        'regime_gate': regime_gate,
        'conformal_predictor': conformal_predictor,
        'sizer': sizer,
        'neutralizer': neutralizer,
        'optimizer': optimizer,
        'performance_summary': performance_summary
    }
    
    return ensemble_components


def main():
    parser = argparse.ArgumentParser(description='Build Ensemble v2')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with model results')
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for ensemble')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build ensemble
    ensemble_components = create_ensemble_v2(
        config, 
        Path(args.results_dir), 
        Path(args.output_dir)
    )
    
    logger.info(f"Ensemble v2 saved to {args.output_dir}")


if __name__ == '__main__':
    main()
