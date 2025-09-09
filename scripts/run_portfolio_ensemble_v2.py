#!/usr/bin/env python3
"""
Run Portfolio with Ensemble v2

Applies the complete Ensemble v2 system to generate portfolio signals:
1. Load ensemble components
2. Make predictions
3. Apply regime gating
4. Quantify uncertainty
5. Neutralize factors
6. Optimize portfolio
7. Export signals
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


def load_ensemble_components(ensemble_dir: Path) -> Dict[str, any]:
    """Load all ensemble components."""
    components = {}
    
    # Load meta-weights
    if (ensemble_dir / 'meta_weights').exists():
        components['blender'] = EnsembleBlender().load(ensemble_dir / 'meta_weights')
        logger.info("Loaded meta-weights")
    
    # Load residual MLP
    if (ensemble_dir / 'residual_mlp').exists():
        components['residual_mlp'] = ResidualMLP().load(ensemble_dir / 'residual_mlp')
        logger.info("Loaded residual MLP")
    
    # Load regime gate
    if (ensemble_dir / 'regime_gate').exists():
        components['regime_gate'] = RegimeGate().load(ensemble_dir / 'regime_gate')
        logger.info("Loaded regime gate")
    
    # Load conformal predictor
    if (ensemble_dir / 'conformal').exists():
        components['conformal_predictor'] = ConformalPredictor().load(ensemble_dir / 'conformal')
        logger.info("Loaded conformal predictor")
    
    # Load sizer
    if (ensemble_dir / 'sizer').exists():
        components['sizer'] = UncertaintyAwareSizer().load(ensemble_dir / 'sizer')
        logger.info("Loaded uncertainty-aware sizer")
    
    # Load neutralizer metadata
    if (ensemble_dir / 'neutralizer_metadata.json').exists():
        with open(ensemble_dir / 'neutralizer_metadata.json', 'r') as f:
            neutralizer_metadata = json.load(f)
        
        components['neutralizer'] = FactorNeutralizer(
            neutralize_market=neutralizer_metadata['neutralize_market'],
            neutralize_sectors=neutralizer_metadata['neutralize_sectors'],
            neutralize_size=neutralizer_metadata['neutralize_size'],
            neutralize_momentum=neutralizer_metadata['neutralize_momentum']
        )
        components['neutralizer'].fit(
            neutralizer_metadata['symbols'], 
            neutralizer_metadata['sector_map']
        )
        logger.info("Loaded factor neutralizer")
    
    # Load optimizer metadata
    if (ensemble_dir / 'optimizer_metadata.json').exists():
        with open(ensemble_dir / 'optimizer_metadata.json', 'r') as f:
            optimizer_metadata = json.load(f)
        
        components['optimizer'] = PortfolioOptimizer(
            lambda_risk=optimizer_metadata['lambda_risk'],
            lambda_turnover=optimizer_metadata['lambda_turnover'],
            weight_cap=optimizer_metadata['weight_cap'],
            long_only=optimizer_metadata['long_only'],
            cov_method=optimizer_metadata['cov_method'],
            cov_lookback=optimizer_metadata['cov_lookback']
        )
        components['optimizer'].initialize(neutralizer_metadata['symbols'])
        logger.info("Loaded portfolio optimizer")
    
    return components


def make_ensemble_predictions(components: Dict[str, any],
                            panel_df: pd.DataFrame,
                            models: List[str]) -> Dict[str, np.ndarray]:
    """Make predictions using ensemble components."""
    predictions = {}
    
    # Get base predictions (simplified - in practice you'd load from model files)
    # For demo, we'll use the panel data to create mock predictions
    n_samples = len(panel_df)
    
    for model in models:
        if model == 'global_ranker':
            # Use some features as mock predictions
            pred = panel_df['ret1'].values + np.random.normal(0, 0.01, n_samples)
        elif model == 'ridge':
            pred = panel_df['ret5'].values + np.random.normal(0, 0.01, n_samples)
        elif model == 'xgboost':
            pred = panel_df['ret10'].values + np.random.normal(0, 0.01, n_samples)
        else:
            pred = np.random.normal(0, 0.01, n_samples)
        
        predictions[model] = pred
    
    # Apply meta-weights if available
    if 'blender' in components:
        ensemble_pred = components['blender'].predict(predictions)
        predictions['ensemble'] = ensemble_pred
        logger.info("Applied meta-weights")
    
    # Apply residual MLP if available
    if 'residual_mlp' in components and 'global_ranker' in predictions:
        # Prepare features
        feature_cols = ['ret1', 'ret5', 'ret10', 'vol_5', 'vol_20', 'rsi_14', 'bb_position']
        available_features = [col for col in feature_cols if col in panel_df.columns]
        
        if available_features:
            X_features = panel_df[available_features].values
            base_pred = predictions['global_ranker']
            
            # Apply residual MLP
            enhanced_pred = components['residual_mlp'].predict(X_features, base_pred)
            predictions['global_ranker_enhanced'] = enhanced_pred
            logger.info("Applied residual MLP")
    
    return predictions


def apply_regime_gating(components: Dict[str, any],
                       predictions: Dict[str, np.ndarray],
                       panel_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Apply regime-aware gating to predictions."""
    if 'regime_gate' not in components:
        logger.warning("Regime gate not available, skipping gating")
        return predictions
    
    # Compute regime features
    regime_features = compute_regime_features(panel_df)
    if regime_features.empty:
        logger.warning("No regime features available, skipping gating")
        return predictions
    
    X_regime = regime_features.values
    
    # Apply regime gating
    gated_predictions = {}
    for name, pred in predictions.items():
        gated_pred = components['regime_gate'].predict({name: pred}, X_regime)
        gated_predictions[f"{name}_gated"] = gated_pred
    
    logger.info("Applied regime gating")
    return gated_predictions


def apply_uncertainty_sizing(components: Dict[str, any],
                           predictions: Dict[str, np.ndarray],
                           panel_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Apply uncertainty-aware position sizing."""
    if 'sizer' not in components:
        logger.warning("Uncertainty sizer not available, skipping sizing")
        return predictions
    
    # Get volatility estimates
    vol_estimates = panel_df['vol_20'].values if 'vol_20' in panel_df.columns else np.ones(len(panel_df)) * 0.02
    
    sized_predictions = {}
    for name, pred in predictions.items():
        sized_pred = components['sizer'].size_positions(pred, vol_estimates)
        sized_predictions[f"{name}_sized"] = sized_pred
    
    logger.info("Applied uncertainty-aware sizing")
    return sized_predictions


def apply_factor_neutralization(components: Dict[str, any],
                              predictions: Dict[str, np.ndarray],
                              panel_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Apply factor neutralization to predictions."""
    if 'neutralizer' not in components:
        logger.warning("Factor neutralizer not available, skipping neutralization")
        return predictions
    
    # Create exposures
    returns = panel_df['ret1'].values if 'ret1' in panel_df.columns else np.zeros(len(panel_df))
    market_returns = panel_df['market_ret1'].values if 'market_ret1' in panel_df.columns else np.zeros(len(panel_df))
    
    exposures = components['neutralizer'].create_exposures(returns, market_returns)
    
    # Apply neutralization
    neutralized_predictions = {}
    for name, pred in predictions.items():
        neutralized_pred = components['neutralizer'].neutralize(pred, exposures)
        neutralized_predictions[f"{name}_neutralized"] = neutralized_pred
    
    logger.info("Applied factor neutralization")
    return neutralized_predictions


def optimize_portfolio(components: Dict[str, any],
                      predictions: Dict[str, np.ndarray],
                      panel_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Optimize portfolio weights."""
    if 'optimizer' not in components:
        logger.warning("Portfolio optimizer not available, skipping optimization")
        return predictions
    
    # Use ensemble predictions as expected returns
    if 'ensemble' in predictions:
        expected_returns = predictions['ensemble']
    else:
        # Use average of available predictions
        pred_values = list(predictions.values())
        expected_returns = np.mean(pred_values, axis=0)
    
    # Create mock returns history for covariance estimation
    returns_history = np.random.normal(0, 0.02, (60, len(expected_returns)))
    
    # Optimize weights
    optimal_weights = components['optimizer'].optimize(expected_returns, returns_history)
    
    # Convert to dictionary
    weight_dict = components['optimizer'].get_weights_dict(optimal_weights)
    
    logger.info("Applied portfolio optimization")
    return {'optimal_weights': weight_dict}


def run_portfolio_ensemble_v2(ensemble_dir: Path,
                            panel_file: Path,
                            output_dir: Path,
                            models: List[str]) -> Dict[str, any]:
    """Run complete Ensemble v2 portfolio system."""
    logger.info("=== Running Portfolio Ensemble v2 ===")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ensemble components
    components = load_ensemble_components(ensemble_dir)
    
    # Load panel data
    panel_df = pd.read_csv(panel_file)
    logger.info(f"Loaded panel data: {len(panel_df)} rows")
    
    # 1. Make ensemble predictions
    logger.info("1. Making ensemble predictions...")
    predictions = make_ensemble_predictions(components, panel_df, models)
    
    # 2. Apply regime gating
    logger.info("2. Applying regime gating...")
    predictions = apply_regime_gating(components, predictions, panel_df)
    
    # 3. Apply uncertainty sizing
    logger.info("3. Applying uncertainty sizing...")
    predictions = apply_uncertainty_sizing(components, predictions, panel_df)
    
    # 4. Apply factor neutralization
    logger.info("4. Applying factor neutralization...")
    predictions = apply_factor_neutralization(components, predictions, panel_df)
    
    # 5. Optimize portfolio
    logger.info("5. Optimizing portfolio...")
    portfolio_results = optimize_portfolio(components, predictions, panel_df)
    
    # 6. Create final signals
    logger.info("6. Creating final signals...")
    
    # Combine all results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'predictions': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in predictions.items()},
        'portfolio_weights': portfolio_results.get('optimal_weights', {}),
        'n_samples': len(panel_df),
        'n_models': len(models)
    }
    
    # Save results
    with open(output_dir / 'portfolio_signals.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Create CSV signals for easy consumption
    signals_df = pd.DataFrame({
        'symbol': panel_df['symbol'],
        'date': panel_df['date'],
        'ensemble_pred': predictions.get('ensemble', np.zeros(len(panel_df))),
        'optimal_weight': [portfolio_results.get('optimal_weights', {}).get(sym, 0.0) 
                          for sym in panel_df['symbol']]
    })
    
    signals_df.to_csv(output_dir / 'portfolio_signals.csv', index=False)
    
    logger.info("=== Portfolio Ensemble v2 Complete ===")
    logger.info(f"Signals saved to {output_dir}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Run Portfolio Ensemble v2')
    parser.add_argument('--ensemble-dir', type=str, required=True,
                       help='Directory with ensemble components')
    parser.add_argument('--panel-file', type=str, required=True,
                       help='Panel dataset file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for signals')
    parser.add_argument('--models', nargs='+', default=['global_ranker', 'ridge', 'xgboost'],
                       help='Models to use')
    
    args = parser.parse_args()
    
    # Run portfolio system
    results = run_portfolio_ensemble_v2(
        Path(args.ensemble_dir),
        Path(args.panel_file),
        Path(args.output_dir),
        args.models
    )
    
    logger.info("Portfolio signals generated successfully")


if __name__ == '__main__':
    main()
