"""
Ensemble Blender - Meta-Learning with Purged OOF

Implements non-negative least squares (NNLS) meta-learning for optimal
ensemble weights using out-of-fold predictions with purged cross-validation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def fit_meta_weights(oof_preds: Dict[str, np.ndarray], 
                    y: np.ndarray,
                    method: str = 'nnls') -> Dict[str, float]:
    """
    Fit optimal ensemble weights using out-of-fold predictions.
    
    Args:
        oof_preds: Dict of model_name -> OOF predictions aligned to y
        y: OOF target/edge aligned to oof_preds
        method: 'nnls' for non-negative least squares, 'ridge' for ridge regression
    
    Returns:
        Dict of model_name -> weight (non-negative, sum to 1)
    """
    if len(oof_preds) == 0:
        raise ValueError("No OOF predictions provided")
    
    # Ensure all predictions have same length
    lengths = [len(preds) for preds in oof_preds.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent prediction lengths: {lengths}")
    
    if len(y) != lengths[0]:
        raise ValueError(f"Target length {len(y)} doesn't match predictions {lengths[0]}")
    
    # Stack predictions into matrix [n_samples, n_models]
    model_names = list(oof_preds.keys())
    M = np.column_stack([oof_preds[name] for name in model_names])
    
    if method == 'nnls':
        # Non-negative least squares
        raw_weights, _ = nnls(M, y)
        
        # Normalize to simplex (sum to 1)
        total_weight = raw_weights.sum()
        if total_weight > 1e-12:
            weights = raw_weights / total_weight
        else:
            # Fallback to equal weights if NNLS fails
            weights = np.ones(len(model_names)) / len(model_names)
            logger.warning("NNLS failed, using equal weights")
    
    elif method == 'ridge':
        # Ridge regression with non-negativity constraint
        from sklearn.linear_model import Ridge
        from scipy.optimize import minimize
        
        # Ridge baseline
        ridge = Ridge(alpha=1.0)
        ridge.fit(M, y)
        baseline_weights = ridge.coef_
        
        # Project to non-negative simplex
        def objective(w):
            pred = M @ w
            mse = np.mean((pred - y) ** 2)
            return mse
        
        def constraint(w):
            return w.sum() - 1.0  # sum to 1
        
        # Initial guess
        x0 = np.maximum(baseline_weights, 0)
        if x0.sum() > 0:
            x0 = x0 / x0.sum()
        else:
            x0 = np.ones(len(model_names)) / len(model_names)
        
        # Optimize with constraints
        bounds = [(0, 1) for _ in model_names]
        constraints = {'type': 'eq', 'fun': constraint}
        
        result = minimize(objective, x0, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
        else:
            weights = x0
            logger.warning("Ridge optimization failed, using initial guess")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create result dictionary
    weight_dict = dict(zip(model_names, weights))
    
    # Log results
    logger.info(f"Meta-weights fitted ({method}):")
    for name, weight in weight_dict.items():
        logger.info(f"  {name}: {weight:.4f}")
    
    return weight_dict


def blend_predictions(preds: Dict[str, np.ndarray], 
                     weights: Dict[str, float]) -> np.ndarray:
    """
    Blend predictions using learned weights.
    
    Args:
        preds: Dict of model_name -> predictions
        weights: Dict of model_name -> weight
    
    Returns:
        Blended predictions
    """
    if not preds:
        raise ValueError("No predictions provided")
    
    # Ensure all predictions have same length
    lengths = [len(pred) for pred in preds.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent prediction lengths: {lengths}")
    
    # Weighted average
    blended = np.zeros(lengths[0])
    total_weight = 0.0
    
    for name, pred in preds.items():
        weight = weights.get(name, 0.0)
        blended += weight * pred
        total_weight += weight
    
    # Normalize by total weight (should be ~1.0)
    if total_weight > 1e-12:
        blended = blended / total_weight
    
    return blended


def evaluate_ensemble_performance(oof_preds: Dict[str, np.ndarray],
                                 weights: Dict[str, float],
                                 y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate ensemble performance using OOF predictions.
    
    Args:
        oof_preds: Dict of model_name -> OOF predictions
        weights: Dict of model_name -> weight
        y: True targets
    
    Returns:
        Dict of performance metrics
    """
    # Individual model performance
    individual_metrics = {}
    for name, pred in oof_preds.items():
        ic = np.corrcoef(pred, y)[0, 1]
        mse = np.mean((pred - y) ** 2)
        individual_metrics[f"{name}_ic"] = float(ic)
        individual_metrics[f"{name}_mse"] = float(mse)
    
    # Ensemble performance
    ensemble_pred = blend_predictions(oof_preds, weights)
    ensemble_ic = np.corrcoef(ensemble_pred, y)[0, 1]
    ensemble_mse = np.mean((ensemble_pred - y) ** 2)
    
    # Improvement metrics
    best_individual_ic = max([ic for name, ic in individual_metrics.items() 
                             if name.endswith('_ic')])
    improvement = ensemble_ic - best_individual_ic
    
    metrics = {
        **individual_metrics,
        'ensemble_ic': float(ensemble_ic),
        'ensemble_mse': float(ensemble_mse),
        'best_individual_ic': float(best_individual_ic),
        'improvement': float(improvement),
        'weight_entropy': float(-sum(w * np.log(w + 1e-12) for w in weights.values())),
        'max_weight': float(max(weights.values())),
        'min_weight': float(min(weights.values()))
    }
    
    return metrics


def save_meta_weights(weights: Dict[str, float], 
                     metrics: Dict[str, float],
                     output_path: Path) -> None:
    """Save meta-weights and performance metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    with open(output_path / 'meta_weights.json', 'w') as f:
        json.dump(weights, f, indent=2)
    
    # Save metrics
    with open(output_path / 'meta_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Meta-weights saved to {output_path}")


def load_meta_weights(weights_path: Path) -> Dict[str, float]:
    """Load meta-weights from file."""
    with open(weights_path / 'meta_weights.json', 'r') as f:
        weights = json.load(f)
    
    logger.info(f"Meta-weights loaded from {weights_path}")
    return weights


class EnsembleBlender:
    """Main ensemble blender class."""
    
    def __init__(self, method: str = 'nnls'):
        self.method = method
        self.weights = None
        self.metrics = None
        self.is_fitted = False
    
    def fit(self, oof_preds: Dict[str, np.ndarray], y: np.ndarray) -> 'EnsembleBlender':
        """Fit ensemble weights using OOF predictions."""
        self.weights = fit_meta_weights(oof_preds, y, self.method)
        self.metrics = evaluate_ensemble_performance(oof_preds, self.weights, y)
        self.is_fitted = True
        
        logger.info(f"Ensemble fitted with {self.method}")
        logger.info(f"Ensemble IC: {self.metrics['ensemble_ic']:.4f}")
        logger.info(f"Improvement: {self.metrics['improvement']:.4f}")
        
        return self
    
    def predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        return blend_predictions(preds, self.weights)
    
    def save(self, output_path: Path) -> None:
        """Save fitted ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        save_meta_weights(self.weights, self.metrics, output_path)
    
    def load(self, input_path: Path) -> 'EnsembleBlender':
        """Load fitted ensemble."""
        self.weights = load_meta_weights(input_path)
        self.is_fitted = True
        return self
