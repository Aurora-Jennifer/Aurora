"""
Conformal Prediction for Uncertainty Quantification

Implements conformal prediction for uncertainty-aware position sizing
and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from sklearn.model_selection import KFold
from sklearn.linear_model import QuantileRegressor

logger = logging.getLogger(__name__)


def conformal_scale(y_true: np.ndarray, 
                   y_pred: np.ndarray, 
                   alpha: float = 0.1) -> float:
    """
    Compute conformal prediction scale using absolute residuals.
    
    Args:
        y_true: True targets
        y_pred: Predictions
        alpha: Significance level (1 - confidence level)
    
    Returns:
        Half-width of prediction interval
    """
    residuals = np.abs(y_true - y_pred)
    quantile = np.quantile(residuals, 1 - alpha)
    
    logger.info(f"Conformal scale (α={alpha}): {quantile:.4f}")
    return float(quantile)


def conformal_intervals(y_pred: np.ndarray, 
                       scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conformal prediction intervals.
    
    Args:
        y_pred: Point predictions
        scale: Conformal scale (half-width)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    lower = y_pred - scale
    upper = y_pred + scale
    
    return lower, upper


def size_positions(edge: np.ndarray, 
                  conf_width: np.ndarray, 
                  vol: np.ndarray,
                  max_weight: float = 0.02,
                  min_weight: float = 0.001) -> np.ndarray:
    """
    Size positions based on expected edge and confidence.
    
    Args:
        edge: Expected edge (predictions)
        conf_width: Confidence width (uncertainty)
        vol: Volatility estimates
        max_weight: Maximum position weight
        min_weight: Minimum position weight (for non-zero edges)
    
    Returns:
        Position weights
    """
    # Confidence-adjusted edge
    confidence_score = edge / (conf_width + 1e-8)
    
    # Volatility adjustment
    vol_adjusted_score = confidence_score / (vol + 1e-8)
    
    # Tanh activation for bounded weights
    raw_weights = 0.5 * np.tanh(vol_adjusted_score)
    
    # Apply min/max constraints
    weights = np.clip(raw_weights, -max_weight, max_weight)
    
    # Set very small weights to zero
    weights[np.abs(weights) < min_weight] = 0.0
    
    return weights


def quantile_conformal_scale(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           alpha: float = 0.1,
                           method: str = 'residual') -> float:
    """
    Compute conformal scale using quantile regression.
    
    Args:
        y_true: True targets
        y_pred: Predictions
        alpha: Significance level
        method: 'residual' or 'quantile'
    
    Returns:
        Conformal scale
    """
    if method == 'residual':
        return conformal_scale(y_true, y_pred, alpha)
    
    elif method == 'quantile':
        # Use quantile regression for more sophisticated uncertainty
        residuals = y_true - y_pred
        
        # Fit quantile regression on residuals
        X_resid = np.abs(residuals).reshape(-1, 1)
        y_resid = np.abs(residuals)
        
        # Quantile regression for upper bound
        quantile_reg = QuantileRegressor(quantile=1-alpha, alpha=0.0)
        quantile_reg.fit(X_resid, y_resid)
        
        # Predict scale
        scale = quantile_reg.predict([[np.median(np.abs(residuals))]])[0]
        
        logger.info(f"Quantile conformal scale (α={alpha}): {scale:.4f}")
        return float(scale)
    
    else:
        raise ValueError(f"Unknown method: {method}")


class ConformalPredictor:
    """
    Conformal prediction for uncertainty quantification.
    """
    
    def __init__(self, alpha: float = 0.1, method: str = 'residual'):
        self.alpha = alpha
        self.method = method
        self.scale = None
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> 'ConformalPredictor':
        """Fit conformal predictor on calibration data."""
        self.scale = quantile_conformal_scale(y_true, y_pred, self.alpha, self.method)
        self.is_fitted = True
        
        logger.info(f"Conformal predictor fitted with scale: {self.scale:.4f}")
        return self
    
    def predict_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted yet")
        
        return conformal_intervals(y_pred, self.scale)
    
    def predict_uncertainty(self, y_pred: np.ndarray) -> np.ndarray:
        """Predict uncertainty (confidence width)."""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted yet")
        
        return np.full_like(y_pred, self.scale)
    
    def save(self, path: Path) -> None:
        """Save conformal predictor."""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted yet")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'alpha': self.alpha,
            'method': self.method,
            'scale': self.scale
        }
        
        with open(path / 'conformal_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Conformal predictor saved to {path}")
    
    def load(self, path: Path) -> 'ConformalPredictor':
        """Load conformal predictor."""
        with open(path / 'conformal_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.alpha = metadata['alpha']
        self.method = metadata['method']
        self.scale = metadata['scale']
        self.is_fitted = True
        
        logger.info(f"Conformal predictor loaded from {path}")
        return self


class UncertaintyAwareSizer:
    """
    Position sizer that uses uncertainty quantification.
    """
    
    def __init__(self, 
                 max_weight: float = 0.02,
                 min_weight: float = 0.001,
                 vol_lookback: int = 20):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.vol_lookback = vol_lookback
        self.conformal_predictor = None
        self.is_fitted = False
    
    def fit(self, 
            y_true: np.ndarray, 
            y_pred: np.ndarray,
            alpha: float = 0.1) -> 'UncertaintyAwareSizer':
        """Fit uncertainty-aware sizer."""
        self.conformal_predictor = ConformalPredictor(alpha=alpha)
        self.conformal_predictor.fit(y_true, y_pred)
        self.is_fitted = True
        
        return self
    
    def size_positions(self, 
                      edge: np.ndarray, 
                      vol: np.ndarray) -> np.ndarray:
        """
        Size positions based on edge and uncertainty.
        
        Args:
            edge: Expected edge
            vol: Volatility estimates
        
        Returns:
            Position weights
        """
        if not self.is_fitted:
            raise ValueError("Sizer not fitted yet")
        
        # Get uncertainty estimates
        conf_width = self.conformal_predictor.predict_uncertainty(edge)
        
        # Size positions
        weights = size_positions(edge, conf_width, vol, 
                               self.max_weight, self.min_weight)
        
        return weights
    
    def save(self, path: Path) -> None:
        """Save sizer."""
        if not self.is_fitted:
            raise ValueError("Sizer not fitted yet")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save conformal predictor
        self.conformal_predictor.save(path)
        
        # Save sizer metadata
        metadata = {
            'max_weight': self.max_weight,
            'min_weight': self.min_weight,
            'vol_lookback': self.vol_lookback
        }
        
        with open(path / 'sizer_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Uncertainty-aware sizer saved to {path}")
    
    def load(self, path: Path) -> 'UncertaintyAwareSizer':
        """Load sizer."""
        # Load conformal predictor
        self.conformal_predictor = ConformalPredictor()
        self.conformal_predictor.load(path)
        
        # Load sizer metadata
        with open(path / 'sizer_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.max_weight = metadata['max_weight']
        self.min_weight = metadata['min_weight']
        self.vol_lookback = metadata['vol_lookback']
        self.is_fitted = True
        
        logger.info(f"Uncertainty-aware sizer loaded from {path}")
        return self


def evaluate_uncertainty_quality(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                conf_width: np.ndarray,
                                alpha: float = 0.1) -> Dict[str, float]:
    """
    Evaluate quality of uncertainty estimates.
    
    Args:
        y_true: True targets
        y_pred: Predictions
        conf_width: Confidence width
        alpha: Significance level
    
    Returns:
        Dict of quality metrics
    """
    # Coverage
    lower, upper = conformal_intervals(y_pred, conf_width)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    # Average width
    avg_width = np.mean(conf_width)
    
    # Efficiency (inverse of average width)
    efficiency = 1.0 / (avg_width + 1e-8)
    
    # Calibration error
    calibration_error = abs(coverage - (1 - alpha))
    
    # Sharpness (standard deviation of widths)
    sharpness = np.std(conf_width)
    
    metrics = {
        'coverage': float(coverage),
        'expected_coverage': float(1 - alpha),
        'calibration_error': float(calibration_error),
        'avg_width': float(avg_width),
        'efficiency': float(efficiency),
        'sharpness': float(sharpness)
    }
    
    logger.info(f"Uncertainty quality metrics:")
    logger.info(f"  Coverage: {coverage:.3f} (expected: {1-alpha:.3f})")
    logger.info(f"  Calibration error: {calibration_error:.3f}")
    logger.info(f"  Average width: {avg_width:.4f}")
    
    return metrics
