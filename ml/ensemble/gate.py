"""
Regime-Aware Gating for Mixture of Experts

Implements regime detection and adaptive weighting for ensemble models
based on market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime features from market data.
    
    Args:
        df: DataFrame with market data (close, volume, etc.)
    
    Returns:
        DataFrame with regime features
    """
    regime_df = df.copy()
    
    # Volatility regime
    regime_df['vol_5'] = regime_df['close'].pct_change().rolling(5).std()
    regime_df['vol_20'] = regime_df['close'].pct_change().rolling(20).std()
    regime_df['vol_regime'] = (regime_df['vol_5'] > regime_df['vol_20']).astype(int)
    
    # Trend regime
    regime_df['ma_fast'] = regime_df['close'].rolling(10).mean()
    regime_df['ma_slow'] = regime_df['close'].rolling(20).mean()
    regime_df['trend_regime'] = (regime_df['ma_fast'] > regime_df['ma_slow']).astype(int)
    
    # Volume regime
    regime_df['volume_ma'] = regime_df['volume'].rolling(20).mean()
    regime_df['volume_regime'] = (regime_df['volume'] > regime_df['volume_ma']).astype(int)
    
    # Momentum regime
    regime_df['momentum_5'] = regime_df['close'].pct_change(5)
    regime_df['momentum_regime'] = (regime_df['momentum_5'] > 0).astype(int)
    
    # Range regime (choppiness)
    regime_df['high_20'] = regime_df['high'].rolling(20).max()
    regime_df['low_20'] = regime_df['low'].rolling(20).min()
    regime_df['range_20'] = (regime_df['high_20'] - regime_df['low_20']) / regime_df['close']
    regime_df['range_ma'] = regime_df['range_20'].rolling(20).mean()
    regime_df['chop_regime'] = (regime_df['range_20'] > regime_df['range_ma']).astype(int)
    
    # Select regime features
    regime_features = [
        'vol_regime', 'trend_regime', 'volume_regime', 
        'momentum_regime', 'chop_regime'
    ]
    
    return regime_df[regime_features].fillna(0)


def fit_regime_gate(X_regime: np.ndarray,
                   oof_members: Dict[str, np.ndarray],
                   y: np.ndarray,
                   regime_threshold: float = 0.7) -> Tuple[LogisticRegression, np.ndarray]:
    """
    Fit regime-aware gating for mixture of experts.
    
    Args:
        X_regime: Regime features [n_samples, n_regime_features]
        oof_members: Dict of model_name -> OOF predictions
        y: True targets
        regime_threshold: Threshold for good vs bad regimes
    
    Returns:
        Tuple of (fitted_gate, regime_weights)
    """
    # Create regime targets (good vs bad performance days)
    y_quantile = np.quantile(y, regime_threshold)
    regime_targets = (y > y_quantile).astype(int)
    
    logger.info(f"Regime targets: {np.sum(regime_targets)}/{len(regime_targets)} good days")
    
    # Fit logistic regression gate
    gate_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    gate_model.fit(X_regime, regime_targets)
    
    # Get regime probabilities
    regime_probs = gate_model.predict_proba(X_regime)[:, 1]
    
    # Map regime probabilities to expert weights
    # High regime prob -> favor global model
    # Low regime prob -> favor local models
    n_models = len(oof_members)
    model_names = list(oof_members.keys())
    
    # Simple linear mapping to weights
    if n_models == 3:  # global, local1, local2
        w_global = 0.4 + 0.4 * (regime_probs - 0.5)  # 0.2 to 0.6
        w_local1 = 0.3 - 0.2 * (regime_probs - 0.5)  # 0.4 to 0.2
        w_local2 = 0.3 - 0.2 * (regime_probs - 0.5)  # 0.4 to 0.2
        
        regime_weights = np.column_stack([w_global, w_local1, w_local2])
    
    elif n_models == 2:  # global, local
        w_global = 0.5 + 0.3 * (regime_probs - 0.5)  # 0.35 to 0.65
        w_local = 0.5 - 0.3 * (regime_probs - 0.5)   # 0.65 to 0.35
        
        regime_weights = np.column_stack([w_global, w_local])
    
    else:
        # Equal weights for other cases
        regime_weights = np.full((len(regime_probs), n_models), 1.0 / n_models)
    
    # Ensure weights are non-negative and sum to 1
    regime_weights = np.clip(regime_weights, 0, 1)
    regime_weights = regime_weights / (regime_weights.sum(axis=1, keepdims=True) + 1e-12)
    
    logger.info(f"Regime gate fitted:")
    logger.info(f"  Gate accuracy: {gate_model.score(X_regime, regime_targets):.3f}")
    logger.info(f"  Weight range: {regime_weights.min():.3f} to {regime_weights.max():.3f}")
    
    return gate_model, regime_weights


def apply_regime_gating(predictions: Dict[str, np.ndarray],
                       regime_weights: np.ndarray,
                       model_names: List[str]) -> np.ndarray:
    """
    Apply regime-aware gating to predictions.
    
    Args:
        predictions: Dict of model_name -> predictions
        regime_weights: Regime weights [n_samples, n_models]
        model_names: List of model names in order
    
    Returns:
        Gated predictions
    """
    n_samples = len(next(iter(predictions.values())))
    gated_preds = np.zeros(n_samples)
    
    for i, model_name in enumerate(model_names):
        if model_name in predictions:
            gated_preds += regime_weights[:, i] * predictions[model_name]
    
    return gated_preds


class RegimeGate:
    """
    Regime-aware gating for mixture of experts.
    """
    
    def __init__(self, regime_threshold: float = 0.7):
        self.regime_threshold = regime_threshold
        self.gate_model = None
        self.scaler = StandardScaler()
        self.model_names = None
        self.is_fitted = False
    
    def fit(self, 
            X_regime: np.ndarray,
            oof_members: Dict[str, np.ndarray],
            y: np.ndarray) -> 'RegimeGate':
        """Fit regime gate."""
        # Scale regime features
        X_regime_scaled = self.scaler.fit_transform(X_regime)
        
        # Fit gate
        self.gate_model, regime_weights = fit_regime_gate(
            X_regime_scaled, oof_members, y, self.regime_threshold
        )
        
        self.model_names = list(oof_members.keys())
        self.is_fitted = True
        
        logger.info(f"Regime gate fitted for {len(self.model_names)} models")
        return self
    
    def predict_weights(self, X_regime: np.ndarray) -> np.ndarray:
        """Predict regime weights for new data."""
        if not self.is_fitted:
            raise ValueError("Regime gate not fitted yet")
        
        # Scale regime features
        X_regime_scaled = self.scaler.transform(X_regime)
        
        # Get regime probabilities
        regime_probs = self.gate_model.predict_proba(X_regime_scaled)[:, 1]
        
        # Map to weights (same logic as in fit)
        n_models = len(self.model_names)
        
        if n_models == 3:
            w_global = 0.4 + 0.4 * (regime_probs - 0.5)
            w_local1 = 0.3 - 0.2 * (regime_probs - 0.5)
            w_local2 = 0.3 - 0.2 * (regime_probs - 0.5)
            regime_weights = np.column_stack([w_global, w_local1, w_local2])
        
        elif n_models == 2:
            w_global = 0.5 + 0.3 * (regime_probs - 0.5)
            w_local = 0.5 - 0.3 * (regime_probs - 0.5)
            regime_weights = np.column_stack([w_global, w_local])
        
        else:
            regime_weights = np.full((len(regime_probs), n_models), 1.0 / n_models)
        
        # Ensure weights are valid
        regime_weights = np.clip(regime_weights, 0, 1)
        regime_weights = regime_weights / (regime_weights.sum(axis=1, keepdims=True) + 1e-12)
        
        return regime_weights
    
    def predict(self, 
                predictions: Dict[str, np.ndarray],
                X_regime: np.ndarray) -> np.ndarray:
        """Make gated predictions."""
        if not self.is_fitted:
            raise ValueError("Regime gate not fitted yet")
        
        regime_weights = self.predict_weights(X_regime)
        return apply_regime_gating(predictions, regime_weights, self.model_names)
    
    def save(self, path: Path) -> None:
        """Save regime gate."""
        if not self.is_fitted:
            raise ValueError("Regime gate not fitted yet")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save gate model
        import joblib
        joblib.dump(self.gate_model, path / 'gate_model.joblib')
        joblib.dump(self.scaler, path / 'scaler.joblib')
        
        # Save metadata
        metadata = {
            'regime_threshold': self.regime_threshold,
            'model_names': self.model_names
        }
        
        with open(path / 'gate_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Regime gate saved to {path}")
    
    def load(self, path: Path) -> 'RegimeGate':
        """Load regime gate."""
        import joblib
        
        # Load gate model
        self.gate_model = joblib.load(path / 'gate_model.joblib')
        self.scaler = joblib.load(path / 'scaler.joblib')
        
        # Load metadata
        with open(path / 'gate_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.regime_threshold = metadata['regime_threshold']
        self.model_names = metadata['model_names']
        self.is_fitted = True
        
        logger.info(f"Regime gate loaded from {path}")
        return self


def evaluate_regime_gate(gate: RegimeGate,
                        X_regime: np.ndarray,
                        predictions: Dict[str, np.ndarray],
                        y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate regime gate performance.
    
    Args:
        gate: Fitted regime gate
        X_regime: Regime features
        predictions: Dict of model predictions
        y: True targets
    
    Returns:
        Dict of evaluation metrics
    """
    # Get gated predictions
    gated_preds = gate.predict(predictions, X_regime)
    
    # Compute IC
    gated_ic = np.corrcoef(gated_preds, y)[0, 1]
    
    # Compare with equal weights
    equal_weights = np.full((len(y), len(predictions)), 1.0 / len(predictions))
    equal_preds = apply_regime_gating(predictions, equal_weights, list(predictions.keys()))
    equal_ic = np.corrcoef(equal_preds, y)[0, 1]
    
    # Improvement
    improvement = gated_ic - equal_ic
    
    # Weight statistics
    regime_weights = gate.predict_weights(X_regime)
    weight_std = np.std(regime_weights, axis=0)
    weight_range = np.max(regime_weights, axis=0) - np.min(regime_weights, axis=0)
    
    metrics = {
        'gated_ic': float(gated_ic),
        'equal_weight_ic': float(equal_ic),
        'improvement': float(improvement),
        'weight_std': [float(w) for w in weight_std],
        'weight_range': [float(w) for w in weight_range],
        'avg_weight_std': float(np.mean(weight_std)),
        'avg_weight_range': float(np.mean(weight_range))
    }
    
    logger.info(f"Regime gate evaluation:")
    logger.info(f"  Gated IC: {gated_ic:.4f}")
    logger.info(f"  Equal weight IC: {equal_ic:.4f}")
    logger.info(f"  Improvement: {improvement:.4f}")
    
    return metrics
