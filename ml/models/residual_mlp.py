"""
Residual MLP - Learning What Trees Miss

A tiny neural network that learns to predict residuals of tree-based models,
capturing non-linear patterns that gradient boosting might miss.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Union
import logging
import joblib
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class ResidualMLP:
    """
    Residual MLP that learns to predict residuals of base models.
    
    This captures non-linear patterns that tree-based models might miss,
    particularly useful for complex interactions and non-additive effects.
    """
    
    def __init__(self, 
                 hidden_layers: Tuple[int, ...] = (64, 32),
                 activation: str = 'relu',
                 learning_rate: float = 0.001,
                 max_iter: int = 1000,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 20,
                 random_state: int = 42,
                 alpha: float = 0.0001):
        """
        Initialize Residual MLP.
        
        Args:
            hidden_layers: Hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'logistic')
            learning_rate: Learning rate for optimizer
            max_iter: Maximum iterations
            early_stopping: Whether to use early stopping
            validation_fraction: Fraction of data for validation
            n_iter_no_change: Patience for early stopping
            random_state: Random seed
            alpha: L2 regularization strength
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.alpha = alpha
        
        # Initialize components
        self.scaler = StandardScaler()
        self.mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            alpha=alpha,
            warm_start=False
        )
        
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = {}
    
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame], 
            y: np.ndarray, 
            base_pred: np.ndarray) -> 'ResidualMLP':
        """
        Fit residual MLP on residuals of base model.
        
        Args:
            X: Feature matrix
            y: True targets
            base_pred: Base model predictions (same length as y)
        
        Returns:
            Self
        """
        if len(y) != len(base_pred):
            raise ValueError(f"Target length {len(y)} != base_pred length {len(base_pred)}")
        
        # Compute residuals
        residuals = y - base_pred
        
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit MLP on residuals
        start_time = time.perf_counter()
        self.mlp.fit(X_scaled, residuals)
        fit_time = time.perf_counter() - start_time
        
        # Compute training metrics
        train_residual_pred = self.mlp.predict(X_scaled)
        train_pred = base_pred + train_residual_pred
        train_ic = np.corrcoef(train_pred, y)[0, 1]
        train_mse = np.mean((train_pred - y) ** 2)
        residual_mse = np.mean(residuals ** 2)
        
        self.training_metrics = {
            'fit_time': fit_time,
            'train_ic': float(train_ic),
            'train_mse': float(train_mse),
            'residual_mse': float(residual_mse),
            'n_iterations': self.mlp.n_iter_,
            'n_layers': len(self.mlp.hidden_layer_sizes) + 1,
            'n_parameters': sum(layer.size for layer in self.mlp.coefs_),
            'converged': self.mlp.n_iter_ < self.max_iter
        }
        
        self.is_fitted = True
        
        logger.info(f"Residual MLP fitted:")
        logger.info(f"  Training IC: {train_ic:.4f}")
        logger.info(f"  Residual MSE: {residual_mse:.4f}")
        logger.info(f"  Iterations: {self.mlp.n_iter_}")
        logger.info(f"  Parameters: {self.training_metrics['n_parameters']}")
        
        return self
    
    def predict(self, 
                X: Union[np.ndarray, pd.DataFrame], 
                base_pred: np.ndarray) -> np.ndarray:
        """
        Make predictions by adding residual predictions to base predictions.
        
        Args:
            X: Feature matrix
            base_pred: Base model predictions
        
        Returns:
            Combined predictions (base + residual)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Prepare features
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict residuals
        residual_pred = self.mlp.predict(X_scaled)
        
        # Combine with base predictions
        combined_pred = base_pred + residual_pred
        
        return combined_pred
    
    def predict_residuals(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict only the residuals (without base predictions)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        return self.mlp.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using connection weights.
        
        Returns:
            Dict of feature_name -> importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Use first layer weights as feature importance
        first_layer_weights = self.mlp.coefs_[0]  # [n_features, n_hidden]
        importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Normalize
        importance = importance / importance.sum()
        
        return dict(zip(self.feature_names, importance))
    
    def cross_validate(self, 
                      X: Union[np.ndarray, pd.DataFrame], 
                      y: np.ndarray, 
                      base_pred: np.ndarray,
                      cv: int = 5) -> Dict[str, float]:
        """
        Cross-validate residual MLP performance.
        
        Args:
            X: Feature matrix
            y: True targets
            base_pred: Base model predictions
            cv: Number of CV folds
        
        Returns:
            Dict of CV metrics
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Compute residuals
        residuals = y - base_pred
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validate on residuals
        cv_scores = cross_val_score(
            self.mlp, X_scaled, residuals, 
            cv=cv, scoring='neg_mean_squared_error'
        )
        
        cv_metrics = {
            'cv_mse_mean': float(-cv_scores.mean()),
            'cv_mse_std': float(cv_scores.std()),
            'cv_mse_scores': [float(score) for score in -cv_scores]
        }
        
        return cv_metrics
    
    def save(self, path: Path) -> None:
        """Save fitted model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save MLP
        joblib.dump(self.mlp, path / 'residual_mlp.joblib')
        
        # Save scaler
        joblib.dump(self.scaler, path / 'scaler.joblib')
        
        # Save metadata
        metadata = {
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'random_state': self.random_state,
            'alpha': self.alpha,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(metadata, path / 'metadata.joblib')
        
        logger.info(f"Residual MLP saved to {path}")
    
    def load(self, path: Path) -> 'ResidualMLP':
        """Load fitted model."""
        # Load MLP
        self.mlp = joblib.load(path / 'residual_mlp.joblib')
        
        # Load scaler
        self.scaler = joblib.load(path / 'scaler.joblib')
        
        # Load metadata
        metadata = joblib.load(path / 'metadata.joblib')
        
        # Restore parameters
        self.hidden_layers = metadata['hidden_layers']
        self.activation = metadata['activation']
        self.learning_rate = metadata['learning_rate']
        self.max_iter = metadata['max_iter']
        self.early_stopping = metadata['early_stopping']
        self.validation_fraction = metadata['validation_fraction']
        self.n_iter_no_change = metadata['n_iter_no_change']
        self.random_state = metadata['random_state']
        self.alpha = metadata['alpha']
        self.feature_names = metadata['feature_names']
        self.training_metrics = metadata['training_metrics']
        
        self.is_fitted = True
        
        logger.info(f"Residual MLP loaded from {path}")
        return self


def create_residual_ensemble(base_models: Dict[str, object],
                           X: Union[np.ndarray, pd.DataFrame],
                           y: np.ndarray,
                           base_preds: Dict[str, np.ndarray],
                           hidden_layers: Tuple[int, ...] = (64, 32)) -> Dict[str, ResidualMLP]:
    """
    Create residual MLPs for multiple base models.
    
    Args:
        base_models: Dict of model_name -> base model
        X: Feature matrix
        y: True targets
        base_preds: Dict of model_name -> base predictions
        hidden_layers: Hidden layer sizes for MLPs
    
    Returns:
        Dict of model_name -> ResidualMLP
    """
    residual_models = {}
    
    for name, base_pred in base_preds.items():
        logger.info(f"Training residual MLP for {name}")
        
        # Create and fit residual MLP
        residual_mlp = ResidualMLP(hidden_layers=hidden_layers)
        residual_mlp.fit(X, y, base_pred)
        
        residual_models[name] = residual_mlp
    
    return residual_models
