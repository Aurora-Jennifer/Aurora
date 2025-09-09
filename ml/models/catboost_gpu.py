"""
GPU-Optimized CatBoost Training

Implements CatBoost with GPU acceleration as a LightGBM alternative
for ensemble diversity and performance.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class CatBoostGPU:
    """
    GPU-optimized CatBoost trainer.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = config.get('params', {})
        self.data_config = config.get('data', {})
        self.training_config = config.get('training', {})
        
        # Ensure GPU parameters
        if self.params.get('task_type') != 'GPU':
            logger.warning("task_type not set to GPU, GPU acceleration may not work")
        
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for CatBoost with GPU optimization."""
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Ensure float32 for GPU efficiency
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        return X, y
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: np.ndarray,
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_val: Optional[np.ndarray] = None) -> 'CatBoostGPU':
        """Fit CatBoost model with GPU acceleration."""
        
        logger.info("Starting CatBoost GPU training...")
        start_time = time.time()
        
        # Prepare training data
        X, y = self._prepare_data(X, y)
        
        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if X_val.dtype != np.float32:
                X_val = X_val.astype(np.float32)
            if y_val.dtype != np.float32:
                y_val = y_val.astype(np.float32)
        
        # Create model
        self.model = CatBoostRegressor(**self.params)
        
        # Fit with early stopping if validation data provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=100
            )
        else:
            self.model.fit(X, y, verbose=100)
        
        self.is_fitted = True
        fit_time = time.time() - start_time
        
        logger.info(f"CatBoost GPU training completed in {fit_time:.2f}s")
        logger.info(f"Best iteration: {getattr(self.model, 'best_iteration_', 'N/A')}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions with GPU acceleration."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Ensure float32
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'PredictionValuesChange') -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        importance = self.model.get_feature_importance(type=importance_type)
        
        # Convert to feature names if available
        if self.feature_names:
            importance = dict(zip(self.feature_names, importance))
        else:
            importance = {f"feature_{i}": imp for i, imp in enumerate(importance)}
        
        return importance
    
    def save(self, path: Path) -> None:
        """Save fitted model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(path / 'model.cbm'))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'params': self.params,
            'config': self.config
        }
        
        joblib.dump(metadata, path / 'metadata.joblib')
        
        logger.info(f"CatBoost GPU model saved to {path}")
    
    def load(self, path: Path) -> 'CatBoostGPU':
        """Load fitted model."""
        # Load model
        self.model = CatBoostRegressor()
        self.model.load_model(str(path / 'model.cbm'))
        
        # Load metadata
        metadata = joblib.load(path / 'metadata.joblib')
        self.feature_names = metadata['feature_names']
        self.params = metadata['params']
        self.config = metadata['config']
        
        self.is_fitted = True
        
        logger.info(f"CatBoost GPU model loaded from {path}")
        return self


def train_global_catboost_gpu(panel_df: pd.DataFrame, 
                             config: Dict,
                             output_dir: Path) -> Dict[str, any]:
    """
    Train global CatBoost model with GPU acceleration.
    
    Args:
        panel_df: Panel dataset
        config: Model configuration
        output_dir: Output directory
    
    Returns:
        Training results
    """
    logger.info("Training global CatBoost with GPU...")
    
    # Prepare features and targets
    feature_cols = [col for col in panel_df.columns if col.startswith(('ret', 'vol', 'bb', 'rsi', 'ma', 'momentum'))]
    target_col = 'ret_fwd_5'
    
    X = panel_df[feature_cols].values
    y = panel_df[target_col].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train model
    catboost_gpu = CatBoostGPU(config)
    catboost_gpu.fit(X, y)
    
    # Make predictions
    predictions = catboost_gpu.predict(X)
    
    # Calculate performance
    ic = np.corrcoef(predictions, y)[0, 1]
    mse = np.mean((predictions - y) ** 2)
    
    # Save model
    catboost_gpu.save(output_dir / 'global_catboost_gpu')
    
    # Save predictions
    np.save(output_dir / 'global_catboost_gpu_predictions.npy', predictions)
    
    results = {
        'model_type': 'global_catboost_gpu',
        'ic': float(ic),
        'mse': float(mse),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'feature_names': feature_cols
    }
    
    logger.info(f"Global CatBoost GPU training completed: IC={ic:.4f}, MSE={mse:.4f}")
    
    return results


def train_per_asset_catboost_gpu(panel_df: pd.DataFrame,
                                config: Dict,
                                output_dir: Path,
                                symbols: List[str]) -> Dict[str, any]:
    """
    Train per-asset CatBoost models with GPU acceleration.
    
    Args:
        panel_df: Panel dataset
        config: Model configuration
        output_dir: Output directory
        symbols: List of symbols to train
    
    Returns:
        Training results
    """
    logger.info(f"Training per-asset CatBoost with GPU for {len(symbols)} symbols...")
    
    results = {}
    
    for symbol in symbols:
        logger.info(f"Training CatBoost GPU for {symbol}")
        
        # Filter data for symbol
        symbol_data = panel_df[panel_df['symbol'] == symbol].copy()
        
        if len(symbol_data) < 100:
            logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} samples")
            continue
        
        # Prepare features and targets
        feature_cols = [col for col in symbol_data.columns if col.startswith(('ret', 'vol', 'bb', 'rsi', 'ma', 'momentum'))]
        target_col = 'ret_fwd_5'
        
        X = symbol_data[feature_cols].values
        y = symbol_data[target_col].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 50:
            logger.warning(f"Insufficient valid data for {symbol}: {len(X)} samples")
            continue
        
        # Train model
        catboost_gpu = CatBoostGPU(config)
        catboost_gpu.fit(X, y)
        
        # Make predictions
        predictions = catboost_gpu.predict(X)
        
        # Calculate performance
        ic = np.corrcoef(predictions, y)[0, 1]
        mse = np.mean((predictions - y) ** 2)
        
        # Save model
        catboost_gpu.save(output_dir / f'{symbol}_catboost_gpu')
        
        # Save predictions
        np.save(output_dir / f'{symbol}_catboost_gpu_predictions.npy', predictions)
        
        results[symbol] = {
            'ic': float(ic),
            'mse': float(mse),
            'n_samples': len(y),
            'n_features': X.shape[1]
        }
        
        logger.info(f"{symbol} CatBoost GPU: IC={ic:.4f}, MSE={mse:.4f}")
    
    logger.info(f"Per-asset CatBoost GPU training completed for {len(results)} symbols")
    
    return results
