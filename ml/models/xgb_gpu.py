"""
GPU-Optimized XGBoost Training

Implements XGBoost with GPU acceleration and memory optimization
for large-scale cross-sectional and per-asset training.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class XGBoostGPU:
    """
    GPU-optimized XGBoost trainer with memory efficiency.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = config.get('params', {})
        self.data_config = config.get('data', {})
        self.training_config = config.get('training', {})
        
        # Ensure GPU parameters
        if self.params.get('tree_method') != 'gpu_hist':
            logger.warning("tree_method not set to gpu_hist, GPU acceleration may not work")
        
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: np.ndarray) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """Prepare data for XGBoost with GPU optimization."""
        
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
        
        # Create DMatrix with GPU optimization
        if self.data_config.get('use_quantile_dmatrix', True):
            # Use QuantileDMatrix for better GPU memory usage
            dtrain = xgb.QuantileDMatrix(X, label=y, feature_names=self.feature_names)
        else:
            dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        
        return dtrain, None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: np.ndarray,
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_val: Optional[np.ndarray] = None) -> 'XGBoostGPU':
        """Fit XGBoost model with GPU acceleration."""
        
        logger.info("Starting XGBoost GPU training...")
        start_time = time.time()
        
        # Prepare training data
        dtrain, _ = self._prepare_data(X, y)
        
        # Prepare validation data if provided
        dval = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if X_val.dtype != np.float32:
                X_val = X_val.astype(np.float32)
            if y_val.dtype != np.float32:
                y_val = y_val.astype(np.float32)
            
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        
        # Create model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Fit with early stopping if validation data provided
        if dval is not None and 'early_stopping_rounds' in self.params:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.params['early_stopping_rounds'],
                verbose=100
            )
        else:
            self.model.fit(X, y, verbose=100)
        
        self.is_fitted = True
        fit_time = time.time() - start_time
        
        logger.info(f"XGBoost GPU training completed in {fit_time:.2f}s")
        logger.info(f"Best iteration: {getattr(self.model, 'best_iteration', 'N/A')}")
        
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
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to feature names if available
        if self.feature_names:
            importance = {self.feature_names[int(k[1:])]: v for k, v in importance.items()}
        
        return importance
    
    def save(self, path: Path) -> None:
        """Save fitted model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(path / 'model.json'))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'params': self.params,
            'config': self.config
        }
        
        joblib.dump(metadata, path / 'metadata.joblib')
        
        logger.info(f"XGBoost GPU model saved to {path}")
    
    def load(self, path: Path) -> 'XGBoostGPU':
        """Load fitted model."""
        # Load model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path / 'model.json'))
        
        # Load metadata
        metadata = joblib.load(path / 'metadata.joblib')
        self.feature_names = metadata['feature_names']
        self.params = metadata['params']
        self.config = metadata['config']
        
        self.is_fitted = True
        
        logger.info(f"XGBoost GPU model loaded from {path}")
        return self


def train_global_xgb_gpu(panel_df: pd.DataFrame, 
                        config: Dict,
                        output_dir: Path) -> Dict[str, any]:
    """
    Train global XGBoost model with GPU acceleration.
    
    Args:
        panel_df: Panel dataset
        config: Model configuration
        output_dir: Output directory
    
    Returns:
        Training results
    """
    logger.info("Training global XGBoost with GPU...")
    
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
    xgb_gpu = XGBoostGPU(config)
    xgb_gpu.fit(X, y)
    
    # Make predictions
    predictions = xgb_gpu.predict(X)
    
    # Calculate performance
    ic = np.corrcoef(predictions, y)[0, 1]
    mse = np.mean((predictions - y) ** 2)
    
    # Save model
    xgb_gpu.save(output_dir / 'global_xgb_gpu')
    
    # Save predictions
    np.save(output_dir / 'global_xgb_gpu_predictions.npy', predictions)
    
    results = {
        'model_type': 'global_xgb_gpu',
        'ic': float(ic),
        'mse': float(mse),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'feature_names': feature_cols
    }
    
    logger.info(f"Global XGBoost GPU training completed: IC={ic:.4f}, MSE={mse:.4f}")
    
    return results


def train_per_asset_xgb_gpu(panel_df: pd.DataFrame,
                           config: Dict,
                           output_dir: Path,
                           symbols: List[str]) -> Dict[str, any]:
    """
    Train per-asset XGBoost models with GPU acceleration.
    
    Args:
        panel_df: Panel dataset
        config: Model configuration
        output_dir: Output directory
        symbols: List of symbols to train
    
    Returns:
        Training results
    """
    logger.info(f"Training per-asset XGBoost with GPU for {len(symbols)} symbols...")
    
    results = {}
    
    for symbol in symbols:
        logger.info(f"Training XGBoost GPU for {symbol}")
        
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
        xgb_gpu = XGBoostGPU(config)
        xgb_gpu.fit(X, y)
        
        # Make predictions
        predictions = xgb_gpu.predict(X)
        
        # Calculate performance
        ic = np.corrcoef(predictions, y)[0, 1]
        mse = np.mean((predictions - y) ** 2)
        
        # Save model
        xgb_gpu.save(output_dir / f'{symbol}_xgb_gpu')
        
        # Save predictions
        np.save(output_dir / f'{symbol}_xgb_gpu_predictions.npy', predictions)
        
        results[symbol] = {
            'ic': float(ic),
            'mse': float(mse),
            'n_samples': len(y),
            'n_features': X.shape[1]
        }
        
        logger.info(f"{symbol} XGBoost GPU: IC={ic:.4f}, MSE={mse:.4f}")
    
    logger.info(f"Per-asset XGBoost GPU training completed for {len(results)} symbols")
    
    return results
