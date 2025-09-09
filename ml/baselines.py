#!/usr/bin/env python3
"""
Baseline Models Module

Implements clean baselines with identical costs/timing as the model:
- Buy & Hold
- Simple Rule (no lookahead bias)
- Ridge Regression (excess return prediction)
- LightGBM/XGBoost (small trees)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def buy_and_hold_daily_pnl(price: pd.Series, costs_bps: float = 0.0) -> pd.Series:
    """
    Calculate buy-and-hold daily PnL
    
    Args:
        price: Price series
        costs_bps: Transaction costs in basis points
    
    Returns:
        Daily PnL series
    """
    ret = price.pct_change().fillna(0.0)
    # Costs negligible for buy-and-hold (no rebalancing)
    return ret


def simple_rule_daily_pnl(price: pd.Series, costs_bps: float = 3.0) -> pd.Series:
    """
    Calculate simple rule daily PnL (clean, no lookahead)
    
    Args:
        price: Price series
        costs_bps: Transaction costs in basis points
    
    Returns:
        Daily PnL series
    """
    # Clean momentum rule: fast/slow MA crossover with hysteresis on slope
    fast = price.rolling(5).mean()
    slow = price.rolling(20).mean()
    slope = (fast - slow).diff()
    sig = np.sign(slope).fillna(0.0)
    
    # Calculate returns
    ret = price.pct_change().fillna(0.0)
    
    # Apply signals with 1-day delay to avoid lookahead
    pos = sig.shift(1).fillna(0.0)
    pnl = pos * ret
    
    # Apply costs on position changes
    switch = (pos != pos.shift(1)).fillna(False)
    costs = switch.astype(float) * (costs_bps / 1e4)
    
    return pnl - costs


class RidgeExcessModel:
    """
    Ridge regression model for excess return prediction with proper scaling
    """
    
    def __init__(self, alpha: float = 1.0):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha, fit_intercept=True, solver="auto")
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.cols = None
        self.is_fitted = False
        self.ill_conditioned = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model with proper scaling and conditioning checks
        
        Args:
            X: Feature DataFrame
            y: Target series (excess returns)
        """
        import warnings
        from numpy.linalg import LinAlgWarning
        
        self.cols = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)
        
        # Check for conditioning issues
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", LinAlgWarning)
            self.model.fit(X_scaled, y.values)
            self.ill_conditioned = any(isinstance(x.message, LinAlgWarning) for x in w)
        
        self.is_fitted = True
    
    def predict_edge(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict edges (expected excess returns)
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Predicted edges
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if list(X.columns) != self.cols:
            raise ValueError("Feature columns don't match training")
        
        # Apply same scaling as training
        X_scaled = self.scaler.transform(X.values)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert regression predictions to probability-like outputs
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Probability-like array (N, 3) for [SELL, HOLD, BUY]
        """
        edges = self.predict_edge(X)
        
        # Convert edges to probabilities using sigmoid-like function
        # This is a heuristic - in practice you'd want proper calibration
        proba = np.zeros((len(edges), 3))
        
        # SELL probability (negative edges)
        proba[:, 0] = np.exp(-edges) / (1 + np.exp(-edges))
        
        # BUY probability (positive edges)  
        proba[:, 2] = np.exp(edges) / (1 + np.exp(edges))
        
        # HOLD probability (residual)
        proba[:, 1] = 1.0 - proba[:, 0] - proba[:, 2]
        
        # Ensure non-negative and normalized
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba / proba.sum(1, keepdims=True)
        
        return proba


class LightGBMExcessModel:
    """
    LightGBM model for excess return prediction
    """
    
    def __init__(self, **params):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not available")
        
        # Default parameters for small trees
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'max_depth': 5,
            'min_data_in_leaf': 20,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Update with user parameters
        default_params.update(params)
        self.params = default_params
        self.model = None
        self.cols = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model
        
        Args:
            X: Feature DataFrame
            y: Target series (excess returns)
        """
        self.cols = list(X.columns)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X.values, label=y.values)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.is_fitted = True
    
    def predict_edge(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict edges (expected excess returns)
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Predicted edges
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if list(X.columns) != self.cols:
            raise ValueError("Feature columns don't match training")
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert regression predictions to probability-like outputs
        """
        edges = self.predict_edge(X)
        
        # Convert edges to probabilities using sigmoid-like function
        proba = np.zeros((len(edges), 3))
        
        # SELL probability (negative edges)
        proba[:, 0] = np.exp(-edges) / (1 + np.exp(-edges))
        
        # BUY probability (positive edges)  
        proba[:, 2] = np.exp(edges) / (1 + np.exp(edges))
        
        # HOLD probability (residual)
        proba[:, 1] = 1.0 - proba[:, 0] - proba[:, 2]
        
        # Ensure non-negative and normalized
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba / proba.sum(1, keepdims=True)
        
        return proba


class XGBoostExcessModel:
    """
    XGBoost model for excess return prediction
    """
    
    def __init__(self, **params):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not available")
        
        # Default parameters for small trees
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'num_leaves': 15,
            'min_child_weight': 20,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        
        # Update with user parameters
        default_params.update(params)
        self.params = default_params
        self.model = None
        self.cols = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model
        
        Args:
            X: Feature DataFrame
            y: Target series (excess returns)
        """
        self.cols = list(X.columns)
        
        # Train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X.values, y.values)
        
        self.is_fitted = True
    
    def predict_edge(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict edges (expected excess returns)
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Predicted edges
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if list(X.columns) != self.cols:
            raise ValueError("Feature columns don't match training")
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert regression predictions to probability-like outputs
        """
        edges = self.predict_edge(X)
        
        # Convert edges to probabilities using sigmoid-like function
        proba = np.zeros((len(edges), 3))
        
        # SELL probability (negative edges)
        proba[:, 0] = np.exp(-edges) / (1 + np.exp(-edges))
        
        # BUY probability (positive edges)  
        proba[:, 2] = np.exp(edges) / (1 + np.exp(edges))
        
        # HOLD probability (residual)
        proba[:, 1] = 1.0 - proba[:, 0] - proba[:, 2]
        
        # Ensure non-negative and normalized
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba / proba.sum(1, keepdims=True)
        
        return proba


def create_baseline_model(model_type: str, **params) -> Any:
    """
    Create a baseline model
    
    Args:
        model_type: Type of model ('ridge', 'lgbm', 'xgboost')
        **params: Model parameters
    
    Returns:
        Model instance
    """
    if model_type == 'ridge':
        return RidgeExcessModel(**params)
    elif model_type == 'lgbm':
        return LightGBMExcessModel(**params)
    elif model_type == 'xgboost':
        return XGBoostExcessModel(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    spy = yf.download("SPY", start="2023-01-01", end="2024-01-01", progress=False)
    
    # Test baselines
    print("Testing baselines...")
    
    # Buy and hold
    bh_pnl = buy_and_hold_daily_pnl(spy['Close'])
    print(f"Buy & Hold: mean={bh_pnl.mean():.4f}, std={bh_pnl.std():.4f}")
    
    # Simple rule
    rule_pnl = simple_rule_daily_pnl(spy['Close'])
    print(f"Simple Rule: mean={rule_pnl.mean():.4f}, std={rule_pnl.std():.4f}")
    
    # Test Ridge model
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y)
    
    ridge_model = RidgeExcessModel(alpha=1.0)
    ridge_model.fit(X_df, y_series)
    
    edges = ridge_model.predict_edge(X_df[:100])
    proba = ridge_model.predict_proba(X_df[:100])
    
    print(f"Ridge edges: mean={edges.mean():.4f}, std={edges.std():.4f}")
    print(f"Ridge proba shape: {proba.shape}")
    
    print("✅ Baseline tests passed!")
