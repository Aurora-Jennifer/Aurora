"""
Robust preprocessing pipeline for multi-symbol trading data.
Ensures consistent feature transformation across train/test splits.
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    """Winsorize extreme values to prevent outliers from dominating."""
    
    def __init__(self, p: float = 0.01):
        self.p = p
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        
        # Calculate percentiles
        self.lower_bounds_ = np.percentile(X, self.p * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, (1 - self.p) * 100, axis=0)
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        
        # Clip values
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped


class DropNearConstant(BaseEstimator, TransformerMixin):
    """Drop features with near-constant values."""
    
    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.constant_features_ = None
        
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        
        # Find near-constant features
        feature_stds = np.std(X, axis=0)
        self.constant_features_ = feature_stds < self.threshold
        
        if np.any(self.constant_features_):
            n_dropped = np.sum(self.constant_features_)
            warnings.warn(f"Dropping {n_dropped} near-constant features")
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        
        # Keep only non-constant features
        return X[:, ~self.constant_features_]


def make_preprocessing_pipeline(n_components: int = 128) -> Pipeline:
    """Create a robust preprocessing pipeline with adaptive PCA."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("winsor", Winsorizer(p=0.01)),
        ("dropconst", DropNearConstant(threshold=1e-6)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=n_components, svd_solver="auto", whiten=False)),
    ])


def fit_and_serialize_preprocessing(X_train: np.ndarray, path: str, n_components: int = 128) -> Pipeline:
    """Fit preprocessing pipeline on training data and serialize it."""
    # Ensure PCA components don't exceed data dimensions
    n_samples, n_features = X_train.shape
    max_components = min(n_samples - 1, n_features)
    actual_components = min(n_components, max_components)
    
    pipe = make_preprocessing_pipeline(actual_components)
    pipe.fit(X_train)
    joblib.dump(pipe, path)
    return pipe


def load_preprocessing(path: str) -> Pipeline:
    """Load a serialized preprocessing pipeline."""
    return joblib.load(path)


def assert_preprocessing_shapes(X_train: np.ndarray, X_processed: np.ndarray, expected_dim: int):
    """Assert that preprocessing shapes are correct."""
    # Allow flexible input dimensions (2-3 symbols: 261-390 features)
    assert X_train.shape[1] in [261, 390], f"Unexpected raw feature dim: {X_train.shape[1]} (expected 261 for 2 symbols or 390 for 3 symbols)"
    assert X_processed.shape[1] == expected_dim, f"Processed dim mismatch: {X_processed.shape[1]} != {expected_dim}"
    assert X_processed.shape[0] == X_train.shape[0], f"Sample count mismatch: {X_processed.shape[0]} != {X_train.shape[0]}"


def compute_class_weights(actions: np.ndarray, max_weight: float = 2.0) -> dict:
    """Compute inverse frequency class weights with clipping."""
    unique, counts = np.unique(actions, return_counts=True)
    n_samples = len(actions)
    
    # Inverse frequency weights
    weights = {}
    for action, count in zip(unique, counts, strict=False):
        weights[action] = n_samples / (len(unique) * count)
    
    # Normalize to mean 1.0
    mean_weight = np.mean(list(weights.values()))
    for action in weights:
        weights[action] /= mean_weight
    
    # Clip extreme weights
    for action in weights:
        weights[action] = min(weights[action], max_weight)
    
    return weights


def create_dead_zone_labels(returns: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """Create labels with dead zone for HOLD actions."""
    # Z-score the returns
    z_scores = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    
    # Create labels with dead zone
    labels = np.zeros_like(z_scores, dtype=int)
    labels[z_scores >= threshold] = 0  # BUY
    labels[z_scores <= -threshold] = 1  # SELL
    labels[np.abs(z_scores) < threshold] = 2  # HOLD
    
    return labels


def calculate_entropy_bonus_schedule(epoch: int, total_epochs: int, 
                                   start_beta: float = 0.05, end_beta: float = 0.01) -> float:
    """Calculate entropy bonus with cosine annealing."""
    progress = epoch / total_epochs
    # Cosine annealing from start_beta to end_beta
    beta = end_beta + (start_beta - end_beta) * 0.5 * (1 + np.cos(np.pi * progress))
    return beta
