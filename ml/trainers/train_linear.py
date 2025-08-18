# ml/trainers/train_linear.py
"""
Train Ridge model for Alpha v1.
Deterministic training with leakage guards.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import yaml
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def load_feature_config() -> dict:
    """Load feature configuration."""
    config_path = Path("config/features.yaml")
    return yaml.safe_load(config_path.read_text())

def load_feature_data(symbols: List[str], feature_dir: str = "artifacts/feature_store") -> pd.DataFrame:
    """
    Load feature data for multiple symbols.
    
    Args:
        symbols: List of symbols to load
        feature_dir: Directory containing feature files
        
    Returns:
        Combined DataFrame with symbol column
    """
    feature_path = Path(feature_dir)
    all_data = []
    
    for symbol in symbols:
        file_path = feature_path / f"{symbol}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df['symbol'] = symbol
            all_data.append(df)
        else:
            logger.warning(f"Feature file not found: {file_path}")
    
    if not all_data:
        raise ValueError("No feature data found")
    
    combined = pd.concat(all_data, ignore_index=False)
    return combined

def prepare_training_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and test data with time-based split.
    
    Args:
        df: Combined feature DataFrame
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by date
    df = df.sort_index()
    
    # Split by time (no leakage)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Training set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df

def create_ridge_pipeline() -> Pipeline:
    """
    Create Ridge pipeline with cross-validation.
    
    Returns:
        sklearn Pipeline
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    ridge = RidgeCV(
        alphas=[0.1, 1.0, 10.0],
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_squared_error'
    )
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('ridge', ridge)
    ])
    
    return pipeline

def train_model(train_df: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """
    Train Ridge model on training data.
    
    Args:
        train_df: Training DataFrame
        random_state: Random seed for reproducibility
        
    Returns:
        Trained pipeline
    """
    config = load_feature_config()
    
    # Set random seed
    np.random.seed(random_state)
    
    # Prepare features and target
    feature_cols = list(config['features'].keys())
    target_col = 'ret_fwd_1d'
    
    # Verify all features exist
    missing_features = [col for col in feature_cols if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    
    # Remove any remaining NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("No valid training data after removing NaN values")
    
    logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features")
    
    # Create and train pipeline
    pipeline = create_ridge_pipeline()
    pipeline.fit(X, y)
    
    # Log model info
    ridge = pipeline.named_steps['ridge']
    logger.info(f"Best alpha: {ridge.alpha_}")
    logger.info(f"R² score: {ridge.score(X, y):.4f}")
    
    return pipeline

def save_model(pipeline: Pipeline, model_path: str = "artifacts/models/linear_v1.pkl"):
    """
    Save trained model.
    
    Args:
        pipeline: Trained pipeline
        model_path: Path to save model
    """
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_file, 'wb') as f:
        pickle.dump(pipeline, f)
    
    logger.info(f"Model saved to {model_file}")

def train_linear_model(symbols: List[str], 
                      feature_dir: str = "artifacts/feature_store",
                      model_path: str = "artifacts/models/linear_v1.pkl",
                      random_state: int = 42) -> Pipeline:
    """
    End-to-end training pipeline.
    
    Args:
        symbols: List of symbols to train on
        feature_dir: Directory containing feature files
        model_path: Path to save trained model
        random_state: Random seed for reproducibility
        
    Returns:
        Trained pipeline
    """
    logger.info(f"Training linear model on {symbols}")
    
    # Load feature data
    df = load_feature_data(symbols, feature_dir)
    
    # Prepare training data
    train_df, test_df = prepare_training_data(df)
    
    # Train model
    pipeline = train_model(train_df, random_state)
    
    # Save model
    save_model(pipeline, model_path)
    
    # Log feature importance
    ridge = pipeline.named_steps['ridge']
    config = load_feature_config()
    feature_cols = list(config['features'].keys())
    
    logger.info("Feature coefficients:")
    for feature, coef in zip(feature_cols, ridge.coef_):
        logger.info(f"  {feature}: {coef:.6f}")
    
    return pipeline

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="SPY,TSLA", help="Comma-separated symbols")
    parser.add_argument("--feature-dir", default="artifacts/feature_store", help="Feature directory")
    parser.add_argument("--model-path", default="artifacts/models/linear_v1.pkl", help="Model output path")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    pipeline = train_linear_model(
        symbols, 
        args.feature_dir, 
        args.model_path, 
        args.random_state
    )
    
    print(f"Training complete. Model saved to {args.model_path}")
