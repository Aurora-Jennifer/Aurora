#!/usr/bin/env python3
"""
Feature Schema & Scaler Persistence Helpers

Ensures feature consistency between training and validation:
1. Persist feature column order and names
2. Enforce scaler consistency
3. Add assertions for feature quality
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def save_feature_schema(feature_columns: List[str], output_path: str):
    """
    Save feature column schema to JSON
    
    Args:
        feature_columns: List of feature column names in order
        output_path: Path to save schema JSON
    """
    schema = {
        "feature_columns": feature_columns,
        "num_features": len(feature_columns)
    }
    
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

def load_feature_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load feature column schema from JSON
    
    Args:
        schema_path: Path to schema JSON
    
    Returns:
        Schema dictionary
    """
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    return schema

def assert_feature_identity(current_cols: List[str], saved_cols: List[str]):
    """
    Assert that current feature columns match saved schema
    
    Args:
        current_cols: Current feature column names
        saved_cols: Saved feature column names
    """
    if list(current_cols) != list(saved_cols):
        raise AssertionError(
            f"Feature column mismatch:\n"
            f"Current: {current_cols}\n"
            f"Saved:   {saved_cols}\n"
            f"Order or names differ!"
        )

def assert_scaled_ok(X_scaled: np.ndarray, min_mean: float = 1e-6, min_std: float = 1e-6):
    """
    Assert that scaled features are non-degenerate
    
    Args:
        X_scaled: Scaled feature matrix
        min_mean: Minimum mean absolute value
        min_std: Minimum standard deviation
    """
    if np.isnan(X_scaled).any():
        raise AssertionError("NaNs in scaled features.")
    
    if np.allclose(X_scaled, 0.0):
        raise AssertionError("All-zero features after scaling.")
    
    # Check for near-constant features
    m = np.nanmean(np.abs(X_scaled))
    s = np.nanstd(X_scaled)
    
    if m < min_mean:
        raise AssertionError(f"Scaled features mean too low: {m:.6f} < {min_mean}")
    
    if s < min_std:
        raise AssertionError(f"Scaled features std too low: {s:.6f} < {min_std}")

def save_training_artifacts(model, scaler, feature_columns: List[str], 
                          model_path: str, scaler_path: str, schema_path: str):
    """
    Save all training artifacts with consistent paths
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_columns: Feature column names
        model_path: Path to save model
        scaler_path: Path to save scaler
        schema_path: Path to save schema
    """
    # Create directories if needed
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    Path(schema_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature schema
    save_feature_schema(feature_columns, schema_path)

def load_training_artifacts(model_path: str, scaler_path: str, schema_path: str):
    """
    Load all training artifacts and validate consistency
    
    Args:
        model_path: Path to model
        scaler_path: Path to scaler
        schema_path: Path to schema
    
    Returns:
        Tuple of (model, scaler, schema)
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load schema
    schema = load_feature_schema(schema_path)
    
    return model, scaler, schema

def validate_feature_pipeline(X: pd.DataFrame, model, scaler, schema: Dict[str, Any]):
    """
    Validate that features are compatible with training pipeline
    
    Args:
        X: Feature DataFrame
        model: Trained model
        scaler: Fitted scaler
        schema: Feature schema
    
    Returns:
        Scaled features ready for prediction
    """
    # Check feature column identity
    assert_feature_identity(list(X.columns), schema["feature_columns"])
    
    # Check feature count
    if X.shape[1] != schema["num_features"]:
        raise AssertionError(
            f"Feature count mismatch: {X.shape[1]} != {schema['num_features']}"
        )
    
    # Validate scaler feature names match schema
    if hasattr(scaler, 'feature_names_in_'):
        if list(scaler.feature_names_in_) != schema["feature_columns"]:
            raise AssertionError(
                f"Scaler feature names mismatch: {list(scaler.feature_names_in_)} != {schema['feature_columns']}"
            )
    
    # Scale features - pass DataFrame to preserve feature names
    X_scaled = scaler.transform(X)
    
    # Assert scaled features are non-degenerate
    assert_scaled_ok(X_scaled)
    
    return X_scaled

def print_feature_diagnostics(X: pd.DataFrame, X_scaled: np.ndarray, schema: Dict[str, Any]):
    """
    Print diagnostic information about features
    
    Args:
        X: Raw features
        X_scaled: Scaled features
        schema: Feature schema
    """
    print(f"Feature schema: {schema['num_features']} features")
    print(f"Raw features shape: {X.shape}")
    print(f"Scaled features shape: {X_scaled.shape}")
    print(f"Scaled features mean: {np.mean(X_scaled):.4f}")
    print(f"Scaled features std: {np.std(X_scaled):.4f}")
    print(f"Scaled features range: [{np.min(X_scaled):.4f}, {np.max(X_scaled):.4f}]")
