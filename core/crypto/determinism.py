#!/usr/bin/env python3
"""
Determinism utilities for crypto model training.

Ensures reproducible results across training runs by controlling
all sources of randomness and ordering.
"""

import os
import logging
import random
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def enforce_global_determinism(seed: int = 42) -> None:
    """
    Enforce deterministic execution across all libraries.
    
    This function sets seeds and environment variables to ensure
    that repeated runs with the same inputs produce identical outputs.
    """
    # Set environment variables (must be done before importing certain modules)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA determinism
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set pandas options for deterministic behavior
    pd.set_option('mode.chained_assignment', 'raise')  # Catch non-deterministic operations
    
    logger.info(f"✅ Global determinism enforced: seed={seed}")


def normalize_dataframe_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame for deterministic comparison.
    
    Ensures consistent ordering of index and columns for reliable
    equality testing across different execution environments.
    """
    return (
        df.copy()
        .sort_index(axis=0)  # Sort by index (time)
        .sort_index(axis=1)  # Sort by column names
        .astype({col: df[col].dtype for col in df.columns})  # Preserve dtypes
    )


def assert_dataframes_equal(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    name: str = "DataFrames",
    normalize: bool = True,
    check_exact: bool = True
) -> None:
    """
    Assert two DataFrames are exactly equal with clear error messages.
    
    Args:
        df1, df2: DataFrames to compare
        name: Descriptive name for error messages
        normalize: Whether to normalize ordering before comparison
        check_exact: Whether to require exact equality (vs approximate)
    """
    if normalize:
        df1 = normalize_dataframe_for_comparison(df1)
        df2 = normalize_dataframe_for_comparison(df2)
    
    try:
        pd.testing.assert_frame_equal(
            df1, df2,
            check_exact=check_exact,
            rtol=0 if check_exact else 1e-10,
            atol=0 if check_exact else 1e-10,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_categorical=True,
            check_like=False,  # Order must match (stricter determinism)
        )
    except AssertionError as e:
        raise AssertionError(f"{name} not deterministic:\n{e}")


def assert_series_equal(
    s1: pd.Series, 
    s2: pd.Series, 
    name: str = "Series",
    check_exact: bool = True
) -> None:
    """
    Assert two Series are exactly equal with clear error messages.
    """
    try:
        pd.testing.assert_series_equal(
            s1, s2,
            check_exact=check_exact,
            rtol=0 if check_exact else 1e-10,
            atol=0 if check_exact else 1e-10,
            check_dtype=True,
            check_index_type=True,
            check_categorical=True,
        )
    except AssertionError as e:
        raise AssertionError(f"{name} not deterministic:\n{e}")


def verify_feature_determinism(
    features1: pd.DataFrame,
    features2: pd.DataFrame,
    labels1: pd.Series,
    labels2: pd.Series,
    strict_ordering: bool = True
) -> None:
    """
    Verify that feature building is deterministic.
    
    Args:
        features1, features2: Feature matrices from two identical runs
        labels1, labels2: Label series from two identical runs  
        strict_ordering: Whether column/index order must match exactly
    """
    # Check shapes first
    if features1.shape != features2.shape:
        raise AssertionError(
            f"Feature shapes differ: {features1.shape} vs {features2.shape}"
        )
    
    if len(labels1) != len(labels2):
        raise AssertionError(
            f"Label lengths differ: {len(labels1)} vs {len(labels2)}"
        )
    
    # Check deterministic features
    assert_dataframes_equal(
        features1, features2, 
        name="Features",
        normalize=not strict_ordering,
        check_exact=True
    )
    
    # Check deterministic labels
    assert_series_equal(
        labels1, labels2,
        name="Labels", 
        check_exact=True
    )
    
    logger.info("✅ Feature determinism verified")


def create_deterministic_symbol_encoding(symbols: pd.Series) -> pd.DataFrame:
    """
    Create deterministic one-hot encoding for symbols.
    
    Ensures consistent column ordering regardless of data ingestion order.
    """
    # Sort categories for deterministic ordering
    unique_symbols = sorted(symbols.unique())
    
    # Create categorical with explicit ordering
    symbol_categorical = pd.Categorical(
        symbols, 
        categories=unique_symbols, 
        ordered=True
    )
    
    # Generate dummies with consistent ordering
    dummies = pd.get_dummies(
        symbol_categorical, 
        prefix="symbol", 
        drop_first=False,
        dtype=float
    )
    
    # Ensure columns are in sorted order
    dummies = dummies.reindex(columns=sorted(dummies.columns))
    
    return dummies


def validate_training_reproducibility(
    model1_path: str,
    model2_path: str,
    metrics1: dict,
    metrics2: dict,
    tolerance: float = 1e-6
) -> None:
    """
    Validate that model training is reproducible.
    
    Checks model file checksums and metric reproducibility.
    """
    import hashlib
    
    # Check model file checksums
    def file_hash(path: str) -> str:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    hash1 = file_hash(model1_path)
    hash2 = file_hash(model2_path)
    
    if hash1 != hash2:
        raise AssertionError(
            f"Model files not deterministic:\n"
            f"Model 1: {hash1}\n"
            f"Model 2: {hash2}"
        )
    
    # Check metric reproducibility
    common_metrics = set(metrics1.keys()) & set(metrics2.keys())
    
    for metric in common_metrics:
        val1, val2 = metrics1[metric], metrics2[metric]
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance:
                raise AssertionError(
                    f"Metric {metric} not deterministic: "
                    f"{val1} vs {val2} (diff: {abs(val1 - val2)})"
                )
    
    logger.info("✅ Training reproducibility verified")


class DeterministicContext:
    """Context manager for deterministic execution."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.original_env = {}
        
    def __enter__(self):
        # Save original environment
        self.original_env = {
            'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        }
        
        # Enforce determinism
        enforce_global_determinism(self.seed)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original environment
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
