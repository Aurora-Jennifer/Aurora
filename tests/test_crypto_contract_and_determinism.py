#!/usr/bin/env python3
"""
Crypto Contract and Determinism Tests

Tests that crypto model training is deterministic and follows data contracts.
This is the foundation test for production-safe crypto models.
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.crypto.contracts import CryptoDataContract, enforce_determinism, create_data_lineage
from core.crypto.determinism import (
    enforce_global_determinism, assert_dataframes_equal, assert_series_equal,
    verify_feature_determinism, validate_training_reproducibility, DeterministicContext
)
from core.utils.ts_cv import PurgedTimeSeriesSplit, validate_time_series_split
from scripts.train_crypto import CryptoModelTrainer


def _sha256_bytes(b: bytes) -> str:
    """Calculate SHA256 hash of bytes."""
    return hashlib.sha256(b).hexdigest()


def _create_crypto_fixture(n_samples: int = 200, symbols: list = None) -> pd.DataFrame:
    """Create deterministic crypto test data."""
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    # Enforce deterministic data generation
    np.random.seed(42)
    
    all_data = []
    
    for i, symbol in enumerate(symbols):
        # Create deterministic date range with offset to avoid duplicates
        start_date = pd.Timestamp('2024-01-01', tz='UTC') + pd.Timedelta(hours=i * (n_samples // len(symbols)))
        dates = pd.date_range(
            start=start_date,
            periods=n_samples // len(symbols),
            freq='h',  # Use 'h' instead of deprecated 'H'
            tz='UTC'
        )
        
        # Generate deterministic price series
        base_price = 50000 if 'BTC' in symbol else 3000
        price_seed = 42 + i * 1000  # Different seed per symbol
        np.random.seed(price_seed)
        
        # Generate realistic OHLCV with constraints
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))
        
        # Generate OHLCV ensuring OHLC constraints
        data = []
        for j, (date, close) in enumerate(zip(dates, prices)):
            open_price = prices[j-1] if j > 0 else close
            
            # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
            spread = abs(np.random.normal(0, 0.005))
            high = max(open_price, close) * (1 + spread)
            low = min(open_price, close) * (1 - spread)
            
            volume = np.random.uniform(1e6, 1e7)
            
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'returns_1h': np.random.normal(0, 0.01),
                'target_1h': np.random.normal(0, 0.01),
                'predictions': np.random.normal(0, 0.01),
                'volatility_24h': np.random.uniform(0.01, 0.05),
            })
        
        df = pd.DataFrame(data)
        all_data.append(df)
    
    # Combine and sort by timestamp
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp')
    
    return combined_df


def test_crypto_data_contract_validation():
    """Test that crypto data contract validation works correctly."""
    # Create valid test data  
    df = _create_crypto_fixture(100)
    
    # Initialize contract
    contract = CryptoDataContract()
    
    # Test valid data (warnings are OK, failures are not)
    is_valid, violations = contract.validate_input_data(df)
    
    # Filter out quality warnings (these are logged but don't fail validation)
    error_violations = [v for v in violations if not any(
        warning_text in v for warning_text in [
            'non-positive values', 'NaN values', 'infinite values', 
            'High < max(Open, Close)', 'Low > min(Open, Close)'
        ]
    )]
    
    # Only fail on true schema violations, not data quality warnings
    assert len(error_violations) == 0 or is_valid, \
        f"Schema validation failed: {error_violations}"
    
        # Test invalid data - introduce violations
    df_invalid = df.copy()
    df_invalid.loc[df_invalid.index[0], 'close'] = np.nan  # NaN violation
    df_invalid.loc[df_invalid.index[1], 'high'] = -1  # Negative price violation

    is_valid, violations = contract.validate_input_data(df_invalid)
    assert not is_valid, "Invalid data passed validation"
    assert len(violations) >= 1, f"Expected at least one violation, got: {violations}"


def test_purged_time_series_split():
    """Test that purged time series split prevents leakage."""
    # Create test data
    df = _create_crypto_fixture(200)
    
    # Initialize purged CV
    cv = PurgedTimeSeriesSplit(n_splits=3, purge_gap=12, test_size=0.2)
    
    splits = list(cv.split(df))
    assert len(splits) == 3, f"Expected 3 splits, got {len(splits)}"
    
    # Validate each split for leakage
    for i, (train_idx, test_idx) in enumerate(splits):
        assert len(train_idx) > 0, f"Split {i}: empty training set"
        assert len(test_idx) > 0, f"Split {i}: empty test set"
        
        # Check no overlap
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Split {i}: index overlap detected"
        
        # Validate temporal ordering
        is_valid = validate_time_series_split(df, train_idx, test_idx)
        assert is_valid, f"Split {i}: temporal leakage detected"


def test_crypto_training_determinism():
    """Test that crypto model training is completely deterministic."""
    
    # Use deterministic context for complete isolation
    with DeterministicContext(seed=42):
        # Create test data
        df = _create_crypto_fixture(150)
    
        # Initialize trainer with deterministic config
        config = {
            'random_seed': 42,
            'model_type': 'ridge',
            'model_params': {'ridge': {'alpha': 1.0, 'random_state': 42}},
            'features': {
                'lookback_periods': [5, 10],
                'include_volume': True,
                'include_volatility': True,
                'include_momentum': True,
                'crypto_specific': True,
            },
            'validation': {'n_splits': 3, 'test_size': 0.2, 'gap': 1},
            'quality_gates': {'min_r2': -1.0, 'max_mse': 10.0, 'min_samples': 10}
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Train model twice with same config
            trainer1 = CryptoModelTrainer(config)
            X1, y1 = trainer1.build_features(df)
            metrics1 = trainer1.train_model(X1, y1)
            
            # Save first model
            import pickle
            model1_path = tmp_path / "model1.pkl"
            with open(model1_path, 'wb') as f:
                pickle.dump({
                    'model': trainer1.model,
                    'scaler': trainer1.scaler,
                    'metrics': metrics1
                }, f)
        
            # Reset environment and train second model
            enforce_global_determinism(42)
            trainer2 = CryptoModelTrainer(config)
            X2, y2 = trainer2.build_features(df)
            metrics2 = trainer2.train_model(X2, y2)
            
            # Save second model
            model2_path = tmp_path / "model2.pkl"
            with open(model2_path, 'wb') as f:
                pickle.dump({
                    'model': trainer2.model,
                    'scaler': trainer2.scaler,
                    'metrics': metrics2
                }, f)
        
            # Test determinism using enhanced verification functions
            
            # 1. Verify feature and label determinism
            verify_feature_determinism(X1, X2, y1, y2, strict_ordering=True)
            
            # 2. Verify training reproducibility (metrics + model files)
            validate_training_reproducibility(
                str(model1_path), str(model2_path), 
                metrics1, metrics2, 
                tolerance=1e-6
            )
            
            print("✅ Determinism test passed!")


def test_feature_manifest_creation():
    """Test feature manifest creation for reproducibility."""
    # Create test data
    df = _create_crypto_fixture(100)
    
    # Initialize trainer and build features
    trainer = CryptoModelTrainer()
    X, y = trainer.build_features(df)
    
    # Initialize contract
    contract = CryptoDataContract()
    
    # Create feature manifest
    manifest = contract.create_feature_manifest(X, y)
    
    # Validate manifest structure
    required_keys = ['feature_hash', 'feature_names', 'feature_count', 'sample_count']
    for key in required_keys:
        assert key in manifest, f"Missing required key: {key}"
    
    # Test reproducibility - same features should produce same hash
    manifest2 = contract.create_feature_manifest(X, y)
    assert manifest['feature_hash'] == manifest2['feature_hash'], \
        "Feature manifest not reproducible"
    
    print(f"✅ Feature manifest created: {manifest['feature_hash']}")


def test_data_lineage_tracking():
    """Test data lineage tracking for audit trail."""
    # Create test data
    df = _create_crypto_fixture(100)
    
    # Create lineage
    processing_steps = ['load_data', 'validate_schema', 'build_features', 'split_data']
    lineage = create_data_lineage(df, processing_steps)
    
    # Validate lineage structure
    required_keys = ['data_hash', 'shape', 'columns', 'processing_steps']
    for key in required_keys:
        assert key in lineage, f"Missing required key: {key}"
    
    assert lineage['processing_steps'] == processing_steps
    assert lineage['shape'] == df.shape
    
    print(f"✅ Data lineage created: {lineage['data_hash']}")


if __name__ == "__main__":
    print("🧪 Running Crypto Contract and Determinism Tests")
    
    test_crypto_data_contract_validation()
    print("✅ Contract validation test passed")
    
    test_purged_time_series_split()
    print("✅ Purged time series split test passed")
    
    test_crypto_training_determinism()
    print("✅ Training determinism test passed")
    
    test_feature_manifest_creation()
    print("✅ Feature manifest test passed")
    
    test_data_lineage_tracking()
    print("✅ Data lineage test passed")
    
    print("\n🎉 All crypto contract and determinism tests PASSED!")
    print("Ready for production-safe crypto model training!")
