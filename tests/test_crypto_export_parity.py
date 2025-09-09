#!/usr/bin/env python3
"""
Crypto ONNX Export Parity Tests

Validates that ONNX-exported models produce identical predictions
to their native sklearn counterparts within tolerance.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from core.crypto.export import (
    to_onnx, 
    onnx_predict, 
    validate_onnx_parity, 
    export_crypto_model_complete,
    create_model_artifact_manifest
)
from core.crypto.determinism import DeterministicContext


@pytest.fixture
def crypto_test_data():
    """Create deterministic test data for crypto model parity testing."""
    with DeterministicContext(seed=42):
        n_samples = 1000
        n_features = 10
        
        # Create feature matrix with realistic crypto-like patterns
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Add some feature correlations
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_samples)  # price correlation
        X[:, 2] = -0.5 * X[:, 0] + 0.5 * np.random.randn(n_samples)  # volatility inverse
        
        # Create target with some signal
        y = (
            0.3 * X[:, 0] + 
            -0.2 * X[:, 1] + 
            0.1 * X[:, 2] + 
            0.05 * np.random.randn(n_samples)
        ).astype(np.float32)
        
        feature_names = [
            'returns_1d', 'returns_5d', 'volatility_10d', 'volume_ratio',
            'rsi_14', 'macd_signal', 'bb_position', 'momentum_5d',
            'sentiment_score', 'funding_rate'
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        
        return df, y, feature_names


@pytest.fixture
def trained_crypto_model(crypto_test_data):
    """Train a Ridge model on crypto test data."""
    X, y, feature_names = crypto_test_data
    
    with DeterministicContext(seed=42):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Optional: fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'model': model,
            'scaler': scaler,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_names': feature_names
        }


class TestONNXExport:
    """Test ONNX model export functionality."""
    
    def test_ridge_to_onnx_export(self, trained_crypto_model):
        """Test basic ONNX export of Ridge model."""
        model_data = trained_crypto_model
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "test_model.onnx"
            
            # Export to ONNX
            metadata = to_onnx(model, feature_names, onnx_path)
            
            # Verify file exists
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0
            
            # Verify metadata
            assert metadata['model_type'] == 'Ridge'
            assert metadata['n_features'] == len(feature_names)
            assert metadata['feature_names'] == feature_names
            assert 'onnx_version' in metadata
    
    def test_onnx_prediction_basic(self, trained_crypto_model):
        """Test basic ONNX prediction functionality."""
        model_data = trained_crypto_model
        model = model_data['model']
        X_test = model_data['X_test']
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "test_model.onnx"
            
            # Export and predict
            to_onnx(model, feature_names, onnx_path)
            predictions = onnx_predict(onnx_path, X_test)
            
            # Verify output shape and type
            assert predictions.shape == (len(X_test),)
            assert predictions.dtype == np.float32
            assert not np.isnan(predictions).any()


class TestONNXParity:
    """Test ONNX model parity with native models."""
    
    def test_ridge_parity_exact(self, trained_crypto_model):
        """Test exact parity between Ridge native and ONNX models."""
        model_data = trained_crypto_model
        model = model_data['model']
        X_test = model_data['X_test']
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "test_model.onnx"
            
            # Export to ONNX
            to_onnx(model, feature_names, onnx_path)
            
            # Validate parity
            results = validate_onnx_parity(
                model, onnx_path, X_test, tolerance=1e-5
            )
            
            # Assert parity passed
            assert bool(results['is_parity']) is True
            assert results['max_absolute_difference'] <= 1e-5
            assert results['n_samples_tested'] == len(X_test)
            
            # Check prediction samples match
            native_preds = model.predict(X_test)
            onnx_preds = onnx_predict(onnx_path, X_test)
            
            np.testing.assert_allclose(
                native_preds, onnx_preds, 
                atol=1e-5, rtol=1e-5,
                err_msg="Native and ONNX predictions must match within tolerance"
            )
    
    def test_parity_with_random_sampling(self, trained_crypto_model):
        """Test parity validation with random sampling."""
        model_data = trained_crypto_model
        model = model_data['model']
        X_test = model_data['X_test']
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "test_model.onnx"
            
            # Export to ONNX
            to_onnx(model, feature_names, onnx_path)
            
            # Test with 100 random samples
            results = validate_onnx_parity(
                model, onnx_path, X_test, 
                tolerance=1e-5, n_samples=100, random_seed=42
            )
            
            assert bool(results['is_parity']) is True
            assert results['n_samples_tested'] == 100
            assert results['max_absolute_difference'] <= 1e-5
    
    def test_parity_failure_detection(self, trained_crypto_model):
        """Test that parity validation correctly detects failures."""
        model_data = trained_crypto_model
        model = model_data['model']
        X_test = model_data['X_test'][:10]  # Small sample
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "test_model.onnx"
            
            # Export to ONNX
            to_onnx(model, feature_names, onnx_path)
            
            # Test with very strict tolerance to force failure
            results = validate_onnx_parity(
                model, onnx_path, X_test, tolerance=1e-10
            )
            
            # Should fail with such strict tolerance
            assert bool(results['is_parity']) is False
            assert results['max_absolute_difference'] > 1e-10


class TestCompleteExport:
    """Test complete model export pipeline."""
    
    def test_complete_export_pipeline(self, trained_crypto_model):
        """Test the complete export pipeline with all artifacts."""
        model_data = trained_crypto_model
        model = model_data['model']
        scaler = model_data['scaler']
        X_test = model_data['X_test']
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "model_artifacts"
            
            # Run complete export
            results = export_crypto_model_complete(
                model=model,
                scaler=scaler,
                feature_names=feature_names,
                X_test=X_test,
                output_dir=output_dir,
                model_name="crypto_test_v1",
                tolerance=1e-5
            )
            
            # Verify all artifacts created
            assert bool(results['export_successful']) is True
            assert bool(results['parity_passed']) is True
            
            # Check files exist
            onnx_path = Path(results['onnx_path'])
            native_path = Path(results['native_path'])
            manifest_path = Path(results['manifest_path'])
            
            assert onnx_path.exists()
            assert native_path.exists()
            assert manifest_path.exists()
            
            # Verify manifest content
            manifest = results['manifest']
            assert manifest['artifact_version'] == '1.0'
            assert manifest['model']['type'] == 'Ridge'
            assert manifest['features']['count'] == len(feature_names)
            assert manifest['features']['names'] == feature_names
            assert bool(manifest['parity']['is_parity']) is True
            assert bool(manifest['quality_gates']['parity_passed']) is True
    
    def test_artifact_manifest_creation(self, trained_crypto_model):
        """Test artifact manifest creation with comprehensive metadata."""
        model_data = trained_crypto_model
        model = model_data['model']
        X_test = model_data['X_test'][:50]  # Small sample
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "test_model.onnx"
            
            # Export to ONNX
            to_onnx(model, feature_names, onnx_path)
            
            # Validate parity
            parity_results = validate_onnx_parity(
                model, onnx_path, X_test, tolerance=1e-5
            )
            
            # Create manifest
            training_metadata = {
                'algorithm': 'Ridge',
                'cv_folds': 5,
                'train_samples': 700,
                'test_samples': 300,
                'random_seed': 42,
            }
            
            manifest = create_model_artifact_manifest(
                model, onnx_path, feature_names, training_metadata, parity_results
            )
            
            # Verify manifest structure
            assert 'artifact_version' in manifest
            assert 'created_at' in manifest
            assert 'model' in manifest
            assert 'features' in manifest
            assert 'training' in manifest
            assert 'parity' in manifest
            assert 'deployment' in manifest
            assert 'quality_gates' in manifest
            
            # Verify model hash is generated
            assert len(manifest['model']['onnx_hash']) == 16
            
            # Verify feature hash is generated
            assert len(manifest['features']['hash']) == 12


class TestRegressionPrevention:
    """Tests to prevent regressions in model export."""
    
    def test_dataframe_vs_numpy_parity(self, trained_crypto_model):
        """Test that DataFrame and numpy inputs produce identical results."""
        model_data = trained_crypto_model
        model = model_data['model']
        X_test = model_data['X_test']
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "test_model.onnx"
            
            # Export to ONNX
            to_onnx(model, feature_names, onnx_path)
            
            # Test with DataFrame
            df_predictions = onnx_predict(onnx_path, X_test)
            
            # Test with numpy array
            array_predictions = onnx_predict(onnx_path, X_test.values)
            
            # Should be identical
            np.testing.assert_array_equal(
                df_predictions, array_predictions,
                err_msg="DataFrame and numpy array inputs must produce identical results"
            )
    
    def test_single_sample_prediction(self, trained_crypto_model):
        """Test prediction on single samples."""
        model_data = trained_crypto_model
        model = model_data['model']
        X_test = model_data['X_test']
        feature_names = model_data['feature_names']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "test_model.onnx"
            
            # Export to ONNX
            to_onnx(model, feature_names, onnx_path)
            
            # Test single sample (1D input)
            single_sample = X_test.iloc[0].values
            prediction = onnx_predict(onnx_path, single_sample)
            
            # Verify output
            assert prediction.shape == (1,)
            assert not np.isnan(prediction[0])
            
            # Compare with native model
            native_pred = model.predict(single_sample.reshape(1, -1))
            np.testing.assert_allclose(
                prediction, native_pred, atol=1e-5,
                err_msg="Single sample prediction must match native model"
            )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
