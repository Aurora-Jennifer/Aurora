#!/usr/bin/env python3
"""
ONNX Export and Parity Testing for Crypto Models

Provides production-safe model export with rigorous parity validation.
Ensures ONNX models produce identical predictions to native models.
"""

import logging
import json
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from sklearn.base import BaseEstimator
import joblib
from datetime import UTC

logger = logging.getLogger(__name__)


def to_onnx(
    model: BaseEstimator,
    feature_names: list[str],
    output_path: str | Path,
    model_name: str = "crypto_model",
    opset_version: int = 11
) -> dict[str, Any]:
    """
    Export sklearn model to ONNX format with full metadata.
    
    Args:
        model: Trained sklearn model
        feature_names: List of feature names in prediction order
        output_path: Path to save ONNX model
        model_name: Name for the ONNX model
        opset_version: ONNX opset version
    
    Returns:
        Export metadata dictionary
    """
    try:
        # Import skl2onnx (lazy import to avoid dependency issues)
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Define input schema
        n_features = len(feature_names)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=opset_version,
            final_types=None,
            options=None
        )
        
        # Set model metadata
        onnx_model.doc_string = f"Crypto model: {model_name}"
        onnx_model.model_version = 1
        onnx_model.producer_name = "aurora_crypto_pipeline"
        
        # Add feature names to metadata
        meta_props = onnx_model.metadata_props
        meta_props.append(onnx.StringStringEntryProto(key="feature_names", value=json.dumps(feature_names)))
        meta_props.append(onnx.StringStringEntryProto(key="model_type", value=type(model).__name__))
        
        # Save ONNX model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        onnx.save(onnx_model, str(output_path))
        
        # Create export metadata
        metadata = {
            'export_path': str(output_path),
            'model_name': model_name,
            'model_type': type(model).__name__,
            'feature_names': feature_names,
            'n_features': n_features,
            'opset_version': opset_version,
            'onnx_version': onnx.__version__,
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
        }
        
        logger.info(f"✅ ONNX export successful: {output_path}")
        logger.info(f"   Model: {model_name} ({type(model).__name__})")
        logger.info(f"   Features: {n_features}")
        logger.info(f"   Opset: {opset_version}")
        
        return metadata
        
    except ImportError as e:
        raise ImportError(f"ONNX export requires skl2onnx: pip install skl2onnx onnx onnxruntime\n{e}")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise


def onnx_predict(
    onnx_path: str | Path,
    X: np.ndarray | pd.DataFrame,
    session_options: ort.SessionOptions | None = None
) -> np.ndarray:
    """
    Make predictions using ONNX model.
    
    Args:
        onnx_path: Path to ONNX model file
        X: Input features (samples x features)
        session_options: ONNX runtime session options
    
    Returns:
        Predictions as numpy array
    """
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float32)
        else:
            X_array = np.asarray(X, dtype=np.float32)
        
        # Ensure 2D array
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        # Create ONNX runtime session
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3  # ERROR level
            
        session = ort.InferenceSession(str(onnx_path), session_options)
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        predictions = session.run(
            [output_name], 
            {input_name: X_array}
        )[0]
        
        # Flatten if single output
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
            
        return predictions
        
    except Exception as e:
        logger.error(f"ONNX prediction failed: {e}")
        raise


def validate_onnx_parity(
    native_model: BaseEstimator,
    onnx_path: str | Path,
    X_test: np.ndarray | pd.DataFrame,
    tolerance: float = 1e-5,
    n_samples: int | None = None,
    random_seed: int = 42
) -> dict[str, Any]:
    """
    Validate that ONNX model produces identical predictions to native model.
    
    Args:
        native_model: Original sklearn model
        onnx_path: Path to ONNX model
        X_test: Test features for validation
        tolerance: Maximum allowed absolute difference
        n_samples: Number of random samples to test (None = use all)
        random_seed: Random seed for sampling
    
    Returns:
        Validation results dictionary
    """
    try:
        # Sample test data if requested
        if n_samples is not None and len(X_test) > n_samples:
            np.random.seed(random_seed)
            idx = np.random.choice(len(X_test), size=n_samples, replace=False)
            if isinstance(X_test, pd.DataFrame):
                X_sample = X_test.iloc[idx]
            else:
                X_sample = X_test[idx]
        else:
            X_sample = X_test
            n_samples = len(X_test)
        
        # Get native predictions
        y_native = native_model.predict(X_sample)
        
        # Get ONNX predictions
        y_onnx = onnx_predict(onnx_path, X_sample)
        
        # Ensure same shape
        if y_native.shape != y_onnx.shape:
            raise ValueError(f"Shape mismatch: native {y_native.shape} vs ONNX {y_onnx.shape}")
        
        # Calculate differences
        abs_diff = np.abs(y_native - y_onnx)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        # Check parity
        is_parity = max_diff <= tolerance
        
        # Create validation results
        results = {
            'is_parity': bool(is_parity),
            'n_samples_tested': int(n_samples),
            'max_absolute_difference': float(max_diff),
            'mean_absolute_difference': float(mean_diff),
            'tolerance': float(tolerance),
            'native_predictions_sample': y_native[:5].tolist() if len(y_native) >= 5 else y_native.tolist(),
            'onnx_predictions_sample': y_onnx[:5].tolist() if len(y_onnx) >= 5 else y_onnx.tolist(),
            'differences_sample': abs_diff[:5].tolist() if len(abs_diff) >= 5 else abs_diff.tolist(),
        }
        
        # Log results
        if is_parity:
            logger.info("✅ ONNX parity validation PASSED")
            logger.info(f"   Samples tested: {n_samples}")
            logger.info(f"   Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
            logger.info(f"   Mean difference: {mean_diff:.2e}")
        else:
            logger.error("❌ ONNX parity validation FAILED")
            logger.error(f"   Max difference: {max_diff:.2e} > tolerance: {tolerance:.2e}")
            logger.error(f"   Mean difference: {mean_diff:.2e}")
            logger.error(f"   Sample differences: {abs_diff[:5]}")
        
        return results
        
    except Exception as e:
        logger.error(f"ONNX parity validation error: {e}")
        raise


def create_model_artifact_manifest(
    native_model: BaseEstimator,
    onnx_path: str | Path,
    feature_names: list[str],
    training_metadata: dict[str, Any],
    parity_results: dict[str, Any]
) -> dict[str, Any]:
    """
    Create comprehensive artifact manifest for model deployment.
    
    Args:
        native_model: Trained sklearn model
        onnx_path: Path to ONNX model
        feature_names: Feature names in order
        training_metadata: Metadata from training process
        parity_results: Results from parity validation
    
    Returns:
        Complete artifact manifest
    """
    import hashlib
    from datetime import datetime
    
    # Calculate model hash for tracking
    with open(onnx_path, 'rb') as f:
        onnx_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    
    # Get model parameters
    model_params = native_model.get_params() if hasattr(native_model, 'get_params') else {}
    
    manifest = {
        'artifact_version': '1.0',
        'created_at': datetime.now(UTC).isoformat(),
        'model': {
            'type': type(native_model).__name__,
            'parameters': model_params,
            'onnx_hash': onnx_hash,
            'onnx_path': str(onnx_path),
        },
        'features': {
            'names': feature_names,
            'count': len(feature_names),
            'hash': hashlib.sha256('|'.join(feature_names).encode()).hexdigest()[:12],
        },
        'training': training_metadata,
        'parity': parity_results,
        'deployment': {
            'requires_preprocessing': True,
            'input_dtype': 'float32',
            'output_dtype': 'float32',
            'runtime': 'onnxruntime',
        },
        'quality_gates': {
            'parity_passed': parity_results.get('is_parity', False),
            'max_inference_time_ms': None,  # To be filled by performance tests
            'memory_usage_mb': None,  # To be filled by performance tests
        }
    }
    
    return manifest


def export_crypto_model_complete(
    model: BaseEstimator,
    scaler: BaseEstimator | None,
    feature_names: list[str],
    X_test: np.ndarray | pd.DataFrame,
    output_dir: str | Path,
    model_name: str = "crypto_v1",
    tolerance: float = 1e-5
) -> dict[str, Any]:
    """
    Complete model export pipeline with all validation and artifacts.
    
    Args:
        model: Trained model
        scaler: Feature scaler (if used)
        feature_names: Feature names
        X_test: Test data for parity validation
        output_dir: Directory to save all artifacts
        model_name: Model name
        tolerance: Parity tolerance
    
    Returns:
        Complete export results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Export ONNX model
        onnx_path = output_dir / f"{model_name}.onnx"
        export_metadata = to_onnx(model, feature_names, onnx_path, model_name)
        
        # Save native model and scaler
        native_path = output_dir / f"{model_name}_native.joblib"
        artifact = {'model': model, 'scaler': scaler, 'feature_names': feature_names}
        joblib.dump(artifact, native_path)
        
        # Validate parity
        parity_results = validate_onnx_parity(
            model, onnx_path, X_test, tolerance=tolerance
        )
        
        # Create manifest
        training_metadata = {
            'model_type': type(model).__name__,
            'feature_count': len(feature_names),
            'scaler_used': scaler is not None,
        }
        
        manifest = create_model_artifact_manifest(
            model, onnx_path, feature_names, training_metadata, parity_results
        )
        
        # Save manifest
        manifest_path = output_dir / f"{model_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Combine all results
        results = {
            'export_successful': True,
            'parity_passed': parity_results['is_parity'],
            'onnx_path': str(onnx_path),
            'native_path': str(native_path),
            'manifest_path': str(manifest_path),
            'export_metadata': export_metadata,
            'parity_results': parity_results,
            'manifest': manifest,
        }
        
        logger.info(f"✅ Complete model export successful: {output_dir}")
        logger.info(f"   ONNX: {onnx_path.name}")
        logger.info(f"   Native: {native_path.name}")
        logger.info(f"   Manifest: {manifest_path.name}")
        logger.info(f"   Parity: {'PASSED' if parity_results['is_parity'] else 'FAILED'}")
        
        return results
        
    except Exception as e:
        logger.error(f"Complete model export failed: {e}")
        raise
