#!/usr/bin/env python3
"""
Demo: ONNX Model Export and Validation

Demonstrates how the Aurora system exports trained models to ONNX format
for production deployment with cross-platform compatibility.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path  
sys.path.insert(0, ".")

def create_sample_model():
    """Create a sample trained model for demo purposes."""
    
    print("🤖 Creating sample model...")
    
    try:
        from sklearn.linear_model import Ridge
        from sklearn.datasets import make_regression
        import pickle
        
        # Generate sample data
        X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
        feature_names = ['returns', 'sma_ratio', 'volatility', 'volume_ratio', 'momentum']
        
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Train model
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_df, y)
        
        # Save model
        Path("models").mkdir(exist_ok=True)
        model_path = "models/demo_ridge.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  ✅ Model trained and saved to {model_path}")
        print(f"  📊 R² Score: {model.score(X_df, y):.4f}")
        
        return model_path, X_df[:10]  # Return path and sample data
        
    except ImportError as e:
        print(f"  ❌ Missing dependency: {e}")
        return None, None

def export_to_onnx(model_path: str, sample_data: pd.DataFrame, output_path: str = None):
    """Export a scikit-learn model to ONNX format."""
    
    if output_path is None:
        output_path = model_path.replace('.pkl', '.onnx')
    
    print("\n📦 Exporting model to ONNX...")
    print(f"  📂 Input: {model_path}")
    print(f"  📂 Output: {output_path}")
    
    try:
        import pickle
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Define input types
        initial_type = [('float_input', FloatTensorType([None, sample_data.shape[1]]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Save ONNX model
        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print("  ✅ ONNX export successful!")
        print(f"  📊 Model size: {Path(output_path).stat().st_size / 1024:.1f} KB")
        
        return output_path
        
    except ImportError as e:
        print(f"  ❌ ONNX export failed - missing dependency: {e}")
        print("  💡 Install with: pip install skl2onnx onnx onnxruntime")
        return None
    except Exception as e:
        print(f"  ❌ ONNX export failed: {e}")
        return None

def validate_onnx_parity(original_path: str, onnx_path: str, test_data: pd.DataFrame):
    """Validate that ONNX model produces identical predictions to original."""
    
    print("\n🔍 Validating ONNX parity...")
    
    try:
        import pickle
        import onnxruntime as ort
        
        # Load original model
        with open(original_path, 'rb') as f:
            original_model = pickle.load(f)
        
        # Load ONNX model
        onnx_session = ort.InferenceSession(onnx_path)
        
        # Get predictions from original model
        original_pred = original_model.predict(test_data)
        
        # Get predictions from ONNX model
        input_name = onnx_session.get_inputs()[0].name
        onnx_pred = onnx_session.run(None, {input_name: test_data.values.astype(np.float32)})[0]
        
        # Compare predictions
        max_diff = np.max(np.abs(original_pred - onnx_pred.flatten()))
        mean_diff = np.mean(np.abs(original_pred - onnx_pred.flatten()))
        
        print(f"  📊 Max difference: {max_diff:.8f}")
        print(f"  📊 Mean difference: {mean_diff:.8f}")
        
        # Check if differences are within acceptable tolerance
        tolerance = 1e-5
        is_parity = max_diff < tolerance
        
        if is_parity:
            print(f"  ✅ ONNX parity validation PASSED (tolerance: {tolerance})")
        else:
            print(f"  ❌ ONNX parity validation FAILED (tolerance: {tolerance})")
        
        return is_parity
        
    except ImportError as e:
        print(f"  ❌ Parity validation failed - missing dependency: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Parity validation failed: {e}")
        return False

def demo_complete_pipeline():
    """Run the complete ONNX export and validation pipeline."""
    
    print("🚀 ONNX Export Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Create sample model
    model_path, sample_data = create_sample_model()
    
    if not model_path:
        print("❌ Cannot proceed without a trained model")
        return False
    
    # Step 2: Export to ONNX
    onnx_path = export_to_onnx(model_path, sample_data)
    
    if not onnx_path:
        print("❌ Cannot proceed without ONNX export")
        return False
    
    # Step 3: Validate parity
    is_parity = validate_onnx_parity(model_path, onnx_path, sample_data)
    
    # Summary
    print("\n📋 Pipeline Summary:")
    print("  ✅ Model training: Success")
    print("  ✅ ONNX export: Success") 
    print(f"  {'✅' if is_parity else '❌'} Parity validation: {'Success' if is_parity else 'Failed'}")
    
    if is_parity:
        print("\n🎉 Complete ONNX pipeline successful!")
        print("  💡 ONNX model ready for production deployment")
    else:
        print("\n⚠️  ONNX pipeline completed with warnings")
    
    return is_parity

def main():
    """Main entry point for ONNX demo."""
    
    try:
        success = demo_complete_pipeline()
        
        print("\n📖 About ONNX Export:")
        print("  🔄 Converts ML models to standardized format")
        print("  🌐 Enables cross-platform deployment")
        print("  ⚡ Optimized for inference performance")
        print("  🔒 Ensures prediction consistency")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
