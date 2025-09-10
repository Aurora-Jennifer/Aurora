#!/usr/bin/env python3
"""
Test script to load and validate the XGBoost model for paper trading.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """Test loading the XGBoost model and feature whitelist."""
    
    print("🧪 Testing XGBoost model loading...")
    
    # Paths
    model_path = project_root / "results" / "production" / "model.json"
    features_path = project_root / "results" / "production" / "features_whitelist.json"
    
    # Check files exist
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
        
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Features file found: {features_path}")
    
    # Load features whitelist
    try:
        with open(features_path, 'r') as f:
            features_config = json.load(f)
        features_whitelist = features_config['feature_cols']
        print(f"✅ Loaded {len(features_whitelist)} features from whitelist")
        print(f"   Features: {features_whitelist[:5]}...")  # Show first 5
        print(f"   Total panel cols: {features_config['total_panel_cols']}")
        print(f"   Feature count: {features_config['feature_count']}")
    except Exception as e:
        print(f"❌ Failed to load features whitelist: {e}")
        return False
    
    # Load XGBoost model
    try:
        model = xgb.Booster()
        model.load_model(str(model_path))
        print(f"✅ XGBoost model loaded successfully")
        
        # Get model info
        print(f"   Model type: {type(model)}")
        print(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")
        
    except Exception as e:
        print(f"❌ Failed to load XGBoost model: {e}")
        return False
    
    # Test prediction with dummy data
    try:
        # Create dummy feature data matching the whitelist
        n_features = len(features_whitelist)
        dummy_data = np.random.randn(10, n_features)  # 10 samples, n_features columns
        
        # Create DataFrame with proper feature names
        dummy_df = pd.DataFrame(dummy_data, columns=features_whitelist)
        
        # Convert to DMatrix (XGBoost format) with feature names
        dmatrix = xgb.DMatrix(dummy_df)
        
        # Make predictions
        predictions = model.predict(dmatrix)
        
        print(f"✅ Model prediction test successful")
        print(f"   Input shape: {dummy_data.shape}")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[:3]}")
        
    except Exception as e:
        print(f"❌ Model prediction test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Model is ready for paper trading.")
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
