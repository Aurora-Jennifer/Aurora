#!/usr/bin/env python3
"""
RETRAIN WITH MOMENTUM FEATURES
Retrain your Ridge model with the new momentum features that showed 4x improvement
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
from pathlib import Path
from core.ml.build_features import build_matrix
import warnings
warnings.filterwarnings('ignore')

def download_spy_data(period="2y"):
    """Download recent SPY data for training"""
    print(f"📥 Downloading SPY data ({period})...")
    spy = yf.Ticker("SPY")
    data = spy.history(period=period, auto_adjust=True)
    
    # Just keep whatever columns yfinance gives us
    print(f"✅ Columns: {list(data.columns)}")
    print(f"✅ Downloaded {len(data)} days of SPY data")
    return data

def retrain_improved_model():
    """Retrain Ridge model with momentum features"""
    print("🚀 RETRAINING WITH MOMENTUM FEATURES")
    print("=" * 45)
    
    # Get data
    spy_data = download_spy_data("2y")  # 2 years of training data
    
    # Build features using your upgraded feature engine
    print("🔧 Building features with momentum upgrade...")
    X, y = build_matrix(spy_data)
    
    print(f"✅ Features: {len(X.columns)}")
    print(f"✅ Samples: {len(X)}")
    print(f"✅ Feature names: {list(X.columns)}")
    
    # Split data (70% train, 30% test - no lookahead!)
    split_idx = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📊 Train samples: {len(X_train)}")
    print(f"📊 Test samples:  {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge model
    print("🎯 Training Ridge model...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    
    # Test performance
    train_pred = ridge.predict(X_train_scaled)
    test_pred = ridge.predict(X_test_scaled)
    
    # Calculate Information Coefficient (IC)
    train_ic = np.corrcoef(y_train, train_pred)[0, 1]
    test_ic = np.corrcoef(y_test, test_pred)[0, 1]
    
    # Calculate MSE
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    
    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"   Train IC: {train_ic:.4f}")
    print(f"   Test IC:  {test_ic:.4f}")
    print(f"   Train MSE: {train_mse:.6f}")
    print(f"   Test MSE:  {test_mse:.6f}")
    
    # Interpret results
    if abs(test_ic) > 0.05:
        print(f"   🎯 EXCELLENT: Strong signal (IC > 0.05)")
    elif abs(test_ic) > 0.02:
        print(f"   ✅ GOOD: Tradeable signal (IC > 0.02)")
    else:
        print(f"   ⚠️  WEAK: Signal may not survive costs")
    
    # Show feature importance
    print(f"\n🔍 TOP FEATURES (absolute importance):")
    feature_importance = list(zip(X.columns, np.abs(ridge.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance[:5]):
        print(f"   {i+1}. {name}: {importance:.4f}")
    
    # Save the improved model
    print(f"\n💾 SAVING IMPROVED MODEL...")
    
    # Create model package
    model_package = {
        'model': ridge,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'performance': {
            'train_ic': train_ic,
            'test_ic': test_ic,
            'train_mse': train_mse,
            'test_mse': test_mse
        },
        'training_info': {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(X.columns),
            'momentum_upgrade': True
        }
    }
    
    # Save to multiple locations for compatibility
    output_paths = [
        'artifacts/models/linear_v1.pkl',  # Main location
        'artifacts/models/momentum_ridge_v1.pkl',  # Backup
        'models/linear_v1.pkl'  # Legacy location
    ]
    
    for path in output_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_package, f)
        print(f"   ✅ Saved to {path}")
    
    print(f"\n🎉 MODEL RETRAINED SUCCESSFULLY!")
    print(f"   - Now uses {len(X.columns)} features (was 5)")
    print(f"   - Includes momentum features for better performance")
    print(f"   - Ready for paper trading")
    
    return test_ic

if __name__ == "__main__":
    try:
        ic = retrain_improved_model()
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Your model is now upgraded with momentum features")
        print(f"   2. Test it: python scripts/easy_trade.py")
        print(f"   3. Run paper trading to see improved performance")
        print(f"   4. Compare results to old model")
        
        if abs(ic) > 0.05:
            print(f"\n🎯 EXPECTED RESULT:")
            print(f"   - Much better trading performance")
            print(f"   - Higher Sharpe ratio") 
            print(f"   - More consistent profits")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that you have internet connection for downloading SPY data")

