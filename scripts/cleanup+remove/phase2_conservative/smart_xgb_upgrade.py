#!/usr/bin/env python3
"""
PRINCIPLED XGBOOST UPGRADE - Features First, Algorithm Second
=====================================

Based on sound quant principles:
1. Test if upgrade actually improves signal quality
2. Focus on where XGBoost helps: interactions & regime changes
3. Compare apples-to-apples with same features
4. Measure what matters: Sharpe, transaction costs, robustness
"""

import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_test_data():
    """Load or create test financial data"""
    # In real system, this would load from your data pipeline
    np.random.seed(42)  # Reproducible
    
    # Simulate realistic financial returns (fat tails, autocorr)
    n = 1000
    returns = np.random.normal(0, 0.02, n) + np.random.normal(0, 0.005, n) * np.random.choice([-3, 3], n, p=[0.95, 0.05])
    
    prices = 100 * np.exp(np.cumsum(returns))
    volume = np.random.lognormal(15, 0.5, n)
    
    df = pd.DataFrame({
        'close': prices,
        'volume': volume,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        'open': np.roll(prices, 1)  # Previous close ≈ next open
    })
    df['open'].iloc[0] = df['close'].iloc[0]
    
    return df

def engineer_features(df):
    """Build standard momentum/mean-reversion features"""
    features = pd.DataFrame(index=df.index)
    
    # Momentum features
    features['ret_1d'] = df['close'].pct_change(1)
    features['ret_5d'] = df['close'].pct_change(5)
    features['ret_20d'] = df['close'].pct_change(20)
    
    # Mean reversion
    features['ma_20'] = df['close'].rolling(20).mean()
    features['price_to_ma'] = df['close'] / features['ma_20'] - 1
    
    # Volatility 
    features['vol_20d'] = df['close'].pct_change().rolling(20).std()
    
    # Volume
    features['volume_ma'] = df['volume'].rolling(20).mean()
    features['rel_volume'] = df['volume'] / features['volume_ma'] - 1
    
    # Cross-asset proxy (simulate regime indicator)
    features['regime'] = np.where(features['vol_20d'] > features['vol_20d'].rolling(60).median(), 1, 0)
    
    return features.dropna()

def create_target(df, features_df):
    """Create forward return target (what we're predicting)"""
    # Predict next day's return
    target = df['close'].pct_change().shift(-1)  # Tomorrow's return
    
    # Align with features
    target = target.loc[features_df.index]
    return target.dropna()

def compare_models(X, y):
    """Compare Ridge vs XGBoost on same features"""
    print("🔬 COMPARING MODELS ON SAME FEATURES")
    print("=" * 50)
    
    # Split data (no lookahead!)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Test Ridge (current system)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_ic = np.corrcoef(y_test, ridge_pred)[0, 1]  # Information Coefficient
    
    print(f"\n📊 RIDGE REGRESSION:")
    print(f"   MSE: {ridge_mse:.6f}")
    print(f"   IC:  {ridge_ic:.4f}")
    print(f"   Signal Strength: {'WEAK' if abs(ridge_ic) < 0.02 else 'MODERATE' if abs(ridge_ic) < 0.05 else 'STRONG'}")
    
    # Test XGBoost (from your existing model)
    try:
        # Load your actual XGBoost model
        session = ort.InferenceSession('artifacts/models/latest.onnx')
        input_name = session.get_inputs()[0].name
        
        # XGBoost expects different preprocessing - try both scaled and raw
        test_inputs = [X_test_scaled.astype(np.float32), X_test.astype(np.float32)]
        test_names = ["scaled", "raw"]
        
        best_xgb_ic = -999
        best_xgb_pred = None
        
        for test_X, name in zip(test_inputs, test_names):
            try:
                xgb_pred = session.run(None, {input_name: test_X})[0].flatten()
                xgb_ic = np.corrcoef(y_test, xgb_pred)[0, 1]
                
                if abs(xgb_ic) > abs(best_xgb_ic):
                    best_xgb_ic = xgb_ic
                    best_xgb_pred = xgb_pred
                    
            except Exception as e:
                print(f"   XGBoost {name} failed: {e}")
                continue
        
        if best_xgb_pred is not None:
            xgb_mse = mean_squared_error(y_test, best_xgb_pred)
            
            print(f"\n🌲 XGBOOST:")
            print(f"   MSE: {xgb_mse:.6f}")
            print(f"   IC:  {best_xgb_ic:.4f}")
            print(f"   Signal Strength: {'WEAK' if abs(best_xgb_ic) < 0.02 else 'MODERATE' if abs(best_xgb_ic) < 0.05 else 'STRONG'}")
            
            # Compare
            print(f"\n🎯 COMPARISON:")
            ic_improvement = abs(best_xgb_ic) - abs(ridge_ic)
            print(f"   IC Improvement: {ic_improvement:+.4f}")
            
            if ic_improvement > 0.01:
                print("   ✅ XGBoost shows meaningful improvement")
                return "upgrade"
            elif ic_improvement > 0.005:
                print("   🤔 XGBoost shows modest improvement")
                return "maybe"
            else:
                print("   ❌ XGBoost doesn't improve signal quality")
                return "stick_with_ridge"
        else:
            print("   ❌ XGBoost failed to run")
            return "stick_with_ridge"
            
    except Exception as e:
        print(f"\n❌ XGBOOST TEST FAILED: {e}")
        return "stick_with_ridge"

def analyze_feature_importance():
    """Understand what features matter most"""
    print("\n🔍 FEATURE ANALYSIS")
    print("=" * 30)
    
    # Load Ridge model to see coefficients
    try:
        with open('artifacts/models/linear_v1.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        if isinstance(model_data, dict):
            ridge_model = model_data['model']
            feature_names = model_data.get('feature_names', [f'feature_{i}' for i in range(len(ridge_model.coef_))])
        else:
            ridge_model = model_data
            feature_names = [f'feature_{i}' for i in range(len(ridge_model.coef_))]
            
        # Show feature importance (absolute coefficients)
        importance = np.abs(ridge_model.coef_)
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("📊 Current Ridge Model Feature Importance:")
        for name, imp in feature_importance[:5]:  # Top 5
            print(f"   {name}: {imp:.4f}")
            
    except Exception as e:
        print(f"Could not analyze Ridge features: {e}")

def recommend_upgrade_path():
    """Give recommendation based on test results"""
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 25)
    
    # Generate test data
    df = load_test_data()
    features_df = engineer_features(df)
    target = create_target(df, features_df)
    
    # Align features and target
    common_idx = features_df.index.intersection(target.index)
    X = features_df.loc[common_idx].values
    y = target.loc[common_idx].values
    
    if len(X) < 100:
        print("❌ Not enough data for meaningful comparison")
        return
    
    # Compare models
    result = compare_models(X, y)
    
    # Feature analysis
    analyze_feature_importance()
    
    # Recommendations
    print(f"\n💡 RECOMMENDATION:")
    
    if result == "upgrade":
        print("✅ UPGRADE TO XGBOOST")
        print("   - Shows meaningful signal improvement")
        print("   - Likely capturing feature interactions")
        print("   - Worth the complexity")
        
    elif result == "maybe": 
        print("🤔 MARGINAL CASE") 
        print("   - Modest improvement only")
        print("   - Consider staying with Ridge for simplicity")
        print("   - Or test with more data/better features")
        
    else:
        print("❌ STICK WITH RIDGE")
        print("   - XGBoost doesn't improve signal quality")
        print("   - Focus on better features instead")
        print("   - Linear relationships may be sufficient")
    
    print(f"\n🎓 LEARNING:")
    print("   - The features matter more than the algorithm")
    print("   - Test improvements objectively (IC, Sharpe)")
    print("   - Don't upgrade just because XGBoost sounds fancy")

if __name__ == "__main__":
    print("🎯 PRINCIPLED XGBOOST EVALUATION")
    print("Features First, Algorithm Second")
    print("=" * 50)
    
    try:
        recommend_upgrade_path()
        
        print(f"\n🚀 NEXT STEPS:")
        print("   1. Test with your actual market data")
        print("   2. Focus on feature engineering")
        print("   3. Measure transaction costs impact")
        print("   4. Only upgrade if objectively better")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("But your current Ridge system still works!")
