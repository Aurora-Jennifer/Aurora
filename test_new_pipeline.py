#!/usr/bin/env python3
"""
Test script for the new ML pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ml.targets import create_targets, validate_targets
from ml.decision import make_decisions, calibrate_decision_parameters
from ml.baselines import RidgeExcessModel, buy_and_hold_daily_pnl, simple_rule_daily_pnl
from ml.features import create_feature_pipeline
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler


def test_new_pipeline():
    """Test the new ML pipeline end-to-end"""
    print("🧪 Testing new ML pipeline...")
    
    # Download data
    print("📥 Downloading data...")
    qqq = yf.download('QQQ', period='500d', progress=False)
    spy = yf.download('SPY', period='500d', progress=False)
    
    # Flatten MultiIndex columns
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    print(f"QQQ shape: {qqq.shape}, SPY shape: {spy.shape}")
    
    # Create features
    print("🔧 Creating features...")
    features = create_feature_pipeline(qqq)
    print(f"Features shape: {features.shape}")
    
    # Calculate returns
    asset_ret = qqq['Close'].pct_change().dropna()
    market_ret = spy['Close'].pct_change().dropna()
    
    # Align data
    common_idx = asset_ret.index.intersection(market_ret.index)
    asset_ret = asset_ret.loc[common_idx]
    market_ret = market_ret.loc[common_idx]
    features = features.loc[common_idx]
    
    print(f"Aligned data shape: {len(common_idx)}")
    
    # Create targets
    print("🎯 Creating targets...")
    train_idx = common_idx[:int(len(common_idx) * 0.7)]
    labels, targets, eps = create_targets(asset_ret, market_ret, H=5, train_idx=train_idx)
    print(f"Epsilon: {eps:.4f}")
    print(f"Label distribution: {labels.value_counts().to_dict()}")
    
    # Validate targets
    validate_targets(labels, targets, eps)
    print("✅ Targets validated")
    
    # Create and train model
    print("🤖 Training model...")
    model = RidgeExcessModel(alpha=0.1)
    train_features = features.loc[train_idx]
    train_targets = targets.loc[train_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_features)
    X_scaled_df = pd.DataFrame(X_scaled, columns=train_features.columns, index=train_features.index)
    
    model.fit(X_scaled_df, train_targets)
    print("✅ Model trained")
    
    # Test predictions
    print("🔮 Testing predictions...")
    test_idx = common_idx[int(len(common_idx) * 0.7):]
    test_features = features.loc[test_idx]
    X_test_scaled = scaler.transform(test_features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=test_features.columns, index=test_features.index)
    
    edges = model.predict_edge(X_test_scaled_df)
    print(f"Edge stats: mean={edges.mean():.4f}, std={edges.std():.4f}")
    
    if np.std(edges) > 1e-6:
        print("✅ Edges have variation")
        
        # Test decision making
        print("🎛️ Testing decision making...")
        decision_params = calibrate_decision_parameters(edges, (0.08, 0.18))
        print(f"Decision params: {decision_params}")
        
        # Make decisions
        positions, edges_out, proba = make_decisions(model, X_test_scaled_df, decision_params, temperature=1.0)
        print(f"Positions: {np.bincount(positions + 1)}")  # Count [-1,0,1] as [0,1,2]
        print("✅ Decision making working")
        
        # Test baselines
        print("📊 Testing baselines...")
        test_prices = qqq['Close'].loc[test_idx]
        
        bh_pnl = buy_and_hold_daily_pnl(test_prices)
        bh_sharpe = np.sqrt(252) * bh_pnl.mean() / bh_pnl.std() if bh_pnl.std() > 0 else 0
        
        rule_pnl = simple_rule_daily_pnl(test_prices)
        rule_sharpe = np.sqrt(252) * rule_pnl.mean() / rule_pnl.std() if rule_pnl.std() > 0 else 0
        
        print(f"Buy & Hold Sharpe: {bh_sharpe:.3f}")
        print(f"Simple Rule Sharpe: {rule_sharpe:.3f}")
        print("✅ Baselines working")
        
        # Test model performance
        print("📈 Testing model performance...")
        test_returns = asset_ret.loc[test_idx]
        model_pnl = positions * test_returns
        model_sharpe = np.sqrt(252) * model_pnl.mean() / model_pnl.std() if model_pnl.std() > 0 else 0
        
        print(f"Model Sharpe: {model_sharpe:.3f}")
        
        # Check if model beats baselines
        best_baseline = max(bh_sharpe, rule_sharpe)
        if model_sharpe > best_baseline + 0.1:
            print(f"🎉 Model beats baselines by {model_sharpe - best_baseline:.3f}!")
        else:
            print(f"📉 Model underperforms baselines by {best_baseline - model_sharpe:.3f}")
        
        print("✅ Pipeline test completed successfully!")
        return True
    else:
        print("❌ Edges are constant - pipeline needs debugging")
        return False


if __name__ == "__main__":
    success = test_new_pipeline()
    if success:
        print("\n🎉 New ML pipeline is working correctly!")
    else:
        print("\n❌ New ML pipeline has issues that need fixing")
        sys.exit(1)
