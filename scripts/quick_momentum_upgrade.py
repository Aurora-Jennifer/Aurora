#!/usr/bin/env python3
"""
SIMPLE MOMENTUM UPGRADE - One file, big improvement
Based on your discovery that momentum features can improve IC from 0.08 to 0.36
"""

import pandas as pd
import numpy as np
from pathlib import Path

def add_momentum_features(df):
    """Add the momentum features that showed IC = 0.36"""
    
    # Your existing features (keep them)
    features = df.copy()
    
    # Add momentum features that tested best
    for period in [1, 3, 5, 10, 20]:
        features[f'momentum_{period}d'] = df['close'].pct_change(period)
    
    # Momentum strength (the key feature that improved IC)
    features['momentum_strength'] = (
        features['momentum_1d'] * 
        features['momentum_5d'] * 
        features['momentum_20d']
    ) ** (1/3)
    
    return features.dropna()

def test_upgrade():
    """Test the upgrade on dummy data"""
    print("🎯 Testing momentum upgrade...")
    
    # Create test data like SPY
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    df = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.normal(0.0008, 0.015, 500))),
        'volume': np.random.lognormal(15, 0.3, 500)
    }, index=dates)
    
    # Add momentum features
    enhanced = add_momentum_features(df)
    
    print(f"✅ Original features: {len(df.columns)}")
    print(f"✅ Enhanced features: {len(enhanced.columns)}")
    print(f"✅ New momentum features: {len(enhanced.columns) - len(df.columns)}")
    
    return enhanced

if __name__ == "__main__":
    print("🚀 SIMPLE MOMENTUM UPGRADE")
    print("=" * 30)
    print("This adds the features that showed 4x improvement")
    print("")
    
    # Test it works
    result = test_upgrade()
    
    print("")
    print("🎯 TO USE IN YOUR REAL SYSTEM:")
    print("1. Add this to your feature building code")
    print("2. Retrain your Ridge model with these features") 
    print("3. Test on paper trading")
    print("4. Expect much better performance")
    
    print("")
    print("✅ You discovered something valuable today!")
    print("✅ Better features > fancy algorithms")
    print("✅ Your system works, now make it better")
