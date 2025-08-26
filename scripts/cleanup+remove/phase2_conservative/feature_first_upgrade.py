#!/usr/bin/env python3
"""
FEATURE-FIRST TRADING IMPROVEMENT
Focus on what actually matters: signal quality

Based on quant principles:
1. Improve features first (momentum, mean reversion, regime detection)
2. Test signal strength objectively  
3. Only change algorithm if features are maxed out
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def test_feature_improvements():
    """Test different feature engineering approaches"""
    print("🔬 TESTING FEATURE IMPROVEMENTS")
    print("=" * 40)
    
    # This would use your real data - simulating for now
    np.random.seed(42)
    n = 1000
    
    # Simulate SPY-like data
    returns = np.random.normal(0.0008, 0.015, n)  # ~20% annual vol
    returns += 0.3 * np.roll(returns, 1)  # Some autocorrelation
    
    prices = 100 * np.exp(np.cumsum(returns))
    volume = np.random.lognormal(15, 0.3, n)
    
    df = pd.DataFrame({
        'close': prices,
        'volume': volume,
        'returns': returns
    })
    
    # Test different feature sets
    feature_sets = {
        'basic': create_basic_features,
        'momentum': create_momentum_features, 
        'regime_aware': create_regime_features,
        'volume_weighted': create_volume_features
    }
    
    results = {}
    
    for name, feature_func in feature_sets.items():
        print(f"\n📊 Testing {name.upper()} features...")
        
        features = feature_func(df)
        target = df['returns'].shift(-1).dropna()  # Tomorrow's return
        
        # Align
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].values
        y = target.loc[common_idx].values
        
        if len(X) < 50:
            print(f"   ❌ Not enough data")
            continue
            
        # Test signal strength
        ic = test_signal_strength(X, y)
        results[name] = ic
        
        print(f"   IC: {ic:.4f} ({'STRONG' if abs(ic) > 0.05 else 'WEAK'})")
    
    # Show best approach
    best = max(results.items(), key=lambda x: abs(x[1]))
    print(f"\n🏆 BEST APPROACH: {best[0].upper()}")
    print(f"   IC: {best[1]:.4f}")
    
    return best[0]

def create_basic_features(df):
    """Basic momentum/mean reversion"""
    features = pd.DataFrame(index=df.index)
    
    features['ret_1d'] = df['close'].pct_change(1)
    features['ret_5d'] = df['close'].pct_change(5)
    features['ma_ratio'] = df['close'] / df['close'].rolling(20).mean() - 1
    
    return features.dropna()

def create_momentum_features(df):
    """Enhanced momentum with multiple timeframes"""
    features = pd.DataFrame(index=df.index)
    
    # Multi-timeframe momentum
    for period in [1, 3, 5, 10, 20]:
        features[f'ret_{period}d'] = df['close'].pct_change(period)
    
    # Momentum strength
    features['momentum_strength'] = (
        features['ret_1d'] * features['ret_5d'] * features['ret_20d']
    ) ** (1/3)
    
    return features.dropna()

def create_regime_features(df):
    """Regime-aware features"""
    features = pd.DataFrame(index=df.index)
    
    # Basic returns
    ret = df['close'].pct_change()
    features['ret_1d'] = ret
    features['ret_5d'] = df['close'].pct_change(5)
    
    # Volatility regime
    vol = ret.rolling(20).std()
    vol_regime = vol > vol.rolling(60).median()
    
    # Features conditional on regime
    features['ret_1d_high_vol'] = ret * vol_regime
    features['ret_1d_low_vol'] = ret * ~vol_regime
    features['vol_regime'] = vol_regime.astype(float)
    
    return features.dropna()

def create_volume_features(df):
    """Volume-weighted features"""
    features = pd.DataFrame(index=df.index)
    
    ret = df['close'].pct_change()
    vol_ma = df['volume'].rolling(20).mean()
    rel_vol = df['volume'] / vol_ma
    
    # Volume-weighted returns
    features['ret_1d'] = ret
    features['ret_5d'] = df['close'].pct_change(5)
    features['vwap_ratio'] = df['close'] / (df['close'] * rel_vol).rolling(5).mean() - 1
    features['high_volume'] = (rel_vol > 1.5).astype(float)
    
    return features.dropna()

def test_signal_strength(X, y):
    """Test signal strength using Ridge regression"""
    # Simple train/test split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    # Test
    pred = ridge.predict(X_test_scaled)
    
    # Information Coefficient (correlation)
    ic = np.corrcoef(y_test, pred)[0, 1]
    
    return ic if not np.isnan(ic) else 0.0

def estimate_trading_performance(ic, costs_bps=5):
    """Estimate trading performance from IC"""
    print(f"\n💰 TRADING PERFORMANCE ESTIMATE")
    print(f"   IC: {ic:.4f}")
    print(f"   Transaction costs: {costs_bps} bps")
    
    # Rule of thumb: Sharpe ≈ IC * sqrt(frequency) * information_ratio
    # For daily trading with good execution
    estimated_sharpe = abs(ic) * np.sqrt(252) * 0.5  # Conservative
    
    print(f"   Estimated Sharpe: {estimated_sharpe:.2f}")
    
    if estimated_sharpe > 1.0:
        print("   🎯 EXCELLENT - Likely profitable after costs")
    elif estimated_sharpe > 0.5:
        print("   ✅ GOOD - Should be profitable")
    else:
        print("   ⚠️  WEAK - May not survive transaction costs")

if __name__ == "__main__":
    print("🎯 FEATURE-FIRST IMPROVEMENT STRATEGY")
    print("=" * 45)
    
    try:
        # Test feature improvements
        best_approach = test_feature_improvements()
        
        # For the best approach, estimate trading performance
        print(f"\n🎯 RECOMMENDED NEXT STEPS:")
        print(f"   1. Implement {best_approach.upper()} features in your real system")
        print(f"   2. Test on your actual SPY/equity data")
        print(f"   3. Measure performance after transaction costs")
        print(f"   4. Only then consider algorithm upgrades")
        
        print(f"\n💡 WHY THIS APPROACH WORKS:")
        print(f"   - Features determine signal quality")
        print(f"   - Algorithm just extracts that signal")
        print(f"   - Ridge with great features > XGBoost with bad features")
        print(f"   - Your current IC=0.0834 suggests good feature work already")
        
    except Exception as e:
        print(f"❌ Error: {e}")
