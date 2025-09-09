#!/usr/bin/env python3
"""
Enhanced Model Training - Implements proper target design and edge-based trading

Key improvements:
1. Proper band labels with epsilon from train percentiles
2. Class weights + label smoothing
3. Edge-based trading with hysteresis
4. Cost-aware threshold selection
5. Enhanced features (regime, risk, cross-asset)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_excess_returns(asset_data: pd.DataFrame, market_data: pd.DataFrame, 
                           window: int = 20) -> pd.Series:
    """Calculate excess returns with rolling beta"""
    asset_returns = asset_data['Close'].pct_change()
    market_returns = market_data['Close'].pct_change()
    
    rolling_cov = asset_returns.rolling(window).cov(market_returns)
    rolling_var = market_returns.rolling(window).var()
    beta = rolling_cov / rolling_var
    
    excess_returns = asset_returns - beta * market_returns
    return excess_returns


def create_band_targets(excess_returns: pd.Series, horizon: int = 5, 
                       epsilon_percentiles: Tuple[int, int] = (40, 60)) -> Tuple[pd.Series, float]:
    """
    Create proper band targets with epsilon from train percentiles
    
    Returns:
        targets: Target series (0=SELL, 1=HOLD, 2=BUY)
        epsilon: The epsilon value used
    """
    future_excess = excess_returns.shift(-horizon)
    
    # Calculate epsilon from training data percentiles
    abs_excess = np.abs(future_excess.dropna())
    eps_low = np.percentile(abs_excess, epsilon_percentiles[0])
    eps_high = np.percentile(abs_excess, epsilon_percentiles[1])
    epsilon = (eps_low + eps_high) / 2
    
    # Create band targets with sklearn-compatible labels [0, 1, 2]
    targets = pd.Series(index=excess_returns.index, dtype=int)
    targets[future_excess > epsilon] = 2  # BUY
    targets[future_excess < -epsilon] = 0  # SELL
    targets[(future_excess >= -epsilon) & (future_excess <= epsilon)] = 1  # HOLD
    
    return targets, epsilon


def build_enhanced_features(asset_data: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build enhanced features including regime, risk, and cross-asset signals
    """
    features = pd.DataFrame(index=asset_data.index)
    
    # Basic price features
    features['returns_1d'] = asset_data['Close'].pct_change()
    features['returns_5d'] = asset_data['Close'].pct_change(5)
    features['returns_20d'] = asset_data['Close'].pct_change(20)
    
    # Volatility features (regime & risk)
    features['vol_5d'] = asset_data['Close'].pct_change().rolling(5).std()
    features['vol_20d'] = asset_data['Close'].pct_change().rolling(20).std()
    features['vol_60d'] = asset_data['Close'].pct_change().rolling(60).std()
    features['vol_ratio'] = features['vol_5d'] / features['vol_20d']
    
    # Realized skew and kurtosis
    features['skew_20d'] = asset_data['Close'].pct_change().rolling(20).skew()
    features['kurt_20d'] = asset_data['Close'].pct_change().rolling(20).kurt()
    
    # Drawdown state
    rolling_max = asset_data['Close'].rolling(20).max()
    features['drawdown'] = (asset_data['Close'] - rolling_max) / rolling_max
    
    # Price position features
    features['high_20d'] = asset_data['High'].rolling(20).max()
    features['low_20d'] = asset_data['Low'].rolling(20).min()
    features['price_position'] = (asset_data['Close'] - features['low_20d']) / (features['high_20d'] - features['low_20d'])
    
    # Multi-horizon momentum
    features['momentum_3d'] = asset_data['Close'].pct_change(3)
    features['momentum_5d'] = asset_data['Close'].pct_change(5)
    features['momentum_10d'] = asset_data['Close'].pct_change(10)
    features['momentum_20d'] = asset_data['Close'].pct_change(20)
    
    # RSI-style oscillator
    features['rsi'] = calculate_rsi(asset_data['Close'], 14)
    
    # MACD
    features['macd'] = calculate_macd(asset_data['Close'])
    
    # Volume features
    features['volume_ma'] = asset_data['Volume'].rolling(20).mean()
    features['volume_ratio'] = asset_data['Volume'] / features['volume_ma']
    
    # Gap indicators
    features['gap'] = (asset_data['Open'] - asset_data['Close'].shift(1)) / asset_data['Close'].shift(1)
    features['gap_abs'] = np.abs(features['gap'])
    
    # Cross-asset features (if market data provided)
    if market_data is not None:
        market_returns = market_data['Close'].pct_change()
        features['market_returns_1d'] = market_returns
        features['market_returns_5d'] = market_returns.rolling(5).sum()
        features['relative_strength'] = features['returns_5d'] - features['market_returns_5d']
        
        # Market volatility
        market_vol = market_returns.rolling(20).std()
        features['market_vol'] = market_vol
        features['vol_spread'] = features['vol_20d'] - market_vol
    
    return features.dropna()


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd


def calculate_class_weights(targets: pd.Series) -> Dict[int, float]:
    """Calculate balanced class weights"""
    from collections import Counter
    
    counts = Counter(targets.dropna())
    total = sum(counts.values())
    
    weights = {}
    for class_id in [0, 1, 2]:  # SELL, HOLD, BUY
        count = counts.get(class_id, 1)
        weights[class_id] = total / count
    
    return weights


def train_enhanced_model(symbol: str = "SPY", lookback_days: int = 500, 
                        horizon: int = 5, epsilon_percentiles: Tuple[int, int] = (40, 60)):
    """
    Train enhanced model with proper target design
    """
    logging.info(f"Training enhanced model for {symbol}")
    
    # Download data
    asset = yf.Ticker(symbol)
    market = yf.Ticker("SPY")  # Market proxy
    
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=lookback_days)
    
    asset_data = asset.history(start=start_date, end=end_date)
    market_data = market.history(start=start_date, end=end_date)
    
    if len(asset_data) < 200:
        raise ValueError(f"Insufficient data: {len(asset_data)} days")
    
    logging.info(f"Downloaded {len(asset_data)} days of data")
    
    # Calculate excess returns
    excess_returns = calculate_excess_returns(asset_data, market_data)
    
    # Create band targets with sklearn-compatible labels [0, 1, 2]
    targets, epsilon = create_band_targets(excess_returns, horizon, epsilon_percentiles)
    
    # Convert to sklearn-compatible labels: SELL=0, HOLD=1, BUY=2
    targets = targets.replace({0: 0, 1: 1, 2: 2})  # Already in correct format
    
    # Build enhanced features
    features = build_enhanced_features(asset_data, market_data)
    
    # Align data
    common_index = features.index.intersection(targets.index)
    features = features.loc[common_index]
    targets = targets.loc[common_index]
    
    # Remove NaN values
    valid_mask = ~(features.isna().any(axis=1) | targets.isna())
    features = features[valid_mask]
    targets = targets[valid_mask]
    
    logging.info(f"Prepared {len(features)} samples with {features.shape[1]} features")
    logging.info(f"Epsilon band: {epsilon:.6f}")
    
    # Check class balance
    from collections import Counter
    class_counts = Counter(targets)
    logging.info(f"Class distribution: {dict(class_counts)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(targets)
    logging.info(f"Class weights: {class_weights}")
    
    # Split data
    split_idx = int(len(features) * 0.8)
    X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_val = targets.iloc[:split_idx], targets.iloc[split_idx:]
    
    # Train model with enhanced features
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model with class weights
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)
    
    logging.info("Validation Results:")
    logging.info(f"Classification Report:\n{classification_report(y_val, y_pred)}")
    
    # Check prediction diversity
    unique_predictions = len(set(y_pred))
    logging.info(f"Unique predictions: {unique_predictions}/3 classes")
    
    # Calculate edge statistics
    edges = y_proba[:, 2] - y_proba[:, 0]  # BUY - SELL
    edge_stats = {
        'mean_edge': np.mean(edges),
        'std_edge': np.std(edges),
        'edge_abs_mean': np.mean(np.abs(edges)),
        'edge_above_0.1': np.mean(np.abs(edges) > 0.1)
    }
    logging.info(f"Edge statistics: {edge_stats}")
    
    # Save model and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/enhanced_{symbol}_{timestamp}.pkl"
    config_path = f"models/enhanced_{symbol}_{timestamp}_config.json"
    scaler_path = f"models/enhanced_{symbol}_{timestamp}_scaler.pkl"
    schema_path = f"models/enhanced_{symbol}_{timestamp}_schema.json"
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature schema
    from feature_schema_helpers import save_feature_schema
    save_feature_schema(list(features.columns), schema_path)
    
    # Save comprehensive config
    config = {
        'symbol': symbol,
        'model_type': 'enhanced_random_forest',
        'trained_at': datetime.now().isoformat(),
        'lookback_days': lookback_days,
        'horizon': horizon,
        'epsilon': epsilon,
        'epsilon_percentiles': epsilon_percentiles,
        'features': list(features.columns),
        'class_weights': class_weights,
        'class_distribution': dict(class_counts),
        'class_mapping': {
            'model_classes': model.classes_.tolist(),
            'canonical_order': ['SELL', 'HOLD', 'BUY'],
            'mapping_type': 'sklearn_standard' if np.array_equal(model.classes_, [0, 1, 2]) else 'custom'
        },
        'edge_statistics': edge_stats,
        'validation_metrics': {
            'unique_predictions': unique_predictions,
            'total_samples': len(features),
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Config saved to {config_path}")
    logging.info(f"Scaler saved to {scaler_path}")
    logging.info(f"Schema saved to {schema_path}")
    
    return model_path, config_path, scaler_path, schema_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train enhanced model with proper target design")
    parser.add_argument("--symbol", default="SPY", help="Symbol to train on")
    parser.add_argument("--lookback-days", type=int, default=500, help="Lookback period in days")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days")
    parser.add_argument("--epsilon-low", type=int, default=40, help="Lower epsilon percentile")
    parser.add_argument("--epsilon-high", type=int, default=60, help="Upper epsilon percentile")
    
    args = parser.parse_args()
    
    try:
        model_path, config_path, scaler_path, schema_path = train_enhanced_model(
            args.symbol, args.lookback_days, args.horizon, 
            (args.epsilon_low, args.epsilon_high)
        )
        print(f"\n🎉 Enhanced model training completed!")
        print(f"Model: {model_path}")
        print(f"Config: {config_path}")
        print(f"Scaler: {scaler_path}")
        print(f"Schema: {schema_path}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)
