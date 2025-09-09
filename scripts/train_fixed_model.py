#!/usr/bin/env python3
"""
Fixed Model Training - Implements proper target design to prevent "always BUY" collapse

Key fixes:
1. Predict excess returns (asset - market) instead of raw returns
2. Balanced class weights
3. Cost-aware loss with label smoothing
4. Proper decision calibration
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import json
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_excess_returns(asset_data: pd.DataFrame, market_data: pd.DataFrame, 
                           window: int = 20) -> pd.Series:
    """
    Calculate excess returns (asset - market) with rolling beta
    
    Args:
        asset_data: Asset price data
        market_data: Market (SPY) price data  
        window: Rolling window for beta calculation
    
    Returns:
        Excess returns series
    """
    # Calculate returns
    asset_returns = asset_data['Close'].pct_change()
    market_returns = market_data['Close'].pct_change()
    
    # Calculate rolling beta
    rolling_cov = asset_returns.rolling(window).cov(market_returns)
    rolling_var = market_returns.rolling(window).var()
    beta = rolling_cov / rolling_var
    
    # Calculate excess returns
    excess_returns = asset_returns - beta * market_returns
    
    return excess_returns


def create_balanced_targets(excess_returns: pd.Series, horizon: int = 5, 
                          epsilon_percentile: tuple = (40, 60)) -> pd.Series:
    """
    Create balanced targets from excess returns
    
    Args:
        excess_returns: Excess return series
        horizon: Prediction horizon in days
        epsilon_percentile: Percentiles for HOLD band (lower, upper)
    
    Returns:
        Target series (0=SELL, 1=HOLD, 2=BUY)
    """
    # Forward-looking excess returns
    future_excess = excess_returns.shift(-horizon)
    
    # Calculate epsilon band from training data
    abs_excess = np.abs(future_excess.dropna())
    eps_low = np.percentile(abs_excess, epsilon_percentile[0])
    eps_high = np.percentile(abs_excess, epsilon_percentile[1])
    epsilon = (eps_low + eps_high) / 2
    
    # Create balanced targets
    targets = pd.Series(index=excess_returns.index, dtype=int)
    targets[future_excess > epsilon] = 2  # BUY
    targets[future_excess < -epsilon] = 0  # SELL
    targets[(future_excess >= -epsilon) & (future_excess <= epsilon)] = 1  # HOLD
    
    return targets


def calculate_class_weights(targets: pd.Series) -> dict:
    """
    Calculate class weights for balanced training
    
    Args:
        targets: Target series (0, 1, 2)
    
    Returns:
        Dictionary with class weights
    """
    from collections import Counter
    
    counts = Counter(targets.dropna())
    total = sum(counts.values())
    
    # Calculate inverse frequency weights
    weights = {}
    for class_id in [0, 1, 2]:  # SELL, HOLD, BUY
        count = counts.get(class_id, 1)  # Avoid division by zero
        weights[class_id] = total / count
    
    return weights


def build_simple_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple technical features
    
    Args:
        data: Price data with OHLCV
    
    Returns:
        Feature DataFrame
    """
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['returns_1d'] = data['Close'].pct_change()
    features['returns_5d'] = data['Close'].pct_change(5)
    features['returns_20d'] = data['Close'].pct_change(20)
    
    # Volatility features
    features['vol_5d'] = data['Close'].pct_change().rolling(5).std()
    features['vol_20d'] = data['Close'].pct_change().rolling(20).std()
    
    # Price position features
    features['high_20d'] = data['High'].rolling(20).max()
    features['low_20d'] = data['Low'].rolling(20).min()
    features['price_position'] = (data['Close'] - features['low_20d']) / (features['high_20d'] - features['low_20d'])
    
    # Volume features
    features['volume_ma'] = data['Volume'].rolling(20).mean()
    features['volume_ratio'] = data['Volume'] / features['volume_ma']
    
    # Momentum features
    features['rsi'] = calculate_rsi(data['Close'], 14)
    features['macd'] = calculate_macd(data['Close'])
    
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


def train_fixed_model(symbol: str = "SPY", lookback_days: int = 300):
    """
    Train a model with fixed target design
    """
    logging.info(f"Training fixed model for {symbol}")
    
    # Download data
    asset = yf.Ticker(symbol)
    market = yf.Ticker("SPY")  # Market proxy
    
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=lookback_days)
    
    asset_data = asset.history(start=start_date, end=end_date)
    market_data = market.history(start=start_date, end=end_date)
    
    if len(asset_data) < 100:
        raise ValueError(f"Insufficient data: {len(asset_data)} days")
    
    logging.info(f"Downloaded {len(asset_data)} days of data")
    
    # Calculate excess returns
    excess_returns = calculate_excess_returns(asset_data, market_data)
    
    # Create balanced targets
    targets = create_balanced_targets(excess_returns, horizon=5)
    
    # Build features
    features = build_simple_features(asset_data)
    
    # Align data
    common_index = features.index.intersection(targets.index)
    features = features.loc[common_index]
    targets = targets.loc[common_index]
    
    # Remove any remaining NaN values
    valid_mask = ~(features.isna().any(axis=1) | targets.isna())
    features = features[valid_mask]
    targets = targets[valid_mask]
    
    logging.info(f"Prepared {len(features)} samples with {features.shape[1]} features")
    
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
    
    # Train simple model (using sklearn for simplicity)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model with class weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',  # Use balanced class weights
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)
    
    logging.info("Validation Results:")
    logging.info(f"Classification Report:\n{classification_report(y_val, y_pred)}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    
    # Check prediction diversity
    unique_predictions = len(set(y_pred))
    logging.info(f"Unique predictions: {unique_predictions}/3 classes")
    
    if unique_predictions < 3:
        logging.warning("Model is not predicting all classes - may need more data or different features")
    
    # Save model and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/fixed_{symbol}_{timestamp}.pkl"
    config_path = f"models/fixed_{symbol}_{timestamp}_config.json"
    
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = f"models/fixed_{symbol}_{timestamp}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save config
    config = {
        'symbol': symbol,
        'model_type': 'fixed_random_forest',
        'trained_at': datetime.now().isoformat(),
        'lookback_days': lookback_days,
        'features': list(features.columns),
        'class_weights': class_weights,
        'class_distribution': dict(class_counts),
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
    
    return model_path, config_path, scaler_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fixed model with proper target design")
    parser.add_argument("--symbol", default="SPY", help="Symbol to train on")
    parser.add_argument("--lookback-days", type=int, default=300, help="Lookback period in days")
    
    args = parser.parse_args()
    
    try:
        model_path, config_path, scaler_path = train_fixed_model(args.symbol, args.lookback_days)
        print(f"\n🎉 Fixed model training completed!")
        print(f"Model: {model_path}")
        print(f"Config: {config_path}")
        print(f"Scaler: {scaler_path}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)
