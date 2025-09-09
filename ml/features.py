#!/usr/bin/env python3
"""
Features Module

Implements 12 robust features with schema/versioning helpers:
- Multi-horizon returns
- Volatility measures
- Moving averages and slopes
- Drawdown
- Overnight vs intraday
- Cross-asset placeholders
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate_proxy_coverage(df_main: pd.DataFrame, df_proxy: pd.DataFrame, 
                          cross_proxies: List[str]) -> List[str]:
    """
    Validate proxy coverage and drop proxies with poor coverage
    
    Args:
        df_main: Main asset data
        df_proxy: Proxy data
        cross_proxies: List of proxy symbols
    
    Returns:
        List of valid proxy symbols
    """
    valid = []
    for px in cross_proxies:
        if px in df_proxy.columns:
            n_non_na = df_proxy[px].dropna().shape[0]
            coverage_pct = n_non_na / len(df_main) * 100
            logger.info(f"Proxy {px} non-NA rows: {n_non_na} ({coverage_pct:.1f}% coverage)")
            
            if coverage_pct >= 80.0:  # >=80% coverage
                valid.append(px)
            else:
                logger.warning(f"Dropping proxy {px} due to poor coverage: {coverage_pct:.1f}%")
        else:
            logger.warning(f"Proxy {px} not found in proxy data")
    
    dropped = set(cross_proxies) - set(valid)
    if dropped:
        logger.warning(f"Dropping proxies with poor coverage: {sorted(dropped)}")
    
    return valid


def add_core_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Add 12 robust core features to DataFrame
    
    Args:
        df: DataFrame with price data
        price_col: Name of price column
    
    Returns:
        DataFrame with added features
    """
    # Ensure we have a copy to avoid modifying original
    df = df.copy()
    
    # Get price series
    px = df[price_col].astype(float)
    # Handle MultiIndex columns from yfinance
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]  # Take first column
    
    # 1. Multi-horizon returns
    ret = px.pct_change()
    df["ret1"] = ret
    
    for k in [3, 5, 10, 20]:
        df[f"ret{k}"] = px.pct_change(k)
    
    # 2. Volatility measures
    for w in [10, 20, 60]:
        df[f"vol{w}"] = ret.rolling(w).std()
    
    # 3. Volatility ratio (short-term vs long-term)
    df["vol_ratio_short_long"] = df["vol10"] / df["vol60"]
    
    # 4. Moving averages and slopes
    df["ma_fast"] = px.rolling(10).mean()
    df["ma_slow"] = px.rolling(50).mean()
    df["ma_slope"] = (df["ma_fast"] - df["ma_slow"]).diff()
    
    # 5. Drawdown
    df["dd"] = (px / px.cummax() - 1.0)
    
    # 6. Overnight vs intraday (if open column exists)
    if {"Open", "Close"}.issubset(df.columns):
        open_price = df["Open"]
        close_price = df["Close"]
        # Handle MultiIndex columns from yfinance
        if isinstance(open_price, pd.DataFrame):
            open_price = open_price.iloc[:, 0]
        if isinstance(close_price, pd.DataFrame):
            close_price = close_price.iloc[:, 0]
        df["overnight"] = (open_price / close_price.shift(1) - 1.0)
        df["intraday"] = (close_price / open_price - 1.0)
    elif {"open", "close"}.issubset(df.columns):
        df["overnight"] = (df["open"] / df["close"].shift(1) - 1.0)
        df["intraday"] = (df["close"] / df["open"] - 1.0)
    else:
        # Fallback: use price gaps as proxy
        df["overnight"] = ret  # Simplified
        df["intraday"] = ret   # Simplified
    
    # 7. Momentum indicators
    df["momentum_5"] = px / px.shift(5) - 1.0
    df["momentum_20"] = px / px.shift(20) - 1.0
    
    # 8. Price position in range
    df["price_position"] = (px - px.rolling(20).min()) / (px.rolling(20).max() - px.rolling(20).min())
    
    # 9. Volume features (if available)
    if "Volume" in df.columns:
        volume = df["Volume"]
        # Handle MultiIndex columns from yfinance
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]  # Take first column
        df["vol_ma"] = volume.rolling(20).mean()
        df["vol_ratio_volume"] = volume / df["vol_ma"]
    elif "volume" in df.columns:
        volume = df["volume"]
        # Handle MultiIndex columns from yfinance
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]  # Take first column
        df["vol_ma"] = volume.rolling(20).mean()
        df["vol_ratio_volume"] = volume / df["vol_ma"]
    else:
        # Placeholder volume features
        df["vol_ma"] = 1.0
        df["vol_ratio_volume"] = 1.0
    
    # 10. Bollinger Bands
    bb_period = 20
    bb_std = 2
    df["bb_middle"] = px.rolling(bb_period).mean()
    bb_std_val = px.rolling(bb_period).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std_val * bb_std)
    df["bb_lower"] = df["bb_middle"] - (bb_std_val * bb_std)
    df["bb_position"] = (px - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # 11. RSI (simplified)
    delta = px.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # 12. MACD
    exp1 = px.ewm(span=12).mean()
    exp2 = px.ewm(span=26).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    
    # Clean up: replace inf/-inf with NaN, then forward/back fill
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward fill then back fill
    df = df.ffill().bfill()
    
    # Drop any remaining NaN rows
    df = df.dropna()
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Get list of feature columns from DataFrame
    
    Args:
        df: DataFrame
        exclude_cols: Columns to exclude from features
    
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                       'open', 'high', 'low', 'close', 'volume', 'adj_close']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return sorted(feature_cols)


def save_feature_schema(feature_columns: List[str], schema_path: str, 
                       metadata: Dict[str, Any] = None) -> None:
    """
    Save feature schema to file
    
    Args:
        feature_columns: List of feature column names
        schema_path: Path to save schema
        metadata: Additional metadata
    """
    schema = {
        'feature_columns': feature_columns,
        'num_features': len(feature_columns),
        'created_at': pd.Timestamp.now().isoformat(),
        'version': '1.0'
    }
    
    if metadata:
        schema.update(metadata)
    
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)


def load_feature_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load feature schema from file
    
    Args:
        schema_path: Path to schema file
    
    Returns:
        Schema dictionary
    """
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_feature_schema(df: pd.DataFrame, schema: Dict[str, Any]) -> None:
    """
    Validate DataFrame against feature schema
    
    Args:
        df: DataFrame to validate
        schema: Feature schema
    
    Raises:
        AssertionError: If validation fails
    """
    expected_cols = schema['feature_columns']
    actual_cols = list(df.columns)
    
    if actual_cols != expected_cols:
        raise AssertionError(f"Feature columns mismatch: {actual_cols} != {expected_cols}")
    
    if df.shape[1] != schema['num_features']:
        raise AssertionError(f"Feature count mismatch: {df.shape[1]} != {schema['num_features']}")


def create_feature_pipeline(df: pd.DataFrame, price_col: str = "Close", 
                          schema_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create complete feature pipeline
    
    Args:
        df: Input DataFrame
        price_col: Price column name
        schema_path: Optional path to save schema
    
    Returns:
        DataFrame with features
    """
    # Add core features
    df_features = add_core_features(df, price_col)
    
    # Get feature columns
    feature_cols = get_feature_columns(df_features)
    
    # Select only feature columns
    df_features = df_features[feature_cols]
    
    # Save schema if requested
    if schema_path:
        save_feature_schema(feature_cols, schema_path)
    
    return df_features


def add_cross_asset_features(df: pd.DataFrame, market_data: pd.DataFrame, 
                           price_col: str = "Close") -> pd.DataFrame:
    """
    Add cross-asset features (placeholder implementation)
    
    Args:
        df: Main DataFrame
        market_data: Market/benchmark data
        price_col: Price column name
    
    Returns:
        DataFrame with cross-asset features
    """
    # This is a placeholder - in practice you'd add:
    # - Relative strength vs market
    # - Sector rotation indicators
    # - Currency/commodity correlations
    # - VIX/volatility regime indicators
    
    # For now, just add market return as a feature
    if price_col in market_data.columns:
        market_ret = market_data[price_col].pct_change()
        df["market_ret"] = market_ret
    
    return df


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    spy = yf.download("SPY", start="2023-01-01", end="2024-01-01", progress=False)
    
    print("Original columns:", list(spy.columns))
    
    # Create features
    df_features = create_feature_pipeline(spy, schema_path="test_schema.json")
    
    print(f"Feature columns ({len(df_features.columns)}):")
    for col in df_features.columns:
        print(f"  {col}")
    
    print(f"Feature shape: {df_features.shape}")
    print(f"Feature stats:")
    print(df_features.describe())
    
    # Test schema
    schema = load_feature_schema("test_schema.json")
    print(f"Schema: {schema}")
    
    # Validate
    validate_feature_schema(df_features, schema)
    print("✅ Feature pipeline test passed!")
    
    # Clean up
    Path("test_schema.json").unlink(missing_ok=True)
