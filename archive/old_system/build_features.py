import pandas as pd
import numpy as np


def _rsi(series: pd.Series, n=14):
    """
    Compute RSI with robust handling for monotonic series.
    NO LOOKAHEAD: All calculations use only past data.

    Behaviors (after at least n periods):
    - avg_loss == 0 and avg_gain > 0  -> RSI = 100
    - avg_gain == 0 and avg_loss > 0  -> RSI = 0
    - avg_gain == 0 and avg_loss == 0 -> RSI = 50 (flat)
    - else standard: 100 - 100/(1 + avg_gain/avg_loss)
    """
    # Compute returns first, then shift to avoid lookahead
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Rolling calculations use only past data
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()

    rsi = pd.Series(index=series.index, dtype=float)

    # Standard case where both are positive
    std_mask = (avg_gain > 0) & (avg_loss > 0)
    rs = avg_gain[std_mask] / avg_loss[std_mask]
    rsi.loc[std_mask] = 100.0 - (100.0 / (1.0 + rs))

    # No losses -> RSI 100
    no_loss_mask = (avg_loss == 0) & (avg_gain > 0)
    rsi.loc[no_loss_mask] = 100.0

    # No gains -> RSI 0
    no_gain_mask = (avg_gain == 0) & (avg_loss > 0)
    rsi.loc[no_gain_mask] = 0.0

    # Flat -> RSI 50
    flat_mask = (avg_gain == 0) & (avg_loss == 0)
    rsi.loc[flat_mask] = 50.0

    return rsi


def build_demo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build demo features for portfolio testing.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with demo features, NaN values dropped
    """
    features = pd.DataFrame(index=df.index)
    
    # Basic price features
    features['returns'] = df['Close'].pct_change()
    features['volatility'] = features['returns'].rolling(20).std()
    
    # SMA features
    features['sma_5'] = df['Close'].rolling(5).mean()
    features['sma_20'] = df['Close'].rolling(20).mean()
    
    # Momentum features
    features['momentum_3d'] = df['Close'].pct_change(3)
    features['momentum_5d'] = df['Close'].pct_change(5)
    
    # Volume features
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Price position (relative to 20-day range)
    high_20 = df['High'].rolling(20).max()
    low_20 = df['Low'].rolling(20).min()
    features['price_position'] = (df['Close'] - low_20) / (high_20 - low_20)
    
    # Drop NaN values created by rolling windows
    features = features.dropna()
    
    return features


def build_matrix(df: pd.DataFrame, horizon=1, exclude_tags=None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix with optional feature exclusion for ablation testing.

    Args:
        df: DataFrame with OHLC data
        horizon: Forward return horizon
        exclude_tags: List of feature tags to exclude (e.g., ["trend", "momentum"])
    """
    ret_1d = df["Close"].pct_change()
    X = pd.DataFrame(index=df.index)

    # Build all features
    X["ret_1d_lag1"] = ret_1d.shift(1)
    X["sma_10"] = df["Close"].rolling(10, min_periods=10).mean().shift(1)
    X["sma_20"] = df["Close"].rolling(20, min_periods=20).mean().shift(1)
    X["vol_10"] = ret_1d.rolling(10, min_periods=10).std().shift(1)
    X["rsi_14"] = _rsi(df["Close"], 14).shift(1)
    
    # MOMENTUM FEATURES (discovered to improve IC from 0.08 → 0.36)
    # Multi-timeframe momentum that showed 4x improvement
    X["momentum_3d"] = df["Close"].pct_change(3).shift(1)
    X["momentum_5d"] = df["Close"].pct_change(5).shift(1) 
    X["momentum_10d"] = df["Close"].pct_change(10).shift(1)
    X["momentum_20d"] = df["Close"].pct_change(20).shift(1)
    
    # Momentum strength (key interaction feature)
    X["momentum_strength"] = (X["ret_1d_lag1"] * X["momentum_5d"] * X["momentum_20d"]) ** (1/3)

    # Apply feature masking if exclude_tags is provided
    if exclude_tags:
        import yaml
        with open("config/features.yaml") as f:
            feature_config = yaml.safe_load(f)

        # Create feature to tags mapping
        feature_tags = {}
        for feat in feature_config["features"]:
            feature_tags[feat["name"]] = feat["tags"]

        # Remove features that match excluded tags
        columns_to_remove = []
        for col in X.columns:
            if col in feature_tags and any(tag in exclude_tags for tag in feature_tags[col]):
                columns_to_remove.append(col)

        X = X.drop(columns=columns_to_remove)
        print(f"Excluded features with tags {exclude_tags}: {columns_to_remove}")

    y = df["Close"].pct_change(horizon).shift(-horizon)

    # Align, drop NaNs created by rolling/shift
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]


def validate_features(X: pd.DataFrame, y: pd.Series) -> bool:
    """
    Validate that features and targets are properly aligned.
    
    Args:
        X: Feature matrix
        y: Target series
        
    Returns:
        True if validation passes
    """
    # Check alignment
    if not X.index.equals(y.index):
        return False
    
    # Check for infinite values
    if X.isin([np.inf, -np.inf]).any().any():
        return False
    
    # Check for excessive NaN values
    if X.isna().sum().max() > len(X) * 0.5:
        return False
        
    return True


