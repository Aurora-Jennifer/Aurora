#!/usr/bin/env python3
"""
Regime Features Module for ML Trading System v0.2

This module computes regime-aware features for multi-asset trading data,
including trend, volatility, liquidity, and binary regime indicators.
All features are computed without forward-looking leakage.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


def compute_regime_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    asset_col: str = "asset",
    ts_col: str = "ts",
) -> pd.DataFrame:
    """
    Compute regime-aware features for multi-asset trading data.

    Features computed per asset & timestamp:
    - Trend: SMA(50), SMA(200), RSI(14), MACD(12,26,9), trend_z = SMA50/SMA200 - 1
    - Volatility: rolling std of 1d returns (20), ATR(14), realized vol (10,20)
    - Liquidity: ADV_20 = mean(dollar_volume, 20), spread_proxy = 1/ADV_20
    - Binary tags: bull (SMA50>SMA200), high_vol (vol_20 > asset's 252d median)

    Args:
        df: Multi-asset dataframe with columns [ts, asset, close, volume]
        price_col: Column name for price data (default: "close")
        volume_col: Column name for volume data (default: "volume")
        asset_col: Column name for asset identifier (default: "asset")
        ts_col: Column name for timestamp (default: "ts")

    Returns:
        DataFrame with original data plus computed features

    Raises:
        ValueError: If required columns are missing
        KeyError: If asset or timestamp data is malformed
    """

    # Validate input
    required_cols = [ts_col, asset_col, price_col, volume_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure proper data types
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df[asset_col] = df[asset_col].astype(str)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")

    # Sort by asset and timestamp to ensure proper rolling calculations
    df = df.sort_values([asset_col, ts_col]).reset_index(drop=True)

    # Initialize result dataframe
    result_df = df.copy()

    # Compute features for each asset
    feature_dfs = []

    for asset in df[asset_col].unique():
        asset_data = df[df[asset_col] == asset].copy()

        if len(asset_data) < 252:  # Need at least 252 days for median calculations
            logger.warning(
                f"Asset {asset} has insufficient data ({len(asset_data)} rows), skipping"
            )
            continue

        # Compute features for this asset
        asset_features = _compute_asset_features(asset_data, price_col, volume_col, ts_col)
        feature_dfs.append(asset_features)

    if not feature_dfs:
        raise ValueError("No assets had sufficient data for feature computation")

    # Combine all asset features
    result_df = pd.concat(feature_dfs, ignore_index=True)

    # Sort final result
    result_df = result_df.sort_values([asset_col, ts_col]).reset_index(drop=True)

    logger.info(
        f"Computed regime features for {len(result_df)} rows across {len(feature_dfs)} assets"
    )

    return result_df


def _compute_asset_features(
    df: pd.DataFrame, price_col: str, volume_col: str, ts_col: str
) -> pd.DataFrame:
    """
    Compute features for a single asset.

    Args:
        df: Single asset dataframe
        price_col: Price column name
        volume_col: Volume column name
        ts_col: Timestamp column name

    Returns:
        DataFrame with computed features
    """

    # Ensure sorted by timestamp
    df = df.sort_values(ts_col).reset_index(drop=True)

    # Compute returns
    df["ret_1d"] = df[price_col].pct_change(fill_method=None)

    # 1. TREND FEATURES
    # Simple Moving Averages
    df["sma_50"] = df[price_col].rolling(window=50, min_periods=50).mean()
    df["sma_200"] = df[price_col].rolling(window=200, min_periods=200).mean()

    # Trend Z-score
    df["trend_z"] = (df["sma_50"] / df["sma_200"]) - 1

    # RSI
    df["rsi_14"] = _compute_rsi(df[price_col], window=14)

    # MACD
    macd_data = _compute_macd(df[price_col], fast=12, slow=26, signal=9)
    df["macd"] = macd_data["macd"]
    df["macd_signal"] = macd_data["signal"]
    df["macd_histogram"] = macd_data["histogram"]

    # 2. VOLATILITY FEATURES
    # Rolling standard deviation of returns
    df["vol_20"] = df["ret_1d"].rolling(window=20, min_periods=20).std()

    # ATR (Average True Range)
    df["atr_14"] = _compute_atr(df, price_col, window=14)

    # Realized volatility (annualized)
    df["realized_vol_10"] = df["ret_1d"].rolling(window=10, min_periods=10).std() * np.sqrt(252)
    df["realized_vol_20"] = df["ret_1d"].rolling(window=20, min_periods=20).std() * np.sqrt(252)

    # 3. LIQUIDITY FEATURES
    # Dollar volume
    df["dollar_volume"] = df[price_col] * df[volume_col]

    # Average Daily Volume (ADV)
    df["adv_20"] = df["dollar_volume"].rolling(window=20, min_periods=20).mean()

    # Spread proxy (inverse of ADV)
    df["spread_proxy"] = 1 / (df["adv_20"] + 1e-8)  # Add small constant to avoid division by zero

    # 4. BINARY REGIME TAGS
    # Bull market indicator
    df["bull"] = (df["sma_50"] > df["sma_200"]).astype(int)

    # High volatility indicator
    vol_median = df["vol_20"].rolling(window=252, min_periods=252).median()
    df["high_vol"] = (df["vol_20"] > vol_median).astype(int)

    # 5. ADDITIONAL FEATURES
    # Price momentum
    df["momentum_5"] = df[price_col].pct_change(periods=5, fill_method=None)
    df["momentum_20"] = df[price_col].pct_change(periods=20, fill_method=None)

    # Volume momentum
    df["volume_momentum"] = df[volume_col].pct_change(periods=5)

    # Price position within range
    high_20 = df[price_col].rolling(window=20, min_periods=20).max()
    low_20 = df[price_col].rolling(window=20, min_periods=20).min()
    df["price_position"] = (df[price_col] - low_20) / (high_20 - low_20 + 1e-8)

    # Z-score of price
    price_mean = df[price_col].rolling(window=20, min_periods=20).mean()
    price_std = df[price_col].rolling(window=20, min_periods=20).std()
    df["price_z_score"] = (df[price_col] - price_mean) / (price_std + 1e-8)

    return df


def _compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        prices: Price series
        window: RSI window (default: 14)

    Returns:
        RSI series
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()

    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))



def _compute_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> dict[str, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Dictionary with MACD, signal, and histogram series
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    return {"macd": macd, "signal": signal_line, "histogram": histogram}


def _compute_atr(df: pd.DataFrame, price_col: str, window: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).

    Args:
        df: DataFrame with high, low, close columns
        price_col: Price column name (assumes OHLC data)
        window: ATR window (default: 14)

    Returns:
        ATR series
    """
    # For simplicity, we'll use close price as proxy for high/low
    # In practice, you'd want actual high/low data
    high = df[price_col]  # Proxy
    low = df[price_col]  # Proxy
    close = df[price_col]

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is the moving average of True Range
    return true_range.rolling(window=window, min_periods=window).mean()



def validate_features(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate computed features for data quality and leakage.

    Args:
        df: DataFrame with computed features

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "total_rows": len(df),
        "total_assets": df["asset"].nunique(),
        "feature_columns": [],
        "missing_values": {},
        "leakage_check": "PASSED",
        "warnings": [],
    }

    # Check for feature columns
    feature_cols = [col for col in df.columns if col not in ["ts", "asset", "close", "volume"]]
    validation_results["feature_columns"] = feature_cols

    # Check for missing values
    for col in feature_cols:
        missing_pct = df[col].isnull().mean()
        validation_results["missing_values"][col] = missing_pct

        if missing_pct > 0.5:
            validation_results["warnings"].append(
                f"High missing values in {col}: {missing_pct:.1%}"
            )

    # Basic leakage check (features should be NaN at the beginning)
    for asset in df["asset"].unique():
        asset_data = df[df["asset"] == asset].sort_values("ts")

        # Check if first few rows have non-NaN features (potential leakage)
        first_features = asset_data[feature_cols].iloc[:10]
        if not first_features.isnull().all().all():
            validation_results["leakage_check"] = "FAILED"
            validation_results["warnings"].append(f"Potential leakage detected in asset {asset}")

    return validation_results


if __name__ == "__main__":
    # Example usage and testing
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    assets = ["SPY", "QQQ", "AAPL"]

    sample_data = []
    for asset in assets:
        for date in dates:
            sample_data.append(
                {
                    "ts": date,
                    "asset": asset,
                    "close": 100 + np.random.randn() * 10,
                    "volume": 1000000 + np.random.randn() * 100000,
                }
            )

    df = pd.DataFrame(sample_data)

    # Compute features
    features_df = compute_regime_features(df)

    # Validate features
    validation = validate_features(features_df)

    print("Feature computation completed!")
    print(f"Total rows: {validation['total_rows']}")
    print(f"Total assets: {validation['total_assets']}")
    print(f"Feature columns: {len(validation['feature_columns'])}")
    print(f"Leakage check: {validation['leakage_check']}")

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
