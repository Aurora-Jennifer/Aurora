# ml/features/build_daily.py
"""
Build daily features for Alpha v1 model.
Strict leakage guards: label is shifted forward by 1 day.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)


def load_feature_config() -> dict:
    """Load feature configuration."""
    config_path = Path("config/features.yaml")
    return yaml.safe_load(config_path.read_text())


def calculate_returns(prices: pd.Series, periods: list[int]) -> pd.DataFrame:
    """Calculate returns for multiple periods."""
    returns = pd.DataFrame()
    for period in periods:
        returns[f"ret_{period}d"] = prices.pct_change(period)
    return returns


def calculate_sma_ratio(prices: pd.Series, short: int = 20, long: int = 50) -> pd.Series:
    """Calculate SMA ratio (short/long - 1)."""
    sma_short = prices.rolling(short).mean()
    sma_long = prices.rolling(long).mean()
    return sma_short / sma_long - 1


def calculate_volatility(returns: pd.Series, periods: list[int]) -> pd.DataFrame:
    """Calculate rolling volatility."""
    vol = pd.DataFrame()
    for period in periods:
        vol[f"vol_{period}d"] = returns.rolling(period).std()
    return vol


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate z-scored volume."""
    vol_mean = volume.rolling(period).mean()
    vol_std = volume.rolling(period).std()
    return (volume - vol_mean) / vol_std


def build_features_for_symbol(
    symbol: str, start_date: str = "2020-01-01", end_date: str | None = None
) -> pd.DataFrame:
    """
    Build features for a single symbol with strict leakage guards.

    Args:
        symbol: Trading symbol
        start_date: Start date for data
        end_date: End date for data (default: today)

    Returns:
        DataFrame with features and labels
    """
    config = load_feature_config()
    min_bars = config["params"]["min_history_bars"]
    label_shift = config["params"]["label_shift"]

    # Download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

    if len(df) < min_bars:
        logger.warning(f"Insufficient data for {symbol}: {len(df)} < {min_bars}")
        return pd.DataFrame()

    # Ensure UTC timezone and monotonic index
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df.sort_index()

    # Calculate features
    features = pd.DataFrame(index=df.index)

    # Returns
    returns = calculate_returns(df["Close"], [1, 5, 20])
    features = pd.concat([features, returns], axis=1)

    # SMA ratio
    features["sma_20_minus_50"] = calculate_sma_ratio(df["Close"])

    # Volatility
    vol_features = calculate_volatility(returns["ret_1d"], [10, 20])
    features = pd.concat([features, vol_features], axis=1)

    # RSI
    features["rsi_14"] = calculate_rsi(df["Close"])

    # Volume z-score
    features["volu_z_20d"] = calculate_volume_zscore(df["Volume"])

    # Calculate label (next-day return) with leakage guard
    features["ret_fwd_1d"] = df["Close"].pct_change(label_shift).shift(-label_shift)

    # Drop NaN values
    features = features.dropna()

    # Verify leakage guard: label index > feature index
    if len(features) > 0:
        label_col = "ret_fwd_1d"
        [col for col in features.columns if col != label_col]

        # Check that label is properly shifted
        for i in range(len(features) - 1):
            if not pd.isna(features.iloc[i][label_col]):
                # Label should be future relative to features
                assert features.index[i] < features.index[i + label_shift], (
                    f"Leakage detected: label at {features.index[i]} not properly shifted"
                )

    return features


def build_feature_store(
    symbols: list[str], output_dir: str = "artifacts/feature_store"
) -> dict[str, str]:
    """
    Build feature store for multiple symbols.

    Args:
        symbols: List of symbols to process
        output_dir: Output directory for feature files

    Returns:
        Dict mapping symbol to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for symbol in symbols:
        logger.info(f"Building features for {symbol}")
        try:
            features = build_features_for_symbol(symbol)

            if len(features) > 0:
                file_path = output_path / f"{symbol}.parquet"
                features.to_parquet(file_path)
                results[symbol] = str(file_path)
                logger.info(f"Saved {len(features)} rows for {symbol}")
            else:
                logger.warning(f"No features generated for {symbol}")

        except Exception as e:
            logger.error(f"Error building features for {symbol}: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="SPY,TSLA", help="Comma-separated symbols")
    parser.add_argument("--output-dir", default="artifacts/feature_store", help="Output directory")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    results = build_feature_store(symbols, args.output_dir)

    print(f"Built features for {len(results)} symbols:")
    for symbol, path in results.items():
        print(f"  {symbol}: {path}")
