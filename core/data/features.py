from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl


def _to_polars_ohlcv(df: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas DataFrame from yfinance to Polars DataFrame with proper column names."""
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-level columns
        df.columns = [
            col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in df.columns
        ]

    # Reset index to get Date as a column
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()

    # Map common column names (handle both original and flattened names)
    col_mapping = {}
    if "Date" in df.columns:
        col_mapping["Date"] = "ts"

    # Handle standard column names
    if "Open" in df.columns:
        col_mapping["Open"] = "open"
    if "High" in df.columns:
        col_mapping["High"] = "high"
    if "Low" in df.columns:
        col_mapping["Low"] = "low"
    if "Close" in df.columns:
        col_mapping["Close"] = "close"
    if "Volume" in df.columns:
        col_mapping["Volume"] = "volume"

    # Handle symbol-specific column names (e.g., Open_SPY, Close_QQQ, etc.)
    for col in df.columns:
        if col.startswith("Open_"):
            col_mapping[col] = "open"
        elif col.startswith("High_"):
            col_mapping[col] = "high"
        elif col.startswith("Low_"):
            col_mapping[col] = "low"
        elif col.startswith("Close_"):
            col_mapping[col] = "close"
        elif col.startswith("Volume_"):
            col_mapping[col] = "volume"

    # Rename columns
    df = df.rename(columns=col_mapping)

    # Convert to Polars
    df_pl = pl.from_pandas(df)

    # Ensure correct dtypes
    df_pl = df_pl.with_columns(
        [
            pl.col("ts").cast(pl.Datetime),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]
    )

    return df_pl


def build_features_parquet(
    symbol: str, df: Any, outdir: str, version: str = "v1"
) -> Path:
    """
    Build features from OHLCV data and save to Parquet.

    Args:
            symbol: Symbol name
            df: DataFrame with OHLCV data (pandas or polars)
            outdir: Output directory
            version: Feature version

    Returns:
            Path to saved Parquet file
    """
    # Convert to Polars DataFrame with proper column names
    if isinstance(df, pd.DataFrame):
        df_pl = _to_polars_ohlcv(df)
    else:
        df_pl = df

    # Sort by timestamp
    df_pl = df_pl.sort("ts")

    # Add features
    df_pl = df_pl.with_columns(
        [
            pl.col("close").pct_change().alias("ret1"),
            pl.col("close").rolling_mean(window_size=20).alias("ma20"),
            pl.col("close").rolling_std(window_size=20).alias("vol20"),
        ]
    ).with_columns(
        [
            ((pl.col("close") - pl.col("ma20")) / (pl.col("vol20") + 1e-12)).alias(
                "zscore20"
            )
        ]
    )

    # Create output directory
    out_path = Path(outdir) / f"{symbol}_{version}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Parquet
    df_pl.write_parquet(out_path)

    return out_path


def load_features(symbol: str, outdir: str, version: str = "v1") -> pl.DataFrame:
    return pl.read_parquet(Path(outdir) / f"{symbol}_{version}.parquet")
