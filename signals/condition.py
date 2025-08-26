#!/usr/bin/env python3
"""
Signal Conditioning Module for ML Trading System v0.2

This module converts ML predictions into trading signals with position sizing,
confidence thresholds, and position decay mechanisms.
"""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


def condition_signal(
    df: pd.DataFrame,
    score_col: str = "y_cal",
    vol_col: str = "vol_20",
    conf_q: float = 0.7,
    max_hold: int = 5,
    decay: Literal["linear", "exponential"] = "linear",
    target_vol: float = 0.20,
    position_cap: float = 1.0,
    ts_col: str = "ts",
    asset_col: str = "asset",
) -> pd.DataFrame:
    """
    Condition ML predictions into trading signals with position sizing and decay.

    Logic:
    - Enter long if score >= conf_q quantile per date; short if <= (1-conf_q)
    - Position size = zscore(score) * (target_vol / vol_20), capped [-1,1]
    - Hold position up to max_hold bars with decay (linear or exponential)

    Args:
        df: DataFrame with predictions and features
        score_col: Column name for prediction scores
        vol_col: Column name for volatility estimates
        conf_q: Confidence quantile for signal generation (default: 0.7)
        max_hold: Maximum holding period in bars (default: 5)
        decay: Position decay method ("linear" or "exponential")
        target_vol: Target annualized volatility (default: 0.20)
        position_cap: Maximum position size (default: 1.0)
        ts_col: Timestamp column name
        asset_col: Asset column name

    Returns:
        DataFrame with ["pos_raw", "pos_sized"] columns

    Raises:
        ValueError: If required columns are missing or parameters are invalid
    """

    # Validate inputs
    required_cols = [ts_col, asset_col, score_col, vol_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not (0 < conf_q < 1):
        raise ValueError(f"conf_q must be between 0 and 1, got {conf_q}")

    if max_hold < 1:
        raise ValueError(f"max_hold must be >= 1, got {max_hold}")

    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")

    if position_cap <= 0:
        raise ValueError(f"position_cap must be > 0, got {position_cap}")

    # Ensure proper data types
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([asset_col, ts_col]).reset_index(drop=True)

    # Initialize result dataframe
    result_df = df.copy()

    # Process each asset separately
    signal_dfs = []

    for asset in df[asset_col].unique():
        asset_data = df[df[asset_col] == asset].copy()

        if len(asset_data) < max_hold:
            logger.warning(
                f"Asset {asset} has insufficient data ({len(asset_data)} rows), skipping"
            )
            continue

        # Generate signals for this asset
        asset_signals = _generate_asset_signals(
            asset_data,
            score_col,
            vol_col,
            conf_q,
            max_hold,
            decay,
            target_vol,
            position_cap,
            ts_col,
        )
        signal_dfs.append(asset_signals)

    if not signal_dfs:
        raise ValueError("No assets had sufficient data for signal generation")

    # Combine all asset signals
    result_df = pd.concat(signal_dfs, ignore_index=True)

    # Sort final result
    result_df = result_df.sort_values([asset_col, ts_col]).reset_index(drop=True)

    logger.info(f"Generated signals for {len(result_df)} rows across {len(signal_dfs)} assets")

    return result_df


def _generate_asset_signals(
    df: pd.DataFrame,
    score_col: str,
    vol_col: str,
    conf_q: float,
    max_hold: int,
    decay: str,
    target_vol: float,
    position_cap: float,
    ts_col: str,
) -> pd.DataFrame:
    """
    Generate trading signals for a single asset.

    Args:
        df: Single asset dataframe
        score_col: Score column name
        vol_col: Volatility column name
        conf_q: Confidence quantile
        max_hold: Maximum holding period
        decay: Decay method
        target_vol: Target volatility
        position_cap: Position cap
        ts_col: Timestamp column name

    Returns:
        DataFrame with signals
    """

    # Ensure sorted by timestamp
    df = df.sort_values(ts_col).reset_index(drop=True)

    # 1. Generate raw position signals based on confidence quantiles
    df["pos_raw"] = _generate_raw_signals(df[score_col], conf_q)

    # 2. Apply position sizing based on volatility targeting
    df["pos_sized"] = _apply_position_sizing(
        df["pos_raw"], df[score_col], df[vol_col], target_vol, position_cap
    )

    # 3. Apply position decay
    df["pos_sized"] = _apply_position_decay(df["pos_sized"], max_hold, decay)

    return df


def _generate_raw_signals(scores: pd.Series, conf_q: float) -> pd.Series:
    """
    Generate raw position signals based on score quantiles.

    Args:
        scores: Prediction scores
        conf_q: Confidence quantile

    Returns:
        Raw position signals (-1, 0, 1)
    """

    # Calculate quantiles for each date
    signals = pd.Series(0, index=scores.index)

    # Group by date if available, otherwise use global quantiles
    if hasattr(scores, "groupby"):
        # If we have a multi-index with dates, group by date
        try:
            date_groups = scores.groupby(level=0)
            for date, group in date_groups:
                if len(group) > 1:  # Need multiple observations for quantiles
                    upper_q = group.quantile(conf_q)
                    lower_q = group.quantile(1 - conf_q)

                    # Long signal
                    long_mask = (scores.loc[date] >= upper_q) & (upper_q > 0.5)
                    signals.loc[date] = np.where(long_mask, 1, signals.loc[date])

                    # Short signal
                    short_mask = (scores.loc[date] <= lower_q) & (lower_q < 0.5)
                    signals.loc[date] = np.where(short_mask, -1, signals.loc[date])
        except Exception:
            # Fallback to global quantiles
            pass

    # Global quantiles (fallback)
    if signals.sum() == 0:  # No signals generated yet
        upper_q = scores.quantile(conf_q)
        lower_q = scores.quantile(1 - conf_q)

        # Long signal
        long_mask = (scores >= upper_q) & (upper_q > 0.5)
        signals = np.where(long_mask, 1, signals)

        # Short signal
        short_mask = (scores <= lower_q) & (lower_q < 0.5)
        signals = np.where(short_mask, -1, signals)

    return pd.Series(signals, index=scores.index)


def _apply_position_sizing(
    pos_raw: pd.Series,
    scores: pd.Series,
    volatility: pd.Series,
    target_vol: float,
    position_cap: float,
) -> pd.Series:
    """
    Apply position sizing based on volatility targeting.

    Args:
        pos_raw: Raw position signals
        scores: Prediction scores
        volatility: Volatility estimates
        target_vol: Target volatility
        position_cap: Position cap

    Returns:
        Sized position signals
    """

    # Calculate z-score of scores
    score_mean = scores.rolling(window=252, min_periods=50).mean()
    score_std = scores.rolling(window=252, min_periods=50).std()
    score_zscore = (scores - score_mean) / (score_std + 1e-8)

    # Volatility scaling factor
    vol_scale = target_vol / (volatility + 1e-8)

    # Position size = zscore * vol_scale * raw_signal
    pos_sized = score_zscore * vol_scale * pos_raw

    # Apply position cap
    pos_sized = np.clip(pos_sized, -position_cap, position_cap)

    return pd.Series(pos_sized, index=pos_raw.index)


def _apply_position_decay(positions: pd.Series, max_hold: int, decay: str) -> pd.Series:
    """
    Apply position decay over time.

    Args:
        positions: Position series
        max_hold: Maximum holding period
        decay: Decay method ("linear" or "exponential")

    Returns:
        Decayed position series
    """

    decayed_positions = positions.copy()

    # Find position entry points (non-zero positions preceded by zero)
    entry_points = (positions != 0) & (positions.shift(1) == 0)

    # Apply decay for each position
    for i in range(len(positions)):
        if entry_points.iloc[i]:
            # New position entered
            current_pos = positions.iloc[i]
            if current_pos != 0:
                # Apply decay for max_hold periods
                for j in range(1, min(max_hold + 1, len(positions) - i)):
                    if i + j < len(positions):
                        if decay == "linear":
                            decay_factor = 1 - (j / max_hold)
                        elif decay == "exponential":
                            decay_factor = np.exp(-j / max_hold)
                        else:
                            decay_factor = 1

                        decayed_positions.iloc[i + j] = current_pos * decay_factor

    return decayed_positions


def compute_signal_metrics(
    df: pd.DataFrame,
    pos_col: str = "pos_sized",
    ret_col: str = "ret_1d",
    ts_col: str = "ts",
    asset_col: str = "asset",
) -> dict[str, Any]:
    """
    Compute signal performance metrics.

    Args:
        df: DataFrame with positions and returns
        pos_col: Position column name
        ret_col: Return column name
        ts_col: Timestamp column name
        asset_col: Asset column name

    Returns:
        Dictionary with signal metrics
    """

    if pos_col not in df.columns:
        raise ValueError(f"Position column '{pos_col}' not found")

    if ret_col not in df.columns:
        logger.warning(f"Return column '{ret_col}' not found, skipping return-based metrics")
        return _compute_basic_signal_metrics(df, pos_col, ts_col, asset_col)

    # Calculate position-weighted returns
    df = df.copy()
    df["weighted_ret"] = df[pos_col] * df[ret_col]

    # Basic signal metrics
    metrics = _compute_basic_signal_metrics(df, pos_col, ts_col, asset_col)

    # Return-based metrics
    return_metrics = _compute_return_metrics(df, pos_col, ret_col, ts_col, asset_col)
    metrics.update(return_metrics)

    return metrics


def _compute_basic_signal_metrics(
    df: pd.DataFrame, pos_col: str, ts_col: str, asset_col: str
) -> dict[str, Any]:
    """
    Compute basic signal statistics.

    Args:
        df: DataFrame with positions
        pos_col: Position column name
        ts_col: Timestamp column name
        asset_col: Asset column name

    Returns:
        Dictionary with basic metrics
    """

    metrics = {
        "total_signals": len(df[df[pos_col] != 0]),
        "long_signals": len(df[df[pos_col] > 0]),
        "short_signals": len(df[df[pos_col] < 0]),
        "avg_position_size": df[pos_col].abs().mean(),
        "max_position_size": df[pos_col].abs().max(),
        "position_std": df[pos_col].std(),
        "signal_frequency": len(df[df[pos_col] != 0]) / len(df),
        "assets_with_signals": df[df[pos_col] != 0][asset_col].nunique(),
        "date_range": {"start": df[ts_col].min(), "end": df[ts_col].max()},
    }

    # Position distribution
    pos_dist = df[pos_col].value_counts().sort_index()
    metrics["position_distribution"] = pos_dist.to_dict()

    return metrics


def _compute_return_metrics(
    df: pd.DataFrame, pos_col: str, ret_col: str, ts_col: str, asset_col: str
) -> dict[str, Any]:
    """
    Compute return-based signal metrics.

    Args:
        df: DataFrame with positions and returns
        pos_col: Position column name
        ret_col: Return column name
        ts_col: Timestamp column name
        asset_col: Asset column name

    Returns:
        Dictionary with return metrics
    """

    # Filter to periods with positions
    pos_df = df[df[pos_col] != 0].copy()

    if len(pos_df) == 0:
        return {"return_metrics": "No positions to analyze"}

    # Calculate metrics
    metrics = {
        "total_return": pos_df["weighted_ret"].sum(),
        "avg_return_per_signal": pos_df["weighted_ret"].mean(),
        "return_std": pos_df["weighted_ret"].std(),
        "sharpe_ratio": pos_df["weighted_ret"].mean() / (pos_df["weighted_ret"].std() + 1e-8),
        "win_rate": len(pos_df[pos_df["weighted_ret"] > 0]) / len(pos_df),
        "max_gain": pos_df["weighted_ret"].max(),
        "max_loss": pos_df["weighted_ret"].min(),
        "avg_gain": pos_df[pos_df["weighted_ret"] > 0]["weighted_ret"].mean(),
        "avg_loss": pos_df[pos_df["weighted_ret"] < 0]["weighted_ret"].mean(),
    }

    # Long vs short performance
    long_pos = pos_df[pos_df[pos_col] > 0]
    short_pos = pos_df[pos_df[pos_col] < 0]

    if len(long_pos) > 0:
        metrics["long_performance"] = {
            "count": len(long_pos),
            "avg_return": long_pos["weighted_ret"].mean(),
            "win_rate": len(long_pos[long_pos["weighted_ret"] > 0]) / len(long_pos),
        }

    if len(short_pos) > 0:
        metrics["short_performance"] = {
            "count": len(short_pos),
            "avg_return": short_pos["weighted_ret"].mean(),
            "win_rate": len(short_pos[short_pos["weighted_ret"] > 0]) / len(short_pos),
        }

    # Time-based analysis
    pos_df["date"] = pd.to_datetime(pos_df[ts_col]).dt.date
    daily_returns = pos_df.groupby("date")["weighted_ret"].sum()

    metrics["daily_metrics"] = {
        "avg_daily_return": daily_returns.mean(),
        "daily_return_std": daily_returns.std(),
        "best_day": daily_returns.max(),
        "worst_day": daily_returns.min(),
        "profitable_days": len(daily_returns[daily_returns > 0]) / len(daily_returns),
    }

    return metrics


def validate_signals(df: pd.DataFrame, pos_col: str = "pos_sized") -> dict[str, Any]:
    """
    Validate generated signals for data quality and consistency.

    Args:
        df: DataFrame with signals
        pos_col: Position column name

    Returns:
        Validation results dictionary
    """

    validation = {
        "total_rows": len(df),
        "signal_quality": "PASSED",
        "warnings": [],
        "errors": [],
    }

    # Check for missing positions
    missing_pct = df[pos_col].isnull().mean()
    if missing_pct > 0.1:
        validation["warnings"].append(f"High missing positions: {missing_pct:.1%}")

    # Check for extreme position values
    extreme_pos = df[df[pos_col].abs() > 2]
    if len(extreme_pos) > 0:
        validation["warnings"].append(f"Extreme position values: {len(extreme_pos)} rows")

    # Check for position consistency
    pos_changes = df[pos_col].diff().abs()
    large_changes = pos_changes > 1.5
    if large_changes.sum() > len(df) * 0.1:
        validation["warnings"].append("Frequent large position changes detected")

    # Check for signal clustering
    consecutive_signals = (df[pos_col] != 0).rolling(window=10).sum()
    if consecutive_signals.max() > 8:
        validation["warnings"].append("Signal clustering detected")

    # Check for data leakage (signals should not be perfectly correlated with future returns)
    if "ret_1d" in df.columns:
        correlation = df[pos_col].corr(df["ret_1d"].shift(-1))
        if abs(correlation) > 0.8:
            validation["errors"].append(f"Potential data leakage: correlation = {correlation:.3f}")
            validation["signal_quality"] = "FAILED"

    return validation


if __name__ == "__main__":
    # Example usage and testing
    import logging

    # Set up logging
    from core.utils import setup_logging
    setup_logging("logs/signals_condition.log", logging.INFO)

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    assets = ["SPY", "QQQ"]

    sample_data = []
    for asset in assets:
        for date in dates:
            # Create sample scores and volatility
            score = np.random.beta(2, 2)  # Beta distribution for scores
            vol = 0.02 + np.random.exponential(0.01)  # Volatility

            sample_data.append(
                {
                    "ts": date,
                    "asset": asset,
                    "y_cal": score,
                    "vol_20": vol,
                    "ret_1d": np.random.normal(0, vol),
                }
            )

    df = pd.DataFrame(sample_data)

    # Generate signals
    signals_df = condition_signal(
        df=df,
        score_col="y_cal",
        vol_col="vol_20",
        conf_q=0.7,
        max_hold=5,
        decay="linear",
    )

    # Compute metrics
    metrics = compute_signal_metrics(signals_df)

    # Validate signals
    validation = validate_signals(signals_df)

    print("Signal conditioning completed!")
    print(f"Total signals: {metrics['total_signals']}")
    print(f"Signal frequency: {metrics['signal_frequency']:.1%}")
    print(f"Average position size: {metrics['avg_position_size']:.3f}")
    print(f"Signal quality: {validation['signal_quality']}")

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    if validation["errors"]:
        print("Errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
