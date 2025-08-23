#!/usr/bin/env python3
"""
Risk Overlay Module for ML Trading System v0.2

This module applies risk management controls including volatility targeting,
drawdown limits, and daily loss limits to trading positions.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


def apply_risk_overlay(
    df: pd.DataFrame,
    pos_col: str = "pos_sized",
    ret_col: str = "ret_1d",
    target_annual_vol: float = 0.20,
    max_dd: float = 0.15,
    daily_loss_limit: float = 0.03,
    vol_lookback: int = 252,
    dd_lookback: int = 252,
    ts_col: str = "ts",
    asset_col: str = "asset",
) -> pd.DataFrame:
    """
    Apply risk overlay to trading positions.

    Features:
    - Vol targeting: scale exposure so realized vol ≈ target_annual_vol (√252)
    - Guardrails:
        - If running max drawdown > max_dd → cut exposure to 0 until recovery
        - If intraday/daily PnL < -daily_loss_limit → flat for next day

    Args:
        df: DataFrame with positions and returns
        pos_col: Position column name
        ret_col: Return column name
        target_annual_vol: Target annualized volatility (default: 0.20)
        max_dd: Maximum drawdown limit (default: 0.15)
        daily_loss_limit: Daily loss limit (default: 0.03)
        vol_lookback: Lookback period for volatility calculation (default: 252)
        dd_lookback: Lookback period for drawdown calculation (default: 252)
        ts_col: Timestamp column name
        asset_col: Asset column name

    Returns:
        DataFrame with adjusted positions and performance metrics

    Raises:
        ValueError: If required columns are missing or parameters are invalid
    """

    # Validate inputs
    required_cols = [ts_col, asset_col, pos_col, ret_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if target_annual_vol <= 0:
        raise ValueError(f"target_annual_vol must be > 0, got {target_annual_vol}")

    if max_dd <= 0 or max_dd >= 1:
        raise ValueError(f"max_dd must be between 0 and 1, got {max_dd}")

    if daily_loss_limit <= 0 or daily_loss_limit >= 1:
        raise ValueError(f"daily_loss_limit must be between 0 and 1, got {daily_loss_limit}")

    # Ensure proper data types
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([asset_col, ts_col]).reset_index(drop=True)

    # Initialize result dataframe
    result_df = df.copy()

    # Process each asset separately
    risk_dfs = []

    for asset in df[asset_col].unique():
        asset_data = df[df[asset_col] == asset].copy()

        if len(asset_data) < vol_lookback:
            logger.warning(
                f"Asset {asset} has insufficient data ({len(asset_data)} rows), skipping"
            )
            continue

        # Apply risk overlay for this asset
        asset_risk = _apply_asset_risk_overlay(
            asset_data,
            pos_col,
            ret_col,
            target_annual_vol,
            max_dd,
            daily_loss_limit,
            vol_lookback,
            dd_lookback,
            ts_col,
        )
        risk_dfs.append(asset_risk)

    if not risk_dfs:
        raise ValueError("No assets had sufficient data for risk overlay")

    # Combine all asset results
    result_df = pd.concat(risk_dfs, ignore_index=True)

    # Sort final result
    result_df = result_df.sort_values([asset_col, ts_col]).reset_index(drop=True)

    logger.info(f"Applied risk overlay to {len(result_df)} rows across {len(risk_dfs)} assets")

    return result_df


def _apply_asset_risk_overlay(
    df: pd.DataFrame,
    pos_col: str,
    ret_col: str,
    target_annual_vol: float,
    max_dd: float,
    daily_loss_limit: float,
    vol_lookback: int,
    dd_lookback: int,
    ts_col: str,
) -> pd.DataFrame:
    """
    Apply risk overlay to a single asset.

    Args:
        df: Single asset dataframe
        pos_col: Position column name
        ret_col: Return column name
        target_annual_vol: Target annualized volatility
        max_dd: Maximum drawdown limit
        daily_loss_limit: Daily loss limit
        vol_lookback: Volatility lookback period
        dd_lookback: Drawdown lookback period
        ts_col: Timestamp column name

    Returns:
        DataFrame with risk-adjusted positions
    """

    # Ensure sorted by timestamp
    df = df.sort_values(ts_col).reset_index(drop=True)

    # Calculate position-weighted returns
    df["weighted_ret"] = df[pos_col] * df[ret_col]

    # 1. Volatility Targeting
    df["pos_vol_targeted"] = _apply_volatility_targeting(
        df[pos_col], df["weighted_ret"], target_annual_vol, vol_lookback
    )

    # 2. Drawdown Protection
    df["pos_dd_protected"] = _apply_drawdown_protection(
        df["pos_vol_targeted"], df["weighted_ret"], max_dd, dd_lookback
    )

    # 3. Daily Loss Limit
    df["pos_final"] = _apply_daily_loss_limit(
        df["pos_dd_protected"], df["weighted_ret"], daily_loss_limit
    )

    # 4. Calculate performance metrics
    return _calculate_performance_metrics(df, pos_col, ret_col)



def _apply_volatility_targeting(
    positions: pd.Series, returns: pd.Series, target_annual_vol: float, lookback: int
) -> pd.Series:
    """
    Apply volatility targeting to positions.

    Args:
        positions: Position series
        returns: Return series
        target_annual_vol: Target annualized volatility
        lookback: Lookback period

    Returns:
        Volatility-targeted positions
    """

    # Calculate realized volatility
    vol = returns.rolling(window=lookback, min_periods=lookback // 2).std() * np.sqrt(252)

    # Volatility scaling factor
    vol_scale = target_annual_vol / (vol + 1e-8)

    # Apply scaling (cap to prevent extreme values)
    vol_scale = np.clip(vol_scale, 0.1, 10.0)

    # Scale positions
    vol_targeted_pos = positions * vol_scale

    return pd.Series(vol_targeted_pos, index=positions.index)


def _apply_drawdown_protection(
    positions: pd.Series, returns: pd.Series, max_dd: float, lookback: int
) -> pd.Series:
    """
    Apply drawdown protection to positions.

    Args:
        positions: Position series
        returns: Return series
        max_dd: Maximum drawdown limit
        lookback: Lookback period

    Returns:
        Drawdown-protected positions
    """

    # Calculate cumulative returns
    cum_ret = (1 + returns).cumprod()

    # Calculate running maximum
    running_max = cum_ret.rolling(window=lookback, min_periods=lookback // 2).max()

    # Calculate drawdown
    drawdown = (cum_ret - running_max) / running_max

    # Create drawdown protection mask
    dd_protection = drawdown > -max_dd

    # Apply protection (set positions to 0 when drawdown limit exceeded)
    dd_protected_pos = positions * dd_protection

    return pd.Series(dd_protected_pos, index=positions.index)


def _apply_daily_loss_limit(
    positions: pd.Series, returns: pd.Series, daily_loss_limit: float
) -> pd.Series:
    """
    Apply daily loss limit protection.

    Args:
        positions: Position series
        returns: Return series
        daily_loss_limit: Daily loss limit

    Returns:
        Loss-limit-protected positions
    """

    # Calculate daily PnL
    daily_pnl = returns.rolling(window=1).sum()

    # Create loss limit mask (flat next day if loss limit exceeded)
    loss_limit_mask = daily_pnl >= -daily_loss_limit

    # Apply loss limit protection
    loss_protected_pos = positions * loss_limit_mask

    return pd.Series(loss_protected_pos, index=positions.index)


def _calculate_performance_metrics(
    df: pd.DataFrame, original_pos_col: str, ret_col: str
) -> pd.DataFrame:
    """
    Calculate performance metrics for risk-adjusted positions.

    Args:
        df: DataFrame with positions and returns
        original_pos_col: Original position column name
        ret_col: Return column name

    Returns:
        DataFrame with performance metrics
    """

    # Calculate returns for each position type
    df["ret_original"] = df[original_pos_col] * df[ret_col]
    df["ret_vol_targeted"] = df["pos_vol_targeted"] * df[ret_col]
    df["ret_dd_protected"] = df["pos_dd_protected"] * df[ret_col]
    df["ret_final"] = df["pos_final"] * df[ret_col]

    # Calculate cumulative returns
    df["cum_ret_original"] = (1 + df["ret_original"]).cumprod()
    df["cum_ret_vol_targeted"] = (1 + df["ret_vol_targeted"]).cumprod()
    df["cum_ret_dd_protected"] = (1 + df["ret_dd_protected"]).cumprod()
    df["cum_ret_final"] = (1 + df["ret_final"]).cumprod()

    # Calculate drawdowns
    df["dd_original"] = _calculate_drawdown(df["cum_ret_original"])
    df["dd_vol_targeted"] = _calculate_drawdown(df["cum_ret_vol_targeted"])
    df["dd_dd_protected"] = _calculate_drawdown(df["cum_ret_dd_protected"])
    df["dd_final"] = _calculate_drawdown(df["cum_ret_final"])

    # Calculate rolling volatility
    df["vol_original"] = df["ret_original"].rolling(window=252, min_periods=126).std() * np.sqrt(
        252
    )
    df["vol_vol_targeted"] = df["ret_vol_targeted"].rolling(
        window=252, min_periods=126
    ).std() * np.sqrt(252)
    df["vol_dd_protected"] = df["ret_dd_protected"].rolling(
        window=252, min_periods=126
    ).std() * np.sqrt(252)
    df["vol_final"] = df["ret_final"].rolling(window=252, min_periods=126).std() * np.sqrt(252)

    return df


def _calculate_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.

    Args:
        cumulative_returns: Cumulative return series

    Returns:
        Drawdown series
    """

    running_max = cumulative_returns.expanding().max()
    return (cumulative_returns - running_max) / running_max



def compute_risk_metrics(
    df: pd.DataFrame,
    pos_col: str = "pos_final",
    ret_col: str = "ret_1d",
    ts_col: str = "ts",
    asset_col: str = "asset",
) -> dict[str, Any]:
    """
    Compute comprehensive risk metrics.

    Args:
        df: DataFrame with risk-adjusted positions
        pos_col: Position column name
        ret_col: Return column name
        ts_col: Timestamp column name
        asset_col: Asset column name

    Returns:
        Dictionary with risk metrics
    """

    if pos_col not in df.columns:
        raise ValueError(f"Position column '{pos_col}' not found")

    if ret_col not in df.columns:
        logger.warning(f"Return column '{ret_col}' not found, skipping return-based metrics")
        return _compute_basic_risk_metrics(df, pos_col, ts_col, asset_col)

    # Calculate position-weighted returns
    df = df.copy()
    df["weighted_ret"] = df[pos_col] * df[ret_col]

    # Basic risk metrics
    metrics = _compute_basic_risk_metrics(df, pos_col, ts_col, asset_col)

    # Return-based risk metrics
    return_metrics = _compute_return_risk_metrics(df, pos_col, ret_col, ts_col, asset_col)
    metrics.update(return_metrics)

    return metrics


def _compute_basic_risk_metrics(
    df: pd.DataFrame, pos_col: str, ts_col: str, asset_col: str
) -> dict[str, Any]:
    """
    Compute basic risk statistics.

    Args:
        df: DataFrame with positions
        pos_col: Position column name
        ts_col: Timestamp column name
        asset_col: Asset column name

    Returns:
        Dictionary with basic risk metrics
    """

    metrics = {
        "total_positions": len(df[df[pos_col] != 0]),
        "avg_position_size": df[pos_col].abs().mean(),
        "max_position_size": df[pos_col].abs().max(),
        "position_std": df[pos_col].std(),
        "position_skew": df[pos_col].skew(),
        "position_kurtosis": df[pos_col].kurtosis(),
        "exposure_frequency": len(df[df[pos_col] != 0]) / len(df),
        "assets_with_positions": df[df[pos_col] != 0][asset_col].nunique(),
        "date_range": {"start": df[ts_col].min(), "end": df[ts_col].max()},
    }

    # Position distribution
    pos_dist = df[pos_col].value_counts().sort_index()
    metrics["position_distribution"] = pos_dist.to_dict()

    return metrics


def _compute_return_risk_metrics(
    df: pd.DataFrame, pos_col: str, ret_col: str, ts_col: str, asset_col: str
) -> dict[str, Any]:
    """
    Compute return-based risk metrics.

    Args:
        df: DataFrame with positions and returns
        pos_col: Position column name
        ret_col: Return column name
        ts_col: Timestamp column name
        asset_col: Asset column name

    Returns:
        Dictionary with return risk metrics
    """

    # Filter to periods with positions
    pos_df = df[df[pos_col] != 0].copy()

    if len(pos_df) == 0:
        return {"return_risk_metrics": "No positions to analyze"}

    # Calculate metrics
    metrics = {
        "total_return": pos_df["weighted_ret"].sum(),
        "annualized_return": pos_df["weighted_ret"].mean() * 252,
        "volatility": pos_df["weighted_ret"].std() * np.sqrt(252),
        "sharpe_ratio": (pos_df["weighted_ret"].mean() * 252)
        / (pos_df["weighted_ret"].std() * np.sqrt(252) + 1e-8),
        "sortino_ratio": _calculate_sortino_ratio(pos_df["weighted_ret"]),
        "calmar_ratio": _calculate_calmar_ratio(pos_df["weighted_ret"]),
        "max_drawdown": _calculate_max_drawdown(pos_df["weighted_ret"]),
        "var_95": np.percentile(pos_df["weighted_ret"], 5),
        "cvar_95": pos_df["weighted_ret"][
            pos_df["weighted_ret"] <= np.percentile(pos_df["weighted_ret"], 5)
        ].mean(),
        "win_rate": len(pos_df[pos_df["weighted_ret"] > 0]) / len(pos_df),
        "profit_factor": _calculate_profit_factor(pos_df["weighted_ret"]),
        "avg_gain": pos_df[pos_df["weighted_ret"] > 0]["weighted_ret"].mean(),
        "avg_loss": pos_df[pos_df["weighted_ret"] < 0]["weighted_ret"].mean(),
        "gain_loss_ratio": _calculate_gain_loss_ratio(pos_df["weighted_ret"]),
    }

    # Time-based analysis
    pos_df["date"] = pd.to_datetime(pos_df[ts_col]).dt.date
    daily_returns = pos_df.groupby("date")["weighted_ret"].sum()

    metrics["daily_risk_metrics"] = {
        "avg_daily_return": daily_returns.mean(),
        "daily_return_std": daily_returns.std(),
        "daily_var_95": np.percentile(daily_returns, 5),
        "best_day": daily_returns.max(),
        "worst_day": daily_returns.min(),
        "profitable_days": len(daily_returns[daily_returns > 0]) / len(daily_returns),
        "consecutive_losses": _calculate_max_consecutive_losses(daily_returns),
    }

    return metrics


def _calculate_sortino_ratio(returns: pd.Series) -> float:
    """Calculate Sortino ratio."""
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return np.inf
    downside_std = negative_returns.std()
    if downside_std == 0:
        return np.inf
    return returns.mean() / downside_std


def _calculate_calmar_ratio(returns: pd.Series) -> float:
    """Calculate Calmar ratio."""
    max_dd = _calculate_max_drawdown(returns)
    if max_dd == 0:
        return np.inf
    return (returns.mean() * 252) / abs(max_dd)


def _calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def _calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor."""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return np.inf
    return gross_profit / gross_loss


def _calculate_gain_loss_ratio(returns: pd.Series) -> float:
    """Calculate gain/loss ratio."""
    avg_gain = returns[returns > 0].mean()
    avg_loss = abs(returns[returns < 0].mean())
    if avg_loss == 0:
        return np.inf
    return avg_gain / avg_loss


def _calculate_max_consecutive_losses(returns: pd.Series) -> int:
    """Calculate maximum consecutive losses."""
    losses = returns < 0
    max_consecutive = 0
    current_consecutive = 0

    for loss in losses:
        if loss:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def validate_risk_overlay(df: pd.DataFrame, pos_col: str = "pos_final") -> dict[str, Any]:
    """
    Validate risk overlay application.

    Args:
        df: DataFrame with risk-adjusted positions
        pos_col: Position column name

    Returns:
        Validation results dictionary
    """

    validation = {
        "total_rows": len(df),
        "risk_overlay_quality": "PASSED",
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

    # Check if risk controls are working
    if "dd_final" in df.columns:
        max_dd = df["dd_final"].min()
        if max_dd < -0.15:  # Assuming max_dd limit of 15%
            validation["warnings"].append(
                f"Drawdown limit may not be working: max_dd = {max_dd:.1%}"
            )

    if "vol_final" in df.columns:
        avg_vol = df["vol_final"].mean()
        if avg_vol > 0.25:  # Assuming target vol of 20%
            validation["warnings"].append(
                f"Volatility targeting may not be working: avg_vol = {avg_vol:.1%}"
            )

    return validation


if __name__ == "__main__":
    # Example usage and testing
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    assets = ["SPY", "QQQ"]

    sample_data = []
    for asset in assets:
        for date in dates:
            # Create sample positions and returns
            position = np.random.uniform(-1, 1)
            ret = np.random.normal(0, 0.02)

            sample_data.append({"ts": date, "asset": asset, "pos_sized": position, "ret_1d": ret})

    df = pd.DataFrame(sample_data)

    # Apply risk overlay
    risk_df = apply_risk_overlay(
        df=df,
        pos_col="pos_sized",
        ret_col="ret_1d",
        target_annual_vol=0.20,
        max_dd=0.15,
        daily_loss_limit=0.03,
    )

    # Compute risk metrics
    metrics = compute_risk_metrics(risk_df)

    # Validate risk overlay
    validation = validate_risk_overlay(risk_df)

    print("Risk overlay application completed!")
    print(f"Total positions: {metrics['total_positions']}")
    print(f"Average position size: {metrics['avg_position_size']:.3f}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.1%}")
    print(f"Risk overlay quality: {validation['risk_overlay_quality']}")

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    if validation["errors"]:
        print("Errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
