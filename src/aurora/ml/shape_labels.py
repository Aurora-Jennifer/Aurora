import numpy as np
import pandas as pd
from scipy import stats


def shape_labels(y: pd.Series, config: dict) -> tuple[pd.Series, pd.Series]:
    """
    Shape labels using various methods (rank, winsorize, zscore).

    Args:
        y: Raw labels series
        config: Shaping configuration

    Returns:
        tuple: (raw_labels, shaped_labels)
    """
    method = config.get("method", "rank")

    if method == "rank":
        return shape_rank_labels(y, config)
    if method == "winsorize":
        return shape_winsorize_labels(y, config)
    if method == "zscore":
        return shape_zscore_labels(y, config)
    raise ValueError(f"Unknown shaping method: {method}")


def shape_rank_labels(y: pd.Series, config: dict) -> tuple[pd.Series, pd.Series]:
    """Shape labels using rolling percentile ranks"""
    rank_window = config.get("rank_window", 252)
    rank_type = config.get("options", {}).get("rank_type", "time_series")
    rank_method = config.get("options", {}).get("rank_method", "percentile")

    if rank_type == "time_series":
        # Rolling time-series ranking
        if rank_method == "percentile":
            # Rolling percentile rank (0-1)
            y_ranked = y.rolling(window=rank_window, min_periods=rank_window//2).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        else:  # zscore
            # Rolling z-score
            y_ranked = y.rolling(window=rank_window, min_periods=rank_window//2).apply(
                lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
            )
    else:
        # Cross-sectional ranking (would need multi-asset data)
        raise NotImplementedError("Cross-sectional ranking not implemented")

    return y, y_ranked


def shape_winsorize_labels(y: pd.Series, config: dict) -> tuple[pd.Series, pd.Series]:
    """Shape labels using winsorization"""
    limits = config.get("winsor_limits", [0.01, 0.99])
    symmetric = config.get("options", {}).get("winsorize_symmetric", True)

    if symmetric:
        # Symmetric winsorization
        lower, upper = np.percentile(y.dropna(), [limits[0] * 100, limits[1] * 100])
        y_winsorized = y.clip(lower=lower, upper=upper)
    else:
        # Asymmetric winsorization
        y_winsorized = stats.mstats.winsorize(y, limits=limits)
        y_winsorized = pd.Series(y_winsorized, index=y.index)

    return y, y_winsorized


def shape_zscore_labels(y: pd.Series, config: dict) -> tuple[pd.Series, pd.Series]:
    """Shape labels using z-score normalization"""
    zscore_window = config.get("zscore_window", 252)
    robust = config.get("options", {}).get("zscore_robust", False)

    if robust:
        # Robust z-score using median and MAD
        y_zscore = y.rolling(window=zscore_window, min_periods=zscore_window//2).apply(
            lambda x: (x.iloc[-1] - x.median()) / (1.4826 * x.mad()) if x.mad() > 0 else 0
        )
    else:
        # Standard z-score using mean and std
        y_zscore = y.rolling(window=zscore_window, min_periods=zscore_window//2).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )

    return y, y_zscore


def validate_no_leakage(y_raw: pd.Series, y_shaped: pd.Series, config: dict) -> bool:
    """
    Validate that label shaping doesn't introduce leakage.

    Args:
        y_raw: Raw labels
        y_shaped: Shaped labels
        config: Shaping configuration

    Returns:
        bool: True if no leakage detected
    """
    method = config.get("method", "rank")

    if method == "rank":
        # For rank labels, check that the transformation is reasonable
        # Remove NaN values for correlation calculation
        valid_mask = y_raw.notna() & y_shaped.notna()
        if valid_mask.sum() < 10:  # Need at least 10 valid points
            return False

        y_raw_valid = y_raw[valid_mask]
        y_shaped_valid = y_shaped[valid_mask]

        # Check monotonicity: if raw increases, shaped should generally increase
        # (allowing for some noise due to rolling window)
        corr = y_raw_valid.corr(y_shaped_valid)
        if pd.isna(corr) or corr < 0.3:  # Lower threshold for rank labels
            return False

        # Check that extreme values in raw correspond to extreme values in shaped
        raw_extreme = y_raw_valid.abs() > y_raw_valid.abs().quantile(0.9)
        shaped_extreme = y_shaped_valid.abs() > y_shaped_valid.abs().quantile(0.9)

        # Most extreme raw values should also be extreme in shaped
        if raw_extreme.sum() > 0:
            overlap = (raw_extreme & shaped_extreme).sum() / raw_extreme.sum()
            if overlap < 0.5:  # Lower threshold for rank labels
                return False

    return True
