import pandas as pd


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


