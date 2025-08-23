"""
Technical Indicators Utilities

Pure functions for computing technical indicators and statistical calculations.
All functions are vectorized and optimized for pandas/numpy operations.
Enhanced with comprehensive indicator coverage and performance optimizations.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


def rolling_mean(
    data: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.Series | pd.DataFrame:
    """
    Compute rolling mean with configurable minimum periods.

    Args:
        data: Input series or dataframe
        window: Rolling window size
        min_periods: Minimum periods for calculation

    Returns:
        Rolling mean series or dataframe
    """
    if min_periods is None:
        min_periods = window
    return data.rolling(window=window, min_periods=min_periods).mean()


def rolling_std(
    data: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.Series | pd.DataFrame:
    """
    Compute rolling standard deviation.

    Args:
        data: Input series or dataframe
        window: Rolling window size
        min_periods: Minimum periods for calculation

    Returns:
        Rolling standard deviation
    """
    if min_periods is None:
        min_periods = window
    return data.rolling(window=window, min_periods=min_periods).std()


def rolling_median(
    data: pd.Series | pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.Series | pd.DataFrame:
    """
    Compute rolling median.

    Args:
        data: Input series or dataframe
        window: Rolling window size
        min_periods: Minimum periods for calculation

    Returns:
        Rolling median
    """
    if min_periods is None:
        min_periods = window
    return data.rolling(window=window, min_periods=min_periods).median()


def zscore(
    data: pd.Series | pd.DataFrame,
    window: int | None = None,
    fill_method: str = "ffill",
) -> pd.Series | pd.DataFrame:
    """
    Compute z-score (standardized values).

    Args:
        data: Input series or dataframe
        window: Rolling window for mean/std calculation (None for global)
        fill_method: Method to fill NaN values

    Returns:
        Z-score series or dataframe
    """
    if window is None:
        # Global z-score
        mean_val = data.mean()
        std_val = data.std()
        zscore_data = (data - mean_val) / (std_val + 1e-8)
    else:
        # Rolling z-score
        rolling_mean_val = rolling_mean(data, window)
        rolling_std_val = rolling_std(data, window)
        zscore_data = (data - rolling_mean_val) / (rolling_std_val + 1e-8)

    return zscore_data.fillna(method=fill_method)


def winsorize(
    data: pd.Series | pd.DataFrame,
    limits: tuple[float, float] = (0.05, 0.05),
    axis: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Winsorize data by capping extreme values.

    Args:
        data: Input series or dataframe
        limits: Tuple of (lower_limit, upper_limit) as fractions
        axis: Axis for dataframe operations

    Returns:
        Winsorized data
    """
    if isinstance(data, pd.Series):
        return pd.Series(stats.mstats.winsorize(data, limits=limits))
    return pd.DataFrame(
        stats.mstats.winsorize(data, limits=limits, axis=axis),
        index=data.index,
        columns=data.columns,
    )


def normalize(
    data: pd.Series | pd.DataFrame,
    method: str = "minmax",
    window: int | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Normalize data using various methods.

    Args:
        data: Input series or dataframe
        method: Normalization method ('minmax', 'zscore', 'robust')
        window: Rolling window for normalization (None for global)

    Returns:
        Normalized data
    """
    if method == "minmax":
        if window is None:
            min_val = data.min()
            max_val = data.max()
            return (data - min_val) / (max_val - min_val + 1e-8)
        rolling_min = data.rolling(window=window, min_periods=window // 2).min()
        rolling_max = data.rolling(window=window, min_periods=window // 2).max()
        return (data - rolling_min) / (rolling_max - rolling_min + 1e-8)

    if method == "zscore":
        return zscore(data, window=window)

    if method == "robust":
        if window is None:
            median_val = data.median()
            mad_val = np.median(np.abs(data - median_val))
            return (data - median_val) / (mad_val + 1e-8)
        median_series = data.rolling(window=window, min_periods=window // 2).median()
        rolling_mad = data.rolling(window=window, min_periods=window // 2).apply(
            lambda x: np.median(np.abs(x - x.median()))
        )
        return (data - median_series) / (rolling_mad + 1e-8)

    raise ValueError(f"Unknown normalization method: {method}")


def rsi(prices: pd.Series, window: int = 14, fill_method: str = "ffill") -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        prices: Price series
        window: RSI window (default: 14)
        fill_method: Method to fill NaN values

    Returns:
        RSI series
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()

    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    if fill_method == "ffill":
        return rsi.ffill()
    if fill_method == "bfill":
        return rsi.bfill()
    return rsi


def macd(
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

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: ATR window (default: 14)

    Returns:
        ATR series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()



def bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> dict[str, pd.Series]:
    """
    Compute Bollinger Bands.

    Args:
        prices: Price series
        window: Rolling window (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        Dictionary with upper, middle, and lower bands
    """
    middle = rolling_mean(prices, window)
    std = rolling_std(prices, window)

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return {"upper": upper, "middle": middle, "lower": lower}


def pct_change(
    data: pd.Series | pd.DataFrame,
    periods: int = 1,
    fill_method: str | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Compute percentage change with configurable fill method.

    Args:
        data: Input series or dataframe
        periods: Number of periods for change calculation
        fill_method: Method to fill NaN values (None for no filling)

    Returns:
        Percentage change series or dataframe
    """
    pct = data.pct_change(periods=periods, fill_method=fill_method)
    if fill_method is not None:
        pct = pct.fillna(method=fill_method)
    return pct


def lag(data: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    """
    Lag data by specified number of periods.

    Args:
        data: Input series or dataframe
        periods: Number of periods to lag

    Returns:
        Lagged data
    """
    return data.shift(periods)


def lead(data: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    """
    Lead data by specified number of periods.

    Args:
        data: Input series or dataframe
        periods: Number of periods to lead

    Returns:
        Led data
    """
    return data.shift(-periods)


def diff(data: pd.Series | pd.DataFrame, periods: int = 1) -> pd.Series | pd.DataFrame:
    """
    Compute difference between current and lagged values.

    Args:
        data: Input series or dataframe
        periods: Number of periods for difference calculation

    Returns:
        Difference series or dataframe
    """
    return data.diff(periods)


def momentum(prices: pd.Series, periods: int = 10) -> pd.Series:
    """
    Compute price momentum.

    Args:
        prices: Price series
        periods: Number of periods for momentum calculation

    Returns:
        Momentum series
    """
    return pct_change(prices, periods=periods)


def volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Compute rolling volatility.

    Args:
        returns: Return series
        window: Rolling window size
        annualize: Whether to annualize volatility (√252)

    Returns:
        Volatility series
    """
    vol = rolling_std(returns, window)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


# NEW ENHANCED INDICATORS


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX).

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ADX period (default: 14)

    Returns:
        ADX series
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low

    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)

    # Smoothed values
    tr_smooth = tr.rolling(window=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()

    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)

    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    return dx.rolling(window=period).mean()



def roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Compute Rate of Change (ROC).

    Args:
        prices: Price series
        period: ROC period (default: 10)

    Returns:
        ROC series
    """
    return ((prices / prices.shift(period)) - 1) * 100


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Money Flow Index (MFI).

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        period: MFI period (default: 14)

    Returns:
        MFI series
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    return 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))



def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, pd.Series]:
    """
    Compute Stochastic Oscillator.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)

    Returns:
        Dictionary with %K and %D series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))
    d_percent = k_percent.rolling(window=d_period).mean()

    return {"k": k_percent, "d": d_percent}


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Williams %R.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Williams %R period (default: 14)

    Returns:
        Williams %R series
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    return -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-8))



def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute Commodity Channel Index (CCI).

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: CCI period (default: 20)

    Returns:
        CCI series
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )

    return (typical_price - sma_tp) / (0.015 * mean_deviation)



def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Compute On-Balance Volume (OBV).

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        OBV series
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    return obv


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int | None = None,
) -> pd.Series:
    """
    Compute Volume Weighted Average Price (VWAP).

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        window: Rolling window (None for cumulative)

    Returns:
        VWAP series
    """
    typical_price = (high + low + close) / 3
    price_volume = typical_price * volume

    if window is None:
        # Cumulative VWAP
        cumulative_pv = price_volume.cumsum()
        cumulative_volume = volume.cumsum()
        vwap = cumulative_pv / cumulative_volume
    else:
        # Rolling VWAP
        rolling_pv = price_volume.rolling(window=window).sum()
        rolling_volume = volume.rolling(window=window).sum()
        vwap = rolling_pv / rolling_volume

    return vwap


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26,
) -> dict[str, pd.Series]:
    """
    Compute Ichimoku Cloud components.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        tenkan_period: Tenkan-sen period (default: 9)
        kijun_period: Kijun-sen period (default: 26)
        senkou_span_b_period: Senkou Span B period (default: 52)
        displacement: Displacement for Senkou spans (default: 26)

    Returns:
        Dictionary with Ichimoku components
    """
    # Tenkan-sen (Conversion Line)
    tenkan = (
        high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()
    ) / 2

    # Kijun-sen (Base Line)
    kijun = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan + kijun) / 2).shift(displacement)

    # Senkou Span B (Leading Span B)
    senkou_span_b = (
        (
            high.rolling(window=senkou_span_b_period).max()
            + low.rolling(window=senkou_span_b_period).min()
        )
        / 2
    ).shift(displacement)

    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-displacement)

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }


def calculate_all_indicators(
    data: pd.DataFrame,
    price_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    volume_col: str = "Volume",
) -> dict[str, pd.Series | dict[str, pd.Series]]:
    """
    Calculate all technical indicators for a given dataset.

    Args:
        data: DataFrame with OHLCV data
        price_col: Column name for close prices
        high_col: Column name for high prices
        low_col: Column name for low prices
        volume_col: Column name for volume

    Returns:
        Dictionary with all calculated indicators
    """
    close = data[price_col]
    high = data[high_col]
    low = data[low_col]
    volume = data[volume_col]

    indicators = {}

    # Basic indicators
    indicators["sma_20"] = rolling_mean(close, 20)
    indicators["sma_50"] = rolling_mean(close, 50)
    indicators["sma_200"] = rolling_mean(close, 200)

    # Price ratios
    indicators["price_to_sma20"] = close / indicators["sma_20"]
    indicators["price_to_sma50"] = close / indicators["sma_50"]
    indicators["sma20_to_sma50"] = indicators["sma_20"] / indicators["sma_50"]

    # Volatility
    returns = pct_change(close)
    indicators["volatility_20"] = volatility(returns, 20)
    indicators["volatility_50"] = volatility(returns, 50)

    # RSI
    indicators["rsi_14"] = rsi(close, 14)
    indicators["rsi_21"] = rsi(close, 21)

    # MACD
    macd_data = macd(close)
    indicators["macd"] = macd_data["macd"]
    indicators["macd_signal"] = macd_data["signal"]
    indicators["macd_histogram"] = macd_data["histogram"]

    # Bollinger Bands
    bb_data = bollinger_bands(close)
    indicators["bb_upper"] = bb_data["upper"]
    indicators["bb_middle"] = bb_data["middle"]
    indicators["bb_lower"] = bb_data["lower"]
    indicators["bb_position"] = (close - bb_data["lower"]) / (bb_data["upper"] - bb_data["lower"])

    # ATR
    indicators["atr_14"] = atr(high, low, close, 14)

    # ADX
    indicators["adx_14"] = adx(high, low, close, 14)

    # ROC
    indicators["roc_10"] = roc(close, 10)

    # MFI
    indicators["mfi_14"] = mfi(high, low, close, volume, 14)

    # Stochastic
    stoch_data = stochastic(high, low, close)
    indicators["stoch_k"] = stoch_data["k"]
    indicators["stoch_d"] = stoch_data["d"]

    # Williams %R
    indicators["williams_r_14"] = williams_r(high, low, close, 14)

    # CCI
    indicators["cci_20"] = cci(high, low, close, 20)

    # OBV
    indicators["obv"] = obv(close, volume)

    # VWAP
    indicators["vwap"] = vwap(high, low, close, volume)

    # Ichimoku
    ichimoku_data = ichimoku(high, low, close)
    indicators["ichimoku"] = ichimoku_data

    return indicators


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    import pandas as pd

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)

    # Test indicators
    print("Testing technical indicators...")

    # Rolling statistics
    sma_20 = rolling_mean(prices, 20)
    print(f"SMA 20: {sma_20.iloc[-1]:.2f}")

    # RSI
    rsi_val = rsi(prices)
    print(f"RSI: {rsi_val.iloc[-1]:.2f}")

    # MACD
    macd_data = macd(prices)
    print(f"MACD: {macd_data['macd'].iloc[-1]:.4f}")

    # Bollinger Bands
    bb_data = bollinger_bands(prices)
    print(f"BB Upper: {bb_data['upper'].iloc[-1]:.2f}")

    print("All indicators computed successfully!")
