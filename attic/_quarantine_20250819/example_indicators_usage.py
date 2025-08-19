#!/usr/bin/env python3
"""
Example Usage of Enhanced Technical Indicators

This script demonstrates how to use the centralized technical indicators
from utils/indicators.py instead of calculating them inline.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.indicators import (
    adx,
    atr,
    bollinger_bands,
    calculate_all_indicators,
    cci,
    ichimoku,
    macd,
    mfi,
    obv,
    roc,
    rsi,
    stochastic,
    vwap,
    williams_r,
)


def create_sample_data(n_days: int = 252) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    np.random.seed(42)

    # Generate price data with some trend and volatility
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

    # Create a trending price series with some volatility
    trend = np.linspace(100, 150, n_days)
    noise = np.random.randn(n_days) * 2
    close_prices = trend + noise

    # Generate OHLC data
    data = pd.DataFrame(
        {
            "Date": dates,
            "Close": close_prices,
            "High": close_prices + np.random.uniform(0, 3, n_days),
            "Low": close_prices - np.random.uniform(0, 3, n_days),
            "Volume": np.random.uniform(1000000, 5000000, n_days),
        }
    )

    # Ensure High >= Close >= Low
    data["High"] = np.maximum(data["High"], data["Close"])
    data["Low"] = np.minimum(data["Low"], data["Close"])

    data.set_index("Date", inplace=True)
    return data


def demonstrate_basic_indicators(data: pd.DataFrame):
    """Demonstrate basic technical indicators."""
    print("=== Basic Technical Indicators ===")

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    data["Volume"]

    # RSI
    rsi_14 = rsi(close, window=14)
    print(f"RSI (14): {rsi_14.iloc[-1]:.2f}")

    # MACD
    macd_data = macd(close)
    print(f"MACD: {macd_data['macd'].iloc[-1]:.4f}")
    print(f"MACD Signal: {macd_data['signal'].iloc[-1]:.4f}")
    print(f"MACD Histogram: {macd_data['histogram'].iloc[-1]:.4f}")

    # Bollinger Bands
    bb_data = bollinger_bands(close)
    print(f"BB Upper: {bb_data['upper'].iloc[-1]:.2f}")
    print(f"BB Middle: {bb_data['middle'].iloc[-1]:.2f}")
    print(f"BB Lower: {bb_data['lower'].iloc[-1]:.2f}")

    # ATR
    atr_14 = atr(high, low, close, window=14)
    print(f"ATR (14): {atr_14.iloc[-1]:.2f}")

    print()


def demonstrate_advanced_indicators(data: pd.DataFrame):
    """Demonstrate advanced technical indicators."""
    print("=== Advanced Technical Indicators ===")

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # ADX
    adx_14 = adx(high, low, close, period=14)
    print(f"ADX (14): {adx_14.iloc[-1]:.2f}")

    # ROC
    roc_10 = roc(close, period=10)
    print(f"ROC (10): {roc_10.iloc[-1]:.2f}%")

    # MFI
    mfi_14 = mfi(high, low, close, volume, period=14)
    print(f"MFI (14): {mfi_14.iloc[-1]:.2f}")

    # Stochastic
    stoch_data = stochastic(high, low, close)
    print(f"Stochastic %K: {stoch_data['k'].iloc[-1]:.2f}")
    print(f"Stochastic %D: {stoch_data['d'].iloc[-1]:.2f}")

    # Williams %R
    williams_r_14 = williams_r(high, low, close, period=14)
    print(f"Williams %R (14): {williams_r_14.iloc[-1]:.2f}")

    # CCI
    cci_20 = cci(high, low, close, period=20)
    print(f"CCI (20): {cci_20.iloc[-1]:.2f}")

    # OBV
    obv_val = obv(close, volume)
    print(f"OBV: {obv_val.iloc[-1]:,.0f}")

    # VWAP
    vwap_val = vwap(high, low, close, volume)
    print(f"VWAP: {vwap_val.iloc[-1]:.2f}")

    print()


def demonstrate_ichimoku(data: pd.DataFrame):
    """Demonstrate Ichimoku Cloud indicators."""
    print("=== Ichimoku Cloud ===")

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    ichimoku_data = ichimoku(high, low, close)

    print(f"Tenkan-sen: {ichimoku_data['tenkan'].iloc[-1]:.2f}")
    print(f"Kijun-sen: {ichimoku_data['kijun'].iloc[-1]:.2f}")
    print(f"Senkou Span A: {ichimoku_data['senkou_span_a'].iloc[-1]:.2f}")
    print(f"Senkou Span B: {ichimoku_data['senkou_span_b'].iloc[-1]:.2f}")
    print(f"Chikou Span: {ichimoku_data['chikou_span'].iloc[-1]:.2f}")

    print()


def demonstrate_bulk_calculation(data: pd.DataFrame):
    """Demonstrate calculating all indicators at once."""
    print("=== Bulk Indicator Calculation ===")

    # Calculate all indicators
    all_indicators = calculate_all_indicators(data)

    print(f"Total indicators calculated: {len(all_indicators)}")
    print("Available indicators:")
    for indicator_name in sorted(all_indicators.keys()):
        if isinstance(all_indicators[indicator_name], pd.Series):
            print(f"  - {indicator_name}")
        elif isinstance(all_indicators[indicator_name], dict):
            print(
                f"  - {indicator_name} (dict with {len(all_indicators[indicator_name])} components)"
            )

    print()


def create_comparison_plot(data: pd.DataFrame):
    """Create a comparison plot showing old vs new indicator calculations."""
    print("=== Creating Comparison Plot ===")

    close = data["Close"]

    # Calculate indicators using centralized functions
    rsi_centralized = rsi(close, window=14)
    macd_centralized = macd(close)
    bb_centralized = bollinger_bands(close)

    # Calculate indicators inline (old way)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / (loss + 1e-8)
    rsi_inline = 100 - (100 / (1 + rs))

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Price and Bollinger Bands
    axes[0, 0].plot(close.index, close, label="Close Price", alpha=0.7)
    axes[0, 0].plot(
        bb_centralized["upper"].index,
        bb_centralized["upper"],
        label="BB Upper",
        alpha=0.7,
        linestyle="--",
    )
    axes[0, 0].plot(
        bb_centralized["lower"].index,
        bb_centralized["lower"],
        label="BB Lower",
        alpha=0.7,
        linestyle="--",
    )
    axes[0, 0].set_title("Price and Bollinger Bands")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # RSI Comparison
    axes[0, 1].plot(rsi_centralized.index, rsi_centralized, label="Centralized RSI", alpha=0.7)
    axes[0, 1].plot(rsi_inline.index, rsi_inline, label="Inline RSI", alpha=0.7, linestyle="--")
    axes[0, 1].axhline(y=70, color="r", linestyle=":", alpha=0.5, label="Overbought")
    axes[0, 1].axhline(y=30, color="g", linestyle=":", alpha=0.5, label="Oversold")
    axes[0, 1].set_title("RSI Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MACD
    axes[1, 0].plot(
        macd_centralized["macd"].index,
        macd_centralized["macd"],
        label="MACD",
        alpha=0.7,
    )
    axes[1, 0].plot(
        macd_centralized["signal"].index,
        macd_centralized["signal"],
        label="Signal",
        alpha=0.7,
    )
    axes[1, 0].bar(
        macd_centralized["histogram"].index,
        macd_centralized["histogram"],
        label="Histogram",
        alpha=0.5,
        width=1,
    )
    axes[1, 0].set_title("MACD")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Volume
    axes[1, 1].bar(data.index, data["Volume"], alpha=0.5, label="Volume")
    axes[1, 1].set_title("Volume")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("indicators_comparison.png", dpi=300, bbox_inches="tight")
    print("Comparison plot saved as 'indicators_comparison.png'")
    plt.show()


def main():
    """Main demonstration function."""
    print("Technical Indicators Usage Example")
    print("=" * 50)

    # Create sample data
    data = create_sample_data(252)
    print(f"Created sample data with {len(data)} days")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print()

    # Demonstrate different types of indicators
    demonstrate_basic_indicators(data)
    demonstrate_advanced_indicators(data)
    demonstrate_ichimoku(data)
    demonstrate_bulk_calculation(data)

    # Create comparison plot
    try:
        create_comparison_plot(data)
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

    print("=== Migration Benefits ===")
    print("✅ Centralized indicator calculations")
    print("✅ Consistent behavior across the codebase")
    print("✅ Easier testing and validation")
    print("✅ Better performance with optimized functions")
    print("✅ Reduced code duplication")
    print("✅ Enhanced maintainability")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
