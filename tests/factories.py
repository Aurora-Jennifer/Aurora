"""
Data factories for DataSanity tests.
Generate test data with composable mutations.
"""

from datetime import UTC

import numpy as np
import pandas as pd


def base_df(n: int = 100, start: str = "2024-01-01", freq: str = "1min") -> pd.DataFrame:
    """Create base OHLCV DataFrame with realistic market data."""
    ts = pd.date_range(start, periods=n, freq=freq, tz=UTC)

    # Generate realistic price data without lookahead
    base_price = 100.0
    prices = [base_price]
    for i in range(1, n):
        # Small random walk to avoid extreme returns
        change = np.random.uniform(-0.02, 0.02)  # Max 2% change
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))

    prices = np.array(prices)

    # Create OHLCV data
    data = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "High": prices * (1 + np.random.uniform(0, 0.01, n)),
            "Low": prices * (1 - np.random.uniform(0, 0.01, n)),
            "Close": prices,
            "Volume": np.random.lognormal(10, 0.5, n),
        },
        index=ts,
    )

    # Ensure OHLC consistency
    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

    return data


def build_case(name: str, **kwargs) -> pd.DataFrame:
    """Build a test case DataFrame with specified mutations."""
    df = base_df(**kwargs)

    mutations = {
        # Price violations
        "neg_prices": lambda d: d.assign(Close=d["Close"].where(d.index != d.index[50], -50.0)),
        "extreme_prices": lambda d: d.assign(Close=d["Close"].where(d.index != d.index[50], 1e12)),
        "zero_prices": lambda d: d.assign(Close=d["Close"].where(d.index != d.index[50], 0.0)),
        # Volume violations
        "neg_volume": lambda d: d.assign(Volume=d["Volume"].where(d.index != d.index[50], -1000.0)),
        "zero_volume": lambda d: d.assign(Volume=d["Volume"].where(d.index != d.index[50], 0.0)),
        "extreme_volume": lambda d: d.assign(
            Volume=d["Volume"].where(d.index != d.index[50], 1e15)
        ),
        # NaN violations
        "nan_burst": lambda d: d.assign(
            Close=d["Close"].where(~d.index.isin(d.index[100:120]), np.nan)
        ),
        "nan_scattered": lambda d: d.assign(
            Close=d["Close"].where([i % 10 != 0 for i in range(len(d))], np.nan)
        ),
        "inf_values": lambda d: d.assign(Close=d["Close"].where(d.index != d.index[50], np.inf)),
        # Time series violations
        "duplicate_timestamps": lambda d: d.set_index(d.index.repeat(2)[::2]),
        "non_monotonic": lambda d: d.reindex(d.index[::-1]),
        "naive_timezone": lambda d: d.set_index(d.index.tz_localize(None)),
        "wrong_timezone": lambda d: d.set_index(
            d.index.tz_localize(None).tz_localize("US/Eastern")
        ),
        "mixed_timezones": lambda d: pd.concat(
            [
                d.iloc[:50].set_index(d.index[:50].tz_localize(None).tz_localize("US/Eastern")),
                d.iloc[50:],
            ]
        ),
        # OHLC violations
        "ohlc_inconsistent": lambda d: d.assign(
            High=d["High"].where(d.index != d.index[50], d["Low"].iloc[50] - 1)
        ),
        "high_below_open": lambda d: d.assign(
            High=d["High"].where(d.index != d.index[50], d["Open"].iloc[50] - 1)
        ),
        "low_above_close": lambda d: d.assign(
            Low=d["Low"].where(d.index != d.index[50], d["Close"].iloc[50] + 1)
        ),
        # Lookahead contamination
        "lookahead_returns": lambda d: d.assign(
            Returns=d["Close"]
            .pct_change()
            .where(d.index != d.index[10], d["Close"].pct_change().iloc[11])
        ),
        "future_data": lambda d: d.assign(Future_Price=d["Close"].shift(-1)),
        # Data type violations
        "string_prices": lambda d: d.assign(
            Close=d["Close"].astype(str).where(d.index.isin(d.index[50:60]), d["Close"])
        ),
        "mixed_dtypes": lambda d: d.assign(
            Open=d["Open"].astype(str).where(d.index.isin(d.index[50:60]), d["Open"])
        ),
        # Missing data
        "missing_columns": lambda d: d.drop(columns=["Volume"]),
        "empty_df": lambda d: pd.DataFrame(columns=d.columns),
        "single_row": lambda d: d.iloc[:1],
        # Edge cases
        "very_small": lambda d: d.iloc[:5],
        "very_large": lambda d: d,
        "single_price": lambda d: d.iloc[:1],
        "two_prices": lambda d: d.iloc[:2],
        "decreasing_prices": lambda d: d.assign(
            Close=pd.Series(range(100, 50, -1), index=d.index[:50]),
            High=pd.Series(range(100, 50, -1), index=d.index[:50]) * 1.01,
            Low=pd.Series(range(100, 50, -1), index=d.index[:50]) * 0.99,
            Open=pd.Series(range(100, 50, -1), index=d.index[:50]) * 1.005,
        ).iloc[:50],
        "constant_prices": lambda d: d.assign(Close=100.0, High=101.0, Low=99.0, Open=100.5),
        "volatile_prices": lambda d: d.assign(
            Close=100 + 50 * np.sin(np.arange(len(d)) * 0.5),
            High=100 + 50 * np.sin(np.arange(len(d)) * 0.5) + 1,
            Low=100 + 50 * np.sin(np.arange(len(d)) * 0.5) - 1,
            Open=100 + 50 * np.sin(np.arange(len(d)) * 0.5) + 0.5,
        ),
        # Clean data (no mutations)
        "clean": lambda d: d,
    }

    if name not in mutations:
        raise ValueError(f"Unknown test case: {name}")

    return mutations[name](df)


def build_edge_case(name: str, **kwargs) -> pd.DataFrame:
    """Build edge case DataFrames."""
    edge_cases = {
        "very_small": lambda: base_df(n=5),
        "very_large": lambda: base_df(n=10000),
        "single_price": lambda: base_df(n=1),
        "two_prices": lambda: base_df(n=2),
        "decreasing_prices": lambda: base_df(n=100).assign(
            Close=pd.Series(range(100, 0, -1), index=base_df(n=100).index)
        ),
        "constant_prices": lambda: base_df(n=100).assign(Close=100.0),
        "volatile_prices": lambda: base_df(n=100).assign(
            Close=100 + 50 * np.sin(np.arange(100) * 0.5)
        ),
    }

    if name not in edge_cases:
        raise ValueError(f"Unknown edge case: {name}")

    return edge_cases[name]()


def build_stress_case(name: str, **kwargs) -> pd.DataFrame:
    """Build stress test DataFrames."""
    stress_cases = {
        "large_dataset": lambda: base_df(n=50000),
        "many_nans": lambda: base_df(n=1000).assign(
            Close=np.where(np.random.random(1000) < 0.1, np.nan, base_df(n=1000)["Close"])
        ),
        "many_duplicates": lambda: pd.concat([base_df(n=100)] * 10),
        "extreme_volatility": lambda: base_df(n=1000).assign(
            Close=100 * np.exp(np.cumsum(np.random.normal(0, 0.1, 1000)))
        ),
    }

    if name not in stress_cases:
        raise ValueError(f"Unknown stress case: {name}")

    return stress_cases[name]()
