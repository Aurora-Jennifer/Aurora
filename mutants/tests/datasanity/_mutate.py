import numpy as np
import pandas as pd


def inject_duplicates(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    dup_idx = df.index[:n]
    return pd.concat([df, df.loc[dup_idx]], axis=0).sort_index()


def inject_non_monotonic(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 3:
        return df
    swapped = df.iloc[[0, 2, 1]].copy()
    rest = df.iloc[3:]
    out = pd.concat([swapped, rest], axis=0)
    out.index = out.index
    return out


def inject_nan_inf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [c for c in ["Open", "High", "Low", "Close"] if c in out.columns][:2]:
        out.iloc[0, out.columns.get_loc(col)] = float("nan")
        out.iloc[1, out.columns.get_loc(col)] = float("inf")
    return out


def inject_lookahead(df: pd.DataFrame, shift: int = 1) -> pd.DataFrame:
    out = df.copy()
    if "Close" in out.columns:
        out["Close"] = out["Close"].shift(-shift)
    return out


def inject_duplicates(df, frac=0.01):
    """Inject duplicate timestamps into DataFrame."""
    n = max(1, int(len(df)*frac))
    pick = df.sample(n, replace=False).index
    return pd.concat([df, df.loc[pick]]).sort_index()


def inject_nans(df, cols=None, frac=0.05):
    """Inject NaN values into specified columns."""
    df_copy = df.copy()
    if cols is None:
        cols = df_copy.select_dtypes(include=np.number).columns
    for col in cols:
        n_nan = max(1, int(len(df_copy) * frac))
        nan_indices = np.random.choice(df_copy.index, n_nan, replace=False)
        df_copy.loc[nan_indices, col] = np.nan
    return df_copy


def inject_infs(df, cols=None, frac=0.05):
    """Inject infinite values into specified columns."""
    df_copy = df.copy()
    if cols is None:
        cols = df_copy.select_dtypes(include=np.number).columns
    for col in cols:
        n_inf = max(1, int(len(df_copy) * frac))
        inf_indices = np.random.choice(df_copy.index, n_inf, replace=False)
        df_copy.loc[inf_indices, col] = np.inf
    return df_copy


def inject_string_dtype(df, col="Close"):
    """Inject string dtype into specified column."""
    df_copy = df.copy()
    if col in df_copy.columns:
        df_copy[col] = df_copy[col].astype(str)
    return df_copy


