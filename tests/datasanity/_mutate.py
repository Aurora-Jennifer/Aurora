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


