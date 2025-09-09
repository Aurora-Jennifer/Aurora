import pandas as pd

from .codes import NEGATIVE_PRICES, OHLC_INVARIANT
from .columnmap import map_ohlcv
from .errors import DataSanityError, estring


def assert_ohlc_invariants(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, low, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o, h = o.align(h, join="inner")
    o, low = o.align(low, join="inner")
    o, c = o.align(c, join="inner")
    h = h.reindex(o.index)
    low = low.reindex(o.index)
    c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (low<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0)
        h = h.clip(lower=0)
        low = low.clip(lower=0)
        c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = low > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        low = pd.concat([low, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o
    df[m["High"]] = h
    df[m["Low"]] = low
    df[m["Close"]] = c
    return df
