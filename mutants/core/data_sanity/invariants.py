from collections.abc import Callable
from inspect import signature as _mutmut_signature
from typing import Annotated, ClassVar

import pandas as pd

from .codes import NEGATIVE_PRICES, OHLC_INVARIANT
from .columnmap import map_ohlcv
from .errors import DataSanityError, estring

MutantDict = Annotated[dict[str, Callable], "Mutant"]


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None):
    """Forward call to original or mutated function, depending on the environment"""
    import os
    mutant_under_test = os.environ['MUTANT_UNDER_TEST']
    if mutant_under_test == 'fail':
        from mutmut.__main__ import MutmutProgrammaticFailException
        raise MutmutProgrammaticFailException('Failed programmatically')      
    elif mutant_under_test == 'stats':
        from mutmut.__main__ import record_trampoline_hit
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__)
        result = orig(*call_args, **call_kwargs)
        return result
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_'
    if not mutant_under_test.startswith(prefix):
        result = orig(*call_args, **call_kwargs)
        return result
    mutant_name = mutant_under_test.rpartition('.')[-1]
    if self_arg:
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)
    return result

def x_assert_ohlc_invariants__mutmut_orig(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_1(df: pd.DataFrame, profile):
    m = None
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_2(df: pd.DataFrame, profile):
    m = map_ohlcv(None)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_3(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_4(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(None):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_5(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k not in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_6(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("XXOpenXX","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_7(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_8(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("OPEN","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_9(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","XXHighXX","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_10(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","high","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_11(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","HIGH","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_12(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","XXLowXX","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_13(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_14(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","LOW","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_15(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","XXCloseXX")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_16(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_17(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","CLOSE")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_18(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = None

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_19(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype(None),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_20(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["XXOpenXX"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_21(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_22(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["OPEN"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_23(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("XXfloat64XX"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_24(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("FLOAT64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_25(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype(None),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_26(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["XXHighXX"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_27(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["high"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_28(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["HIGH"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_29(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("XXfloat64XX"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_30(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("FLOAT64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_31(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype(None),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_32(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["XXLowXX"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_33(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_34(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["LOW"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_35(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("XXfloat64XX"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_36(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("FLOAT64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_37(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype(None))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_38(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["XXCloseXX"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_39(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_40(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["CLOSE"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_41(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("XXfloat64XX"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_42(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("FLOAT64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_43(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = None; o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_44(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(None, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_45(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join=None); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_46(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_47(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, ); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_48(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="XXinnerXX"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_49(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="INNER"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_50(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = None; o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_51(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(None, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_52(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join=None); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_53(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_54(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, ); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_55(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="XXinnerXX"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_56(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="INNER"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_57(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = None
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_58(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(None, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_59(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join=None)
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_60(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_61(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, )
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_62(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="XXinnerXX")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_63(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="INNER")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_64(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = None; l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_65(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(None); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_66(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = None; c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_67(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(None); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_68(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = None

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_69(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(None)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_70(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() and (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_71(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() and (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_72(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() and (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_73(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o <= 0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_74(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<1).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_75(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h <= 0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_76(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<1).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_77(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l <= 0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_78(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<1).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_79(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c <= 0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_80(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<1).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_81(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_82(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get(None,True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_83(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",None):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_84(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get(True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_85(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_86(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("XXallow_repairsXX",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_87(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("ALLOW_REPAIRS",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_88(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",False):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_89(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(None)
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_90(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(None, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_91(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, None))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_92(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring("negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_93(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, ))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_94(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "XXnegative values in OHLCXX"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_95(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in ohlc"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_96(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "NEGATIVE VALUES IN OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_97(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = None; h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_98(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=None); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_99(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=1); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_100(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = None; l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_101(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=None); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_102(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=1); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_103(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = None; c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_104(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=None); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_105(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=1); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_106(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = None

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_107(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=None)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_108(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=1)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_109(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = None
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_110(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h <= pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_111(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=None)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_112(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat(None, axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_113(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=None).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_114(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat(axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_115(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], ).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_116(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=2).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_117(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=2)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_118(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = None
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_119(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l >= pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_120(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=None)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_121(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat(None, axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_122(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=None).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_123(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat(axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_124(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], ).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_125(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=2).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_126(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=2)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_127(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() and bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_128(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_129(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get(None,True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_130(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",None):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_131(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get(True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_132(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_133(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("XXallow_repairsXX",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_134(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("ALLOW_REPAIRS",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_135(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",False):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_136(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(None)
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_137(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(None,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_138(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                None))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_139(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_140(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                ))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_141(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(None)}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_142(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(None)}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_143(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = None
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_144(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=None)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_145(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat(None, axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_146(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=None).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_147(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat(axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_148(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], ).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_149(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=2).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_150(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=2)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_151(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = None

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_152(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=None)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_153(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat(None, axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_154(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=None).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_155(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat(axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_156(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], ).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_157(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=2).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_158(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=2)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_159(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = None
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_160(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = None; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_161(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["XXOpenXX"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_162(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_163(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["OPEN"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_164(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = None; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_165(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["XXHighXX"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_166(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["high"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_167(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["HIGH"]] = h; df[m["Low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_168(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = None; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_169(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["XXLowXX"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_170(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["low"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_171(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["LOW"]] = l; df[m["Close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_172(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["Close"]] = None
    return df

def x_assert_ohlc_invariants__mutmut_173(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["XXCloseXX"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_174(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["close"]] = c
    return df

def x_assert_ohlc_invariants__mutmut_175(df: pd.DataFrame, profile):
    m = map_ohlcv(df)
    if not all(k in m for k in ("Open","High","Low","Close")):
        return df  # earlier stages handle missing cols

    o, h, l, c = (df[m["Open"]].astype("float64"),
                  df[m["High"]].astype("float64"),
                  df[m["Low"]].astype("float64"),
                  df[m["Close"]].astype("float64"))

    # Align indexes just in case (defensive)
    o,h = o.align(h, join="inner"); o,l = o.align(l, join="inner"); o,c = o.align(c, join="inner")
    h = h.reindex(o.index); l = l.reindex(o.index); c = c.reindex(o.index)

    if (o<0).any() or (h<0).any() or (l<0).any() or (c<0).any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(NEGATIVE_PRICES, "negative values in OHLC"))
        o = o.clip(lower=0); h = h.clip(lower=0); l = l.clip(lower=0); c = c.clip(lower=0)

    bad_hi = h < pd.concat([o,c], axis=1).max(axis=1)
    bad_lo = l > pd.concat([o,c], axis=1).min(axis=1)
    if bad_hi.any() or bad_lo.any():
        if not profile.get("allow_repairs",True):
            raise DataSanityError(estring(OHLC_INVARIANT,
                f"High < max(Open,Close): {int(bad_hi.sum())}, Low > min(Open,Close): {int(bad_lo.sum())}"))
        # lenient: repair
        h = pd.concat([h, o, c], axis=1).max(axis=1)
        l = pd.concat([l, o, c], axis=1).min(axis=1)

    # write back (preserve column keys)
    df = df.copy()
    df[m["Open"]] = o; df[m["High"]] = h; df[m["Low"]] = l; df[m["CLOSE"]] = c
    return df

x_assert_ohlc_invariants__mutmut_mutants : ClassVar[MutantDict] = {
'x_assert_ohlc_invariants__mutmut_1': x_assert_ohlc_invariants__mutmut_1, 
    'x_assert_ohlc_invariants__mutmut_2': x_assert_ohlc_invariants__mutmut_2, 
    'x_assert_ohlc_invariants__mutmut_3': x_assert_ohlc_invariants__mutmut_3, 
    'x_assert_ohlc_invariants__mutmut_4': x_assert_ohlc_invariants__mutmut_4, 
    'x_assert_ohlc_invariants__mutmut_5': x_assert_ohlc_invariants__mutmut_5, 
    'x_assert_ohlc_invariants__mutmut_6': x_assert_ohlc_invariants__mutmut_6, 
    'x_assert_ohlc_invariants__mutmut_7': x_assert_ohlc_invariants__mutmut_7, 
    'x_assert_ohlc_invariants__mutmut_8': x_assert_ohlc_invariants__mutmut_8, 
    'x_assert_ohlc_invariants__mutmut_9': x_assert_ohlc_invariants__mutmut_9, 
    'x_assert_ohlc_invariants__mutmut_10': x_assert_ohlc_invariants__mutmut_10, 
    'x_assert_ohlc_invariants__mutmut_11': x_assert_ohlc_invariants__mutmut_11, 
    'x_assert_ohlc_invariants__mutmut_12': x_assert_ohlc_invariants__mutmut_12, 
    'x_assert_ohlc_invariants__mutmut_13': x_assert_ohlc_invariants__mutmut_13, 
    'x_assert_ohlc_invariants__mutmut_14': x_assert_ohlc_invariants__mutmut_14, 
    'x_assert_ohlc_invariants__mutmut_15': x_assert_ohlc_invariants__mutmut_15, 
    'x_assert_ohlc_invariants__mutmut_16': x_assert_ohlc_invariants__mutmut_16, 
    'x_assert_ohlc_invariants__mutmut_17': x_assert_ohlc_invariants__mutmut_17, 
    'x_assert_ohlc_invariants__mutmut_18': x_assert_ohlc_invariants__mutmut_18, 
    'x_assert_ohlc_invariants__mutmut_19': x_assert_ohlc_invariants__mutmut_19, 
    'x_assert_ohlc_invariants__mutmut_20': x_assert_ohlc_invariants__mutmut_20, 
    'x_assert_ohlc_invariants__mutmut_21': x_assert_ohlc_invariants__mutmut_21, 
    'x_assert_ohlc_invariants__mutmut_22': x_assert_ohlc_invariants__mutmut_22, 
    'x_assert_ohlc_invariants__mutmut_23': x_assert_ohlc_invariants__mutmut_23, 
    'x_assert_ohlc_invariants__mutmut_24': x_assert_ohlc_invariants__mutmut_24, 
    'x_assert_ohlc_invariants__mutmut_25': x_assert_ohlc_invariants__mutmut_25, 
    'x_assert_ohlc_invariants__mutmut_26': x_assert_ohlc_invariants__mutmut_26, 
    'x_assert_ohlc_invariants__mutmut_27': x_assert_ohlc_invariants__mutmut_27, 
    'x_assert_ohlc_invariants__mutmut_28': x_assert_ohlc_invariants__mutmut_28, 
    'x_assert_ohlc_invariants__mutmut_29': x_assert_ohlc_invariants__mutmut_29, 
    'x_assert_ohlc_invariants__mutmut_30': x_assert_ohlc_invariants__mutmut_30, 
    'x_assert_ohlc_invariants__mutmut_31': x_assert_ohlc_invariants__mutmut_31, 
    'x_assert_ohlc_invariants__mutmut_32': x_assert_ohlc_invariants__mutmut_32, 
    'x_assert_ohlc_invariants__mutmut_33': x_assert_ohlc_invariants__mutmut_33, 
    'x_assert_ohlc_invariants__mutmut_34': x_assert_ohlc_invariants__mutmut_34, 
    'x_assert_ohlc_invariants__mutmut_35': x_assert_ohlc_invariants__mutmut_35, 
    'x_assert_ohlc_invariants__mutmut_36': x_assert_ohlc_invariants__mutmut_36, 
    'x_assert_ohlc_invariants__mutmut_37': x_assert_ohlc_invariants__mutmut_37, 
    'x_assert_ohlc_invariants__mutmut_38': x_assert_ohlc_invariants__mutmut_38, 
    'x_assert_ohlc_invariants__mutmut_39': x_assert_ohlc_invariants__mutmut_39, 
    'x_assert_ohlc_invariants__mutmut_40': x_assert_ohlc_invariants__mutmut_40, 
    'x_assert_ohlc_invariants__mutmut_41': x_assert_ohlc_invariants__mutmut_41, 
    'x_assert_ohlc_invariants__mutmut_42': x_assert_ohlc_invariants__mutmut_42, 
    'x_assert_ohlc_invariants__mutmut_43': x_assert_ohlc_invariants__mutmut_43, 
    'x_assert_ohlc_invariants__mutmut_44': x_assert_ohlc_invariants__mutmut_44, 
    'x_assert_ohlc_invariants__mutmut_45': x_assert_ohlc_invariants__mutmut_45, 
    'x_assert_ohlc_invariants__mutmut_46': x_assert_ohlc_invariants__mutmut_46, 
    'x_assert_ohlc_invariants__mutmut_47': x_assert_ohlc_invariants__mutmut_47, 
    'x_assert_ohlc_invariants__mutmut_48': x_assert_ohlc_invariants__mutmut_48, 
    'x_assert_ohlc_invariants__mutmut_49': x_assert_ohlc_invariants__mutmut_49, 
    'x_assert_ohlc_invariants__mutmut_50': x_assert_ohlc_invariants__mutmut_50, 
    'x_assert_ohlc_invariants__mutmut_51': x_assert_ohlc_invariants__mutmut_51, 
    'x_assert_ohlc_invariants__mutmut_52': x_assert_ohlc_invariants__mutmut_52, 
    'x_assert_ohlc_invariants__mutmut_53': x_assert_ohlc_invariants__mutmut_53, 
    'x_assert_ohlc_invariants__mutmut_54': x_assert_ohlc_invariants__mutmut_54, 
    'x_assert_ohlc_invariants__mutmut_55': x_assert_ohlc_invariants__mutmut_55, 
    'x_assert_ohlc_invariants__mutmut_56': x_assert_ohlc_invariants__mutmut_56, 
    'x_assert_ohlc_invariants__mutmut_57': x_assert_ohlc_invariants__mutmut_57, 
    'x_assert_ohlc_invariants__mutmut_58': x_assert_ohlc_invariants__mutmut_58, 
    'x_assert_ohlc_invariants__mutmut_59': x_assert_ohlc_invariants__mutmut_59, 
    'x_assert_ohlc_invariants__mutmut_60': x_assert_ohlc_invariants__mutmut_60, 
    'x_assert_ohlc_invariants__mutmut_61': x_assert_ohlc_invariants__mutmut_61, 
    'x_assert_ohlc_invariants__mutmut_62': x_assert_ohlc_invariants__mutmut_62, 
    'x_assert_ohlc_invariants__mutmut_63': x_assert_ohlc_invariants__mutmut_63, 
    'x_assert_ohlc_invariants__mutmut_64': x_assert_ohlc_invariants__mutmut_64, 
    'x_assert_ohlc_invariants__mutmut_65': x_assert_ohlc_invariants__mutmut_65, 
    'x_assert_ohlc_invariants__mutmut_66': x_assert_ohlc_invariants__mutmut_66, 
    'x_assert_ohlc_invariants__mutmut_67': x_assert_ohlc_invariants__mutmut_67, 
    'x_assert_ohlc_invariants__mutmut_68': x_assert_ohlc_invariants__mutmut_68, 
    'x_assert_ohlc_invariants__mutmut_69': x_assert_ohlc_invariants__mutmut_69, 
    'x_assert_ohlc_invariants__mutmut_70': x_assert_ohlc_invariants__mutmut_70, 
    'x_assert_ohlc_invariants__mutmut_71': x_assert_ohlc_invariants__mutmut_71, 
    'x_assert_ohlc_invariants__mutmut_72': x_assert_ohlc_invariants__mutmut_72, 
    'x_assert_ohlc_invariants__mutmut_73': x_assert_ohlc_invariants__mutmut_73, 
    'x_assert_ohlc_invariants__mutmut_74': x_assert_ohlc_invariants__mutmut_74, 
    'x_assert_ohlc_invariants__mutmut_75': x_assert_ohlc_invariants__mutmut_75, 
    'x_assert_ohlc_invariants__mutmut_76': x_assert_ohlc_invariants__mutmut_76, 
    'x_assert_ohlc_invariants__mutmut_77': x_assert_ohlc_invariants__mutmut_77, 
    'x_assert_ohlc_invariants__mutmut_78': x_assert_ohlc_invariants__mutmut_78, 
    'x_assert_ohlc_invariants__mutmut_79': x_assert_ohlc_invariants__mutmut_79, 
    'x_assert_ohlc_invariants__mutmut_80': x_assert_ohlc_invariants__mutmut_80, 
    'x_assert_ohlc_invariants__mutmut_81': x_assert_ohlc_invariants__mutmut_81, 
    'x_assert_ohlc_invariants__mutmut_82': x_assert_ohlc_invariants__mutmut_82, 
    'x_assert_ohlc_invariants__mutmut_83': x_assert_ohlc_invariants__mutmut_83, 
    'x_assert_ohlc_invariants__mutmut_84': x_assert_ohlc_invariants__mutmut_84, 
    'x_assert_ohlc_invariants__mutmut_85': x_assert_ohlc_invariants__mutmut_85, 
    'x_assert_ohlc_invariants__mutmut_86': x_assert_ohlc_invariants__mutmut_86, 
    'x_assert_ohlc_invariants__mutmut_87': x_assert_ohlc_invariants__mutmut_87, 
    'x_assert_ohlc_invariants__mutmut_88': x_assert_ohlc_invariants__mutmut_88, 
    'x_assert_ohlc_invariants__mutmut_89': x_assert_ohlc_invariants__mutmut_89, 
    'x_assert_ohlc_invariants__mutmut_90': x_assert_ohlc_invariants__mutmut_90, 
    'x_assert_ohlc_invariants__mutmut_91': x_assert_ohlc_invariants__mutmut_91, 
    'x_assert_ohlc_invariants__mutmut_92': x_assert_ohlc_invariants__mutmut_92, 
    'x_assert_ohlc_invariants__mutmut_93': x_assert_ohlc_invariants__mutmut_93, 
    'x_assert_ohlc_invariants__mutmut_94': x_assert_ohlc_invariants__mutmut_94, 
    'x_assert_ohlc_invariants__mutmut_95': x_assert_ohlc_invariants__mutmut_95, 
    'x_assert_ohlc_invariants__mutmut_96': x_assert_ohlc_invariants__mutmut_96, 
    'x_assert_ohlc_invariants__mutmut_97': x_assert_ohlc_invariants__mutmut_97, 
    'x_assert_ohlc_invariants__mutmut_98': x_assert_ohlc_invariants__mutmut_98, 
    'x_assert_ohlc_invariants__mutmut_99': x_assert_ohlc_invariants__mutmut_99, 
    'x_assert_ohlc_invariants__mutmut_100': x_assert_ohlc_invariants__mutmut_100, 
    'x_assert_ohlc_invariants__mutmut_101': x_assert_ohlc_invariants__mutmut_101, 
    'x_assert_ohlc_invariants__mutmut_102': x_assert_ohlc_invariants__mutmut_102, 
    'x_assert_ohlc_invariants__mutmut_103': x_assert_ohlc_invariants__mutmut_103, 
    'x_assert_ohlc_invariants__mutmut_104': x_assert_ohlc_invariants__mutmut_104, 
    'x_assert_ohlc_invariants__mutmut_105': x_assert_ohlc_invariants__mutmut_105, 
    'x_assert_ohlc_invariants__mutmut_106': x_assert_ohlc_invariants__mutmut_106, 
    'x_assert_ohlc_invariants__mutmut_107': x_assert_ohlc_invariants__mutmut_107, 
    'x_assert_ohlc_invariants__mutmut_108': x_assert_ohlc_invariants__mutmut_108, 
    'x_assert_ohlc_invariants__mutmut_109': x_assert_ohlc_invariants__mutmut_109, 
    'x_assert_ohlc_invariants__mutmut_110': x_assert_ohlc_invariants__mutmut_110, 
    'x_assert_ohlc_invariants__mutmut_111': x_assert_ohlc_invariants__mutmut_111, 
    'x_assert_ohlc_invariants__mutmut_112': x_assert_ohlc_invariants__mutmut_112, 
    'x_assert_ohlc_invariants__mutmut_113': x_assert_ohlc_invariants__mutmut_113, 
    'x_assert_ohlc_invariants__mutmut_114': x_assert_ohlc_invariants__mutmut_114, 
    'x_assert_ohlc_invariants__mutmut_115': x_assert_ohlc_invariants__mutmut_115, 
    'x_assert_ohlc_invariants__mutmut_116': x_assert_ohlc_invariants__mutmut_116, 
    'x_assert_ohlc_invariants__mutmut_117': x_assert_ohlc_invariants__mutmut_117, 
    'x_assert_ohlc_invariants__mutmut_118': x_assert_ohlc_invariants__mutmut_118, 
    'x_assert_ohlc_invariants__mutmut_119': x_assert_ohlc_invariants__mutmut_119, 
    'x_assert_ohlc_invariants__mutmut_120': x_assert_ohlc_invariants__mutmut_120, 
    'x_assert_ohlc_invariants__mutmut_121': x_assert_ohlc_invariants__mutmut_121, 
    'x_assert_ohlc_invariants__mutmut_122': x_assert_ohlc_invariants__mutmut_122, 
    'x_assert_ohlc_invariants__mutmut_123': x_assert_ohlc_invariants__mutmut_123, 
    'x_assert_ohlc_invariants__mutmut_124': x_assert_ohlc_invariants__mutmut_124, 
    'x_assert_ohlc_invariants__mutmut_125': x_assert_ohlc_invariants__mutmut_125, 
    'x_assert_ohlc_invariants__mutmut_126': x_assert_ohlc_invariants__mutmut_126, 
    'x_assert_ohlc_invariants__mutmut_127': x_assert_ohlc_invariants__mutmut_127, 
    'x_assert_ohlc_invariants__mutmut_128': x_assert_ohlc_invariants__mutmut_128, 
    'x_assert_ohlc_invariants__mutmut_129': x_assert_ohlc_invariants__mutmut_129, 
    'x_assert_ohlc_invariants__mutmut_130': x_assert_ohlc_invariants__mutmut_130, 
    'x_assert_ohlc_invariants__mutmut_131': x_assert_ohlc_invariants__mutmut_131, 
    'x_assert_ohlc_invariants__mutmut_132': x_assert_ohlc_invariants__mutmut_132, 
    'x_assert_ohlc_invariants__mutmut_133': x_assert_ohlc_invariants__mutmut_133, 
    'x_assert_ohlc_invariants__mutmut_134': x_assert_ohlc_invariants__mutmut_134, 
    'x_assert_ohlc_invariants__mutmut_135': x_assert_ohlc_invariants__mutmut_135, 
    'x_assert_ohlc_invariants__mutmut_136': x_assert_ohlc_invariants__mutmut_136, 
    'x_assert_ohlc_invariants__mutmut_137': x_assert_ohlc_invariants__mutmut_137, 
    'x_assert_ohlc_invariants__mutmut_138': x_assert_ohlc_invariants__mutmut_138, 
    'x_assert_ohlc_invariants__mutmut_139': x_assert_ohlc_invariants__mutmut_139, 
    'x_assert_ohlc_invariants__mutmut_140': x_assert_ohlc_invariants__mutmut_140, 
    'x_assert_ohlc_invariants__mutmut_141': x_assert_ohlc_invariants__mutmut_141, 
    'x_assert_ohlc_invariants__mutmut_142': x_assert_ohlc_invariants__mutmut_142, 
    'x_assert_ohlc_invariants__mutmut_143': x_assert_ohlc_invariants__mutmut_143, 
    'x_assert_ohlc_invariants__mutmut_144': x_assert_ohlc_invariants__mutmut_144, 
    'x_assert_ohlc_invariants__mutmut_145': x_assert_ohlc_invariants__mutmut_145, 
    'x_assert_ohlc_invariants__mutmut_146': x_assert_ohlc_invariants__mutmut_146, 
    'x_assert_ohlc_invariants__mutmut_147': x_assert_ohlc_invariants__mutmut_147, 
    'x_assert_ohlc_invariants__mutmut_148': x_assert_ohlc_invariants__mutmut_148, 
    'x_assert_ohlc_invariants__mutmut_149': x_assert_ohlc_invariants__mutmut_149, 
    'x_assert_ohlc_invariants__mutmut_150': x_assert_ohlc_invariants__mutmut_150, 
    'x_assert_ohlc_invariants__mutmut_151': x_assert_ohlc_invariants__mutmut_151, 
    'x_assert_ohlc_invariants__mutmut_152': x_assert_ohlc_invariants__mutmut_152, 
    'x_assert_ohlc_invariants__mutmut_153': x_assert_ohlc_invariants__mutmut_153, 
    'x_assert_ohlc_invariants__mutmut_154': x_assert_ohlc_invariants__mutmut_154, 
    'x_assert_ohlc_invariants__mutmut_155': x_assert_ohlc_invariants__mutmut_155, 
    'x_assert_ohlc_invariants__mutmut_156': x_assert_ohlc_invariants__mutmut_156, 
    'x_assert_ohlc_invariants__mutmut_157': x_assert_ohlc_invariants__mutmut_157, 
    'x_assert_ohlc_invariants__mutmut_158': x_assert_ohlc_invariants__mutmut_158, 
    'x_assert_ohlc_invariants__mutmut_159': x_assert_ohlc_invariants__mutmut_159, 
    'x_assert_ohlc_invariants__mutmut_160': x_assert_ohlc_invariants__mutmut_160, 
    'x_assert_ohlc_invariants__mutmut_161': x_assert_ohlc_invariants__mutmut_161, 
    'x_assert_ohlc_invariants__mutmut_162': x_assert_ohlc_invariants__mutmut_162, 
    'x_assert_ohlc_invariants__mutmut_163': x_assert_ohlc_invariants__mutmut_163, 
    'x_assert_ohlc_invariants__mutmut_164': x_assert_ohlc_invariants__mutmut_164, 
    'x_assert_ohlc_invariants__mutmut_165': x_assert_ohlc_invariants__mutmut_165, 
    'x_assert_ohlc_invariants__mutmut_166': x_assert_ohlc_invariants__mutmut_166, 
    'x_assert_ohlc_invariants__mutmut_167': x_assert_ohlc_invariants__mutmut_167, 
    'x_assert_ohlc_invariants__mutmut_168': x_assert_ohlc_invariants__mutmut_168, 
    'x_assert_ohlc_invariants__mutmut_169': x_assert_ohlc_invariants__mutmut_169, 
    'x_assert_ohlc_invariants__mutmut_170': x_assert_ohlc_invariants__mutmut_170, 
    'x_assert_ohlc_invariants__mutmut_171': x_assert_ohlc_invariants__mutmut_171, 
    'x_assert_ohlc_invariants__mutmut_172': x_assert_ohlc_invariants__mutmut_172, 
    'x_assert_ohlc_invariants__mutmut_173': x_assert_ohlc_invariants__mutmut_173, 
    'x_assert_ohlc_invariants__mutmut_174': x_assert_ohlc_invariants__mutmut_174, 
    'x_assert_ohlc_invariants__mutmut_175': x_assert_ohlc_invariants__mutmut_175
}

def assert_ohlc_invariants(*args, **kwargs):
    result = _mutmut_trampoline(x_assert_ohlc_invariants__mutmut_orig, x_assert_ohlc_invariants__mutmut_mutants, args, kwargs)
    return result 

assert_ohlc_invariants.__signature__ = _mutmut_signature(x_assert_ohlc_invariants__mutmut_orig)
x_assert_ohlc_invariants__mutmut_orig.__name__ = 'x_assert_ohlc_invariants'
