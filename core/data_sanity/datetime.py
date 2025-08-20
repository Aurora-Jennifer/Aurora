import pandas as pd
from .errors import DataSanityError, estring
from .codes import (TZ_NAIVE, TZ_NON_UTC, INDEX_NON_MONOTONIC,
                    DUPLICATE_TSTAMPS, NO_VALID_DT_INDEX)

def _coerce_any_to_utc_index(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors="coerce", utc=True)
    return pd.DatetimeIndex(idx)

def _ensure_singlelevel_utc_index(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def canonicalize_datetime_index(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "timestamp" in names:
            lvl = names.index("timestamp")
            raw = out.index.get_level_values(lvl)
            idx = _coerce_any_to_utc_index(raw)
            arrays = [idx if i == lvl else out.index.get_level_values(i) for i in range(len(names))]
            out.index = pd.MultiIndex.from_arrays(arrays, names=names)
        else:
            best_i, best_idx, best_valid = None, None, -1
            for i in range(len(names)):
                cand = _coerce_any_to_utc_index(out.index.get_level_values(i))
                valid = (~cand.isna()).sum()
                if valid > best_valid:
                    best_i, best_idx, best_valid = i, cand, valid
            if best_idx is None or best_valid == 0:
                if strict:
                    raise DataSanityError(estring(NO_VALID_DT_INDEX, "cannot parse MultiIndex"))
                # lenient → empty
                return out.iloc[0:0]
            arrays = [best_idx if j == best_i else out.index.get_level_values(j) for j in range(len(names))]
            names[best_i] = "timestamp"
            out.index = pd.MultiIndex.from_arrays(arrays, names=names)
        idx = out.index.get_level_values("timestamp")
        # idx is already UTC
    else:
        idx = _ensure_singlelevel_utc_index(out.index, strict)

    # Strict checks (no repairs)
    if strict:
        if any(pd.isna(idx)):
            raise DataSanityError(estring(NO_VALID_DT_INDEX, "NaT present"))
        if idx.has_duplicates:
            raise DataSanityError(estring(DUPLICATE_TSTAMPS, "duplicate timestamps"))
        if not idx.is_monotonic_increasing:
            raise DataSanityError(estring(INDEX_NON_MONOTONIC, "timestamps not sorted"))

    # Lenient repairs
    keep = ~pd.isna(idx)
    if (~keep).any():
        out = out.loc[keep]
        idx = idx[keep]
    order = idx.argsort()
    out = out.iloc[order]
    idx = pd.DatetimeIndex(idx[order], tz="UTC", name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out
