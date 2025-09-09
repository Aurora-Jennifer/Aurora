import pandas as pd

_CANON = ("Open", "High", "Low", "Close", "Volume")
_SYNONYMS = {
    "Close": {"Close", "close", "Adj Close", "adj close", "settle", "last", "price"},
    "Open": {"Open", "open"},
    "High": {"High", "high"},
    "Low": {"Low", "low"},
    "Volume": {"Volume", "volume", "vol"}
}

def map_ohlcv(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(_SYNONYMS.get(name, []))):
            k = str(cand).lower()
            if k in lower:
                return lower[k]
        return None

    if isinstance(df.columns, pd.MultiIndex):
        # Prefer last-level match; otherwise any-level synonym
        for canon in _CANON:
            target = None
            syns = {s.lower() for s in _SYNONYMS.get(canon, {canon})}
            # last-level first
            for col in df.columns:
                if str(col[-1]).lower() in syns:
                    target = col
                    break
            if target is None:
                for col in df.columns:
                    if any(str(part).lower() in syns for part in col):
                        target = col
                        break
            if target is not None:
                mapping[canon] = target
    else:
        for canon in _CANON:
            hit = _match_simple(canon)
            if hit is not None:
                mapping[canon] = hit

    return mapping
