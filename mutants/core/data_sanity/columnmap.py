import pandas as pd

_CANON = ("Open", "High", "Low", "Close", "Volume")
_SYNONYMS = {
    "Close": {"Close", "close", "Adj Close", "adj close", "settle", "last", "price"},
    "Open": {"Open", "open"},
    "High": {"High", "high"},
    "Low": {"Low", "low"},
    "Volume": {"Volume", "volume", "vol"}
}
from inspect import signature as _mutmut_signature
from typing import Annotated
from typing import Callable
from typing import ClassVar


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

def x_map_ohlcv__mutmut_orig(df: pd.DataFrame) -> dict[str, object]:
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

def x_map_ohlcv__mutmut_1(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = None

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

def x_map_ohlcv__mutmut_2(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = None
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

def x_map_ohlcv__mutmut_3(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).upper(): c for c in df.columns}
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

def x_map_ohlcv__mutmut_4(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(None).lower(): c for c in df.columns}
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

def x_map_ohlcv__mutmut_5(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] - list(_SYNONYMS.get(name, []))):
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

def x_map_ohlcv__mutmut_6(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(None)):
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

def x_map_ohlcv__mutmut_7(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(_SYNONYMS.get(None, []))):
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

def x_map_ohlcv__mutmut_8(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(_SYNONYMS.get(name, None))):
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

def x_map_ohlcv__mutmut_9(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(_SYNONYMS.get([]))):
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

def x_map_ohlcv__mutmut_10(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(_SYNONYMS.get(name, ))):
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

def x_map_ohlcv__mutmut_11(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(_SYNONYMS.get(name, []))):
            k = None
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

def x_map_ohlcv__mutmut_12(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(_SYNONYMS.get(name, []))):
            k = str(cand).upper()
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

def x_map_ohlcv__mutmut_13(df: pd.DataFrame) -> dict[str, object]:
    """
    Return a mapping { 'Open': col_key, ... } where col_key is the actual
    column label (works for simple or MultiIndex columns).
    """
    mapping: dict[str, object] = {}

    def _match_simple(name: str):
        # exact, case-insensitive, and synonyms
        lower = {str(c).lower(): c for c in df.columns}
        for cand in ([name] + list(_SYNONYMS.get(name, []))):
            k = str(None).lower()
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

def x_map_ohlcv__mutmut_14(df: pd.DataFrame) -> dict[str, object]:
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
            if k not in lower:
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

def x_map_ohlcv__mutmut_15(df: pd.DataFrame) -> dict[str, object]:
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
            target = ""
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

def x_map_ohlcv__mutmut_16(df: pd.DataFrame) -> dict[str, object]:
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
            syns = None
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

def x_map_ohlcv__mutmut_17(df: pd.DataFrame) -> dict[str, object]:
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
            syns = {s.upper() for s in _SYNONYMS.get(canon, {canon})}
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

def x_map_ohlcv__mutmut_18(df: pd.DataFrame) -> dict[str, object]:
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
            syns = {s.lower() for s in _SYNONYMS.get(None, {canon})}
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

def x_map_ohlcv__mutmut_19(df: pd.DataFrame) -> dict[str, object]:
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
            syns = {s.lower() for s in _SYNONYMS.get(canon, None)}
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

def x_map_ohlcv__mutmut_20(df: pd.DataFrame) -> dict[str, object]:
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
            syns = {s.lower() for s in _SYNONYMS.get({canon})}
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

def x_map_ohlcv__mutmut_21(df: pd.DataFrame) -> dict[str, object]:
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
            syns = {s.lower() for s in _SYNONYMS.get(canon, )}
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

def x_map_ohlcv__mutmut_22(df: pd.DataFrame) -> dict[str, object]:
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
                if str(col[-1]).upper() in syns:
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

def x_map_ohlcv__mutmut_23(df: pd.DataFrame) -> dict[str, object]:
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
                if str(None).lower() in syns:
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

def x_map_ohlcv__mutmut_24(df: pd.DataFrame) -> dict[str, object]:
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
                if str(col[+1]).lower() in syns:
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

def x_map_ohlcv__mutmut_25(df: pd.DataFrame) -> dict[str, object]:
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
                if str(col[-2]).lower() in syns:
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

def x_map_ohlcv__mutmut_26(df: pd.DataFrame) -> dict[str, object]:
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
                if str(col[-1]).lower() not in syns:
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

def x_map_ohlcv__mutmut_27(df: pd.DataFrame) -> dict[str, object]:
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
                    target = None
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

def x_map_ohlcv__mutmut_28(df: pd.DataFrame) -> dict[str, object]:
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
                    return
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

def x_map_ohlcv__mutmut_29(df: pd.DataFrame) -> dict[str, object]:
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
            if target is not None:
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

def x_map_ohlcv__mutmut_30(df: pd.DataFrame) -> dict[str, object]:
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
                    if any(None):
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

def x_map_ohlcv__mutmut_31(df: pd.DataFrame) -> dict[str, object]:
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
                    if any(str(part).upper() in syns for part in col):
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

def x_map_ohlcv__mutmut_32(df: pd.DataFrame) -> dict[str, object]:
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
                    if any(str(None).lower() in syns for part in col):
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

def x_map_ohlcv__mutmut_33(df: pd.DataFrame) -> dict[str, object]:
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
                    if any(str(part).lower() not in syns for part in col):
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

def x_map_ohlcv__mutmut_34(df: pd.DataFrame) -> dict[str, object]:
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
                        target = None
                        break
            if target is not None:
                mapping[canon] = target
    else:
        for canon in _CANON:
            hit = _match_simple(canon)
            if hit is not None:
                mapping[canon] = hit

    return mapping

def x_map_ohlcv__mutmut_35(df: pd.DataFrame) -> dict[str, object]:
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
                        return
            if target is not None:
                mapping[canon] = target
    else:
        for canon in _CANON:
            hit = _match_simple(canon)
            if hit is not None:
                mapping[canon] = hit

    return mapping

def x_map_ohlcv__mutmut_36(df: pd.DataFrame) -> dict[str, object]:
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
            if target is None:
                mapping[canon] = target
    else:
        for canon in _CANON:
            hit = _match_simple(canon)
            if hit is not None:
                mapping[canon] = hit

    return mapping

def x_map_ohlcv__mutmut_37(df: pd.DataFrame) -> dict[str, object]:
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
                mapping[canon] = None
    else:
        for canon in _CANON:
            hit = _match_simple(canon)
            if hit is not None:
                mapping[canon] = hit

    return mapping

def x_map_ohlcv__mutmut_38(df: pd.DataFrame) -> dict[str, object]:
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
            hit = None
            if hit is not None:
                mapping[canon] = hit

    return mapping

def x_map_ohlcv__mutmut_39(df: pd.DataFrame) -> dict[str, object]:
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
            hit = _match_simple(None)
            if hit is not None:
                mapping[canon] = hit

    return mapping

def x_map_ohlcv__mutmut_40(df: pd.DataFrame) -> dict[str, object]:
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
            if hit is None:
                mapping[canon] = hit

    return mapping

def x_map_ohlcv__mutmut_41(df: pd.DataFrame) -> dict[str, object]:
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
                mapping[canon] = None

    return mapping

x_map_ohlcv__mutmut_mutants : ClassVar[MutantDict] = {
'x_map_ohlcv__mutmut_1': x_map_ohlcv__mutmut_1, 
    'x_map_ohlcv__mutmut_2': x_map_ohlcv__mutmut_2, 
    'x_map_ohlcv__mutmut_3': x_map_ohlcv__mutmut_3, 
    'x_map_ohlcv__mutmut_4': x_map_ohlcv__mutmut_4, 
    'x_map_ohlcv__mutmut_5': x_map_ohlcv__mutmut_5, 
    'x_map_ohlcv__mutmut_6': x_map_ohlcv__mutmut_6, 
    'x_map_ohlcv__mutmut_7': x_map_ohlcv__mutmut_7, 
    'x_map_ohlcv__mutmut_8': x_map_ohlcv__mutmut_8, 
    'x_map_ohlcv__mutmut_9': x_map_ohlcv__mutmut_9, 
    'x_map_ohlcv__mutmut_10': x_map_ohlcv__mutmut_10, 
    'x_map_ohlcv__mutmut_11': x_map_ohlcv__mutmut_11, 
    'x_map_ohlcv__mutmut_12': x_map_ohlcv__mutmut_12, 
    'x_map_ohlcv__mutmut_13': x_map_ohlcv__mutmut_13, 
    'x_map_ohlcv__mutmut_14': x_map_ohlcv__mutmut_14, 
    'x_map_ohlcv__mutmut_15': x_map_ohlcv__mutmut_15, 
    'x_map_ohlcv__mutmut_16': x_map_ohlcv__mutmut_16, 
    'x_map_ohlcv__mutmut_17': x_map_ohlcv__mutmut_17, 
    'x_map_ohlcv__mutmut_18': x_map_ohlcv__mutmut_18, 
    'x_map_ohlcv__mutmut_19': x_map_ohlcv__mutmut_19, 
    'x_map_ohlcv__mutmut_20': x_map_ohlcv__mutmut_20, 
    'x_map_ohlcv__mutmut_21': x_map_ohlcv__mutmut_21, 
    'x_map_ohlcv__mutmut_22': x_map_ohlcv__mutmut_22, 
    'x_map_ohlcv__mutmut_23': x_map_ohlcv__mutmut_23, 
    'x_map_ohlcv__mutmut_24': x_map_ohlcv__mutmut_24, 
    'x_map_ohlcv__mutmut_25': x_map_ohlcv__mutmut_25, 
    'x_map_ohlcv__mutmut_26': x_map_ohlcv__mutmut_26, 
    'x_map_ohlcv__mutmut_27': x_map_ohlcv__mutmut_27, 
    'x_map_ohlcv__mutmut_28': x_map_ohlcv__mutmut_28, 
    'x_map_ohlcv__mutmut_29': x_map_ohlcv__mutmut_29, 
    'x_map_ohlcv__mutmut_30': x_map_ohlcv__mutmut_30, 
    'x_map_ohlcv__mutmut_31': x_map_ohlcv__mutmut_31, 
    'x_map_ohlcv__mutmut_32': x_map_ohlcv__mutmut_32, 
    'x_map_ohlcv__mutmut_33': x_map_ohlcv__mutmut_33, 
    'x_map_ohlcv__mutmut_34': x_map_ohlcv__mutmut_34, 
    'x_map_ohlcv__mutmut_35': x_map_ohlcv__mutmut_35, 
    'x_map_ohlcv__mutmut_36': x_map_ohlcv__mutmut_36, 
    'x_map_ohlcv__mutmut_37': x_map_ohlcv__mutmut_37, 
    'x_map_ohlcv__mutmut_38': x_map_ohlcv__mutmut_38, 
    'x_map_ohlcv__mutmut_39': x_map_ohlcv__mutmut_39, 
    'x_map_ohlcv__mutmut_40': x_map_ohlcv__mutmut_40, 
    'x_map_ohlcv__mutmut_41': x_map_ohlcv__mutmut_41
}

def map_ohlcv(*args, **kwargs):
    result = _mutmut_trampoline(x_map_ohlcv__mutmut_orig, x_map_ohlcv__mutmut_mutants, args, kwargs)
    return result 

map_ohlcv.__signature__ = _mutmut_signature(x_map_ohlcv__mutmut_orig)
x_map_ohlcv__mutmut_orig.__name__ = 'x_map_ohlcv'
