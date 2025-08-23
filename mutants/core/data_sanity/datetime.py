from collections.abc import Callable
from inspect import signature as _mutmut_signature
from typing import Annotated, ClassVar

import pandas as pd

from .codes import (
    DUPLICATE_TSTAMPS,
    INDEX_NON_MONOTONIC,
    NO_VALID_DT_INDEX,
    TZ_NAIVE,
    TZ_NON_UTC,
)
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

def x__coerce_any_to_utc_index__mutmut_orig(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors="coerce", utc=True)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_1(idx_like) -> pd.DatetimeIndex:
    idx = None
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_2(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(None, errors="coerce", utc=True)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_3(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors=None, utc=True)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_4(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors="coerce", utc=None)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_5(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(errors="coerce", utc=True)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_6(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, utc=True)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_7(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors="coerce", )
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_8(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors="XXcoerceXX", utc=True)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_9(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors="COERCE", utc=True)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_10(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors="coerce", utc=False)
    return pd.DatetimeIndex(idx)

def x__coerce_any_to_utc_index__mutmut_11(idx_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx_like, errors="coerce", utc=True)
    return pd.DatetimeIndex(None)

x__coerce_any_to_utc_index__mutmut_mutants : ClassVar[MutantDict] = {
'x__coerce_any_to_utc_index__mutmut_1': x__coerce_any_to_utc_index__mutmut_1, 
    'x__coerce_any_to_utc_index__mutmut_2': x__coerce_any_to_utc_index__mutmut_2, 
    'x__coerce_any_to_utc_index__mutmut_3': x__coerce_any_to_utc_index__mutmut_3, 
    'x__coerce_any_to_utc_index__mutmut_4': x__coerce_any_to_utc_index__mutmut_4, 
    'x__coerce_any_to_utc_index__mutmut_5': x__coerce_any_to_utc_index__mutmut_5, 
    'x__coerce_any_to_utc_index__mutmut_6': x__coerce_any_to_utc_index__mutmut_6, 
    'x__coerce_any_to_utc_index__mutmut_7': x__coerce_any_to_utc_index__mutmut_7, 
    'x__coerce_any_to_utc_index__mutmut_8': x__coerce_any_to_utc_index__mutmut_8, 
    'x__coerce_any_to_utc_index__mutmut_9': x__coerce_any_to_utc_index__mutmut_9, 
    'x__coerce_any_to_utc_index__mutmut_10': x__coerce_any_to_utc_index__mutmut_10, 
    'x__coerce_any_to_utc_index__mutmut_11': x__coerce_any_to_utc_index__mutmut_11
}

def _coerce_any_to_utc_index(*args, **kwargs):
    result = _mutmut_trampoline(x__coerce_any_to_utc_index__mutmut_orig, x__coerce_any_to_utc_index__mutmut_mutants, args, kwargs)
    return result 

_coerce_any_to_utc_index.__signature__ = _mutmut_signature(x__coerce_any_to_utc_index__mutmut_orig)
x__coerce_any_to_utc_index__mutmut_orig.__name__ = 'x__coerce_any_to_utc_index'

def x__ensure_singlelevel_utc_index__mutmut_orig(idx: pd.Index, strict: bool):
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

def x__ensure_singlelevel_utc_index__mutmut_1(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
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

def x__ensure_singlelevel_utc_index__mutmut_2(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(None)
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_3(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(None, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_4(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, None))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_5(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring("timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_6(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, ))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_7(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "XXtimestamps must be tz-aware UTCXX"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_8(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware utc"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_9(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "TIMESTAMPS MUST BE TZ-AWARE UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_10(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = None
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_11(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize(None)
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_12(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("XXUTCXX")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_13(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("utc")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_14(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(None) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_15(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) == "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_16(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "XXUTCXX":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_17(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "utc":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_18(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(None)
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_19(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(None, f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_20(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, None))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_21(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(f"{idx.tz}"))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_22(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, ))
                idx = idx.tz_convert("UTC")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_23(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = None
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_24(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert(None)
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_25(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("XXUTCXX")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_26(idx: pd.Index, strict: bool):
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if strict:
                raise DataSanityError(estring(TZ_NAIVE, "timestamps must be tz-aware UTC"))
            idx = idx.tz_localize("UTC")
        else:
            if str(idx.tz) != "UTC":
                if strict:
                    raise DataSanityError(estring(TZ_NON_UTC, f"{idx.tz}"))
                idx = idx.tz_convert("utc")
        return pd.DatetimeIndex(idx, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_27(idx: pd.Index, strict: bool):
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
        return pd.DatetimeIndex(None, tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_28(idx: pd.Index, strict: bool):
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
        return pd.DatetimeIndex(idx, tz=None)
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_29(idx: pd.Index, strict: bool):
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
        return pd.DatetimeIndex(tz="UTC")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_30(idx: pd.Index, strict: bool):
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
        return pd.DatetimeIndex(idx, )
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_31(idx: pd.Index, strict: bool):
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
        return pd.DatetimeIndex(idx, tz="XXUTCXX")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_32(idx: pd.Index, strict: bool):
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
        return pd.DatetimeIndex(idx, tz="utc")
    # object/index → try coercion
    return _coerce_any_to_utc_index(idx)

def x__ensure_singlelevel_utc_index__mutmut_33(idx: pd.Index, strict: bool):
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
    return _coerce_any_to_utc_index(None)

x__ensure_singlelevel_utc_index__mutmut_mutants : ClassVar[MutantDict] = {
'x__ensure_singlelevel_utc_index__mutmut_1': x__ensure_singlelevel_utc_index__mutmut_1, 
    'x__ensure_singlelevel_utc_index__mutmut_2': x__ensure_singlelevel_utc_index__mutmut_2, 
    'x__ensure_singlelevel_utc_index__mutmut_3': x__ensure_singlelevel_utc_index__mutmut_3, 
    'x__ensure_singlelevel_utc_index__mutmut_4': x__ensure_singlelevel_utc_index__mutmut_4, 
    'x__ensure_singlelevel_utc_index__mutmut_5': x__ensure_singlelevel_utc_index__mutmut_5, 
    'x__ensure_singlelevel_utc_index__mutmut_6': x__ensure_singlelevel_utc_index__mutmut_6, 
    'x__ensure_singlelevel_utc_index__mutmut_7': x__ensure_singlelevel_utc_index__mutmut_7, 
    'x__ensure_singlelevel_utc_index__mutmut_8': x__ensure_singlelevel_utc_index__mutmut_8, 
    'x__ensure_singlelevel_utc_index__mutmut_9': x__ensure_singlelevel_utc_index__mutmut_9, 
    'x__ensure_singlelevel_utc_index__mutmut_10': x__ensure_singlelevel_utc_index__mutmut_10, 
    'x__ensure_singlelevel_utc_index__mutmut_11': x__ensure_singlelevel_utc_index__mutmut_11, 
    'x__ensure_singlelevel_utc_index__mutmut_12': x__ensure_singlelevel_utc_index__mutmut_12, 
    'x__ensure_singlelevel_utc_index__mutmut_13': x__ensure_singlelevel_utc_index__mutmut_13, 
    'x__ensure_singlelevel_utc_index__mutmut_14': x__ensure_singlelevel_utc_index__mutmut_14, 
    'x__ensure_singlelevel_utc_index__mutmut_15': x__ensure_singlelevel_utc_index__mutmut_15, 
    'x__ensure_singlelevel_utc_index__mutmut_16': x__ensure_singlelevel_utc_index__mutmut_16, 
    'x__ensure_singlelevel_utc_index__mutmut_17': x__ensure_singlelevel_utc_index__mutmut_17, 
    'x__ensure_singlelevel_utc_index__mutmut_18': x__ensure_singlelevel_utc_index__mutmut_18, 
    'x__ensure_singlelevel_utc_index__mutmut_19': x__ensure_singlelevel_utc_index__mutmut_19, 
    'x__ensure_singlelevel_utc_index__mutmut_20': x__ensure_singlelevel_utc_index__mutmut_20, 
    'x__ensure_singlelevel_utc_index__mutmut_21': x__ensure_singlelevel_utc_index__mutmut_21, 
    'x__ensure_singlelevel_utc_index__mutmut_22': x__ensure_singlelevel_utc_index__mutmut_22, 
    'x__ensure_singlelevel_utc_index__mutmut_23': x__ensure_singlelevel_utc_index__mutmut_23, 
    'x__ensure_singlelevel_utc_index__mutmut_24': x__ensure_singlelevel_utc_index__mutmut_24, 
    'x__ensure_singlelevel_utc_index__mutmut_25': x__ensure_singlelevel_utc_index__mutmut_25, 
    'x__ensure_singlelevel_utc_index__mutmut_26': x__ensure_singlelevel_utc_index__mutmut_26, 
    'x__ensure_singlelevel_utc_index__mutmut_27': x__ensure_singlelevel_utc_index__mutmut_27, 
    'x__ensure_singlelevel_utc_index__mutmut_28': x__ensure_singlelevel_utc_index__mutmut_28, 
    'x__ensure_singlelevel_utc_index__mutmut_29': x__ensure_singlelevel_utc_index__mutmut_29, 
    'x__ensure_singlelevel_utc_index__mutmut_30': x__ensure_singlelevel_utc_index__mutmut_30, 
    'x__ensure_singlelevel_utc_index__mutmut_31': x__ensure_singlelevel_utc_index__mutmut_31, 
    'x__ensure_singlelevel_utc_index__mutmut_32': x__ensure_singlelevel_utc_index__mutmut_32, 
    'x__ensure_singlelevel_utc_index__mutmut_33': x__ensure_singlelevel_utc_index__mutmut_33
}

def _ensure_singlelevel_utc_index(*args, **kwargs):
    result = _mutmut_trampoline(x__ensure_singlelevel_utc_index__mutmut_orig, x__ensure_singlelevel_utc_index__mutmut_mutants, args, kwargs)
    return result 

_ensure_singlelevel_utc_index.__signature__ = _mutmut_signature(x__ensure_singlelevel_utc_index__mutmut_orig)
x__ensure_singlelevel_utc_index__mutmut_orig.__name__ = 'x__ensure_singlelevel_utc_index'

def x_canonicalize_datetime_index__mutmut_orig(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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

def x_canonicalize_datetime_index__mutmut_1(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = None
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

def x_canonicalize_datetime_index__mutmut_2(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = profile.get("allow_repairs", True) if profile is not None else False
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

def x_canonicalize_datetime_index__mutmut_3(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get(None, True) if profile is not None else False
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

def x_canonicalize_datetime_index__mutmut_4(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", None) if profile is not None else False
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

def x_canonicalize_datetime_index__mutmut_5(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get(True) if profile is not None else False
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

def x_canonicalize_datetime_index__mutmut_6(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", ) if profile is not None else False
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

def x_canonicalize_datetime_index__mutmut_7(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("XXallow_repairsXX", True) if profile is not None else False
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

def x_canonicalize_datetime_index__mutmut_8(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("ALLOW_REPAIRS", True) if profile is not None else False
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

def x_canonicalize_datetime_index__mutmut_9(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", False) if profile is not None else False
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

def x_canonicalize_datetime_index__mutmut_10(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is None else False
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

def x_canonicalize_datetime_index__mutmut_11(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else True
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

def x_canonicalize_datetime_index__mutmut_12(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = None

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

def x_canonicalize_datetime_index__mutmut_13(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col or ts_col in out.columns:
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

def x_canonicalize_datetime_index__mutmut_14(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col not in out.columns:
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

def x_canonicalize_datetime_index__mutmut_15(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = None
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

def x_canonicalize_datetime_index__mutmut_16(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(None)
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

def x_canonicalize_datetime_index__mutmut_17(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = None
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

def x_canonicalize_datetime_index__mutmut_18(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(None)
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

def x_canonicalize_datetime_index__mutmut_19(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "XXtimestampXX" in names:
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

def x_canonicalize_datetime_index__mutmut_20(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "TIMESTAMP" in names:
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

def x_canonicalize_datetime_index__mutmut_21(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "timestamp" not in names:
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

def x_canonicalize_datetime_index__mutmut_22(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "timestamp" in names:
            lvl = None
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

def x_canonicalize_datetime_index__mutmut_23(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "timestamp" in names:
            lvl = names.index(None)
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

def x_canonicalize_datetime_index__mutmut_24(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "timestamp" in names:
            lvl = names.rindex("timestamp")
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

def x_canonicalize_datetime_index__mutmut_25(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "timestamp" in names:
            lvl = names.index("XXtimestampXX")
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

def x_canonicalize_datetime_index__mutmut_26(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
    strict = not profile.get("allow_repairs", True) if profile is not None else False
    out = df.copy()

    # Select timestamp source
    if ts_col and ts_col in out.columns:
        idx = _coerce_any_to_utc_index(out[ts_col])
    elif isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        # try 'timestamp' level; else pick level with most valid timestamps
        if "timestamp" in names:
            lvl = names.index("TIMESTAMP")
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

def x_canonicalize_datetime_index__mutmut_27(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raw = None
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

def x_canonicalize_datetime_index__mutmut_28(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raw = out.index.get_level_values(None)
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

def x_canonicalize_datetime_index__mutmut_29(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            idx = None
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

def x_canonicalize_datetime_index__mutmut_30(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            idx = _coerce_any_to_utc_index(None)
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

def x_canonicalize_datetime_index__mutmut_31(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            arrays = None
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

def x_canonicalize_datetime_index__mutmut_32(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            arrays = [idx if i != lvl else out.index.get_level_values(i) for i in range(len(names))]
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

def x_canonicalize_datetime_index__mutmut_33(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            arrays = [idx if i == lvl else out.index.get_level_values(None) for i in range(len(names))]
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

def x_canonicalize_datetime_index__mutmut_34(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            arrays = [idx if i == lvl else out.index.get_level_values(i) for i in range(None)]
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

def x_canonicalize_datetime_index__mutmut_35(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = None
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

def x_canonicalize_datetime_index__mutmut_36(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = pd.MultiIndex.from_arrays(None, names=names)
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

def x_canonicalize_datetime_index__mutmut_37(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = pd.MultiIndex.from_arrays(arrays, names=None)
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

def x_canonicalize_datetime_index__mutmut_38(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = pd.MultiIndex.from_arrays(names=names)
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

def x_canonicalize_datetime_index__mutmut_39(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = pd.MultiIndex.from_arrays(arrays, )
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

def x_canonicalize_datetime_index__mutmut_40(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            best_i, best_idx, best_valid = None
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

def x_canonicalize_datetime_index__mutmut_41(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            best_i, best_idx, best_valid = None, None, +1
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

def x_canonicalize_datetime_index__mutmut_42(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            best_i, best_idx, best_valid = None, None, -2
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

def x_canonicalize_datetime_index__mutmut_43(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            for i in range(None):
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

def x_canonicalize_datetime_index__mutmut_44(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                cand = None
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

def x_canonicalize_datetime_index__mutmut_45(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                cand = _coerce_any_to_utc_index(None)
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

def x_canonicalize_datetime_index__mutmut_46(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                cand = _coerce_any_to_utc_index(out.index.get_level_values(None))
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

def x_canonicalize_datetime_index__mutmut_47(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                valid = None
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

def x_canonicalize_datetime_index__mutmut_48(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                valid = cand.isna().sum()
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

def x_canonicalize_datetime_index__mutmut_49(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                if valid >= best_valid:
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

def x_canonicalize_datetime_index__mutmut_50(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    best_i, best_idx, best_valid = None
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

def x_canonicalize_datetime_index__mutmut_51(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            if best_idx is None and best_valid == 0:
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

def x_canonicalize_datetime_index__mutmut_52(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            if best_idx is not None or best_valid == 0:
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

def x_canonicalize_datetime_index__mutmut_53(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            if best_idx is None or best_valid != 0:
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

def x_canonicalize_datetime_index__mutmut_54(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            if best_idx is None or best_valid == 1:
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

def x_canonicalize_datetime_index__mutmut_55(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    raise DataSanityError(None)
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

def x_canonicalize_datetime_index__mutmut_56(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    raise DataSanityError(estring(None, "cannot parse MultiIndex"))
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

def x_canonicalize_datetime_index__mutmut_57(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    raise DataSanityError(estring(NO_VALID_DT_INDEX, None))
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

def x_canonicalize_datetime_index__mutmut_58(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    raise DataSanityError(estring("cannot parse MultiIndex"))
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

def x_canonicalize_datetime_index__mutmut_59(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    raise DataSanityError(estring(NO_VALID_DT_INDEX, ))
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

def x_canonicalize_datetime_index__mutmut_60(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    raise DataSanityError(estring(NO_VALID_DT_INDEX, "XXcannot parse MultiIndexXX"))
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

def x_canonicalize_datetime_index__mutmut_61(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    raise DataSanityError(estring(NO_VALID_DT_INDEX, "cannot parse multiindex"))
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

def x_canonicalize_datetime_index__mutmut_62(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                    raise DataSanityError(estring(NO_VALID_DT_INDEX, "CANNOT PARSE MULTIINDEX"))
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

def x_canonicalize_datetime_index__mutmut_63(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                return out.iloc[1:0]
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

def x_canonicalize_datetime_index__mutmut_64(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
                return out.iloc[0:1]
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

def x_canonicalize_datetime_index__mutmut_65(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            arrays = None
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

def x_canonicalize_datetime_index__mutmut_66(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            arrays = [best_idx if j != best_i else out.index.get_level_values(j) for j in range(len(names))]
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

def x_canonicalize_datetime_index__mutmut_67(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            arrays = [best_idx if j == best_i else out.index.get_level_values(None) for j in range(len(names))]
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

def x_canonicalize_datetime_index__mutmut_68(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            arrays = [best_idx if j == best_i else out.index.get_level_values(j) for j in range(None)]
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

def x_canonicalize_datetime_index__mutmut_69(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            names[best_i] = None
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

def x_canonicalize_datetime_index__mutmut_70(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            names[best_i] = "XXtimestampXX"
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

def x_canonicalize_datetime_index__mutmut_71(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            names[best_i] = "TIMESTAMP"
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

def x_canonicalize_datetime_index__mutmut_72(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = None
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

def x_canonicalize_datetime_index__mutmut_73(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = pd.MultiIndex.from_arrays(None, names=names)
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

def x_canonicalize_datetime_index__mutmut_74(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = pd.MultiIndex.from_arrays(arrays, names=None)
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

def x_canonicalize_datetime_index__mutmut_75(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = pd.MultiIndex.from_arrays(names=names)
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

def x_canonicalize_datetime_index__mutmut_76(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            out.index = pd.MultiIndex.from_arrays(arrays, )
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

def x_canonicalize_datetime_index__mutmut_77(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = None
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

def x_canonicalize_datetime_index__mutmut_78(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = out.index.get_level_values(None)
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

def x_canonicalize_datetime_index__mutmut_79(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = out.index.get_level_values("XXtimestampXX")
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

def x_canonicalize_datetime_index__mutmut_80(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = out.index.get_level_values("TIMESTAMP")
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

def x_canonicalize_datetime_index__mutmut_81(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = None

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

def x_canonicalize_datetime_index__mutmut_82(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = _ensure_singlelevel_utc_index(None, strict)

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

def x_canonicalize_datetime_index__mutmut_83(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = _ensure_singlelevel_utc_index(out.index, None)

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

def x_canonicalize_datetime_index__mutmut_84(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = _ensure_singlelevel_utc_index(strict)

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

def x_canonicalize_datetime_index__mutmut_85(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = _ensure_singlelevel_utc_index(out.index, )

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

def x_canonicalize_datetime_index__mutmut_86(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        if any(None):
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

def x_canonicalize_datetime_index__mutmut_87(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        if any(pd.isna(None)):
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

def x_canonicalize_datetime_index__mutmut_88(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(None)
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

def x_canonicalize_datetime_index__mutmut_89(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(None, "NaT present"))
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

def x_canonicalize_datetime_index__mutmut_90(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(NO_VALID_DT_INDEX, None))
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

def x_canonicalize_datetime_index__mutmut_91(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring("NaT present"))
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

def x_canonicalize_datetime_index__mutmut_92(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(NO_VALID_DT_INDEX, ))
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

def x_canonicalize_datetime_index__mutmut_93(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(NO_VALID_DT_INDEX, "XXNaT presentXX"))
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

def x_canonicalize_datetime_index__mutmut_94(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(NO_VALID_DT_INDEX, "nat present"))
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

def x_canonicalize_datetime_index__mutmut_95(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(NO_VALID_DT_INDEX, "NAT PRESENT"))
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

def x_canonicalize_datetime_index__mutmut_96(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(None)
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

def x_canonicalize_datetime_index__mutmut_97(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(None, "duplicate timestamps"))
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

def x_canonicalize_datetime_index__mutmut_98(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(DUPLICATE_TSTAMPS, None))
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

def x_canonicalize_datetime_index__mutmut_99(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring("duplicate timestamps"))
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

def x_canonicalize_datetime_index__mutmut_100(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(DUPLICATE_TSTAMPS, ))
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

def x_canonicalize_datetime_index__mutmut_101(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(DUPLICATE_TSTAMPS, "XXduplicate timestampsXX"))
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

def x_canonicalize_datetime_index__mutmut_102(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(DUPLICATE_TSTAMPS, "DUPLICATE TIMESTAMPS"))
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

def x_canonicalize_datetime_index__mutmut_103(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        if idx.is_monotonic_increasing:
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

def x_canonicalize_datetime_index__mutmut_104(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(None)

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

def x_canonicalize_datetime_index__mutmut_105(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(None, "timestamps not sorted"))

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

def x_canonicalize_datetime_index__mutmut_106(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(INDEX_NON_MONOTONIC, None))

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

def x_canonicalize_datetime_index__mutmut_107(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring("timestamps not sorted"))

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

def x_canonicalize_datetime_index__mutmut_108(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(INDEX_NON_MONOTONIC, ))

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

def x_canonicalize_datetime_index__mutmut_109(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(INDEX_NON_MONOTONIC, "XXtimestamps not sortedXX"))

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

def x_canonicalize_datetime_index__mutmut_110(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
            raise DataSanityError(estring(INDEX_NON_MONOTONIC, "TIMESTAMPS NOT SORTED"))

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

def x_canonicalize_datetime_index__mutmut_111(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    keep = None
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

def x_canonicalize_datetime_index__mutmut_112(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    keep = pd.isna(idx)
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

def x_canonicalize_datetime_index__mutmut_113(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    keep = ~pd.isna(None)
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

def x_canonicalize_datetime_index__mutmut_114(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    if keep.any():
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

def x_canonicalize_datetime_index__mutmut_115(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        out = None
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

def x_canonicalize_datetime_index__mutmut_116(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = None
    order = idx.argsort()
    out = out.iloc[order]
    idx = pd.DatetimeIndex(idx[order], tz="UTC", name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_117(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    order = None
    out = out.iloc[order]
    idx = pd.DatetimeIndex(idx[order], tz="UTC", name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_118(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    out = None
    idx = pd.DatetimeIndex(idx[order], tz="UTC", name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_119(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = None
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_120(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(None, tz="UTC", name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_121(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(idx[order], tz=None, name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_122(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(idx[order], tz="UTC", name=None)
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_123(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(tz="UTC", name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_124(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(idx[order], name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_125(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(idx[order], tz="UTC", )
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_126(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(idx[order], tz="XXUTCXX", name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_127(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(idx[order], tz="utc", name="timestamp")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_128(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(idx[order], tz="UTC", name="XXtimestampXX")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_129(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    idx = pd.DatetimeIndex(idx[order], tz="UTC", name="TIMESTAMP")
    if idx.has_duplicates:
        first = ~idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_130(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        first = None
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_131(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        first = idx.duplicated(keep="first")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_132(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        first = ~idx.duplicated(keep=None)
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_133(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        first = ~idx.duplicated(keep="XXfirstXX")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_134(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        first = ~idx.duplicated(keep="FIRST")
        out = out.loc[first]
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_135(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        out = None
        idx = idx[first]
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_136(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
        idx = None
    out.index = idx
    return out

def x_canonicalize_datetime_index__mutmut_137(df: pd.DataFrame, profile=None, ts_col: str | None = None) -> pd.DataFrame:
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
    out.index = None
    return out

x_canonicalize_datetime_index__mutmut_mutants : ClassVar[MutantDict] = {
'x_canonicalize_datetime_index__mutmut_1': x_canonicalize_datetime_index__mutmut_1, 
    'x_canonicalize_datetime_index__mutmut_2': x_canonicalize_datetime_index__mutmut_2, 
    'x_canonicalize_datetime_index__mutmut_3': x_canonicalize_datetime_index__mutmut_3, 
    'x_canonicalize_datetime_index__mutmut_4': x_canonicalize_datetime_index__mutmut_4, 
    'x_canonicalize_datetime_index__mutmut_5': x_canonicalize_datetime_index__mutmut_5, 
    'x_canonicalize_datetime_index__mutmut_6': x_canonicalize_datetime_index__mutmut_6, 
    'x_canonicalize_datetime_index__mutmut_7': x_canonicalize_datetime_index__mutmut_7, 
    'x_canonicalize_datetime_index__mutmut_8': x_canonicalize_datetime_index__mutmut_8, 
    'x_canonicalize_datetime_index__mutmut_9': x_canonicalize_datetime_index__mutmut_9, 
    'x_canonicalize_datetime_index__mutmut_10': x_canonicalize_datetime_index__mutmut_10, 
    'x_canonicalize_datetime_index__mutmut_11': x_canonicalize_datetime_index__mutmut_11, 
    'x_canonicalize_datetime_index__mutmut_12': x_canonicalize_datetime_index__mutmut_12, 
    'x_canonicalize_datetime_index__mutmut_13': x_canonicalize_datetime_index__mutmut_13, 
    'x_canonicalize_datetime_index__mutmut_14': x_canonicalize_datetime_index__mutmut_14, 
    'x_canonicalize_datetime_index__mutmut_15': x_canonicalize_datetime_index__mutmut_15, 
    'x_canonicalize_datetime_index__mutmut_16': x_canonicalize_datetime_index__mutmut_16, 
    'x_canonicalize_datetime_index__mutmut_17': x_canonicalize_datetime_index__mutmut_17, 
    'x_canonicalize_datetime_index__mutmut_18': x_canonicalize_datetime_index__mutmut_18, 
    'x_canonicalize_datetime_index__mutmut_19': x_canonicalize_datetime_index__mutmut_19, 
    'x_canonicalize_datetime_index__mutmut_20': x_canonicalize_datetime_index__mutmut_20, 
    'x_canonicalize_datetime_index__mutmut_21': x_canonicalize_datetime_index__mutmut_21, 
    'x_canonicalize_datetime_index__mutmut_22': x_canonicalize_datetime_index__mutmut_22, 
    'x_canonicalize_datetime_index__mutmut_23': x_canonicalize_datetime_index__mutmut_23, 
    'x_canonicalize_datetime_index__mutmut_24': x_canonicalize_datetime_index__mutmut_24, 
    'x_canonicalize_datetime_index__mutmut_25': x_canonicalize_datetime_index__mutmut_25, 
    'x_canonicalize_datetime_index__mutmut_26': x_canonicalize_datetime_index__mutmut_26, 
    'x_canonicalize_datetime_index__mutmut_27': x_canonicalize_datetime_index__mutmut_27, 
    'x_canonicalize_datetime_index__mutmut_28': x_canonicalize_datetime_index__mutmut_28, 
    'x_canonicalize_datetime_index__mutmut_29': x_canonicalize_datetime_index__mutmut_29, 
    'x_canonicalize_datetime_index__mutmut_30': x_canonicalize_datetime_index__mutmut_30, 
    'x_canonicalize_datetime_index__mutmut_31': x_canonicalize_datetime_index__mutmut_31, 
    'x_canonicalize_datetime_index__mutmut_32': x_canonicalize_datetime_index__mutmut_32, 
    'x_canonicalize_datetime_index__mutmut_33': x_canonicalize_datetime_index__mutmut_33, 
    'x_canonicalize_datetime_index__mutmut_34': x_canonicalize_datetime_index__mutmut_34, 
    'x_canonicalize_datetime_index__mutmut_35': x_canonicalize_datetime_index__mutmut_35, 
    'x_canonicalize_datetime_index__mutmut_36': x_canonicalize_datetime_index__mutmut_36, 
    'x_canonicalize_datetime_index__mutmut_37': x_canonicalize_datetime_index__mutmut_37, 
    'x_canonicalize_datetime_index__mutmut_38': x_canonicalize_datetime_index__mutmut_38, 
    'x_canonicalize_datetime_index__mutmut_39': x_canonicalize_datetime_index__mutmut_39, 
    'x_canonicalize_datetime_index__mutmut_40': x_canonicalize_datetime_index__mutmut_40, 
    'x_canonicalize_datetime_index__mutmut_41': x_canonicalize_datetime_index__mutmut_41, 
    'x_canonicalize_datetime_index__mutmut_42': x_canonicalize_datetime_index__mutmut_42, 
    'x_canonicalize_datetime_index__mutmut_43': x_canonicalize_datetime_index__mutmut_43, 
    'x_canonicalize_datetime_index__mutmut_44': x_canonicalize_datetime_index__mutmut_44, 
    'x_canonicalize_datetime_index__mutmut_45': x_canonicalize_datetime_index__mutmut_45, 
    'x_canonicalize_datetime_index__mutmut_46': x_canonicalize_datetime_index__mutmut_46, 
    'x_canonicalize_datetime_index__mutmut_47': x_canonicalize_datetime_index__mutmut_47, 
    'x_canonicalize_datetime_index__mutmut_48': x_canonicalize_datetime_index__mutmut_48, 
    'x_canonicalize_datetime_index__mutmut_49': x_canonicalize_datetime_index__mutmut_49, 
    'x_canonicalize_datetime_index__mutmut_50': x_canonicalize_datetime_index__mutmut_50, 
    'x_canonicalize_datetime_index__mutmut_51': x_canonicalize_datetime_index__mutmut_51, 
    'x_canonicalize_datetime_index__mutmut_52': x_canonicalize_datetime_index__mutmut_52, 
    'x_canonicalize_datetime_index__mutmut_53': x_canonicalize_datetime_index__mutmut_53, 
    'x_canonicalize_datetime_index__mutmut_54': x_canonicalize_datetime_index__mutmut_54, 
    'x_canonicalize_datetime_index__mutmut_55': x_canonicalize_datetime_index__mutmut_55, 
    'x_canonicalize_datetime_index__mutmut_56': x_canonicalize_datetime_index__mutmut_56, 
    'x_canonicalize_datetime_index__mutmut_57': x_canonicalize_datetime_index__mutmut_57, 
    'x_canonicalize_datetime_index__mutmut_58': x_canonicalize_datetime_index__mutmut_58, 
    'x_canonicalize_datetime_index__mutmut_59': x_canonicalize_datetime_index__mutmut_59, 
    'x_canonicalize_datetime_index__mutmut_60': x_canonicalize_datetime_index__mutmut_60, 
    'x_canonicalize_datetime_index__mutmut_61': x_canonicalize_datetime_index__mutmut_61, 
    'x_canonicalize_datetime_index__mutmut_62': x_canonicalize_datetime_index__mutmut_62, 
    'x_canonicalize_datetime_index__mutmut_63': x_canonicalize_datetime_index__mutmut_63, 
    'x_canonicalize_datetime_index__mutmut_64': x_canonicalize_datetime_index__mutmut_64, 
    'x_canonicalize_datetime_index__mutmut_65': x_canonicalize_datetime_index__mutmut_65, 
    'x_canonicalize_datetime_index__mutmut_66': x_canonicalize_datetime_index__mutmut_66, 
    'x_canonicalize_datetime_index__mutmut_67': x_canonicalize_datetime_index__mutmut_67, 
    'x_canonicalize_datetime_index__mutmut_68': x_canonicalize_datetime_index__mutmut_68, 
    'x_canonicalize_datetime_index__mutmut_69': x_canonicalize_datetime_index__mutmut_69, 
    'x_canonicalize_datetime_index__mutmut_70': x_canonicalize_datetime_index__mutmut_70, 
    'x_canonicalize_datetime_index__mutmut_71': x_canonicalize_datetime_index__mutmut_71, 
    'x_canonicalize_datetime_index__mutmut_72': x_canonicalize_datetime_index__mutmut_72, 
    'x_canonicalize_datetime_index__mutmut_73': x_canonicalize_datetime_index__mutmut_73, 
    'x_canonicalize_datetime_index__mutmut_74': x_canonicalize_datetime_index__mutmut_74, 
    'x_canonicalize_datetime_index__mutmut_75': x_canonicalize_datetime_index__mutmut_75, 
    'x_canonicalize_datetime_index__mutmut_76': x_canonicalize_datetime_index__mutmut_76, 
    'x_canonicalize_datetime_index__mutmut_77': x_canonicalize_datetime_index__mutmut_77, 
    'x_canonicalize_datetime_index__mutmut_78': x_canonicalize_datetime_index__mutmut_78, 
    'x_canonicalize_datetime_index__mutmut_79': x_canonicalize_datetime_index__mutmut_79, 
    'x_canonicalize_datetime_index__mutmut_80': x_canonicalize_datetime_index__mutmut_80, 
    'x_canonicalize_datetime_index__mutmut_81': x_canonicalize_datetime_index__mutmut_81, 
    'x_canonicalize_datetime_index__mutmut_82': x_canonicalize_datetime_index__mutmut_82, 
    'x_canonicalize_datetime_index__mutmut_83': x_canonicalize_datetime_index__mutmut_83, 
    'x_canonicalize_datetime_index__mutmut_84': x_canonicalize_datetime_index__mutmut_84, 
    'x_canonicalize_datetime_index__mutmut_85': x_canonicalize_datetime_index__mutmut_85, 
    'x_canonicalize_datetime_index__mutmut_86': x_canonicalize_datetime_index__mutmut_86, 
    'x_canonicalize_datetime_index__mutmut_87': x_canonicalize_datetime_index__mutmut_87, 
    'x_canonicalize_datetime_index__mutmut_88': x_canonicalize_datetime_index__mutmut_88, 
    'x_canonicalize_datetime_index__mutmut_89': x_canonicalize_datetime_index__mutmut_89, 
    'x_canonicalize_datetime_index__mutmut_90': x_canonicalize_datetime_index__mutmut_90, 
    'x_canonicalize_datetime_index__mutmut_91': x_canonicalize_datetime_index__mutmut_91, 
    'x_canonicalize_datetime_index__mutmut_92': x_canonicalize_datetime_index__mutmut_92, 
    'x_canonicalize_datetime_index__mutmut_93': x_canonicalize_datetime_index__mutmut_93, 
    'x_canonicalize_datetime_index__mutmut_94': x_canonicalize_datetime_index__mutmut_94, 
    'x_canonicalize_datetime_index__mutmut_95': x_canonicalize_datetime_index__mutmut_95, 
    'x_canonicalize_datetime_index__mutmut_96': x_canonicalize_datetime_index__mutmut_96, 
    'x_canonicalize_datetime_index__mutmut_97': x_canonicalize_datetime_index__mutmut_97, 
    'x_canonicalize_datetime_index__mutmut_98': x_canonicalize_datetime_index__mutmut_98, 
    'x_canonicalize_datetime_index__mutmut_99': x_canonicalize_datetime_index__mutmut_99, 
    'x_canonicalize_datetime_index__mutmut_100': x_canonicalize_datetime_index__mutmut_100, 
    'x_canonicalize_datetime_index__mutmut_101': x_canonicalize_datetime_index__mutmut_101, 
    'x_canonicalize_datetime_index__mutmut_102': x_canonicalize_datetime_index__mutmut_102, 
    'x_canonicalize_datetime_index__mutmut_103': x_canonicalize_datetime_index__mutmut_103, 
    'x_canonicalize_datetime_index__mutmut_104': x_canonicalize_datetime_index__mutmut_104, 
    'x_canonicalize_datetime_index__mutmut_105': x_canonicalize_datetime_index__mutmut_105, 
    'x_canonicalize_datetime_index__mutmut_106': x_canonicalize_datetime_index__mutmut_106, 
    'x_canonicalize_datetime_index__mutmut_107': x_canonicalize_datetime_index__mutmut_107, 
    'x_canonicalize_datetime_index__mutmut_108': x_canonicalize_datetime_index__mutmut_108, 
    'x_canonicalize_datetime_index__mutmut_109': x_canonicalize_datetime_index__mutmut_109, 
    'x_canonicalize_datetime_index__mutmut_110': x_canonicalize_datetime_index__mutmut_110, 
    'x_canonicalize_datetime_index__mutmut_111': x_canonicalize_datetime_index__mutmut_111, 
    'x_canonicalize_datetime_index__mutmut_112': x_canonicalize_datetime_index__mutmut_112, 
    'x_canonicalize_datetime_index__mutmut_113': x_canonicalize_datetime_index__mutmut_113, 
    'x_canonicalize_datetime_index__mutmut_114': x_canonicalize_datetime_index__mutmut_114, 
    'x_canonicalize_datetime_index__mutmut_115': x_canonicalize_datetime_index__mutmut_115, 
    'x_canonicalize_datetime_index__mutmut_116': x_canonicalize_datetime_index__mutmut_116, 
    'x_canonicalize_datetime_index__mutmut_117': x_canonicalize_datetime_index__mutmut_117, 
    'x_canonicalize_datetime_index__mutmut_118': x_canonicalize_datetime_index__mutmut_118, 
    'x_canonicalize_datetime_index__mutmut_119': x_canonicalize_datetime_index__mutmut_119, 
    'x_canonicalize_datetime_index__mutmut_120': x_canonicalize_datetime_index__mutmut_120, 
    'x_canonicalize_datetime_index__mutmut_121': x_canonicalize_datetime_index__mutmut_121, 
    'x_canonicalize_datetime_index__mutmut_122': x_canonicalize_datetime_index__mutmut_122, 
    'x_canonicalize_datetime_index__mutmut_123': x_canonicalize_datetime_index__mutmut_123, 
    'x_canonicalize_datetime_index__mutmut_124': x_canonicalize_datetime_index__mutmut_124, 
    'x_canonicalize_datetime_index__mutmut_125': x_canonicalize_datetime_index__mutmut_125, 
    'x_canonicalize_datetime_index__mutmut_126': x_canonicalize_datetime_index__mutmut_126, 
    'x_canonicalize_datetime_index__mutmut_127': x_canonicalize_datetime_index__mutmut_127, 
    'x_canonicalize_datetime_index__mutmut_128': x_canonicalize_datetime_index__mutmut_128, 
    'x_canonicalize_datetime_index__mutmut_129': x_canonicalize_datetime_index__mutmut_129, 
    'x_canonicalize_datetime_index__mutmut_130': x_canonicalize_datetime_index__mutmut_130, 
    'x_canonicalize_datetime_index__mutmut_131': x_canonicalize_datetime_index__mutmut_131, 
    'x_canonicalize_datetime_index__mutmut_132': x_canonicalize_datetime_index__mutmut_132, 
    'x_canonicalize_datetime_index__mutmut_133': x_canonicalize_datetime_index__mutmut_133, 
    'x_canonicalize_datetime_index__mutmut_134': x_canonicalize_datetime_index__mutmut_134, 
    'x_canonicalize_datetime_index__mutmut_135': x_canonicalize_datetime_index__mutmut_135, 
    'x_canonicalize_datetime_index__mutmut_136': x_canonicalize_datetime_index__mutmut_136, 
    'x_canonicalize_datetime_index__mutmut_137': x_canonicalize_datetime_index__mutmut_137
}

def canonicalize_datetime_index(*args, **kwargs):
    result = _mutmut_trampoline(x_canonicalize_datetime_index__mutmut_orig, x_canonicalize_datetime_index__mutmut_mutants, args, kwargs)
    return result 

canonicalize_datetime_index.__signature__ = _mutmut_signature(x_canonicalize_datetime_index__mutmut_orig)
x_canonicalize_datetime_index__mutmut_orig.__name__ = 'x_canonicalize_datetime_index'
