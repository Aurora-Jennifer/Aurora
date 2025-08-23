# Error codes for consistent error messages across DataSanity
NONFINITE = "Non-finite values"
INVALID_DTYPE = "Non-numeric prices"
MISSING_COLS = "Missing required columns"
INDEX_NON_MONOTONIC = "Index is not monotonic"
DUPLICATE_TSTAMPS = "Duplicate timestamps not allowed"
TZ_NAIVE = "Naive timezone not allowed"
TZ_NON_UTC = "Non-UTC timezone not allowed"
NO_VALID_DT_INDEX = "No valid datetime index found"
PRICES_GT = "Prices >"
NEGATIVE_PRICES = "Negative prices"
OHLC_INVARIANT = "OHLC invariant violation"
LOOKAHEAD = "Lookahead contamination"
FUTURE_DATA = "Future data present"
from collections.abc import Callable
from typing import Annotated

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
