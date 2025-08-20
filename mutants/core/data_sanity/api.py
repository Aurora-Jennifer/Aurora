# core/data_sanity/api.py
# NOTE: Keep this file very light; import only what you must re-export.

# Import from the main module (the original core/data_sanity.py content)
from .main import (
    DataSanityError,
    DataSanityValidator, 
    ValidationResult,
    DataSanityGuard,
    DataSanityWrapper,
    validate_market_data,
    get_data_sanity_wrapper,
    attach_guard,
    get_guard,
    assert_validated,
    # Note: SanityProfile, estring are not defined in main.py
)

# Import from the new modular functions
from .columnmap import map_ohlcv
from .group import enforce_groupwise_time_order
from .clean import repair_nonfinite_ohlc, coerce_ohlcv_numeric
from .datetime import canonicalize_datetime_index

# Add any other modular functions as they're created
# from .invariants import assert_ohlc_invariants
# from .lookahead import detect_lookahead
# from .decorator import data_sanity_enforced

__all__ = [
    # Main module exports
    "DataSanityError", "DataSanityValidator", "ValidationResult",
    "DataSanityGuard", "DataSanityWrapper", "validate_market_data",
    "get_data_sanity_wrapper", "attach_guard", "get_guard", "assert_validated",
    # Modular function exports
    "map_ohlcv", "enforce_groupwise_time_order", "repair_nonfinite_ohlc", "coerce_ohlcv_numeric", "canonicalize_datetime_index",
]
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
