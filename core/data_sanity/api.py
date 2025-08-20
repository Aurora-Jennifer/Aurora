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
