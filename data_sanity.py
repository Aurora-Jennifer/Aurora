"""
Data Sanity - Public API shim.
Re-exports from the core.data_sanity package.
"""

# Import from the core.data_sanity package
from core.data_sanity import (
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
    # Also export the new modular functions
    map_ohlcv,
    enforce_groupwise_time_order,
    repair_nonfinite_ohlc,
)
