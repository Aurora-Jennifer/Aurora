# core/data_sanity/api.py
# NOTE: Keep this file very light; import only what you must re-export.

# Import from the main module (the original core/data_sanity.py content)
from .clean import coerce_ohlcv_numeric, repair_nonfinite_ohlc

# Import from the new modular functions
from .columnmap import map_ohlcv
from .datetime import canonicalize_datetime_index
from .group import enforce_groupwise_time_order
from .main import (
    DataSanityError,
    DataSanityGuard,
    DataSanityValidator,
    DataSanityWrapper,
    ValidationResult,
    assert_validated,
    # Note: SanityProfile, estring are not defined in main.py
    attach_guard,
    get_data_sanity_wrapper,
    get_guard,
    validate_market_data,
)
from .metrics import bump, export_metrics, get_metrics, reset_metrics

# Import telemetry and metrics
from .telemetry import emit_validation_telemetry, get_telemetry_stats

# Add any other modular functions as they're created
# from .invariants import assert_ohlc_invariants
# from .lookahead import detect_lookahead
# from .decorator import data_sanity_enforced

# Engine switch facade
def validate_and_repair_with_engine_switch(
    data, symbol: str = "UNKNOWN", profile: str = "strict", run_id: str = None
):
    """
    Validate and repair with engine selection based on config.

    This is the main entry point that switches between v1 and v2 engines
    based on the datasanity.engine configuration.

    Args:
        data: DataFrame to validate
        symbol: Symbol name for logging
        profile: Validation profile to use
        run_id: Optional run identifier for telemetry

    Returns:
        Tuple of (cleaned DataFrame, ValidationResult)
    """
    from .config import get_cfg

    # Get engine configuration
    engine = get_cfg("datasanity.engine", "v1")

    # Create validator with appropriate engine
    if engine == "v2":
        # For now, v2 is the same as v1 - this is where future improvements go
        validator = DataSanityValidator(profile=profile)
        result = validator.validate_and_repair(data, symbol)
    else:
        # Default to v1 engine
        validator = DataSanityValidator(profile=profile)
        result = validator.validate_and_repair(data, symbol)

    # Emit telemetry
    emit_validation_telemetry(symbol, profile, result, run_id)

    return result

__all__ = [
    # Main module exports
    "DataSanityError", "DataSanityValidator", "ValidationResult",
    "DataSanityGuard", "DataSanityWrapper", "validate_market_data",
    "get_data_sanity_wrapper", "attach_guard", "get_guard", "assert_validated",
    # Modular function exports
    "map_ohlcv", "enforce_groupwise_time_order", "repair_nonfinite_ohlc", "coerce_ohlcv_numeric", "canonicalize_datetime_index",
    # Telemetry and metrics exports
    "emit_validation_telemetry", "get_telemetry_stats", "bump", "get_metrics", "reset_metrics", "export_metrics",
    # Engine switch facade
    "validate_and_repair_with_engine_switch",
]
