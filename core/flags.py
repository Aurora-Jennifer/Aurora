"""
Centralized flag system for DataSanity rules.
Use environment variables to gate specific validation rules.
"""

import os


def enabled(flag: str, default: bool = True) -> bool:
    """
    Check if a flag is enabled.

    Args:
        flag: Flag name (e.g., 'DS_DISABLE_LOOKAHEAD')
        default: Default value if flag is not set

    Returns:
        True if flag is enabled, False otherwise
    """
    env_value = os.getenv(flag)
    if env_value is None:
        return default

    # Parse boolean values
    if env_value.lower() in ["true", "1", "yes", "on"]:
        return True
    elif env_value.lower() in ["false", "0", "no", "off"]:
        return False
    else:
        # If not a boolean, treat as enabled if set
        return True


def disabled(flag: str, default: bool = False) -> bool:
    """
    Check if a flag is disabled.

    Args:
        flag: Flag name (e.g., 'DS_DISABLE_LOOKAHEAD')
        default: Default value if flag is not set

    Returns:
        True if flag is disabled, False otherwise
    """
    return not enabled(flag, not default)


def get_flag_value(flag: str, default: str | None = None) -> str | None:
    """
    Get the raw value of a flag.

    Args:
        flag: Flag name
        default: Default value if flag is not set

    Returns:
        Flag value as string, or default
    """
    return os.getenv(flag, default)


# Common DataSanity flags
class DataSanityFlags:
    """Common DataSanity flag names."""

    # Lookahead detection
    DISABLE_LOOKAHEAD = "DS_DISABLE_LOOKAHEAD"

    # Strict mode overrides
    ALLOW_REPAIRS = "DS_ALLOW_REPAIRS"
    ALLOW_CLIP_PRICES = "DS_ALLOW_CLIP_PRICES"
    ALLOW_DROP_DUPES = "DS_ALLOW_DROP_DUPES"
    ALLOW_FFILL_NANS = "DS_ALLOW_FFILL_NANS"

    # Performance flags
    ENABLE_PERF_MODE = "DS_ENABLE_PERF_MODE"
    PERF_MODE = "DS_PERF_MODE"  # RELAXED or STRICT

    # Testing flags
    TEST_MODE = "DS_TEST_MODE"
    MOCK_NETWORK = "DS_MOCK_NETWORK"

    # Debug flags
    VERBOSE_LOGGING = "DS_VERBOSE_LOGGING"
    DEBUG_MODE = "DS_DEBUG_MODE"


# Convenience functions for common flags
def lookahead_enabled() -> bool:
    """Check if lookahead detection is enabled."""
    return enabled(DataSanityFlags.DISABLE_LOOKAHEAD, default=True)


def repairs_allowed() -> bool:
    """Check if repairs are allowed."""
    return enabled(DataSanityFlags.ALLOW_REPAIRS, default=True)


def price_clipping_allowed() -> bool:
    """Check if price clipping is allowed."""
    return enabled(DataSanityFlags.ALLOW_CLIP_PRICES, default=True)


def duplicate_dropping_allowed() -> bool:
    """Check if duplicate dropping is allowed."""
    return enabled(DataSanityFlags.ALLOW_DROP_DUPES, default=True)


def nan_filling_allowed() -> bool:
    """Check if NaN filling is allowed."""
    return enabled(DataSanityFlags.ALLOW_FFILL_NANS, default=True)


def perf_mode() -> str:
    """Get performance mode setting."""
    return get_flag_value(DataSanityFlags.PERF_MODE, "RELAXED")


def is_test_mode() -> bool:
    """Check if running in test mode."""
    return enabled(DataSanityFlags.TEST_MODE, default=False)


def is_debug_mode() -> bool:
    """Check if running in debug mode."""
    return enabled(DataSanityFlags.DEBUG_MODE, default=False)


def verbose_logging() -> bool:
    """Check if verbose logging is enabled."""
    return enabled(DataSanityFlags.VERBOSE_LOGGING, default=False)
