"""
Configuration management for DataSanity.

Provides centralized access to DataSanity configuration settings
from the main config files.
"""

from pathlib import Path
from typing import Any

import yaml


def get_cfg(path: str, default: Any = None) -> Any:
    """
    Get configuration value by dot-separated path.

    Args:
        path: Dot-separated path to config value (e.g., "datasanity.engine")
        default: Default value if path not found

    Returns:
        Configuration value or default
    """
    try:
        # Load base config
        config_path = Path("config/base.yaml")
        if not config_path.exists():
            return default

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Navigate to the requested path
        keys = path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value
    except Exception:
        return default


def get_datasanity_config() -> dict[str, Any]:
    """
    Get the complete DataSanity configuration section.

    Returns:
        Dictionary with DataSanity configuration
    """
    return get_cfg("datasanity", {})


def get_profile_config(profile: str) -> dict[str, Any]:
    """
    Get configuration for a specific DataSanity profile.

    Args:
        profile: Profile name (e.g., "strict", "walkforward_ci")

    Returns:
        Dictionary with profile configuration
    """
    # Load profile-specific config
    profile_path = Path("config/data_sanity.yaml")
    if profile_path.exists():
        with open(profile_path) as f:
            profiles = yaml.safe_load(f)
            if profile in profiles:
                return profiles[profile]

    # Fall back to default profile config
    return {
        "allow_repairs": True,
        "fail_on_lookahead_flag": False,
        "fail_on_nonfinite": True,
        "fail_on_duplicates": True,
        "fail_on_non_monotonic": True,
        "fail_on_naive_timezone": True,
        "fail_on_negative_prices": True,
        "fail_on_extreme_prices": True,
        "fail_on_zero_volume": False,
        "fail_on_dtype_drift": True,
    }
