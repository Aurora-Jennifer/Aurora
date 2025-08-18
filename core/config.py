"""
Configuration loader with deep merging support.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

logger = logging.getLogger(__name__)


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with overlay taking precedence.

    Args:
        base: Base configuration dictionary
        overlay: Overlay configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_paths: List[Union[str, Path]]) -> Dict[str, Any]:
    """
    Load and merge configuration from multiple files.

    Args:
        config_paths: List of configuration file paths (base first, then overlays)

    Returns:
        Merged configuration dictionary
    """
    if not config_paths:
        raise ValueError("At least one configuration file path must be provided")

    config = {}

    for path in config_paths:
        path = Path(path)
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            continue

        try:
            with open(path) as f:
                file_config = yaml.safe_load(f)

            if file_config is None:
                logger.warning(f"Empty configuration file: {path}")
                continue

            config = deep_merge(config, file_config)
            logger.info(f"Loaded configuration from: {path}")

        except Exception as e:
            logger.error(f"Error loading configuration from {path}: {e}")
            raise

    if not config:
        raise ValueError("No valid configuration files found")

    return config


def get_cfg(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation path.

    Args:
        config: Configuration dictionary
        path: Dot-separated path to configuration value
        default: Default value if path not found

    Returns:
        Configuration value or default
    """
    keys = path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_sections = ["engine", "walkforward", "data", "risk", "composer"]

    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False

    # Validate composer configuration
    composer = config.get("composer", {})
    if composer.get("use_composer", False):
        if "min_history_bars" not in composer:
            logger.error("Composer enabled but min_history_bars not specified")
            return False

    # Validate walkforward configuration
    walkforward = config.get("walkforward", {})
    if "fold_length" not in walkforward or "step_size" not in walkforward:
        logger.error("Walkforward configuration missing fold_length or step_size")
        return False

    return True
