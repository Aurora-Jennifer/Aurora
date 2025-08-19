#!/usr/bin/env python3
"""
Centralized configuration loading and merging utility.
Handles base config, profile overrides, asset-specific overrides, and CLI overrides.
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Handles loading and merging of configuration files."""

    def __init__(self, base_config_path: str = "config/base.json"):
        self.base_config_path = base_config_path
        self.base_config = self._load_json(base_config_path)

    def _load_json(self, path: str) -> dict[str, Any]:
        """Load JSON file safely."""
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {path}: {e}")
            return {}

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def load_config(
        self,
        profile: str | None = None,
        asset: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Load configuration with optional profile and asset overrides.

        Args:
            profile: Risk profile name (e.g., 'risk_low', 'risk_balanced', 'risk_strict')
            asset: Asset-specific config to apply
            cli_overrides: CLI-provided overrides

        Returns:
            Merged configuration dictionary
        """
        config = self.base_config.copy()

        # Apply profile override
        if profile and profile in config.get("profiles", {}):
            profile_config = config["profiles"][profile]
            config = self._deep_merge(config, profile_config)
            logger.info(f"Applied profile override: {profile}")

        # Apply asset-specific override
        if asset:
            asset_class = self._get_asset_class(asset)
            if asset_class and asset_class in config.get("asset_classes", {}):
                asset_config = config["asset_classes"][asset_class]
                config = self._deep_merge(config, asset_config)
                logger.info(f"Applied asset class override: {asset_class} for {asset}")

        # Apply CLI overrides (highest priority)
        if cli_overrides:
            config = self._deep_merge(config, cli_overrides)
            logger.info("Applied CLI overrides")

        return config

    def _get_asset_class(self, symbol: str) -> str | None:
        """Determine asset class from symbol."""
        symbol_upper = symbol.upper()

        if any(crypto in symbol_upper for crypto in ["BTC", "ETH"]):
            return "crypto"
        elif any(etf in symbol_upper for etf in ["SPY", "QQQ"]):
            return "etf"
        else:
            return "equity"

    def get_symbols(self, config: dict[str, Any]) -> list:
        """Get list of symbols from config."""
        return config.get("symbols", [config.get("default_symbol", "SPY")])

    def get_walkforward_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get walkforward parameters from config."""
        return config.get("walkforward", {})

    def get_risk_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get risk parameters from config."""
        return config.get("risk_params", {})

    def get_composer_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get composer configuration from config."""
        return config.get("composer", {})

    def get_optimization_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Get optimization configuration from config."""
        return config.get("optimization", {})

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration structure and values."""
        required_sections = ["symbols", "risk_params", "walkforward"]

        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required config section: {section}")
                return False

        # Validate risk parameters
        risk_params = config["risk_params"]
        if not (0 < risk_params.get("max_drawdown_pct", 0) <= 100):
            logger.error("max_drawdown_pct must be between 0 and 100")
            return False

        if not (0 < risk_params.get("max_position_size", 0) <= 1):
            logger.error("max_position_size must be between 0 and 1")
            return False

        # Validate walkforward parameters
        wf_params = config["walkforward"]
        if wf_params.get("train_len", 0) <= 0:
            logger.error("train_len must be positive")
            return False

        if wf_params.get("test_len", 0) <= 0:
            logger.error("test_len must be positive")
            return False

        return True


def load_config(
    profile: str | None = None,
    asset: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    base_config_path: str = "config/base.json",
) -> dict[str, Any]:
    """
    Convenience function to load configuration.

    Args:
        profile: Risk profile name
        asset: Asset symbol for asset-specific config
        cli_overrides: CLI-provided overrides
        base_config_path: Path to base config file

    Returns:
        Merged configuration dictionary
    """
    loader = ConfigLoader(base_config_path)
    config = loader.load_config(profile, asset, cli_overrides)

    if not loader.validate_config(config):
        logger.warning("Configuration validation failed, using base config")
        return loader.base_config

    return config


def save_config(config: dict[str, Any], path: str) -> None:
    """Save configuration to file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {path}: {e}")


def create_profile_config(
    profile_name: str,
    overrides: dict[str, Any],
    base_config_path: str = "config/base.json",
) -> str:
    """
    Create a new profile configuration file.

    Args:
        profile_name: Name of the profile
        overrides: Configuration overrides for this profile
        base_config_path: Path to base config file

    Returns:
        Path to created profile config file
    """
    loader = ConfigLoader(base_config_path)
    config = loader.load_config(profile=profile_name, cli_overrides=overrides)

    profile_path = f"config/{profile_name}.json"
    save_config(config, profile_path)

    return profile_path


# Example usage and testing
if __name__ == "__main__":
    # Test config loading
    loader = ConfigLoader()

    # Test base config
    base_config = loader.load_config()
    print("Base config symbols:", loader.get_symbols(base_config))

    # Test with profile
    risk_low_config = loader.load_config(profile="risk_low")
    print("Risk low max drawdown:", risk_low_config["risk_params"]["max_drawdown_pct"])

    # Test with asset
    btc_config = loader.load_config(asset="BTC-USD")
    print("BTC asset class:", loader._get_asset_class("BTC-USD"))

    # Test with CLI overrides
    cli_overrides = {"symbols": ["AAPL", "MSFT"], "initial_capital": 200000}
    override_config = loader.load_config(cli_overrides=cli_overrides)
    print("Override config symbols:", override_config["symbols"])
    print("Override config capital:", override_config["initial_capital"])
