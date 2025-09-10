"""
Configuration loading and validation for hierarchical risk policy.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .types import PolicyConfig

logger = logging.getLogger(__name__)

class PolicyConfigLoader:
    """Loads and validates hierarchical risk policy configuration."""
    
    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, Any]:
        """
        Load policy configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Policy config file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded policy configuration from {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in policy config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load policy config: {e}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate policy configuration for required fields and reasonable values.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ["risk", "policy"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate risk section
        risk_config = config["risk"]
        required_risk_sections = ["per_trade", "per_symbol", "groups", "portfolio"]
        for section in required_risk_sections:
            if section not in risk_config:
                raise ValueError(f"Missing required risk section: {section}")
        
        # Validate per_trade config
        per_trade = risk_config["per_trade"]
        if per_trade.get("max_notional", 0) <= 0:
            raise ValueError("per_trade.max_notional must be positive")
        if per_trade.get("lot_size", 0) <= 0:
            raise ValueError("per_trade.lot_size must be positive")
        if per_trade.get("min_order_notional", 0) <= 0:
            raise ValueError("per_trade.min_order_notional must be positive")
        
        # Validate per_symbol config
        per_symbol = risk_config["per_symbol"]
        if per_symbol.get("default_cap", 0) <= 0:
            raise ValueError("per_symbol.default_cap must be positive")
        if not 0 < per_symbol.get("band_pct", 0) < 1:
            raise ValueError("per_symbol.band_pct must be between 0 and 1")
        
        # Validate groups config
        groups = risk_config["groups"]
        if groups.get("type") not in ["sector", "factor"]:
            raise ValueError("groups.type must be 'sector' or 'factor'")
        
        # Validate portfolio config
        portfolio = risk_config["portfolio"]
        if portfolio.get("gross_cap", 0) <= 0:
            raise ValueError("portfolio.gross_cap must be positive")
        if portfolio.get("net_cap", 0) <= 0:
            raise ValueError("portfolio.net_cap must be positive")
        
        # Validate policy section
        policy_config = config["policy"]
        rebalance_triggers = policy_config.get("rebalance_triggers", {})
        if not isinstance(rebalance_triggers.get("signal_change_threshold", 0.05), (int, float)):
            raise ValueError("rebalance_triggers.signal_change_threshold must be numeric")
        
        logger.info("Policy configuration validation passed")
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """
        Create a default policy configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "risk": {
                "per_trade": {
                    "max_notional": 1500,
                    "lot_size": 5,
                    "min_order_notional": 200
                },
                "per_symbol": {
                    "default_cap": 15000,
                    "overrides": {},
                    "band_pct": 0.05,
                    "rebalance_cadence": "30m"
                },
                "groups": {
                    "type": "sector",
                    "map_file": "data/sectors/sector_map.parquet",
                    "cap_by_group": {
                        "Technology": 60000,
                        "Financial": 50000,
                        "Default": 40000
                    }
                },
                "portfolio": {
                    "gross_cap": 100000,
                    "net_cap": 20000,
                    "vol_target": {
                        "enabled": True,
                        "lookback_days": 20,
                        "target_daily_vol": 0.01,
                        "clamp": [0.5, 1.5]
                    }
                }
            },
            "policy": {
                "rebalance_triggers": {
                    "on_signal_change": True,
                    "on_cadence_tick": True,
                    "on_threshold_breach": True,
                    "signal_change_threshold": 0.05
                },
                "rounding": {
                    "shares_lot": 5,
                    "cash_buffer_pct": 0.01
                },
                "abstain_rules": {
                    "expected_ret_minus_cost_bps": 0,
                    "uncertainty_z_max": 1.0,
                    "min_confidence_threshold": 0.1
                }
            },
            "position_intent": {
                "enabled": True,
                "store_file": "state/position_intents.json",
                "max_age_hours": 24,
                "material_change_threshold": 0.05
            },
            "telemetry": {
                "enabled": True,
                "log_level": "INFO",
                "structured_logging": True,
                "metrics_interval": 300
            }
        }
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved policy configuration to {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save policy config: {e}")

def load_policy_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load policy configuration from file or create default.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path:
        try:
            config = PolicyConfigLoader.load_from_file(config_path)
            PolicyConfigLoader.validate_config(config)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    
    # Return default configuration
    config = PolicyConfigLoader.create_default_config()
    PolicyConfigLoader.validate_config(config)
    return config
