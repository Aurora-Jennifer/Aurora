"""
DataSanity Rule Registry
Central registry for all validation rules.
"""

from typing import Dict, Callable, Any
from .rules.prices import create_price_positivity_rule
from .rules.ohlc import create_ohlc_consistency_rule
from .rules.finite import create_finite_numbers_rule
from .rules.lookahead import create_lookahead_contamination_rule
from .rules.volume import create_volume_rule


# Central registry of all available rules
RULES: Dict[str, Callable[[dict], Any]] = {
    "price_positivity": create_price_positivity_rule,
    "ohlc_consistency": create_ohlc_consistency_rule,
    "finite_numbers": create_finite_numbers_rule,
    "lookahead_contamination": create_lookahead_contamination_rule,
    "volume": create_volume_rule,
}


def get_rule(rule_name: str, config: dict):
    """Get a rule instance by name and config."""
    if rule_name not in RULES:
        raise ValueError(f"Unknown rule: {rule_name}. Available: {list(RULES.keys())}")
    
    return RULES[rule_name](config)


def list_rules() -> list[str]:
    """List all available rules."""
    return list(RULES.keys())


def register_rule(name: str, factory_func: Callable[[dict], Any]):
    """Register a new rule."""
    RULES[name] = factory_func
