"""
Aurora Trading System - Unified Decision Core
"""

from .decision_core import (
    BUY,
    HOLD,
    SELL,
    DecisionCfg,
    apply_decision,
    ensure_unified_decision_core,
    get_code_hash,
    get_config_hash,
    next_position,
    simulate_step,
    validate_decision_inputs,
    validate_system_startup,
)

__version__ = "0.0.0"
__all__ = [
    "BUY", "SELL", "HOLD",
    "DecisionCfg",
    "apply_decision",
    "next_position", 
    "simulate_step",
    "validate_decision_inputs",
    "ensure_unified_decision_core",
    "get_code_hash",
    "get_config_hash",
    "validate_system_startup"
]