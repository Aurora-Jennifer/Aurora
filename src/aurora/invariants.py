"""
Runtime invariants and kill switches
"""
import os

from .config import Config


def enforce_runtime(cfg: Config) -> bool:
    """
    Enforce runtime invariants - hard fail if violated
    """
    assert cfg.runtime.single_decision_core, "Multiple decision paths forbidden - single_decision_core must be True"
    assert not cfg.runtime.allow_legacy_paths, "Legacy paths must be disabled - allow_legacy_paths must be False"
    
    # Orders disabled unless explicitly enabled
    if os.getenv("ENABLE_ORDERS", "false").lower() != "true":
        print("🚨 KILL SWITCH: Orders disabled (ENABLE_ORDERS != 'true')")
        return False
    
    return True


def enforce_runtime_invariants(cfg: Config) -> bool:
    """Legacy alias for enforce_runtime"""
    return enforce_runtime(cfg)


def validate_system_startup(cfg: Config) -> bool:
    """
    Validate system startup with code/config hashing
    """
    from .decision_core import get_code_hash, get_config_hash
    
    # Enforce runtime invariants
    if not enforce_runtime_invariants(cfg):
        return False
    
    # Get hashes for reproducibility
    code_hash = get_code_hash()
    config_hash = get_config_hash(cfg.__dict__)
    
    print("🚀 System Startup Validation")
    print(f"   Code Hash: {code_hash}")
    print(f"   Config Hash: {config_hash}")
    print(f"   Single Decision Core: {cfg.runtime.single_decision_core}")
    print(f"   Legacy Paths Allowed: {cfg.runtime.allow_legacy_paths}")
    
    return True
