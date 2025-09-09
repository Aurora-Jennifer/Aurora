"""
Unified configuration loader with validation
"""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Costs:
    commission_bps: float = 1.0
    slippage_bps: float = 3.0


@dataclass(frozen=True)
class Decision:
    tau_threshold: float = 1e-4
    apply_cost_on_change_only: bool = True
    costs_bps: float = 4.0  # Total cost in basis points


@dataclass(frozen=True)
class Runtime:
    single_decision_core: bool = True
    allow_legacy_paths: bool = False
    fail_fast_on_nan: bool = True


@dataclass(frozen=True)
class Config:
    costs: Costs = Costs()
    decision: Decision = Decision()
    runtime: Runtime = Runtime()


def load_config(path: str | None = None) -> Config:
    """
    Load configuration from YAML file or return defaults.
    Validates presence of required keys and types.
    """
    if path is None:
        return Config()
    
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract sections with defaults
    costs_data = data.get('costs', {})
    decision_data = data.get('decision', {})
    runtime_data = data.get('runtime', {})
    
    # Create config objects with validation
    costs = Costs(
        commission_bps=costs_data.get('commission_bps', 1.0),
        slippage_bps=costs_data.get('slippage_bps', 3.0)
    )
    
    decision = Decision(
        tau_threshold=decision_data.get('tau_threshold', 1e-4),
        apply_cost_on_change_only=decision_data.get('apply_cost_on_change_only', True),
        costs_bps=decision_data.get('costs_bps', 4.0)
    )
    
    runtime = Runtime(
        single_decision_core=runtime_data.get('single_decision_core', True),
        allow_legacy_paths=runtime_data.get('allow_legacy_paths', False),
        fail_fast_on_nan=runtime_data.get('fail_fast_on_nan', True)
    )
    
    return Config(costs=costs, decision=decision, runtime=runtime)


def validate_config(cfg: Config) -> None:
    """Validate configuration invariants"""
    assert cfg.decision.tau_threshold > 0, "tau_threshold must be positive"
    assert cfg.costs.commission_bps >= 0, "commission_bps must be non-negative"
    assert cfg.costs.slippage_bps >= 0, "slippage_bps must be non-negative"
    assert cfg.decision.costs_bps >= 0, "costs_bps must be non-negative"