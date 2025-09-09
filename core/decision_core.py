"""
Decision Core - Unified decision logic for consistent evaluation

This module ensures both the main evaluator and falsification harness
use identical decision logic, preventing the "always-HOLD" bug.
"""

from dataclasses import dataclass
from typing import Any
import torch

# Action constants
BUY, SELL, HOLD = 0, 1, 2
ACTION_TO_POS = {BUY: +1, SELL: -1, HOLD: None}  # None = keep prev position

@dataclass
class DecisionCfg:
    """Decision configuration for consistent gating"""
    tau: float           # threshold in the same units used for gating
    temperature: float   # 1.0 = no scaling
    gate_on: str         # "adv", "prob_gap", or "edge_bp"
    cost_bps: float      # for cost-aware veto; 0 if not used
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tau": self.tau,
            "temperature": self.temperature,
            "gate_on": self.gate_on,
            "cost_bps": self.cost_bps
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DecisionCfg':
        """Create from dictionary"""
        return cls(
            tau=data["tau"],
            temperature=data["temperature"],
            gate_on=data["gate_on"],
            cost_bps=data["cost_bps"]
        )


def _temper(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Apply temperature scaling to logits"""
    return logits / max(T, 1e-8)


def decide(logits: torch.Tensor, advantage: torch.Tensor, cfg: DecisionCfg) -> int:
    """
    Return BUY/SELL/HOLD with consistent gating.
    
    Args:
        logits: Raw model logits [batch_size, num_actions]
        advantage: Advantage values [batch_size, num_actions] 
        cfg: Decision configuration
        
    Returns:
        Action index (0=BUY, 1=SELL, 2=HOLD)
    """
    # Ensure tensors are on CPU and 1D for single decisions
    if logits.dim() > 1:
        logits = logits.squeeze()
    if advantage.dim() > 1:
        advantage = advantage.squeeze()
    
    # 1) Apply temperature scaling
    logits = _temper(logits, cfg.temperature)
    probs = torch.softmax(logits, dim=-1)
    a_hat = int(probs.argmax().item())
    
    # 2) Compute gate metric
    if cfg.gate_on == "adv":
        metr = float(advantage[a_hat])
    elif cfg.gate_on == "prob_gap":
        top, second = torch.topk(probs, 2).values.tolist()
        metr = top - second
    elif cfg.gate_on == "edge_bp":
        metr = float(advantage[a_hat])  # must be in bp if you choose this
    else:
        raise ValueError(f"Unknown gate_on={cfg.gate_on}")
    
    # 3) Cost-aware / tau gating BEFORE position update
    if abs(metr) < cfg.tau or (cfg.gate_on == "edge_bp" and metr <= cfg.cost_bps):
        return HOLD
    return a_hat


def next_position(prev_pos: int, action: int) -> int:
    """
    Calculate next position based on action.
    
    Args:
        prev_pos: Previous position (-1, 0, 1)
        action: Action taken (BUY=0, SELL=1, HOLD=2)
        
    Returns:
        New position (-1, 0, 1)
    """
    if action == HOLD:
        return prev_pos
    if action == BUY:
        return 1
    if action == SELL:
        return -1
    raise ValueError(f"Invalid action: {action}")


def simulate_step(prev_pos: int, action: int, price: float, cost_bps: float) -> tuple[int, float]:
    """
    Simulate one trading step with position and cost calculation.
    
    Args:
        prev_pos: Previous position
        action: Action taken
        price: Current price (for cost calculation)
        cost_bps: Cost in basis points
        
    Returns:
        (new_position, cost_incurred)
    """
    new_pos = next_position(prev_pos, action)
    traded = (new_pos != prev_pos)
    # Cost is proportional to the absolute position change
    position_change = abs(new_pos - prev_pos)
    cost = position_change * (cost_bps / 10000.0)
    return new_pos, float(cost)


def validate_decision_inputs(logits: torch.Tensor, advantage: torch.Tensor, cfg: DecisionCfg) -> None:
    """
    Validate inputs to decision function.
    
    Raises:
        AssertionError: If inputs are invalid
    """
    # Check logits
    assert torch.isfinite(logits).all(), "Non-finite logits in decision"
    assert logits.max() - logits.min() > 1e-8, "Flat logits in decision"
    
    # Check advantage
    assert torch.isfinite(advantage).all(), "Non-finite advantage in decision"
    
    # Check config
    assert cfg.tau > 0, f"Invalid tau: {cfg.tau}"
    assert cfg.temperature > 0, f"Invalid temperature: {cfg.temperature}"
    assert cfg.gate_on in ["adv", "prob_gap", "edge_bp"], f"Invalid gate_on: {cfg.gate_on}"


def create_default_decision_cfg() -> DecisionCfg:
    """Create default decision configuration"""
    return DecisionCfg(
        tau=0.0001,  # Conservative default
        temperature=1.0,
        gate_on="adv",
        cost_bps=4.0
    )


def print_decision_legend():
    """Print action legend for debugging"""
    print(f"Decision Legend: BUY={BUY}, SELL={SELL}, HOLD={HOLD}")
    print(f"Position Mapping: {ACTION_TO_POS}")


def ensure_unified_decision_core(cfg: dict[str, Any]) -> bool:
    """
    Ensure unified decision core is the only active decision path.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If multiple decision paths are allowed
    """
    runtime = cfg.get('runtime', {})
    
    assert runtime.get('single_decision_core', False), "Multiple decision paths forbidden - single_decision_core must be True"
    assert not runtime.get('allow_legacy_paths', True), "Legacy paths must be disabled - allow_legacy_paths must be False"
    
    return True


def apply_decision(edge: float, tau: float, prev_pos: int, costs: dict[str, float], fail_fast: bool = True) -> tuple[int, int, float]:
    """
    Apply decision logic with runtime guards.
    
    Args:
        edge: Advantage/edge value
        tau: Threshold for action
        prev_pos: Previous position
        costs: Cost configuration
        fail_fast: Whether to fail fast on NaN/Inf
        
    Returns:
        (action, new_position, cost_applied)
    """
    import math
    
    # Guard against NaN/Inf
    if math.isnan(edge) or math.isinf(edge):
        if fail_fast:
            raise ValueError(f"NaN/Inf edge detected: {edge}")
        return HOLD, prev_pos, 0.0
    
    # Decision logic
    action = HOLD
    if abs(edge) >= tau:
        action = BUY if edge > 0 else SELL
    
    # Position update
    new_pos = next_position(prev_pos, action)
    
    # Cost calculation
    cost = 0.0
    if new_pos != prev_pos:
        commission_bps = costs.get('commission_bps', 1.0)
        slippage_bps = costs.get('slippage_bps', 3.0)
        cost = (commission_bps + slippage_bps) / 10000.0
    
    return action, new_pos, cost


def get_code_hash() -> str:
    """Get current git commit hash"""
    import subprocess
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_config_hash(cfg: dict[str, Any]) -> str:
    """Get SHA256 hash of configuration"""
    import hashlib
    import json
    
    # Sort keys for consistent hashing
    cfg_str = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(cfg_str.encode()).hexdigest()


def validate_system_startup(cfg: dict[str, Any]) -> bool:
    """
    Validate system startup invariants.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        AssertionError: If invariants violated
    """
    # Ensure unified decision core
    ensure_unified_decision_core(cfg)
    
    # Validate configuration
    assert 'decision' in cfg, "Decision configuration missing"
    assert 'runtime' in cfg, "Runtime configuration missing"
    
    # Validate decision parameters
    decision = cfg['decision']
    assert 'tau_threshold' in decision, "tau_threshold missing"
    assert decision['tau_threshold'] > 0, "tau_threshold must be positive"
    
    # Validate runtime parameters
    runtime = cfg['runtime']
    assert runtime.get('fail_fast_on_nan', True), "fail_fast_on_nan must be True"
    
    # Log system state
    code_hash = get_code_hash()
    config_hash = get_config_hash(cfg)
    
    print("🚀 System Startup Validation")
    print(f"   Code Hash: {code_hash}")
    print(f"   Config Hash: {config_hash}")
    print(f"   Single Decision Core: {runtime.get('single_decision_core', False)}")
    print(f"   Legacy Paths Allowed: {runtime.get('allow_legacy_paths', True)}")
    
    return True
