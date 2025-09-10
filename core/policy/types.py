"""
Core policy types and data structures for hierarchical risk management.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime, timezone
import json

# Action types for policy decisions
Action = Literal["ALLOW", "CLIP", "DENY", "NOOP"]

@dataclass
class PolicyDecision:
    """Result of a policy evaluation with clear action and reasoning."""
    action: Action
    qty_delta: int = 0
    reason: str = ""
    layer: Optional[str] = None  # "per_trade" | "per_symbol" | "group" | "portfolio"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "action": self.action,
            "qty_delta": self.qty_delta,
            "reason": self.reason,
            "layer": self.layer,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"PolicyDecision(action={self.action}, qty_delta={self.qty_delta:+d}, reason='{self.reason}', layer={self.layer})"

@dataclass
class RebalanceTrigger:
    """Configuration for when to trigger rebalancing."""
    on_signal_change: bool = True
    on_cadence_tick: bool = True
    on_threshold_breach: bool = True
    signal_change_threshold: float = 0.05  # 5% material change
    cadence_minutes: int = 30

@dataclass
class PerTradeConfig:
    """Per-trade risk constraints."""
    max_notional: float = 1500.0
    lot_size: int = 5
    min_order_notional: float = 200.0

@dataclass
class PerSymbolConfig:
    """Per-symbol risk constraints."""
    default_cap: float = 15000.0
    overrides: Dict[str, float] = field(default_factory=dict)
    band_pct: float = 0.05  # 5% buffer zone
    rebalance_cadence: str = "30m"

@dataclass
class GroupConfig:
    """Group-level (sector/factor) risk constraints."""
    type: str = "sector"  # "sector" or "factor"
    map_file: str = "data/sectors/sector_map.parquet"
    cap_by_group: Dict[str, float] = field(default_factory=dict)

@dataclass
class PortfolioConfig:
    """Portfolio-level risk constraints."""
    gross_cap: float = 100000.0
    net_cap: float = 20000.0
    vol_target: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyConfig:
    """Complete policy configuration."""
    rebalance_triggers: RebalanceTrigger = field(default_factory=RebalanceTrigger)
    per_trade: PerTradeConfig = field(default_factory=PerTradeConfig)
    per_symbol: PerSymbolConfig = field(default_factory=PerSymbolConfig)
    groups: GroupConfig = field(default_factory=GroupConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    rounding: Dict[str, Any] = field(default_factory=dict)
    abstain_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PositionIntent:
    """Tracks committed position targets for signal change detection."""
    symbol: str
    target_notional: float
    timestamp: datetime
    signal_strength: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_notional": self.target_notional,
            "timestamp": self.timestamp.isoformat(),
            "signal_strength": self.signal_strength
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionIntent':
        return cls(
            symbol=data["symbol"],
            target_notional=data["target_notional"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            signal_strength=data["signal_strength"]
        )

@dataclass
class PolicyContext:
    """Context for policy evaluation including current state and configuration."""
    symbol: str
    price: float
    current_shares: int
    signal_target_notional: float
    config: PolicyConfig
    portfolio_state: Dict[str, Any] = field(default_factory=dict)
    group_state: Dict[str, Any] = field(default_factory=dict)
    position_intents: Dict[str, PositionIntent] = field(default_factory=dict)
    last_rebalance_time: Optional[datetime] = None
    
    def get_symbol_cap(self) -> float:
        """Get the cap for this symbol (override or default)."""
        return self.config.per_symbol.overrides.get(
            self.symbol, 
            self.config.per_symbol.default_cap
        )
    
    def get_group_cap(self, group: str) -> float:
        """Get the cap for this group."""
        return self.config.groups.cap_by_group.get(
            group,
            self.config.groups.cap_by_group.get("Default", float("inf"))
        )
    
    def should_rebalance(self) -> bool:
        """Determine if rebalancing should be triggered."""
        triggers = self.config.rebalance_triggers
        
        # Check signal change
        if triggers.on_signal_change:
            if self.symbol in self.position_intents:
                last_intent = self.position_intents[self.symbol]
                change_pct = abs(
                    (self.signal_target_notional - last_intent.target_notional) / 
                    max(last_intent.target_notional, 1.0)
                )
                if change_pct > triggers.signal_change_threshold:
                    return True
        
        # Check cadence
        if triggers.on_cadence_tick and self.last_rebalance_time:
            time_since = datetime.now(timezone.utc) - self.last_rebalance_time
            if time_since.total_seconds() > triggers.cadence_minutes * 60:
                return True
        
        # Check threshold breach (will be handled by individual guards)
        if triggers.on_threshold_breach:
            return True
        
        return False
