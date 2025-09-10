"""
Policy orchestrator that coordinates hierarchical risk guards and makes final decisions.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from .types import PolicyDecision, PolicyContext, PositionIntent
from .guards import PerTradeGuard, PerSymbolGuard, GroupGuard, PortfolioGuard

logger = logging.getLogger(__name__)

class PolicyOrchestrator:
    """
    Orchestrates hierarchical risk policy evaluation.
    
    Evaluates policies in order: per-symbol → per-trade → group → portfolio
    Each layer can ALLOW, CLIP, or DENY the proposed action.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize policy orchestrator.
        
        Args:
            config: Policy configuration dictionary
        """
        self.config = self._load_config(config)
        self.per_trade_guard = PerTradeGuard()
        self.per_symbol_guard = PerSymbolGuard()
        self.group_guard = GroupGuard()
        self.portfolio_guard = PortfolioGuard()
        
        # Position intent tracking
        self.position_intents: Dict[str, PositionIntent] = {}
        
        logger.info("PolicyOrchestrator initialized with hierarchical risk controls")
    
    def _load_config(self, config: Dict[str, Any]) -> 'PolicyConfig':
        """Load configuration into PolicyConfig object."""
        from .types import PolicyConfig, RebalanceTrigger, PerTradeConfig, PerSymbolConfig, GroupConfig, PortfolioConfig
        
        # Extract nested configs
        risk_config = config.get("risk", {})
        policy_config = config.get("policy", {})
        
        return PolicyConfig(
            rebalance_triggers=RebalanceTrigger(
                on_signal_change=policy_config.get("rebalance_triggers", {}).get("on_signal_change", True),
                on_cadence_tick=policy_config.get("rebalance_triggers", {}).get("on_cadence_tick", True),
                on_threshold_breach=policy_config.get("rebalance_triggers", {}).get("on_threshold_breach", True),
                signal_change_threshold=policy_config.get("rebalance_triggers", {}).get("signal_change_threshold", 0.05),
                cadence_minutes=30  # Default 30 minutes
            ),
            per_trade=PerTradeConfig(
                max_notional=risk_config.get("per_trade", {}).get("max_notional", 1500.0),
                lot_size=risk_config.get("per_trade", {}).get("lot_size", 5),
                min_order_notional=risk_config.get("per_trade", {}).get("min_order_notional", 200.0)
            ),
            per_symbol=PerSymbolConfig(
                default_cap=risk_config.get("per_symbol", {}).get("default_cap", 15000.0),
                overrides=risk_config.get("per_symbol", {}).get("overrides", {}),
                band_pct=risk_config.get("per_symbol", {}).get("band_pct", 0.05),
                rebalance_cadence=risk_config.get("per_symbol", {}).get("rebalance_cadence", "30m")
            ),
            groups=GroupConfig(
                type=risk_config.get("groups", {}).get("type", "sector"),
                map_file=risk_config.get("groups", {}).get("map_file", "data/sectors/sector_map.parquet"),
                cap_by_group=risk_config.get("groups", {}).get("cap_by_group", {})
            ),
            portfolio=PortfolioConfig(
                gross_cap=risk_config.get("portfolio", {}).get("gross_cap", 100000.0),
                net_cap=risk_config.get("portfolio", {}).get("net_cap", 20000.0),
                vol_target=risk_config.get("portfolio", {}).get("vol_target", {})
            ),
            rounding=policy_config.get("rounding", {}),
            abstain_rules=policy_config.get("abstain_rules", {})
        )
    
    def evaluate_policy(
        self,
        symbol: str,
        price: float,
        current_shares: int,
        signal_target_notional: float,
        portfolio_state: Dict[str, Any],
        group_state: Dict[str, Any],
        last_rebalance_time: Optional[datetime] = None
    ) -> PolicyDecision:
        """
        Evaluate hierarchical risk policy for a trading decision.
        
        Args:
            symbol: Trading symbol
            price: Current market price
            current_shares: Current position in shares
            signal_target_notional: Target position value from signal
            portfolio_state: Current portfolio state
            group_state: Current group/sector state
            last_rebalance_time: Last rebalancing time
            
        Returns:
            PolicyDecision with final action and reasoning
        """
        # Create policy context
        ctx = PolicyContext(
            symbol=symbol,
            price=price,
            current_shares=current_shares,
            signal_target_notional=signal_target_notional,
            config=self.config,
            portfolio_state=portfolio_state,
            group_state=group_state,
            position_intents=self.position_intents,
            last_rebalance_time=last_rebalance_time
        )
        
        # Check if rebalancing should be triggered
        if not ctx.should_rebalance():
            return PolicyDecision(
                action="NOOP",
                reason="no_trigger",
                layer="orchestrator",
                metadata={
                    "symbol": symbol,
                    "signal_target": signal_target_notional,
                    "last_rebalance": last_rebalance_time.isoformat() if last_rebalance_time else None
                }
            )
        
        # Step 1: Per-symbol guard (determines if rebalancing needed and target)
        symbol_decision = self.per_symbol_guard.evaluate(ctx)
        
        if symbol_decision.action in ("NOOP", "DENY"):
            return symbol_decision
        
        # Step 2: Per-trade guard (enforces order size limits)
        trade_decision = self.per_trade_guard.evaluate(ctx, symbol_decision.qty_delta)
        
        if trade_decision.action == "DENY":
            return trade_decision
        
        qty_delta = trade_decision.qty_delta
        
        # Step 3: Group guard (enforces sector/factor limits)
        group_decision = self.group_guard.evaluate(ctx, qty_delta)
        
        if group_decision.action == "DENY":
            return group_decision
        
        qty_delta = group_decision.qty_delta
        
        # Step 4: Portfolio guard (enforces gross/net exposure limits)
        portfolio_decision = self.portfolio_guard.evaluate(ctx, qty_delta)
        
        # Log the decision chain
        self._log_decision_chain(symbol, [
            symbol_decision,
            trade_decision, 
            group_decision,
            portfolio_decision
        ])
        
        # Update position intent if action is taken
        if portfolio_decision.action in ("ALLOW", "CLIP") and portfolio_decision.qty_delta != 0:
            self._update_position_intent(symbol, signal_target_notional, ctx)
        
        return portfolio_decision
    
    def _log_decision_chain(self, symbol: str, decisions: list) -> None:
        """Log the complete decision chain for debugging."""
        logger.debug(f"Policy decision chain for {symbol}:")
        for i, decision in enumerate(decisions):
            logger.debug(f"  Step {i+1} ({decision.layer}): {decision.action} - {decision.reason}")
            if decision.metadata:
                logger.debug(f"    Metadata: {decision.metadata}")
    
    def _update_position_intent(self, symbol: str, target_notional: float, ctx: PolicyContext) -> None:
        """Update position intent tracking for signal change detection."""
        self.position_intents[symbol] = PositionIntent(
            symbol=symbol,
            target_notional=target_notional,
            timestamp=datetime.now(timezone.utc),
            signal_strength=target_notional / ctx.portfolio_state.get("total_value", 1.0)
        )
    
    def get_position_intent(self, symbol: str) -> Optional[PositionIntent]:
        """Get the last position intent for a symbol."""
        return self.position_intents.get(symbol)
    
    def clear_old_intents(self, max_age_hours: int = 24) -> None:
        """Clear position intents older than specified age."""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for symbol, intent in self.position_intents.items():
            if intent.timestamp.timestamp() < cutoff_time:
                to_remove.append(symbol)
        
        for symbol in to_remove:
            del self.position_intents[symbol]
        
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old position intents")
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of current policy configuration and state."""
        return {
            "config": {
                "per_trade": {
                    "max_notional": self.config.per_trade.max_notional,
                    "lot_size": self.config.per_trade.lot_size,
                    "min_order_notional": self.config.per_trade.min_order_notional
                },
                "per_symbol": {
                    "default_cap": self.config.per_symbol.default_cap,
                    "band_pct": self.config.per_symbol.band_pct,
                    "overrides_count": len(self.config.per_symbol.overrides)
                },
                "groups": {
                    "type": self.config.groups.type,
                    "cap_count": len(self.config.groups.cap_by_group)
                },
                "portfolio": {
                    "gross_cap": self.config.portfolio.gross_cap,
                    "net_cap": self.config.portfolio.net_cap
                }
            },
            "state": {
                "position_intents_count": len(self.position_intents),
                "active_symbols": list(self.position_intents.keys())
            }
        }
