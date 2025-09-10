"""
Integration layer for hierarchical risk policy system with existing execution engine.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ..policy.orchestrator import PolicyOrchestrator
from ..policy.types import PolicyDecision
from ..policy.config import load_policy_config

logger = logging.getLogger(__name__)

class HierarchicalPolicyIntegration:
    """
    Integrates hierarchical risk policy with existing execution engine.
    
    This class acts as a bridge between the new policy system and the existing
    OrderManager, PortfolioManager, and ExecutionEngine components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize hierarchical policy integration.
        
        Args:
            config_path: Optional path to policy configuration file
        """
        # Load policy configuration
        self.policy_config = load_policy_config(config_path)
        
        # Initialize policy orchestrator
        self.orchestrator = PolicyOrchestrator(self.policy_config)
        
        # Integration state
        self.last_rebalance_times: Dict[str, datetime] = {}
        self.portfolio_state_cache: Dict[str, Any] = {}
        self.group_state_cache: Dict[str, Any] = {}
        
        logger.info("HierarchicalPolicyIntegration initialized")
    
    def evaluate_trading_decision(
        self,
        symbol: str,
        price: float,
        current_shares: int,
        signal_target_notional: float,
        portfolio_manager,
        order_manager
    ) -> PolicyDecision:
        """
        Evaluate a trading decision using hierarchical risk policy.
        
        Args:
            symbol: Trading symbol
            price: Current market price
            current_shares: Current position in shares
            signal_target_notional: Target position value from signal
            portfolio_manager: Portfolio manager instance
            order_manager: Order manager instance
            
        Returns:
            PolicyDecision with final action and reasoning
        """
        try:
            # Get current portfolio state
            portfolio_state = self._get_portfolio_state(portfolio_manager)
            
            # Get current group state
            group_state = self._get_group_state(symbol, portfolio_manager)
            
            # Get last rebalance time for this symbol
            last_rebalance_time = self.last_rebalance_times.get(symbol)
            
            # Evaluate policy
            decision = self.orchestrator.evaluate_policy(
                symbol=symbol,
                price=price,
                current_shares=current_shares,
                signal_target_notional=signal_target_notional,
                portfolio_state=portfolio_state,
                group_state=group_state,
                last_rebalance_time=last_rebalance_time
            )
            
            # Update last rebalance time if action was taken
            if decision.action in ("ALLOW", "CLIP") and decision.qty_delta != 0:
                self.last_rebalance_times[symbol] = datetime.now(timezone.utc)
            
            # Log decision
            self._log_decision(symbol, decision, price, current_shares, signal_target_notional)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error evaluating policy for {symbol}: {e}")
            # Return safe default decision
            return PolicyDecision(
                action="DENY",
                reason="policy_evaluation_error",
                layer="integration",
                metadata={"error": str(e)}
            )
    
    def _get_portfolio_state(self, portfolio_manager) -> Dict[str, Any]:
        """Get current portfolio state for policy evaluation."""
        try:
            # Get portfolio metrics
            metrics = portfolio_manager.calculate_portfolio_metrics()
            
            # Calculate gross and net exposure
            positions = portfolio_manager.get_all_positions()
            gross_exposure = sum(abs(pos.market_value) for pos in positions.values())
            net_exposure = sum(pos.market_value for pos in positions.values())
            
            return {
                "total_value": metrics.total_value,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "cash": metrics.cash,
                "positions_count": len(positions)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get portfolio state: {e}")
            return {
                "total_value": 100000.0,  # Default values
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "cash": 100000.0,
                "positions_count": 0
            }
    
    def _get_group_state(self, symbol: str, portfolio_manager) -> Dict[str, Any]:
        """Get current group/sector state for policy evaluation."""
        try:
            # Load sector mapping from config
            sector_mapping = self.policy_config.get("sector_mapping", {})
            
            # Get current positions
            positions = portfolio_manager.get_all_positions()
            
            # Calculate group exposures
            symbol_to_group = {}
            group_exposures = {}
            
            for pos_symbol, position in positions.items():
                group = sector_mapping.get(pos_symbol, "Default")
                symbol_to_group[pos_symbol] = group
                
                if group not in group_exposures:
                    group_exposures[group] = 0.0
                group_exposures[group] += position.market_value
            
            return {
                "symbol_to_group": symbol_to_group,
                "group_exposures": group_exposures
            }
            
        except Exception as e:
            logger.warning(f"Failed to get group state: {e}")
            return {
                "symbol_to_group": {symbol: "Default"},
                "group_exposures": {"Default": 0.0}
            }
    
    def _log_decision(
        self,
        symbol: str,
        decision: PolicyDecision,
        price: float,
        current_shares: int,
        signal_target_notional: float
    ) -> None:
        """Log policy decision with structured format."""
        log_data = {
            "timestamp": decision.timestamp.isoformat(),
            "symbol": symbol,
            "price": price,
            "current_shares": current_shares,
            "signal_target_notional": signal_target_notional,
            "action": decision.action,
            "qty_delta": decision.qty_delta,
            "reason": decision.reason,
            "layer": decision.layer,
            "metadata": decision.metadata
        }
        
        if decision.action == "ALLOW":
            logger.info(f"POLICY_ALLOW {symbol}: {decision.qty_delta:+d} shares, reason={decision.reason}")
        elif decision.action == "CLIP":
            logger.info(f"POLICY_CLIP {symbol}: {decision.qty_delta:+d} shares, reason={decision.reason}")
        elif decision.action == "DENY":
            logger.info(f"POLICY_DENY {symbol}: reason={decision.reason}")
        elif decision.action == "NOOP":
            logger.debug(f"POLICY_NOOP {symbol}: reason={decision.reason}")
        
        # Log structured data for telemetry
        if self.policy_config.get("telemetry", {}).get("structured_logging", False):
            logger.info(f"POLICY_DECISION: {log_data}")
    
    def get_policy_metrics(self) -> Dict[str, Any]:
        """Get policy performance metrics."""
        return {
            "orchestrator_summary": self.orchestrator.get_policy_summary(),
            "last_rebalance_times": {
                symbol: time.isoformat() 
                for symbol, time in self.last_rebalance_times.items()
            },
            "position_intents_count": len(self.orchestrator.position_intents),
            "config_summary": {
                "per_trade_max_notional": self.policy_config["risk"]["per_trade"]["max_notional"],
                "per_symbol_default_cap": self.policy_config["risk"]["per_symbol"]["default_cap"],
                "band_pct": self.policy_config["risk"]["per_symbol"]["band_pct"],
                "portfolio_gross_cap": self.policy_config["risk"]["portfolio"]["gross_cap"],
                "portfolio_net_cap": self.policy_config["risk"]["portfolio"]["net_cap"]
            }
        }
    
    def clear_old_intents(self, max_age_hours: int = 24) -> None:
        """Clear old position intents."""
        self.orchestrator.clear_old_intents(max_age_hours)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update policy configuration (for dynamic reconfiguration)."""
        try:
            # Validate new configuration
            from ..policy.config import PolicyConfigLoader
            PolicyConfigLoader.validate_config(new_config)
            
            # Update configuration
            self.policy_config = new_config
            
            # Reinitialize orchestrator with new config
            self.orchestrator = PolicyOrchestrator(new_config)
            
            logger.info("Policy configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update policy configuration: {e}")
            raise
    
    def enable_shadow_mode(self) -> None:
        """Enable shadow mode for testing (evaluate but don't act)."""
        self.policy_config["integration"]["order_manager"]["shadow_mode"] = True
        logger.info("Policy shadow mode enabled")
    
    def disable_shadow_mode(self) -> None:
        """Disable shadow mode."""
        self.policy_config["integration"]["order_manager"]["shadow_mode"] = False
        logger.info("Policy shadow mode disabled")
