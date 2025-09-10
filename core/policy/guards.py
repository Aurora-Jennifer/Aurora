"""
Hierarchical risk guards implementing per-trade, per-symbol, group, and portfolio controls.
"""

import logging
from typing import Dict, Any, Optional
from .types import PolicyDecision, PolicyContext, Action

logger = logging.getLogger(__name__)

class PerTradeGuard:
    """Per-trade risk guard: enforces order size limits and lot rounding."""
    
    def evaluate(self, ctx: PolicyContext, proposed_delta: int) -> PolicyDecision:
        """
        Evaluate per-trade constraints on a proposed order delta.
        
        Args:
            ctx: Policy evaluation context
            proposed_delta: Proposed order delta in shares
            
        Returns:
            PolicyDecision with ALLOW/CLIP/DENY action
        """
        if proposed_delta == 0:
            return PolicyDecision(
                action="NOOP",
                reason="zero_delta",
                layer="per_trade"
            )
        
        # Calculate order notional
        order_notional = abs(proposed_delta) * ctx.price
        
        # Check minimum order size
        min_notional = ctx.config.per_trade.min_order_notional
        if order_notional < min_notional:
            return PolicyDecision(
                action="DENY",
                reason="below_min_order",
                layer="per_trade",
                metadata={
                    "order_notional": order_notional,
                    "min_notional": min_notional
                }
            )
        
        # Check maximum order size
        max_notional = ctx.config.per_trade.max_notional
        if order_notional <= max_notional:
            # Within limits, apply lot rounding
            lot_size = ctx.config.per_trade.lot_size
            adjusted_delta = int(round(proposed_delta / lot_size)) * lot_size
            
            return PolicyDecision(
                action="ALLOW",
                qty_delta=adjusted_delta,
                reason="ok_per_trade",
                layer="per_trade",
                metadata={
                    "order_notional": abs(adjusted_delta) * ctx.price,
                    "lot_rounded": True
                }
            )
        
        # Clip to maximum notional
        lot_size = ctx.config.per_trade.lot_size
        max_shares = int(max_notional // ctx.price)
        clipped_shares = (max_shares // lot_size) * lot_size
        
        if clipped_shares <= 0:
            return PolicyDecision(
                action="DENY",
                reason="cannot_form_lot",
                layer="per_trade",
                metadata={
                    "max_notional": max_notional,
                    "price": ctx.price,
                    "max_shares": max_shares
                }
            )
        
        # Apply direction
        clipped_delta = clipped_shares if proposed_delta > 0 else -clipped_shares
        
        return PolicyDecision(
            action="CLIP",
            qty_delta=clipped_delta,
            reason="clip_per_trade",
            layer="per_trade",
            metadata={
                "original_delta": proposed_delta,
                "clipped_notional": abs(clipped_delta) * ctx.price,
                "clip_ratio": abs(clipped_delta) / abs(proposed_delta)
            }
        )

class PerSymbolGuard:
    """Per-symbol risk guard: enforces symbol caps and band-based rebalancing."""
    
    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        """
        Evaluate per-symbol constraints and determine if rebalancing is needed.
        
        Args:
            ctx: Policy evaluation context
            
        Returns:
            PolicyDecision with ALLOW/CLIP/NOOP action
        """
        # Get symbol cap
        cap_val = ctx.get_symbol_cap()
        
        # Clip target by per-symbol cap
        clipped_target_val = min(ctx.signal_target_notional, cap_val)
        
        # Calculate buffer zone
        band_pct = ctx.config.per_symbol.band_pct
        lower_bound = clipped_target_val * (1 - band_pct)
        upper_bound = clipped_target_val * (1 + band_pct)
        
        # Current position value
        current_val = ctx.current_shares * ctx.price
        
        # Check if within buffer zone
        if lower_bound <= current_val <= upper_bound:
            return PolicyDecision(
                action="NOOP",
                reason="within_band",
                layer="per_symbol",
                metadata={
                    "current_val": current_val,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "band_pct": band_pct
                }
            )
        
        # Determine target value (move to nearest band edge)
        if current_val < lower_bound:
            new_target_val = lower_bound
            action_reason = "buy_to_lower_bound"
        else:
            new_target_val = upper_bound
            action_reason = "sell_to_upper_bound"
        
        # Convert to shares with lot rounding
        lot_size = ctx.config.per_trade.lot_size
        target_shares = int(round(new_target_val / ctx.price / lot_size)) * lot_size
        delta = target_shares - ctx.current_shares
        
        if delta == 0:
            return PolicyDecision(
                action="NOOP",
                reason="rounding_noop",
                layer="per_symbol",
                metadata={
                    "target_shares": target_shares,
                    "current_shares": ctx.current_shares
                }
            )
        
        return PolicyDecision(
            action="ALLOW",
            qty_delta=delta,
            reason=action_reason,
            layer="per_symbol",
            metadata={
                "current_val": current_val,
                "target_val": new_target_val,
                "band_pct": band_pct,
                "cap_used": clipped_target_val / cap_val
            }
        )

class GroupGuard:
    """Group-level risk guard: enforces sector/factor exposure limits."""
    
    def evaluate(self, ctx: PolicyContext, proposed_delta: int) -> PolicyDecision:
        """
        Evaluate group-level constraints on a proposed order delta.
        
        Args:
            ctx: Policy evaluation context
            proposed_delta: Proposed order delta in shares
            
        Returns:
            PolicyDecision with ALLOW/CLIP/DENY action
        """
        # Get group for this symbol
        symbol_to_group = ctx.group_state.get("symbol_to_group", {})
        group = symbol_to_group.get(ctx.symbol, "Default")
        
        # Get current group exposure
        group_exposures = ctx.group_state.get("group_exposures", {})
        current_group_val = group_exposures.get(group, 0.0)
        
        # Calculate new group exposure
        order_notional = proposed_delta * ctx.price
        new_group_val = current_group_val + order_notional
        
        # Get group cap
        group_cap = ctx.get_group_cap(group)
        
        # Check if within group cap
        if abs(new_group_val) <= group_cap:
            return PolicyDecision(
                action="ALLOW",
                qty_delta=proposed_delta,
                reason="ok_group",
                layer="group",
                metadata={
                    "group": group,
                    "current_exposure": current_group_val,
                    "new_exposure": new_group_val,
                    "group_cap": group_cap
                }
            )
        
        # Clip to remaining headroom
        headroom = max(0.0, group_cap - abs(current_group_val))
        if headroom <= 0:
            return PolicyDecision(
                action="DENY",
                reason="group_cap_exceeded",
                layer="group",
                metadata={
                    "group": group,
                    "current_exposure": current_group_val,
                    "group_cap": group_cap,
                    "headroom": headroom
                }
            )
        
        # Calculate clipped order size
        clipped_notional = headroom
        clipped_shares = int(clipped_notional // ctx.price)
        
        # Apply lot rounding
        lot_size = ctx.config.per_trade.lot_size
        clipped_shares = (clipped_shares // lot_size) * lot_size
        
        if clipped_shares <= 0:
            return PolicyDecision(
                action="DENY",
                reason="group_cap_insufficient_for_lot",
                layer="group",
                metadata={
                    "group": group,
                    "headroom": headroom,
                    "clipped_shares": clipped_shares
                }
            )
        
        # Apply direction
        clipped_delta = clipped_shares if proposed_delta > 0 else -clipped_shares
        
        return PolicyDecision(
            action="CLIP",
            qty_delta=clipped_delta,
            reason="clip_group",
            layer="group",
            metadata={
                "group": group,
                "original_delta": proposed_delta,
                "clipped_delta": clipped_delta,
                "clip_ratio": abs(clipped_delta) / abs(proposed_delta),
                "headroom_used": abs(clipped_delta) * ctx.price
            }
        )

class PortfolioGuard:
    """Portfolio-level risk guard: enforces gross/net exposure and volatility targeting."""
    
    def evaluate(self, ctx: PolicyContext, proposed_delta: int) -> PolicyDecision:
        """
        Evaluate portfolio-level constraints on a proposed order delta.
        
        Args:
            ctx: Policy evaluation context
            proposed_delta: Proposed order delta in shares
            
        Returns:
            PolicyDecision with ALLOW/CLIP/DENY action
        """
        # Get portfolio state
        gross_exposure = ctx.portfolio_state.get("gross_exposure", 0.0)
        net_exposure = ctx.portfolio_state.get("net_exposure", 0.0)
        gross_cap = ctx.config.portfolio.gross_cap
        net_cap = ctx.config.portfolio.net_cap
        
        # Calculate new exposures
        order_notional = proposed_delta * ctx.price
        new_gross = gross_exposure + abs(order_notional)
        new_net = net_exposure + order_notional
        
        # Check gross and net caps
        if new_gross <= gross_cap and abs(new_net) <= net_cap:
            return PolicyDecision(
                action="ALLOW",
                qty_delta=proposed_delta,
                reason="ok_portfolio",
                layer="portfolio",
                metadata={
                    "current_gross": gross_exposure,
                    "new_gross": new_gross,
                    "current_net": net_exposure,
                    "new_net": new_net,
                    "gross_cap": gross_cap,
                    "net_cap": net_cap
                }
            )
        
        # Determine which cap is binding
        gross_headroom = max(0.0, gross_cap - gross_exposure)
        net_headroom = max(0.0, net_cap - abs(net_exposure))
        
        # Use the more restrictive headroom
        max_dollars = min(gross_headroom, net_headroom)
        
        if max_dollars <= 0:
            return PolicyDecision(
                action="DENY",
                reason="portfolio_cap_exceeded",
                layer="portfolio",
                metadata={
                    "gross_exposure": gross_exposure,
                    "net_exposure": net_exposure,
                    "gross_cap": gross_cap,
                    "net_cap": net_cap,
                    "gross_headroom": gross_headroom,
                    "net_headroom": net_headroom
                }
            )
        
        # Calculate clipped order size
        clipped_shares = int(max_dollars // ctx.price)
        
        # Apply lot rounding
        lot_size = ctx.config.per_trade.lot_size
        clipped_shares = (clipped_shares // lot_size) * lot_size
        
        if clipped_shares <= 0:
            return PolicyDecision(
                action="DENY",
                reason="portfolio_cap_insufficient_for_lot",
                layer="portfolio",
                metadata={
                    "max_dollars": max_dollars,
                    "clipped_shares": clipped_shares
                }
            )
        
        # Apply direction
        clipped_delta = clipped_shares if proposed_delta > 0 else -clipped_shares
        
        return PolicyDecision(
            action="CLIP",
            qty_delta=clipped_delta,
            reason="clip_portfolio",
            layer="portfolio",
            metadata={
                "original_delta": proposed_delta,
                "clipped_delta": clipped_delta,
                "clip_ratio": abs(clipped_delta) / abs(proposed_delta),
                "binding_cap": "gross" if gross_headroom < net_headroom else "net",
                "headroom_used": abs(clipped_delta) * ctx.price
            }
        )
