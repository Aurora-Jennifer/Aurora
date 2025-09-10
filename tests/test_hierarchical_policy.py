"""
Comprehensive unit tests for hierarchical risk policy system.
"""

import pytest
import yaml
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from core.policy.types import PolicyDecision, PolicyContext, PositionIntent, PolicyConfig
from core.policy.guards import PerTradeGuard, PerSymbolGuard, GroupGuard, PortfolioGuard
from core.policy.orchestrator import PolicyOrchestrator
from core.policy.config import PolicyConfigLoader

class TestPerTradeGuard:
    """Test per-trade risk guard."""
    
    def setup_method(self):
        self.guard = PerTradeGuard()
        self.config = PolicyConfig()
        self.ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            config=self.config
        )
    
    def test_zero_delta_noop(self):
        """Test that zero delta returns NOOP."""
        decision = self.guard.evaluate(self.ctx, 0)
        assert decision.action == "NOOP"
        assert decision.reason == "zero_delta"
        assert decision.layer == "per_trade"
    
    def test_below_min_order_deny(self):
        """Test that orders below minimum are denied."""
        # Order of 1 share = $150, below $200 minimum
        decision = self.guard.evaluate(self.ctx, 1)
        assert decision.action == "DENY"
        assert decision.reason == "below_min_order"
        assert decision.layer == "per_trade"
    
    def test_within_limits_allow(self):
        """Test that orders within limits are allowed with lot rounding."""
        # Order of 10 shares = $1500, within $1500 limit
        decision = self.guard.evaluate(self.ctx, 10)
        assert decision.action == "ALLOW"
        assert decision.reason == "ok_per_trade"
        assert decision.qty_delta == 10  # Already multiple of 5
        assert decision.layer == "per_trade"
    
    def test_lot_rounding(self):
        """Test that orders are rounded to lot size."""
        # Order of 7 shares, should round to 5 or 10
        decision = self.guard.evaluate(self.ctx, 7)
        assert decision.action == "ALLOW"
        assert decision.qty_delta in [5, 10]  # Rounded to nearest lot
        assert decision.qty_delta % 5 == 0
    
    def test_above_max_clip(self):
        """Test that orders above maximum are clipped."""
        # Order of 20 shares = $3000, above $1500 limit
        decision = self.guard.evaluate(self.ctx, 20)
        assert decision.action == "CLIP"
        assert decision.reason == "clip_per_trade"
        assert decision.qty_delta == 10  # Clipped to $1500 / $150 = 10 shares
        assert decision.layer == "per_trade"
    
    def test_cannot_form_lot_deny(self):
        """Test that orders that can't form a lot are denied."""
        # Very high price that makes max notional < 1 share
        ctx = PolicyContext(
            symbol="EXPENSIVE",
            price=2000.0,  # $2000 per share
            current_shares=0,
            signal_target_notional=1000.0,
            config=self.config
        )
        decision = self.guard.evaluate(ctx, 1)
        assert decision.action == "DENY"
        assert decision.reason == "cannot_form_lot"

class TestPerSymbolGuard:
    """Test per-symbol risk guard."""
    
    def setup_method(self):
        self.guard = PerSymbolGuard()
        self.config = PolicyConfig()
        self.config.per_symbol.band_pct = 0.05  # 5% band
        self.config.per_symbol.default_cap = 15000.0
        self.config.per_trade.lot_size = 5
    
    def test_within_band_noop(self):
        """Test that positions within band return NOOP."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,  # $15,000 position
            signal_target_notional=15000.0,  # Target $15,000
            config=self.config
        )
        decision = self.guard.evaluate(ctx)
        assert decision.action == "NOOP"
        assert decision.reason == "within_band"
        assert decision.layer == "per_symbol"
    
    def test_below_band_buy(self):
        """Test that positions below band trigger buy to lower bound."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=90,  # $13,500 position (below 5% band)
            signal_target_notional=15000.0,  # Target $15,000
            config=self.config
        )
        decision = self.guard.evaluate(ctx)
        assert decision.action == "ALLOW"
        assert decision.reason == "buy_to_lower_bound"
        assert decision.qty_delta > 0  # Should buy shares
        assert decision.layer == "per_symbol"
    
    def test_above_band_sell(self):
        """Test that positions above band trigger sell to upper bound."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=110,  # $16,500 position (above 5% band)
            signal_target_notional=15000.0,  # Target $15,000
            config=self.config
        )
        decision = self.guard.evaluate(ctx)
        assert decision.action == "ALLOW"
        assert decision.reason == "sell_to_upper_bound"
        assert decision.qty_delta < 0  # Should sell shares
        assert decision.layer == "per_symbol"
    
    def test_cap_override(self):
        """Test that symbol cap overrides are respected."""
        self.config.per_symbol.overrides = {"AAPL": 20000.0}
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=25000.0,  # Above override cap
            config=self.config
        )
        decision = self.guard.evaluate(ctx)
        # Should be clipped to $20,000 cap, then band logic applied
        assert decision.action in ("ALLOW", "NOOP")

class TestGroupGuard:
    """Test group-level risk guard."""
    
    def setup_method(self):
        self.guard = GroupGuard()
        self.config = PolicyConfig()
        self.config.groups.cap_by_group = {"Technology": 50000.0, "Default": 30000.0}
        self.config.per_trade.lot_size = 5
    
    def test_within_group_cap_allow(self):
        """Test that orders within group cap are allowed."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            config=self.config,
            group_state={
                "symbol_to_group": {"AAPL": "Technology"},
                "group_exposures": {"Technology": 30000.0}  # $30k current exposure
            }
        )
        decision = self.guard.evaluate(ctx, 50)  # $7,500 order
        assert decision.action == "ALLOW"
        assert decision.reason == "ok_group"
        assert decision.qty_delta == 50
    
    def test_exceed_group_cap_clip(self):
        """Test that orders exceeding group cap are clipped."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            config=self.config,
            group_state={
                "symbol_to_group": {"AAPL": "Technology"},
                "group_exposures": {"Technology": 45000.0}  # $45k current exposure
            }
        )
        decision = self.guard.evaluate(ctx, 100)  # $15,000 order would exceed $50k cap
        assert decision.action == "CLIP"
        assert decision.reason == "clip_group"
        assert decision.qty_delta < 100  # Should be clipped
        assert decision.layer == "group"
    
    def test_group_cap_exceeded_deny(self):
        """Test that orders are denied when group cap is already exceeded."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            config=self.config,
            group_state={
                "symbol_to_group": {"AAPL": "Technology"},
                "group_exposures": {"Technology": 50000.0}  # Already at cap
            }
        )
        decision = self.guard.evaluate(ctx, 10)  # Any positive order
        assert decision.action == "DENY"
        assert decision.reason == "group_cap_exceeded"
        assert decision.layer == "group"

class TestPortfolioGuard:
    """Test portfolio-level risk guard."""
    
    def setup_method(self):
        self.guard = PortfolioGuard()
        self.config = PolicyConfig()
        self.config.portfolio.gross_cap = 100000.0
        self.config.portfolio.net_cap = 20000.0
        self.config.per_trade.lot_size = 5
    
    def test_within_portfolio_caps_allow(self):
        """Test that orders within portfolio caps are allowed."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            config=self.config,
            portfolio_state={
                "gross_exposure": 50000.0,  # $50k gross
                "net_exposure": 10000.0     # $10k net
            }
        )
        decision = self.guard.evaluate(ctx, 50)  # $7,500 order
        assert decision.action == "ALLOW"
        assert decision.reason == "ok_portfolio"
        assert decision.qty_delta == 50
    
    def test_exceed_gross_cap_clip(self):
        """Test that orders exceeding gross cap are clipped."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            config=self.config,
            portfolio_state={
                "gross_exposure": 95000.0,  # $95k gross, close to $100k cap
                "net_exposure": 10000.0
            }
        )
        decision = self.guard.evaluate(ctx, 100)  # $15,000 order would exceed gross cap
        assert decision.action == "CLIP"
        assert decision.reason == "clip_portfolio"
        assert decision.qty_delta < 100  # Should be clipped
        assert decision.layer == "portfolio"
    
    def test_exceed_net_cap_clip(self):
        """Test that orders exceeding net cap are clipped."""
        ctx = PolicyContext(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            config=self.config,
            portfolio_state={
                "gross_exposure": 50000.0,
                "net_exposure": 18000.0     # $18k net, close to $20k cap
            }
        )
        decision = self.guard.evaluate(ctx, 50)  # $7,500 order would exceed net cap
        assert decision.action == "CLIP"
        assert decision.reason == "clip_portfolio"
        assert decision.qty_delta < 50  # Should be clipped
        assert decision.layer == "portfolio"

class TestPolicyOrchestrator:
    """Test policy orchestrator integration."""
    
    def setup_method(self):
        self.config = {
            "risk": {
                "per_trade": {
                    "max_notional": 1500.0,
                    "lot_size": 5,
                    "min_order_notional": 200.0
                },
                "per_symbol": {
                    "default_cap": 15000.0,
                    "overrides": {"AAPL": 20000.0},
                    "band_pct": 0.05,
                    "rebalance_cadence": "30m"
                },
                "groups": {
                    "type": "sector",
                    "cap_by_group": {"Technology": 50000.0, "Default": 30000.0}
                },
                "portfolio": {
                    "gross_cap": 100000.0,
                    "net_cap": 20000.0
                }
            },
            "policy": {
                "rebalance_triggers": {
                    "on_signal_change": True,
                    "on_cadence_tick": True,
                    "on_threshold_breach": True,
                    "signal_change_threshold": 0.05
                }
            }
        }
        self.orchestrator = PolicyOrchestrator(self.config)
    
    def test_no_trigger_noop(self):
        """Test that no action is taken when no trigger conditions are met."""
        # No position intent, no last rebalance time
        decision = self.orchestrator.evaluate_policy(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            portfolio_state={"gross_exposure": 50000.0, "net_exposure": 10000.0},
            group_state={"symbol_to_group": {"AAPL": "Technology"}, "group_exposures": {"Technology": 30000.0}},
            last_rebalance_time=None
        )
        # Should trigger on first evaluation (no previous intent)
        assert decision.action in ("ALLOW", "CLIP", "NOOP")
    
    def test_signal_change_trigger(self):
        """Test that signal changes trigger rebalancing."""
        # Add position intent
        self.orchestrator.position_intents["AAPL"] = PositionIntent(
            symbol="AAPL",
            target_notional=10000.0,  # Previous target
            timestamp=datetime.now(timezone.utc),
            signal_strength=0.1
        )
        
        decision = self.orchestrator.evaluate_policy(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=20000.0,  # New target (100% change)
            portfolio_state={"gross_exposure": 50000.0, "net_exposure": 10000.0},
            group_state={"symbol_to_group": {"AAPL": "Technology"}, "group_exposures": {"Technology": 30000.0}},
            last_rebalance_time=datetime.now(timezone.utc) - timedelta(minutes=10)
        )
        assert decision.action in ("ALLOW", "CLIP", "NOOP")
    
    def test_cadence_trigger(self):
        """Test that cadence triggers rebalancing."""
        decision = self.orchestrator.evaluate_policy(
            symbol="AAPL",
            price=150.0,
            current_shares=100,
            signal_target_notional=15000.0,
            portfolio_state={"gross_exposure": 50000.0, "net_exposure": 10000.0},
            group_state={"symbol_to_group": {"AAPL": "Technology"}, "group_exposures": {"Technology": 30000.0}},
            last_rebalance_time=datetime.now(timezone.utc) - timedelta(minutes=35)  # 35 minutes ago
        )
        assert decision.action in ("ALLOW", "CLIP", "NOOP")
    
    def test_hierarchical_clipping(self):
        """Test that hierarchical clipping works correctly."""
        # Set up scenario where multiple layers will clip
        decision = self.orchestrator.evaluate_policy(
            symbol="AAPL",
            price=150.0,
            current_shares=50,  # $7,500 current position
            signal_target_notional=25000.0,  # High target
            portfolio_state={"gross_exposure": 95000.0, "net_exposure": 18000.0},  # Near caps
            group_state={"symbol_to_group": {"AAPL": "Technology"}, "group_exposures": {"Technology": 45000.0}},  # Near group cap
            last_rebalance_time=None
        )
        # Should be clipped by multiple layers
        assert decision.action in ("CLIP", "DENY")
        assert decision.layer in ("group", "portfolio")

class TestPolicyConfig:
    """Test policy configuration loading and validation."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = PolicyConfigLoader.create_default_config()
        assert "risk" in config
        assert "policy" in config
        assert "per_trade" in config["risk"]
        assert "per_symbol" in config["risk"]
        assert "groups" in config["risk"]
        assert "portfolio" in config["risk"]
    
    def test_validate_config(self):
        """Test configuration validation."""
        config = PolicyConfigLoader.create_default_config()
        # Should not raise any exceptions
        PolicyConfigLoader.validate_config(config)
    
    def test_validate_invalid_config(self):
        """Test validation of invalid configuration."""
        invalid_config = {
            "risk": {
                "per_trade": {
                    "max_notional": -1000,  # Invalid negative value
                    "lot_size": 5,
                    "min_order_notional": 200
                }
            }
        }
        
        with pytest.raises(ValueError, match="per_trade.max_notional must be positive"):
            PolicyConfigLoader.validate_config(invalid_config)

if __name__ == "__main__":
    pytest.main([__file__])
