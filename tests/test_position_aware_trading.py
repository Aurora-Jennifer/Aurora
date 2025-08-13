"""Unit tests for position-aware trading and PnL calculation fixes."""

from datetime import date
from unittest.mock import Mock, patch

import pytest

from enhanced_paper_trading import EnhancedPaperTradingSystem


class TestPositionAwareTrading:
    """Test position-aware trading logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "symbols": ["SPY", "AAPL"],
            "initial_capital": 100000,
            "risk_params": {
                "max_weight_per_symbol": 0.25,
                "max_drawdown": 0.15,
                "max_daily_loss": 0.02,
            },
            "execution_params": {"max_slippage_bps": 10},
        }

        # Mock the system components and config loading
        with patch("enhanced_paper_trading.RegimeDetector"), patch(
            "enhanced_paper_trading.FeatureReweighter"
        ), patch("enhanced_paper_trading.AdaptiveFeatureEngine"), patch(
            "enhanced_paper_trading.TradingLogger"
        ), patch(
            "enhanced_paper_trading.DiscordNotifier"
        ), patch.object(
            EnhancedPaperTradingSystem, "load_config", return_value=self.config
        ):
            self.trading_system = EnhancedPaperTradingSystem("dummy_config.json")
            self.trading_system.positions = {}
            self.trading_system.capital = 100000
            self.trading_system._previous_total_value = 100000

    def test_no_sell_from_flat_when_short_off(self):
        """Test that selling is prevented when no position exists."""
        # Setup: No position in SPY
        symbol = "SPY"
        self.trading_system.positions[symbol] = 0.0

        # Mock signal that suggests selling
        signals = {"regime_ensemble": -0.5}  # Strong sell signal

        # Mock regime params
        regime_params = Mock()
        regime_params.confidence_threshold = 0.3
        regime_params.position_sizing_multiplier = 1.0
        regime_params.regime_name = "trend"

        # Mock current price
        with patch.object(
            self.trading_system, "_get_current_price", return_value=500.0
        ):
            # Execute trade
            self.trading_system._execute_trades(
                symbol, signals, date.today(), regime_params
            )

            # Verify no position was created (should remain 0.0)
            assert self.trading_system.positions[symbol] == 0.0

    def test_reduce_only_clamp(self):
        """Test that position size is clamped to maximum allowed."""
        # Setup: Small position in SPY
        symbol = "SPY"
        self.trading_system.positions[symbol] = 0.1  # 10% position

        # Mock signal that suggests very large position
        signals = {"regime_ensemble": 0.8}  # Very strong buy signal

        # Mock regime params
        regime_params = Mock()
        regime_params.confidence_threshold = 0.3
        regime_params.position_sizing_multiplier = 1.0
        regime_params.regime_name = "trend"

        # Mock current price
        with patch.object(
            self.trading_system, "_get_current_price", return_value=500.0
        ):
            # Execute trade
            self.trading_system._execute_trades(
                symbol, signals, date.today(), regime_params
            )

            # Verify position is clamped to max_weight_per_symbol (0.25)
            assert self.trading_system.positions[symbol] == 0.25

    def test_daily_pnl_matches_equity_delta(self):
        """Test that daily PnL calculation matches equity delta."""
        # Setup: Position in SPY
        symbol = "SPY"
        self.trading_system.positions[symbol] = 0.2  # 20% position
        self.trading_system.capital = 100000
        self.trading_system._previous_total_value = 100000

        # Mock price change
        initial_price = 500.0
        new_price = 510.0  # 2% increase

        with patch.object(
            self.trading_system, "_get_current_price", return_value=new_price
        ):
            # Update performance tracking
            self.trading_system._update_performance_tracking(date.today())

            # Calculate expected return
            # Total value: 100000 + (0.2 * 100000) = 120,000
            # Previous total value: 100000
            # Expected return: (120000 - 100000) / 100000 = 0.2 (20%)

            # Verify daily return is calculated correctly
            assert len(self.trading_system.daily_returns) == 1
            daily_return = self.trading_system.daily_returns[0]["return"]

            # Should be 20% (position size)
            assert abs(daily_return - 0.2) < 0.001

    def test_position_limits_enforced(self):
        """Test that position limits are properly enforced."""
        # Setup: Already at max position
        symbol = "SPY"
        self.trading_system.positions[symbol] = 0.25  # At max weight

        # Mock signal that suggests increasing position
        signals = {"regime_ensemble": 0.6}  # Strong buy signal

        # Mock regime params
        regime_params = Mock()
        regime_params.confidence_threshold = 0.3
        regime_params.position_sizing_multiplier = 1.0
        regime_params.regime_name = "trend"

        # Mock current price
        with patch.object(
            self.trading_system, "_get_current_price", return_value=500.0
        ):
            # Execute trade
            self.trading_system._execute_trades(
                symbol, signals, date.today(), regime_params
            )

            # Verify position remains at max (no increase)
            assert self.trading_system.positions[symbol] == 0.25

    def test_fees_deducted_from_capital(self):
        """Test that transaction fees are properly deducted."""
        # Setup: No position
        symbol = "SPY"
        self.trading_system.positions[symbol] = 0.0
        self.trading_system.capital = 100000

        # Mock signal for buying
        signals = {"regime_ensemble": 0.4}  # Buy signal

        # Mock regime params
        regime_params = Mock()
        regime_params.confidence_threshold = 0.3
        regime_params.position_sizing_multiplier = 1.0
        regime_params.regime_name = "trend"

        # Mock current price
        with patch.object(
            self.trading_system, "_get_current_price", return_value=500.0
        ):
            # Execute trade
            self.trading_system._execute_trades(
                symbol, signals, date.today(), regime_params
            )

            # Verify fees were deducted (10 bps = 0.1%)
            # Trade value: 0.4 * 100000 = 40,000
            # Fees: 40,000 * 0.001 = 40
            # Expected capital: 100,000 - 40 = 99,960
            assert self.trading_system.capital < 100000

    def test_minimum_trade_size_enforced(self):
        """Test that minimum trade size is enforced."""
        # Setup: Small position
        symbol = "SPY"
        self.trading_system.positions[symbol] = 0.001  # Very small position

        # Mock signal for tiny change
        signals = {"regime_ensemble": 0.002}  # Very small signal

        # Mock regime params
        regime_params = Mock()
        regime_params.confidence_threshold = 0.3
        regime_params.position_sizing_multiplier = 1.0
        regime_params.regime_name = "trend"

        # Mock current price
        with patch.object(
            self.trading_system, "_get_current_price", return_value=500.0
        ):
            # Execute trade
            self.trading_system._execute_trades(
                symbol, signals, date.today(), regime_params
            )

            # Verify position didn't change (trade too small)
            assert self.trading_system.positions[symbol] == 0.001

    def test_proper_position_calculation(self):
        """Test that position calculations are mathematically correct."""
        # Setup: No position
        symbol = "SPY"
        self.trading_system.positions[symbol] = 0.0
        self.trading_system.capital = 100000

        # Mock signal
        signals = {"regime_ensemble": 0.3}  # 30% position signal

        # Mock regime params
        regime_params = Mock()
        regime_params.confidence_threshold = 0.3
        regime_params.position_sizing_multiplier = 1.0
        regime_params.regime_name = "trend"

        # Mock current price
        with patch.object(
            self.trading_system, "_get_current_price", return_value=500.0
        ):
            # Execute trade
            self.trading_system._execute_trades(
                symbol, signals, date.today(), regime_params
            )

            # Verify position is clamped to max_weight_per_symbol (0.25)
            # The signal suggests 0.3 but max is 0.25
            assert self.trading_system.positions[symbol] == 0.25

            # Verify trade value calculation
            # Expected trade value: 0.25 * 100000 = 25,000
            # Trade size: 25,000 / 500 = 50 shares
            # This should be reflected in the position


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])
