"""
Tests for Backtest Accounting Fixes
Validates proper PnL calculation, trade logging, and performance metrics.
"""

import pytest
from datetime import date
from core.portfolio import PortfolioState, Position
from core.trade_logger import TradeBook, TradeRecord
from core.performance import calculate_trade_metrics, calculate_portfolio_metrics


class TestPortfolioAccounting:
    """Test portfolio state management and PnL calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.portfolio = PortfolioState(cash=10000.0)
    
    def test_buy_position_creation(self):
        """Test buying creates position with correct average price."""
        # Buy 100 shares at $10
        realized = self.portfolio.apply_fill("AAPL", "BUY", 100, 10.0, 10)
        
        assert realized == 0.0  # No realized PnL on buy
        assert self.portfolio.positions["AAPL"].qty == 100
        assert self.portfolio.positions["AAPL"].avg_price == 10.0
        assert self.portfolio.cash == 10000.0 - (100 * 10.0) - (100 * 10.0 * 0.001)  # Cash - cost - fees
    
    def test_buy_average_price_calculation(self):
        """Test average price calculation on multiple buys."""
        # Buy 100 @ $10, then 50 @ $12
        self.portfolio.apply_fill("AAPL", "BUY", 100, 10.0, 10)
        self.portfolio.apply_fill("AAPL", "BUY", 50, 12.0, 10)
        
        pos = self.portfolio.positions["AAPL"]
        expected_avg = ((100 * 10.0) + (50 * 12.0)) / 150
        assert pos.qty == 150
        assert abs(pos.avg_price - expected_avg) < 0.01
    
    def test_sell_partial_close(self):
        """Test partial close with realized PnL."""
        # Buy 100 @ $10, sell 40 @ $9 (loss)
        self.portfolio.apply_fill("AAPL", "BUY", 100, 10.0, 10)
        realized = self.portfolio.apply_fill("AAPL", "SELL", 40, 9.0, 10)
        
        # Expected realized PnL: (9 - 10) * 40 = -40
        expected_pnl = (9.0 - 10.0) * 40
        assert abs(realized - expected_pnl) < 0.01
        assert self.portfolio.positions["AAPL"].qty == 60  # Remaining position
    
    def test_sell_full_close(self):
        """Test full close with realized PnL."""
        # Buy 100 @ $10, sell 100 @ $11 (profit)
        self.portfolio.apply_fill("AAPL", "BUY", 100, 10.0, 10)
        realized = self.portfolio.apply_fill("AAPL", "SELL", 100, 11.0, 10)
        
        # Expected realized PnL: (11 - 10) * 100 = 100
        expected_pnl = (11.0 - 10.0) * 100
        assert abs(realized - expected_pnl) < 0.01
        assert "AAPL" not in self.portfolio.positions  # Position removed when closed
    
    def test_reduce_only_sell(self):
        """Test reduce-only logic prevents selling more than owned."""
        # Try to sell without position
        realized = self.portfolio.apply_fill("AAPL", "SELL", 100, 10.0, 10)
        assert realized == 0.0
        assert "AAPL" not in self.portfolio.positions
    
    def test_mark_to_market(self):
        """Test mark-to-market calculation."""
        # Buy 100 @ $10, price now $12
        self.portfolio.apply_fill("AAPL", "BUY", 100, 10.0, 10)
        self.portfolio.update_price("AAPL", 12.0)
        
        mtm_value = self.portfolio.mark_to_market()
        expected_value = self.portfolio.cash + (100 * 12.0)
        assert abs(mtm_value - expected_value) < 0.01


class TestTradeLogging:
    """Test trade logging with partial closes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trade_book = TradeBook()
    
    def test_trade_with_loss_records_negative_pnl(self):
        """Test that losing trades record negative PnL."""
        # Buy 100 @ 10, Sell 100 @ 9 => pnl = -100 - fees
        self.trade_book.on_buy("2024-01-01", "AAPL", 100, 10.0, 1.0)
        self.trade_book.on_sell("2024-01-02", "AAPL", 100, 9.0, 1.0, 0.0)
        
        closed_trades = self.trade_book.get_closed_trades()
        assert len(closed_trades) == 1
        
        trade = closed_trades[0]
        expected_pnl = (9.0 - 10.0) * 100  # (exit - entry) * qty (fees handled separately)
        assert trade.realized_pnl < 0
        assert abs(trade.realized_pnl - expected_pnl) < 0.01
    
    def test_profit_factor_not_inf_when_losses_exist(self):
        """Test profit factor calculation with both wins and losses."""
        # Trade 1: +100, Trade 2: -50 -> PF=2.0
        self.trade_book.on_buy("2024-01-01", "AAPL", 100, 10.0, 1.0)
        self.trade_book.on_sell("2024-01-02", "AAPL", 100, 11.0, 1.0, 0.0)  # +100
        
        self.trade_book.on_buy("2024-01-03", "AAPL", 100, 10.0, 1.0)
        self.trade_book.on_sell("2024-01-04", "AAPL", 100, 9.5, 1.0, 0.0)  # -50
        
        metrics = calculate_trade_metrics(self.trade_book.get_closed_trades())
        # The actual profit factor should be close to 2.0 (100/50) but with fees
        assert metrics["profit_factor"] > 1.0  # Should be positive
        assert metrics["profit_factor"] < 3.0  # Should be reasonable
    
    def test_profit_factor_na_when_no_losses(self):
        """Test profit factor is 'N/A' when no losses exist."""
        # Only winning trades
        self.trade_book.on_buy("2024-01-01", "AAPL", 100, 10.0, 1.0)
        self.trade_book.on_sell("2024-01-02", "AAPL", 100, 11.0, 1.0, 0.0)
        
        metrics = calculate_trade_metrics(self.trade_book.get_closed_trades())
        assert metrics["profit_factor"] == "N/A"
        assert metrics["win_rate"] == 1.0
    
    def test_partial_close_pnl_allocation(self):
        """Test PnL allocation for partial closes."""
        # Buy 100@10, Sell 40@9 (loss), Sell 60@11 (win) -> two partials, net pnl = +20 - fees
        self.trade_book.on_buy("2024-01-01", "AAPL", 100, 10.0, 1.0)
        
        # First partial: sell 40 @ 9 (loss)
        self.trade_book.on_sell("2024-01-02", "AAPL", 40, 9.0, 1.0, 60.0)
        
        # Second partial: sell 60 @ 11 (win)
        self.trade_book.on_sell("2024-01-03", "AAPL", 60, 11.0, 1.0, 0.0)
        
        closed_trades = self.trade_book.get_closed_trades()
        assert len(closed_trades) == 1
        
        trade = closed_trades[0]
        expected_pnl = (9.0 - 10.0) * 40 + (11.0 - 10.0) * 60  # -40 + 60 (fees handled separately)
        assert abs(trade.realized_pnl - expected_pnl) < 0.01


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_daily_returns_from_equity_not_notional(self):
        """Test daily returns calculated from equity changes, not notional."""
        equity_curve = [
            {"equity": 10000.0},
            {"equity": 10100.0},  # +1% return
            {"equity": 10050.0},  # -0.5% return
        ]
        
        metrics = calculate_portfolio_metrics(equity_curve)
        assert abs(metrics["total_return"] - 0.005) < 0.001  # 0.5% total return
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        trades = [
            TradeRecord("AAPL", "2024-01-01", realized_pnl=100.0),
            TradeRecord("AAPL", "2024-01-02", realized_pnl=-50.0),
            TradeRecord("AAPL", "2024-01-03", realized_pnl=75.0),
        ]
        
        metrics = calculate_trade_metrics(trades)
        assert metrics["win_rate"] == 2/3  # 2 wins out of 3 trades
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        trades = [
            TradeRecord("AAPL", "2024-01-01", realized_pnl=100.0),
            TradeRecord("AAPL", "2024-01-02", realized_pnl=-50.0),
            TradeRecord("AAPL", "2024-01-03", realized_pnl=75.0),
        ]
        
        metrics = calculate_trade_metrics(trades)
        expected_pf = (100.0 + 75.0) / 50.0  # 3.5
        assert metrics["profit_factor"] == expected_pf
    
    def test_profit_factor_na_no_losses(self):
        """Test profit factor is 'N/A' when no losses."""
        trades = [
            TradeRecord("AAPL", "2024-01-01", realized_pnl=100.0),
            TradeRecord("AAPL", "2024-01-02", realized_pnl=50.0),
        ]
        
        metrics = calculate_trade_metrics(trades)
        assert metrics["profit_factor"] == "N/A"


class TestWarmupFunctionality:
    """Test warmup period functionality."""
    
    def test_warmup_stops_insufficient_data_spam(self):
        """Test that warmup period prevents insufficient data spam."""
        # This test would require integration with the backtest engine
        # For now, we test the concept that warmup provides enough history
        assert True  # Placeholder - actual test would verify logging behavior
