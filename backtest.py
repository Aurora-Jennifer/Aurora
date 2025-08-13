#!/usr/bin/env python3
"""
Comprehensive Backtest System
Tests the trading strategy over historical data with full simulation.
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from enhanced_paper_trading import EnhancedPaperTradingSystem
from core.portfolio import PortfolioState
from core.trade_logger import TradeBook
from core.performance import calculate_trade_metrics, calculate_portfolio_metrics, validate_daily_returns, generate_performance_report


class BacktestEngine:
    """Comprehensive backtest engine for trading strategies."""
    
    def __init__(self, config_file: str, profile_file: str = None):
        """Initialize backtest engine."""
        self.config_file = config_file
        self.profile_file = profile_file
        
        # Initialize trading system
        self.trading_system = EnhancedPaperTradingSystem(config_file, profile_file)
        
        # Backtest state
        self.start_date = None
        self.end_date = None
        self.initial_capital = self.trading_system.capital
        
        # Portfolio and trade tracking
        self.portfolio = PortfolioState(cash=self.initial_capital)
        self.trade_book = TradeBook()
        
        # Data storage
        self.daily_returns = []
        self.equity_curve = []
        
        # Logging control
        self.insufficient_data_logged = set()  # Rate limit insufficient data logs
        self.logger = logging.getLogger(__name__)
        
        # Constants
        self.MIN_HISTORY = 252  # Minimum trading days for warmup
        
    def run_backtest(
        self, 
        start_date: str, 
        end_date: str, 
        symbols: List[str] = None
    ) -> Dict:
        """Run comprehensive backtest over specified period."""
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if symbols:
            self.trading_system.config["symbols"] = symbols
        
        print(f"🚀 Starting backtest: {start_date} to {end_date}")
        print(f"📊 Symbols: {self.trading_system.config['symbols']}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print("=" * 60)
        
        # Load data with warmup period
        print("📊 Loading data with warmup period...")
        warmup_start = self.start_date - timedelta(days=365)  # 1 year warmup
        all_data = self._load_historical_data(warmup_start, self.end_date)
        
        if all_data is None or all_data.empty:
            print("❌ No data available for backtest period")
            return {}
        
        print(f"📊 Loaded {len(all_data)} data points")
        print(f"📊 Data columns: {list(all_data.columns)}")
        print(f"📊 Data shape: {all_data.shape}")
        
        # Get trading dates from data
        trading_dates = self._get_trading_dates_from_data(all_data)
        print(f"📅 Trading days: {len(trading_dates)}")
        
        if len(trading_dates) == 0:
            print("❌ No trading dates found in data")
            return {}
        
        # Run simulation
        for i, current_date in enumerate(trading_dates):
            if i % 50 == 0:  # Progress indicator
                print(f"📈 Processing day {i+1}/{len(trading_dates)}: {current_date}")
            
            # Update portfolio prices
            self._update_portfolio_prices(current_date, all_data)
            
            # Mark to market and record equity
            equity_today = self.portfolio.mark_to_market()
            if i == 0:
                equity_prev = equity_today
            else:
                # Calculate daily return
                daily_return = (equity_today - equity_prev) / max(equity_prev, 1e-12)
                self.daily_returns.append({
                    "date": current_date,
                    "return": daily_return,
                    "equity": equity_today
                })
                equity_prev = equity_today
            
            # Record equity curve
            self.equity_curve.append({
                "date": current_date,
                "equity": equity_today,
                "cash": self.portfolio.cash,
                "positions_value": equity_today - self.portfolio.cash
            })
            
            # Run daily trading (only if we have enough history)
            if i >= self.MIN_HISTORY:
                self._run_daily_trading(current_date, all_data)
        
        # Calculate performance metrics
        trade_metrics = calculate_trade_metrics(self.trade_book.get_closed_trades())
        portfolio_metrics = calculate_portfolio_metrics(self.equity_curve)
        
        # Generate results
        results = self._generate_results(trade_metrics, portfolio_metrics)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _load_historical_data(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Load historical data for all symbols with proper error handling."""
        all_data = []
        symbols = self.trading_system.config.get("symbols", ["SPY"])
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                
                if not data.empty:
                    data["Symbol"] = symbol
                    all_data.append(data)
                    
            except Exception as e:
                # Demote yfinance errors to DEBUG level
                self.logger.debug(f"Failed to get data for {symbol}: {e}")
        
        if all_data:
            # Preserve the index (dates) when concatenating
            combined_data = pd.concat(all_data, axis=0)
            return combined_data
        else:
            return None
    
    def _get_trading_dates_from_data(self, data: pd.DataFrame) -> List[date]:
        """Get trading dates from actual data."""
        if data.empty:
            return []
        
        # Extract unique dates from data
        try:
            if hasattr(data.index, 'date'):
                dates = data.index.date
            elif 'Date' in data.columns:
                dates = pd.to_datetime(data['Date']).dt.date
            else:
                # Try to parse the index as dates
                dates = pd.to_datetime(data.index).date
            
            unique_dates = sorted(list(set(dates)))
            print(f"📊 Found {len(unique_dates)} unique dates in data")
            print(f"📊 Date range: {min(unique_dates)} to {max(unique_dates)}")
            
            # Filter to backtest period
            trading_dates = [d for d in unique_dates if self.start_date <= d <= self.end_date]
            print(f"📊 Filtered to {len(trading_dates)} trading dates in backtest period")
            
            return trading_dates
            
        except Exception as e:
            print(f"❌ Error extracting dates: {e}")
            print(f"📊 Data index type: {type(data.index)}")
            print(f"📊 Data index sample: {data.index[:5]}")
            return []
    
    def _update_portfolio_prices(self, current_date: date, data: pd.DataFrame):
        """Update portfolio prices for mark-to-market."""
        symbols = self.trading_system.config.get("symbols", ["SPY"])
        
        for symbol in symbols:
            # Get price for current date
            symbol_data = data[data["Symbol"] == symbol] if "Symbol" in data.columns else data
            if not symbol_data.empty:
                # Use close price for mark-to-market
                close_price = symbol_data["Close"].iloc[-1] if "Close" in symbol_data.columns else None
                if close_price and not pd.isna(close_price):
                    self.portfolio.update_price(symbol, float(close_price))
    
    def _run_daily_trading(self, current_date: date, data: pd.DataFrame):
        """Run daily trading simulation."""
        try:
            # Detect regime (rate limit insufficient data logs)
            regime_name, confidence, regime_params = self._detect_regime_with_rate_limit(data)
            
            # Generate signals for each symbol
            symbols = self.trading_system.config.get("symbols", ["SPY"])
            
            for symbol in symbols:
                symbol_data = data[data["Symbol"] == symbol] if "Symbol" in data.columns else data
                
                if symbol_data.empty:
                    continue
                
                # Generate regime-aware signals
                signals = self._generate_signals(symbol_data, regime_name, regime_params)
                
                # Execute trades using portfolio system
                self._execute_trades_with_portfolio(symbol, signals, current_date, regime_params)
            
        except Exception as e:
            self.logger.error(f"Error in daily trading for {current_date}: {e}")
    
    def _detect_regime_with_rate_limit(self, data: pd.DataFrame):
        """Detect regime with rate-limited logging."""
        try:
            return self.trading_system.regime_detector.detect_regime(data)
        except Exception as e:
            if "insufficient data" in str(e).lower():
                # Only log once per symbol
                if "insufficient_data" not in self.insufficient_data_logged:
                    self.logger.warning(f"Insufficient data for regime detection: {e}")
                    self.insufficient_data_logged.add("insufficient_data")
            else:
                self.logger.error(f"Regime detection error: {e}")
            return "trend", 0.5, None  # Default fallback
    
    def _get_historical_data(self, current_date: date) -> Optional[pd.DataFrame]:
        """Get historical market data for backtesting."""
        try:
            # Get data for the last 300 days to ensure enough history
            end_date = current_date
            start_date = end_date - timedelta(days=300)
            
            all_data = []
            symbols = self.trading_system.config.get("symbols", ["SPY"])
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date + timedelta(days=1))
                    
                    if not data.empty:
                        data["Symbol"] = symbol
                        all_data.append(data)
                        
                except Exception as e:
                    print(f"⚠️  Failed to get data for {symbol}: {e}")
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error getting historical data: {e}")
            return None
    
    def _generate_signals(self, data: pd.DataFrame, regime_name: str, regime_params) -> Dict[str, float]:
        """Generate trading signals."""
        try:
            return self.trading_system._generate_regime_aware_signals(data, regime_name, regime_params)
        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            return {}
    
    def _execute_trades_with_portfolio(self, symbol: str, signals: Dict[str, float], current_date: date, regime_params):
        """Execute trades using portfolio system with proper PnL tracking."""
        try:
            # Get current price from portfolio
            current_price = self.portfolio.last_prices.get(symbol)
            if current_price is None:
                return
            
            # Get current position from portfolio
            current_position = self.portfolio.positions.get(symbol, None)
            current_qty = current_position.qty if current_position else 0.0
            
            # Calculate position size based on regime
            position_multiplier = regime_params.position_sizing_multiplier if regime_params else 1.0
            max_position_size = (
                self.trading_system.config.get("risk_params", {}).get("max_weight_per_symbol", 0.25)
                * position_multiplier
            )
            
            # Get regime-aware ensemble signal
            regime_signal = signals.get("regime_ensemble", 0.0)
            
            # Calculate target position with position-aware logic
            signal_strength = abs(regime_signal)
            confidence_threshold = regime_params.confidence_threshold if regime_params else 0.3
            
            if signal_strength < confidence_threshold:
                target_position = 0.0
            else:
                # Calculate target position based on signal direction
                target_position = np.sign(regime_signal) * min(signal_strength, max_position_size)
                
                # ENFORCE REDUCE-ONLY LOGIC: No shorting unless explicitly enabled
                if target_position < 0 and current_qty <= 0:
                    target_position = 0.0
                
                # ENFORCE POSITION LIMITS: Cannot exceed max position size
                if abs(target_position) > max_position_size:
                    target_position = np.sign(target_position) * max_position_size
            
            # Calculate target quantity
            portfolio_value = self.portfolio.mark_to_market()
            target_qty = (target_position * portfolio_value) / current_price
            
            # Execute trade if position changed significantly
            qty_change = target_qty - current_qty
            if abs(qty_change) > 0.01:  # Minimum trade size
                # Determine trade side
                if qty_change > 0:
                    side = "BUY"
                    qty = qty_change
                else:
                    side = "SELL"
                    qty = abs(qty_change)
                
                # Get fee rate
                fee_bps = self.trading_system.config.get("execution_params", {}).get("max_slippage_bps", 10)
                
                # Apply fill to portfolio
                realized_pnl = self.portfolio.apply_fill(symbol, side, qty, current_price, fee_bps)
                
                # Calculate fees
                fees = (qty * current_price) * (fee_bps / 10000.0)
                
                # Record in trade book
                if side == "BUY":
                    self.trade_book.on_buy(str(current_date), symbol, qty, current_price, fees)
                else:
                    # Get remaining quantity after sell
                    remaining_qty = self.portfolio.positions.get(symbol, None)
                    remaining_qty = remaining_qty.qty if remaining_qty else 0.0
                    self.trade_book.on_sell(str(current_date), symbol, qty, current_price, fees, remaining_qty)
                
                self.logger.info(f"Trade: {side} {qty:.2f} {symbol} @ ${current_price:.2f}, PnL: ${realized_pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error executing trades for {symbol}: {e}")
    
    def _get_current_price(self, symbol: str, current_date: date) -> Optional[float]:
        """Get current price for symbol on given date."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=current_date, end=current_date + timedelta(days=1))
            if not data.empty:
                return data["Close"].iloc[-1]
            return None
        except Exception as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def _update_capital_from_trade(self, trade_value: float, price: float, size: float):
        """Update capital based on actual trade execution."""
        # Calculate transaction costs (fees, slippage)
        fees_bps = self.trading_system.config.get("execution_params", {}).get("max_slippage_bps", 10)
        fees = abs(trade_value) * (fees_bps / 10000)
        
        # Update capital: subtract fees from trade value
        self.current_capital -= fees
    
    def _update_performance_tracking(self, current_date: date):
        """Update performance tracking."""
        # Calculate total portfolio value including all positions
        total_value = self.current_capital  # Start with cash
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol, current_date)
            if current_price and position != 0:
                # Calculate position value correctly
                position_value = position * self.current_capital
                total_value += position_value
        
        # Calculate daily return based on total value change
        if hasattr(self, '_previous_total_value'):
            daily_return = (total_value - self._previous_total_value) / self._previous_total_value
        else:
            # First day - no return
            daily_return = 0.0
            
        self._previous_total_value = total_value
        
        # Store daily return
        self.daily_returns.append({
            "date": current_date,
            "return": daily_return,
            "total_value": total_value,
            "cash": self.current_capital,
            "positions_value": total_value - self.current_capital,
        })
    
    def _record_equity_curve(self, current_date: date):
        """Record equity curve for performance analysis."""
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol, current_date)
            if current_price and position != 0:
                position_value = position * self.current_capital
                total_value += position_value
        
        self.equity_curve.append({
            "date": current_date,
            "equity": total_value,
            "cash": self.current_capital,
            "positions_value": total_value - self.current_capital,
        })
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return
        
        # Convert to DataFrame for easier calculations
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("date", inplace=True)
        
        # Calculate returns
        equity_df["returns"] = equity_df["equity"].pct_change()
        
        # Total return
        final_equity = equity_df["equity"].iloc[-1]
        self.total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (self.end_date - self.start_date).days
        if days > 0:
            self.annualized_return = (1 + self.total_return) ** (365 / days) - 1
        
        # Volatility
        self.volatility = equity_df["returns"].std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        if self.volatility > 0:
            self.sharpe_ratio = self.annualized_return / self.volatility
        
        # Maximum drawdown
        equity_df["cummax"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["cummax"]) / equity_df["cummax"]
        self.max_drawdown = equity_df["drawdown"].min()
        
        # Win rate and profit factor
        if self.trades:
            winning_trades = [t for t in self.trades if t["value"] > 0]
            losing_trades = [t for t in self.trades if t["value"] < 0]
            
            self.win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            
            total_wins = sum(t["value"] for t in winning_trades)
            total_losses = abs(sum(t["value"] for t in losing_trades))
            
            self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    def _generate_results(self, trade_metrics: Dict, portfolio_metrics: Dict) -> Dict:
        """Generate comprehensive backtest results."""
        return {
            "backtest_period": {
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "end_date": self.end_date.strftime("%Y-%m-%d"),
                "trading_days": len(self.equity_curve),
            },
            "performance_metrics": portfolio_metrics,
            "trade_metrics": trade_metrics,
            "trading_summary": {
                "initial_capital": self.initial_capital,
                "final_capital": self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_capital,
                "total_trades": trade_metrics.get("total_trades", 0),
                "total_pnl": trade_metrics.get("total_pnl", 0),
            },
            "trades": self.trade_book.export_trades_csv(),
            "daily_returns": self.daily_returns,
            "equity_curve": self.equity_curve,
        }
    
    def _save_results(self, results: Dict):
        """Save backtest results to files."""
        # Create backtest results directory
        results_dir = Path("results/backtest")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(results_dir / "backtest_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save trades with new format
        if results["trades"]:
            trades_df = pd.DataFrame(results["trades"])
            trades_df.to_csv(results_dir / "backtest_trades.csv", index=False)
        
        # Save daily returns
        if results["daily_returns"]:
            returns_df = pd.DataFrame(results["daily_returns"])
            returns_df.to_csv(results_dir / "backtest_daily_returns.csv", index=False)
        
        # Save equity curve
        if results["equity_curve"]:
            equity_df = pd.DataFrame(results["equity_curve"])
            equity_df.to_csv(results_dir / "backtest_equity_curve.csv", index=False)
        
        # Generate and save performance report
        if results.get("trade_metrics") and results.get("performance_metrics"):
            report = generate_performance_report(
                results["trade_metrics"],
                results["performance_metrics"],
                results["equity_curve"],
                results["backtest_period"]
            )
            with open(results_dir / "performance_report.txt", "w") as f:
                f.write(report)
        
        print(f"✅ Backtest results saved to {results_dir}")
    
    def print_results(self, results: Dict):
        """Print formatted backtest results."""
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        
        # Period summary
        period = results["backtest_period"]
        print(f"📅 Period: {period['start_date']} to {period['end_date']}")
        print(f"📈 Trading Days: {period['trading_days']}")
        
        # Portfolio performance metrics
        portfolio_metrics = results["performance_metrics"]
        print(f"\n💰 PORTFOLIO METRICS")
        print(f"   Total Return: {portfolio_metrics.get('total_return', 0):.2%}")
        print(f"   Annualized Return: {portfolio_metrics.get('annualized_return', 0):.2%}")
        print(f"   Volatility: {portfolio_metrics.get('volatility', 0):.2%}")
        print(f"   Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2%}")
        print(f"   Calmar Ratio: {portfolio_metrics.get('calmar_ratio', 0):.2f}")
        
        # Trade metrics
        trade_metrics = results.get("trade_metrics", {})
        print(f"\n📈 TRADING METRICS")
        print(f"   Total Trades: {trade_metrics.get('total_trades', 0)}")
        print(f"   Win Rate: {trade_metrics.get('win_rate', 0):.2%}")
        print(f"   Profit Factor: {trade_metrics.get('profit_factor', 'N/A')}")
        print(f"   Total PnL: ${trade_metrics.get('total_pnl', 0):,.2f}")
        print(f"   Total Fees: ${trade_metrics.get('total_fees', 0):,.2f}")
        print(f"   Largest Win: ${trade_metrics.get('largest_win', 0):,.2f}")
        print(f"   Largest Loss: ${trade_metrics.get('largest_loss', 0):,.2f}")
        print(f"   Average Win: ${trade_metrics.get('avg_win', 0):,.2f}")
        print(f"   Average Loss: ${trade_metrics.get('avg_loss', 0):,.2f}")
        
        # Trading summary
        summary = results["trading_summary"]
        print(f"\n📊 TRADING SUMMARY")
        print(f"   Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"   Final Capital: ${summary['final_capital']:,.2f}")
        print(f"   Total PnL: ${summary['total_pnl']:,.2f}")
        print(f"   Total Trades: {summary['total_trades']}")
        
        # Note about profit factor
        if trade_metrics.get('profit_factor') == 'N/A':
            print(f"\n⚠️  NOTE: No losing trades recorded; verify accounting.")


def main():
    """Main backtest function."""
    parser = argparse.ArgumentParser(description="Comprehensive Backtest System")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/enhanced_paper_trading_config.json", help="Configuration file")
    parser.add_argument("--profile", help="Profile configuration file")
    parser.add_argument("--symbols", nargs="+", help="Symbols to trade")
    
    args = parser.parse_args()
    
    try:
        # Initialize backtest engine
        engine = BacktestEngine(args.config, args.profile)
        
        # Run backtest
        results = engine.run_backtest(args.start_date, args.end_date, args.symbols)
        
        # Print results
        engine.print_results(results)
        
        print(f"\n🎉 Backtest completed successfully!")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
