#!/usr/bin/env python3
"""
Comprehensive Backtest System
Tests the trading strategy over historical data with full simulation.
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from enhanced_paper_trading import EnhancedPaperTradingSystem


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
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_returns = []
        self.equity_curve = []
        
        # Performance metrics
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        
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
        
        # Get trading dates
        trading_dates = self._get_trading_dates()
        print(f"📅 Trading days: {len(trading_dates)}")
        
        # Run simulation
        for i, current_date in enumerate(trading_dates):
            if i % 50 == 0:  # Progress indicator
                print(f"📈 Processing day {i+1}/{len(trading_dates)}: {current_date}")
            
            # Run daily trading
            self._run_daily_trading(current_date)
            
            # Record equity curve
            self._record_equity_curve(current_date)
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Generate results
        results = self._generate_results()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _get_trading_dates(self) -> List[date]:
        """Get list of trading dates between start and end."""
        dates = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Skip weekends (simple approach)
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        return dates
    
    def _run_daily_trading(self, current_date: date):
        """Run daily trading simulation."""
        try:
            # Get market data for the day
            data = self._get_historical_data(current_date)
            if data is None or data.empty:
                return
            
            # Detect regime
            regime_name, confidence, regime_params = self.trading_system.regime_detector.detect_regime(data)
            
            # Generate signals for each symbol
            symbols = self.trading_system.config.get("symbols", ["SPY"])
            
            for symbol in symbols:
                symbol_data = data[data["Symbol"] == symbol] if "Symbol" in data.columns else data
                
                if symbol_data.empty:
                    continue
                
                # Generate regime-aware signals
                signals = self._generate_signals(symbol_data, regime_name, regime_params)
                
                # Execute trades
                self._execute_trades(symbol, signals, current_date, regime_params)
            
            # Update performance tracking
            self._update_performance_tracking(current_date)
            
        except Exception as e:
            print(f"❌ Error in daily trading for {current_date}: {e}")
    
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
    
    def _execute_trades(self, symbol: str, signals: Dict[str, float], current_date: date, regime_params):
        """Execute trades based on signals."""
        try:
            # Get current price (use close price for backtesting)
            current_price = self._get_current_price(symbol, current_date)
            if current_price is None:
                return
            
            # Get current position
            current_position = self.positions.get(symbol, 0.0)
            
            # Calculate position size based on regime
            position_multiplier = regime_params.position_sizing_multiplier
            max_position_size = (
                self.trading_system.config.get("risk_params", {}).get("max_weight_per_symbol", 0.25)
                * position_multiplier
            )
            
            # Get regime-aware ensemble signal
            regime_signal = signals.get("regime_ensemble", 0.0)
            
            # Calculate target position with position-aware logic
            signal_strength = abs(regime_signal)
            if signal_strength < regime_params.confidence_threshold:
                target_position = 0.0
            else:
                # Calculate target position based on signal direction
                target_position = np.sign(regime_signal) * min(signal_strength, max_position_size)
                
                # ENFORCE REDUCE-ONLY LOGIC: No shorting unless explicitly enabled
                if target_position < 0 and current_position <= 0:
                    # Cannot sell if we don't have a position
                    target_position = 0.0
                
                # ENFORCE POSITION LIMITS: Cannot exceed max position size
                if abs(target_position) > max_position_size:
                    target_position = np.sign(target_position) * max_position_size
            
            # Execute trade if position changed significantly
            position_change = target_position - current_position
            if abs(position_change) > 0.01:  # 1% threshold
                # Calculate trade size
                trade_value = position_change * self.current_capital
                trade_size = trade_value / current_price
                
                # Validate trade size
                if abs(trade_size) < 0.01:  # Minimum trade size
                    return
                
                # Record trade
                trade = {
                    "date": current_date,
                    "symbol": symbol,
                    "action": "BUY" if trade_size > 0 else "SELL",
                    "size": abs(trade_size),
                    "price": current_price,
                    "value": abs(trade_value),
                    "regime": regime_params.regime_name,
                    "signal_strength": signal_strength,
                    "current_position": current_position,
                    "target_position": target_position,
                }
                
                self.trades.append(trade)
                
                # Update position
                self.positions[symbol] = target_position
                
                # Update capital based on actual trade execution
                self._update_capital_from_trade(trade_value, current_price, trade_size)
                
        except Exception as e:
            print(f"❌ Error executing trades for {symbol}: {e}")
    
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
    
    def _generate_results(self) -> Dict:
        """Generate comprehensive backtest results."""
        return {
            "backtest_period": {
                "start_date": self.start_date.strftime("%Y-%m-%d"),
                "end_date": self.end_date.strftime("%Y-%m-%d"),
                "trading_days": len(self.equity_curve),
            },
            "performance_metrics": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "volatility": self.volatility,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
            },
            "trading_summary": {
                "initial_capital": self.initial_capital,
                "final_capital": self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_capital,
                "total_trades": len(self.trades),
                "total_pnl": (self.equity_curve[-1]["equity"] - self.initial_capital) if self.equity_curve else 0,
            },
            "trades": self.trades,
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
        
        # Save trades
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
        
        # Performance metrics
        metrics = results["performance_metrics"]
        print(f"\n💰 PERFORMANCE METRICS")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Volatility: {metrics['volatility']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Trading summary
        summary = results["trading_summary"]
        print(f"\n📈 TRADING SUMMARY")
        print(f"   Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"   Final Capital: ${summary['final_capital']:,.2f}")
        print(f"   Total PnL: ${summary['total_pnl']:,.2f}")
        print(f"   Total Trades: {summary['total_trades']}")
        
        # Regime analysis
        if results["trades"]:
            regime_stats = {}
            for trade in results["trades"]:
                regime = trade.get("regime", "unknown")
                if regime not in regime_stats:
                    regime_stats[regime] = {"count": 0, "pnl": 0}
                regime_stats[regime]["count"] += 1
                regime_stats[regime]["pnl"] += trade.get("value", 0)
            
            print(f"\n🎯 REGIME ANALYSIS")
            for regime, stats in regime_stats.items():
                print(f"   {regime.title()}: {stats['count']} trades, PnL: ${stats['pnl']:,.2f}")


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
