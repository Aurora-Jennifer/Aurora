#!/usr/bin/env python3
"""
Paper Trading System
Simulates live trading with realistic execution and monitoring
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Paper trading engine with realistic execution simulation."""
    
    def __init__(self, initial_capital: float = 100000, 
                 commission_bps: float = 3.0, 
                 slippage_bps: float = 1.0):
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # {symbol: shares}
        self.portfolio_value = initial_capital
        self.trade_history = []
        self.daily_pnl = []
        
        # Performance tracking
        self.total_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        position_value = sum(shares * prices.get(symbol, 0) 
                           for symbol, shares in self.positions.items())
        return self.cash + position_value
    
    def execute_trade(self, symbol: str, target_shares: int, 
                     current_price: float, timestamp: datetime) -> Dict:
        """Execute a trade with realistic costs."""
        
        current_shares = self.positions.get(symbol, 0)
        shares_to_trade = target_shares - current_shares
        
        if shares_to_trade == 0:
            return {'status': 'no_trade', 'shares': 0, 'cost': 0}
        
        # Calculate execution price with slippage
        slippage_factor = self.slippage_bps / 10000
        if shares_to_trade > 0:  # Buy
            execution_price = current_price * (1 + slippage_factor)
        else:  # Sell
            execution_price = current_price * (1 - slippage_factor)
        
        # Calculate costs
        trade_value = abs(shares_to_trade) * execution_price
        commission = trade_value * (self.commission_bps / 10000)
        slippage_cost = abs(shares_to_trade) * current_price * slippage_factor
        
        # Check if we have enough cash for buy orders
        if shares_to_trade > 0:
            total_cost = trade_value + commission
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for {symbol} trade: need ${total_cost:.2f}, have ${self.cash:.2f}")
                return {'status': 'insufficient_cash', 'shares': 0, 'cost': 0}
        
        # Execute trade
        self.positions[symbol] = target_shares
        self.cash -= trade_value + commission
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += slippage_cost
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'buy' if shares_to_trade > 0 else 'sell',
            'shares': abs(shares_to_trade),
            'price': execution_price,
            'value': trade_value,
            'commission': commission,
            'slippage': slippage_cost,
            'cash_after': self.cash,
            'position_after': target_shares
        }
        
        self.trade_history.append(trade_record)
        
        logger.info(f"Trade: {trade_record['action'].upper()} {shares_to_trade} {symbol} @ ${execution_price:.2f}")
        
        return {
            'status': 'executed',
            'shares': shares_to_trade,
            'cost': trade_value + commission,
            'commission': commission,
            'slippage': slippage_cost
        }
    
    def rebalance_portfolio(self, target_weights: Dict[str, float], 
                          prices: Dict[str, float], timestamp: datetime) -> Dict:
        """Rebalance portfolio to target weights."""
        
        current_value = self.get_portfolio_value(prices)
        rebalance_results = {}
        
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                logger.warning(f"No price data for {symbol}")
                continue
            
            target_value = current_value * target_weight
            target_shares = int(target_value / prices[symbol])
            
            result = self.execute_trade(symbol, target_shares, prices[symbol], timestamp)
            rebalance_results[symbol] = result
        
        return rebalance_results
    
    def get_performance_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """Calculate performance metrics."""
        
        if not self.daily_pnl:
            return {'error': 'No performance data'}
        
        pnl_series = pd.Series(self.daily_pnl)
        returns = pnl_series.pct_change().dropna()
        
        if len(returns) == 0:
            return {'error': 'Insufficient return data'}
        
        # Basic metrics
        total_return = (pnl_series.iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        avg_trade_size = np.mean([abs(t['value']) for t in self.trade_history]) if self.trade_history else 0
        turnover = sum(abs(t['value']) for t in self.trade_history) / self.initial_capital
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'avg_trade_size': avg_trade_size,
            'turnover': turnover,
            'final_portfolio_value': pnl_series.iloc[-1] if pnl_series.size > 0 else self.initial_capital
        }
        
        # Benchmark comparison
        if benchmark_returns is not None:
            aligned_returns = returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
            
            if len(aligned_returns) > 0 and len(aligned_benchmark) > 0:
                excess_returns = aligned_returns - aligned_benchmark
                alpha = excess_returns.mean() * 252
                beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var()
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = alpha / tracking_error if tracking_error > 0 else 0
                
                metrics.update({
                    'alpha': alpha,
                    'beta': beta,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio
                })
        
        return metrics


def load_portfolio_weights(weights_file: str) -> pd.DataFrame:
    """Load portfolio weights from CSV file."""
    df = pd.read_csv(weights_file)
    return df.set_index('strategy_id')


def fetch_live_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices for symbols."""
    prices = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                prices[symbol] = data['Close'].iloc[-1]
            else:
                logger.warning(f"No price data for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
    
    return prices


def run_paper_trading(weights_file: str, start_date: str, end_date: str,
                     rebalance_freq: str = 'weekly', output_dir: str = 'paper_trading') -> None:
    """Run paper trading simulation."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load portfolio weights
    weights_df = load_portfolio_weights(weights_file)
    logger.info(f"Loaded {len(weights_df)} strategies")
    
    # Initialize trading engine
    engine = PaperTradingEngine()
    
    # Generate rebalancing dates
    dates = pd.date_range(start_date, end_date, freq=rebalance_freq)
    
    # Track daily performance
    daily_performance = []
    
    for date in dates:
        try:
            # Get current prices
            symbols = weights_df.index.tolist()
            prices = fetch_live_prices(symbols)
            
            if not prices:
                logger.warning(f"No price data for {date}")
                continue
            
            # Calculate target weights (simplified - equal weight for now)
            target_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
            
            # Rebalance portfolio
            rebalance_results = engine.rebalance_portfolio(target_weights, prices, date)
            
            # Record daily performance
            portfolio_value = engine.get_portfolio_value(prices)
            engine.daily_pnl.append(portfolio_value)
            
            daily_performance.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': engine.cash,
                'positions': dict(engine.positions),
                'trades_executed': sum(1 for r in rebalance_results.values() if r['status'] == 'executed')
            })
            
            logger.info(f"Rebalanced on {date}: Portfolio value = ${portfolio_value:,.2f}")
            
        except Exception as e:
            logger.error(f"Error on {date}: {e}")
    
    # Calculate final performance metrics
    performance_metrics = engine.get_performance_metrics()
    
    # Save results
    pd.DataFrame(daily_performance).to_csv(output_path / 'daily_performance.csv', index=False)
    pd.DataFrame(engine.trade_history).to_csv(output_path / 'trade_history.csv', index=False)
    
    with open(output_path / 'performance_metrics.json', 'w') as f:
        json.dump(performance_metrics, f, indent=2, default=str)
    
    # Generate report
    generate_paper_trading_report(performance_metrics, engine, output_path)
    
    logger.info(f"Paper trading complete! Final portfolio value: ${performance_metrics.get('final_portfolio_value', 0):,.2f}")


def generate_paper_trading_report(metrics: Dict, engine: PaperTradingEngine, output_path: Path) -> None:
    """Generate paper trading performance report."""
    
    report = f"""# Paper Trading Performance Report

## Summary

- **Initial Capital**: ${engine.initial_capital:,.2f}
- **Final Portfolio Value**: ${metrics.get('final_portfolio_value', 0):,.2f}
- **Total Return**: {metrics.get('total_return', 0):.2%}
- **Annualized Return**: {metrics.get('annualized_return', 0):.2%}
- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.3f}
- **Max Drawdown**: {metrics.get('max_drawdown', 0):.2%}

## Trading Statistics

- **Total Trades**: {metrics.get('total_trades', 0)}
- **Total Commission**: ${metrics.get('total_commission', 0):,.2f}
- **Total Slippage**: ${metrics.get('total_slippage', 0):,.2f}
- **Average Trade Size**: ${metrics.get('avg_trade_size', 0):,.2f}
- **Portfolio Turnover**: {metrics.get('turnover', 0):.2%}

## Risk Metrics

- **Volatility**: {metrics.get('volatility', 0):.2%}
- **Max Drawdown**: {metrics.get('max_drawdown', 0):.2%}

## Cost Analysis

- **Commission Cost**: {metrics.get('total_commission', 0) / engine.initial_capital:.2%} of initial capital
- **Slippage Cost**: {metrics.get('total_slippage', 0) / engine.initial_capital:.2%} of initial capital
- **Total Trading Costs**: {(metrics.get('total_commission', 0) + metrics.get('total_slippage', 0)) / engine.initial_capital:.2%} of initial capital

---
*Generated by paper_trading.py*
"""
    
    with open(output_path / 'performance_report.md', 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description='Run paper trading simulation')
    parser.add_argument('--weights-file', required=True, help='Portfolio weights CSV file')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--rebalance-freq', default='weekly', help='Rebalancing frequency')
    parser.add_argument('--output-dir', default='paper_trading', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        run_paper_trading(
            args.weights_file,
            args.start_date,
            args.end_date,
            args.rebalance_freq,
            args.output_dir
        )
        
    except Exception as e:
        logger.error(f"Error in paper trading: {e}")
        raise


if __name__ == "__main__":
    main()
