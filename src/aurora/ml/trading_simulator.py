"""
Trading Simulation Engine

Simulates actual trading to test model profitability and provide
realistic feedback for learning.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    quantity: float
    value: float
    commission: float
    net_value: float


@dataclass
class Portfolio:
    """Represents portfolio state"""
    cash: float
    shares: float
    total_value: float
    trades: list[Trade]
    daily_returns: list[float]
    max_drawdown: float
    peak_value: float


class TradingSimulator:
    """
    Simulates realistic trading with commissions, slippage, and risk management
    """
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.initial_cash = config.get('initial_cash', 10000)
        self.commission_per_trade = config.get('commission_per_trade', 1.0)
        self.slippage_bps = config.get('slippage_bps', 5)  # 5 basis points
        self.max_position_size = config.get('max_position_size', 0.3)  # 30% max
        self.min_trade_value = config.get('min_trade_value', 100)  # $100 minimum
        
    def simulate_trading(self, 
                        data: pd.DataFrame,
                        predictions: np.ndarray,
                        confidence: np.ndarray,
                        model_name: str = "Model") -> dict[str, Any]:
        """
        Simulate trading based on model predictions
        
        Args:
            data: Market data with OHLCV
            predictions: Model predictions (0=BUY, 1=SELL, 2=HOLD)
            confidence: Model confidence scores
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with trading results
        """
        
        # Initialize portfolio
        portfolio = Portfolio(
            cash=self.initial_cash,
            shares=0.0,
            total_value=self.initial_cash,
            trades=[],
            daily_returns=[],
            max_drawdown=0.0,
            peak_value=self.initial_cash
        )
        
        # Simulate trading day by day
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            current_date = data.index[i]
            action = predictions[i]
            conf = confidence[i]
            
            # Calculate position size based on confidence
            position_size = min(conf * self.max_position_size, self.max_position_size)
            
            # Execute trade
            self._execute_trade(portfolio, current_date, action, current_price, 
                              position_size, conf)
            
            # Update portfolio value
            portfolio.total_value = portfolio.cash + (portfolio.shares * current_price)
            
            # Track daily returns
            if i > 0:
                daily_return = (portfolio.total_value - portfolio.peak_value) / portfolio.peak_value
                portfolio.daily_returns.append(daily_return)
            
            # Update peak and drawdown
            if portfolio.total_value > portfolio.peak_value:
                portfolio.peak_value = portfolio.total_value
            else:
                current_drawdown = (portfolio.peak_value - portfolio.total_value) / portfolio.peak_value
                portfolio.max_drawdown = max(portfolio.max_drawdown, current_drawdown)
        
        # Calculate final results
        results = self._calculate_results(portfolio, data, model_name)
        
        return results
    
    def _execute_trade(self, portfolio: Portfolio, date: datetime, action: int, 
                      price: float, position_size: float, confidence: float):
        """Execute a single trade"""
        
        # Apply slippage
        slippage = price * (self.slippage_bps / 10000)
        if action == 0:  # BUY
            execution_price = price + slippage
        elif action == 1:  # SELL
            execution_price = price - slippage
        else:  # HOLD
            return
        
        # Calculate trade value
        if action == 0:  # BUY
            trade_value = portfolio.cash * position_size
            if trade_value < self.min_trade_value:
                return  # Skip small trades
            
            shares_to_buy = trade_value / execution_price
            commission = self.commission_per_trade
            
            # Check if we have enough cash
            total_cost = (shares_to_buy * execution_price) + commission
            if total_cost > portfolio.cash:
                shares_to_buy = (portfolio.cash - commission) / execution_price
                total_cost = (shares_to_buy * execution_price) + commission
            
            # Execute buy
            portfolio.cash -= total_cost
            portfolio.shares += shares_to_buy
            
            trade = Trade(
                timestamp=date,
                action='BUY',
                price=execution_price,
                quantity=shares_to_buy,
                value=shares_to_buy * execution_price,
                commission=commission,
                net_value=-total_cost
            )
            
        elif action == 1:  # SELL
            if portfolio.shares <= 0:
                return  # Nothing to sell
            
            shares_to_sell = portfolio.shares * position_size
            if shares_to_sell * execution_price < self.min_trade_value:
                return  # Skip small trades
            
            commission = self.commission_per_trade
            gross_proceeds = shares_to_sell * execution_price
            net_proceeds = gross_proceeds - commission
            
            # Execute sell
            portfolio.cash += net_proceeds
            portfolio.shares -= shares_to_sell
            
            trade = Trade(
                timestamp=date,
                action='SELL',
                price=execution_price,
                quantity=shares_to_sell,
                value=gross_proceeds,
                commission=commission,
                net_value=net_proceeds
            )
        
        else:  # HOLD
            return
        
        portfolio.trades.append(trade)
    
    def _calculate_results(self, portfolio: Portfolio, data: pd.DataFrame, 
                          model_name: str) -> dict[str, Any]:
        """Calculate trading results"""
        
        # Final portfolio value
        final_price = data['Close'].iloc[-1]
        final_value = portfolio.cash + (portfolio.shares * final_price)
        
        # Calculate returns
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # Buy and hold comparison
        buy_hold_shares = self.initial_cash / data['Close'].iloc[0]
        buy_hold_value = buy_hold_shares * final_price
        buy_hold_return = (buy_hold_value - self.initial_cash) / self.initial_cash
        
        # Calculate Sharpe ratio
        if len(portfolio.daily_returns) > 1 and np.std(portfolio.daily_returns) > 0:
            sharpe_ratio = np.mean(portfolio.daily_returns) / np.std(portfolio.daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate win rate
        profitable_trades = [t for t in portfolio.trades if t.net_value > 0]
        win_rate = len(profitable_trades) / max(1, len(portfolio.trades))
        
        # Calculate average trade
        if portfolio.trades:
            avg_trade = np.mean([t.net_value for t in portfolio.trades])
        else:
            avg_trade = 0.0
        
        results = {
            'model_name': model_name,
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': portfolio.max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(portfolio.trades),
            'avg_trade': avg_trade,
            'final_cash': portfolio.cash,
            'final_shares': portfolio.shares,
            'trades': portfolio.trades,
            'daily_returns': portfolio.daily_returns
        }
        
        return results
    
    def compare_models(self, data: pd.DataFrame, 
                      model_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare multiple model results"""
        
        comparison = {
            'models': [],
            'best_model': None,
            'best_return': -np.inf,
            'best_sharpe': -np.inf
        }
        
        for result in model_results:
            comparison['models'].append({
                'name': result['model_name'],
                'return': result['total_return'],
                'sharpe': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'trades': result['total_trades']
            })
            
            # Track best performers
            if result['total_return'] > comparison['best_return']:
                comparison['best_return'] = result['total_return']
                comparison['best_model'] = result['model_name']
            
            if result['sharpe_ratio'] > comparison['best_sharpe']:
                comparison['best_sharpe'] = result['sharpe_ratio']
        
        return comparison


class PerformanceTracker:
    """
    Tracks model performance over time and provides feedback for learning
    """
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.performance_history = []
        self.best_performance = None
        self.learning_rate = config.get('learning_rate', 0.01)
        
    def record_performance(self, results: dict[str, Any]):
        """Record model performance"""
        
        performance = {
            'timestamp': datetime.now(),
            'model_name': results['model_name'],
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate'],
            'total_trades': results['total_trades']
        }
        
        self.performance_history.append(performance)
        
        # Update best performance
        if (self.best_performance is None or 
            results['total_return'] > self.best_performance['total_return']):
            self.best_performance = performance
    
    def get_performance_feedback(self) -> dict[str, Any]:
        """Get feedback for model improvement"""
        
        if len(self.performance_history) < 2:
            return {'feedback': 'Need more data for feedback'}
        
        recent_performance = self.performance_history[-5:]  # Last 5 runs
        avg_return = np.mean([p['total_return'] for p in recent_performance])
        avg_sharpe = np.mean([p['sharpe_ratio'] for p in recent_performance])
        avg_win_rate = np.mean([p['win_rate'] for p in recent_performance])
        
        feedback = {
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'trend': 'improving' if avg_return > 0 else 'declining',
            'recommendations': []
        }
        
        # Generate recommendations
        if avg_return < 0:
            feedback['recommendations'].append('Consider reducing position sizes')
        if avg_win_rate < 0.4:
            feedback['recommendations'].append('Improve entry/exit timing')
        if avg_sharpe < 0.5:
            feedback['recommendations'].append('Reduce volatility in returns')
        
        return feedback
    
    def should_continue_learning(self) -> bool:
        """Determine if model should continue learning"""
        
        if len(self.performance_history) < 10:
            return True
        
        recent_performance = self.performance_history[-10:]
        returns = [p['total_return'] for p in recent_performance]
        
        # Stop if performance is consistently poor
        if np.mean(returns) < -0.1 and np.std(returns) < 0.05:
            return False
        
        # Stop if performance has plateaued
        if len(returns) >= 5:
            recent_trend = np.polyfit(range(len(returns)), returns, 1)[0]
            if abs(recent_trend) < 0.001:  # Very flat trend
                return False
        
        return True
