"""
Portfolio Manager

Manages multi-symbol portfolio with position sizing, risk management,
and portfolio-level decision making.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


@dataclass
class Position:
    """Represents a position in a symbol"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    position_type: str  # 'long', 'short', 'cash'
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management"""
    max_positions: int = 10
    max_position_size: float = 0.2  # 20% max per position
    max_portfolio_risk: float = 0.15  # 15% max portfolio risk
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    risk_model: str = 'equal_weight'  # 'equal_weight', 'risk_parity', 'momentum'
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 3.0


class PortfolioManager:
    """
    Manages multi-symbol portfolio with sophisticated risk management
    
    Features:
    - Position sizing and risk management
    - Portfolio rebalancing
    - Risk budgeting and allocation
    - Performance tracking
    - Transaction cost management
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.positions: dict[str, Position] = {}
        self.cash = 100000.0  # Starting cash
        self.total_value = 100000.0
        self.portfolio_history: list[dict[str, Any]] = []
        self.last_rebalance = None
        
    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Calculate current portfolio value"""
        
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                total_value += position.quantity * position.current_price
        
        self.total_value = total_value
        return total_value
    
    def calculate_position_sizes(self, 
                               predictions: dict[str, int], 
                               confidence: dict[str, float],
                               current_prices: dict[str, float],
                               features: pd.DataFrame) -> dict[str, float]:
        """Calculate optimal position sizes based on predictions and risk model"""
        
        position_sizes = {}
        
        if self.config.risk_model == 'equal_weight':
            position_sizes = self._equal_weight_sizing(predictions, confidence)
        elif self.config.risk_model == 'risk_parity':
            position_sizes = self._risk_parity_sizing(predictions, confidence, features)
        elif self.config.risk_model == 'momentum':
            position_sizes = self._momentum_sizing(predictions, confidence, features)
        else:
            position_sizes = self._equal_weight_sizing(predictions, confidence)
        
        # Apply risk constraints
        position_sizes = self._apply_risk_constraints(position_sizes, current_prices)
        
        return position_sizes
    
    def _equal_weight_sizing(self, predictions: dict[str, int], confidence: dict[str, float]) -> dict[str, float]:
        """Equal weight position sizing"""
        
        position_sizes = {}
        
        # Filter for actionable predictions (BUY/SELL with sufficient confidence)
        actionable_symbols = []
        for symbol, pred in predictions.items():
            if pred in [0, 1] and confidence.get(symbol, 0) > 0.3:  # BUY or SELL with >30% confidence
                actionable_symbols.append(symbol)
        
        if actionable_symbols:
            # Equal weight allocation
            weight_per_symbol = min(1.0 / len(actionable_symbols), self.config.max_position_size)
            
            for symbol in actionable_symbols:
                if predictions[symbol] == 0:  # BUY
                    position_sizes[symbol] = weight_per_symbol
                elif predictions[symbol] == 1:  # SELL
                    position_sizes[symbol] = -weight_per_symbol  # Negative for short
        
        return position_sizes
    
    def _risk_parity_sizing(self, predictions: dict[str, int], confidence: dict[str, float], features: pd.DataFrame) -> dict[str, float]:
        """Risk parity position sizing based on volatility"""
        
        position_sizes = {}
        
        # Get volatility estimates for each symbol
        volatilities = {}
        for symbol in predictions:
            # Look for volatility features
            vol_cols = [col for col in features.columns if f'{symbol}_' in col and 'volatility' in col.lower()]
            if vol_cols:
                volatilities[symbol] = features[vol_cols[0]].iloc[-1] if not pd.isna(features[vol_cols[0]].iloc[-1]) else 0.02
            else:
                volatilities[symbol] = 0.02  # Default 2% daily volatility
        
        # Filter for actionable predictions
        actionable_symbols = []
        for symbol, pred in predictions.items():
            if pred in [0, 1] and confidence.get(symbol, 0) > 0.3:
                actionable_symbols.append(symbol)
        
        if actionable_symbols:
            # Risk parity: inverse volatility weighting
            inv_volatilities = {symbol: 1.0 / max(volatilities[symbol], 0.01) for symbol in actionable_symbols}
            total_inv_vol = sum(inv_volatilities.values())
            
            for symbol in actionable_symbols:
                weight = inv_volatilities[symbol] / total_inv_vol
                weight = min(weight, self.config.max_position_size)  # Cap at max position size
                
                if predictions[symbol] == 0:  # BUY
                    position_sizes[symbol] = weight
                elif predictions[symbol] == 1:  # SELL
                    position_sizes[symbol] = -weight
        
        return position_sizes
    
    def _momentum_sizing(self, predictions: dict[str, int], confidence: dict[str, float], features: pd.DataFrame) -> dict[str, float]:
        """Momentum-based position sizing"""
        
        position_sizes = {}
        
        # Get momentum scores for each symbol
        momentum_scores = {}
        for symbol in predictions:
            # Look for momentum features
            mom_cols = [col for col in features.columns if f'{symbol}_' in col and 'momentum' in col.lower()]
            if mom_cols:
                momentum_scores[symbol] = features[mom_cols[0]].iloc[-1] if not pd.isna(features[mom_cols[0]].iloc[-1]) else 0.0
            else:
                momentum_scores[symbol] = 0.0
        
        # Filter for actionable predictions
        actionable_symbols = []
        for symbol, pred in predictions.items():
            if pred in [0, 1] and confidence.get(symbol, 0) > 0.3:
                actionable_symbols.append(symbol)
        
        if actionable_symbols:
            # Momentum weighting: higher momentum = larger position
            total_momentum = sum(abs(momentum_scores[symbol]) for symbol in actionable_symbols)
            
            if total_momentum > 0:
                for symbol in actionable_symbols:
                    momentum_weight = abs(momentum_scores[symbol]) / total_momentum
                    momentum_weight = min(momentum_weight, self.config.max_position_size)
                    
                    if predictions[symbol] == 0:  # BUY
                        position_sizes[symbol] = momentum_weight
                    elif predictions[symbol] == 1:  # SELL
                        position_sizes[symbol] = -momentum_weight
        
        return position_sizes
    
    def _apply_risk_constraints(self, position_sizes: dict[str, float], current_prices: dict[str, float]) -> dict[str, float]:
        """Apply risk constraints to position sizes"""
        
        # 1. Individual position size limits
        constrained_sizes = {}
        for symbol, size in position_sizes.items():
            constrained_sizes[symbol] = np.clip(size, -self.config.max_position_size, self.config.max_position_size)
        
        # 2. Portfolio risk limits
        total_risk = sum(abs(size) for size in constrained_sizes.values())
        if total_risk > self.config.max_portfolio_risk:
            # Scale down all positions proportionally
            scale_factor = self.config.max_portfolio_risk / total_risk
            for symbol in constrained_sizes:
                constrained_sizes[symbol] *= scale_factor
        
        # 3. Maximum number of positions
        if len(constrained_sizes) > self.config.max_positions:
            # Keep only the largest positions
            sorted_positions = sorted(constrained_sizes.items(), key=lambda x: abs(x[1]), reverse=True)
            constrained_sizes = dict(sorted_positions[:self.config.max_positions])
        
        return constrained_sizes
    
    def execute_trades(self, position_sizes: dict[str, float], current_prices: dict[str, float]) -> dict[str, Any]:
        """Execute trades to achieve target position sizes"""
        
        trade_results = {
            'trades_executed': [],
            'total_cost': 0.0,
            'portfolio_value_before': self.total_value,
            'portfolio_value_after': 0.0
        }
        
        for symbol, target_size in position_sizes.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            current_position = self.positions.get(symbol)
            
            # Calculate target quantity
            target_value = target_size * self.total_value
            target_quantity = target_value / current_price
            
            # Calculate current quantity
            current_quantity = current_position.quantity if current_position else 0.0
            
            # Calculate trade quantity
            trade_quantity = target_quantity - current_quantity
            
            if abs(trade_quantity) > 0.01:  # Minimum trade size
                # Execute trade
                trade_value = abs(trade_quantity) * current_price
                transaction_cost = trade_value * (self.config.transaction_cost_bps / 10000)
                slippage_cost = trade_value * (self.config.slippage_bps / 10000)
                total_cost = transaction_cost + slippage_cost
                
                # Update cash
                if trade_quantity > 0:  # Buy
                    self.cash -= (trade_value + total_cost)
                else:  # Sell
                    self.cash += (trade_value - total_cost)
                
                # Update position
                if symbol in self.positions:
                    self.positions[symbol].quantity += trade_quantity
                    if self.positions[symbol].quantity == 0:
                        del self.positions[symbol]
                else:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=trade_quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        entry_time=datetime.now(),
                        position_type='long' if trade_quantity > 0 else 'short'
                    )
                
                trade_results['trades_executed'].append({
                    'symbol': symbol,
                    'quantity': trade_quantity,
                    'price': current_price,
                    'value': trade_value,
                    'cost': total_cost
                })
                
                trade_results['total_cost'] += total_cost
        
        # Update portfolio value
        trade_results['portfolio_value_after'] = self.get_portfolio_value(current_prices)
        
        return trade_results
    
    def get_portfolio_summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """Get comprehensive portfolio summary"""
        
        portfolio_value = self.get_portfolio_value(current_prices)
        
        # Calculate returns
        if self.portfolio_history:
            initial_value = self.portfolio_history[0]['total_value']
            total_return = (portfolio_value - initial_value) / initial_value
        else:
            total_return = 0.0
        
        # Position summary
        position_summary = {}
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value = position.quantity * current_prices[symbol]
                position_weight = position_value / portfolio_value
                
                position_summary[symbol] = {
                    'quantity': position.quantity,
                    'current_price': current_prices[symbol],
                    'entry_price': position.entry_price,
                    'value': position_value,
                    'weight': position_weight,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl / (position.quantity * position.entry_price) if position.quantity != 0 else 0
                }
        
        # Risk metrics
        position_weights = [pos['weight'] for pos in position_summary.values()]
        concentration = max(position_weights) if position_weights else 0.0
        diversification = 1.0 - sum(w**2 for w in position_weights)  # Herfindahl index
        
        summary = {
            'total_value': portfolio_value,
            'cash': self.cash,
            'cash_weight': self.cash / portfolio_value,
            'total_return': total_return,
            'num_positions': len(self.positions),
            'concentration': concentration,
            'diversification': diversification,
            'positions': position_summary
        }
        
        return summary
    
    def rebalance_portfolio(self, 
                          predictions: dict[str, int], 
                          confidence: dict[str, float],
                          current_prices: dict[str, float],
                          features: pd.DataFrame) -> dict[str, Any]:
        """Rebalance portfolio based on predictions and risk model"""
        
        # Check if rebalancing is needed
        if not self._should_rebalance():
            return {'rebalanced': False, 'reason': 'Not time for rebalancing'}
        
        # Calculate new position sizes
        position_sizes = self.calculate_position_sizes(predictions, confidence, current_prices, features)
        
        # Execute trades
        trade_results = self.execute_trades(position_sizes, current_prices)
        
        # Update rebalance time
        self.last_rebalance = datetime.now()
        
        # Record portfolio state
        portfolio_summary = self.get_portfolio_summary(current_prices)
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'total_value': portfolio_summary['total_value'],
            'total_return': portfolio_summary['total_return'],
            'num_positions': portfolio_summary['num_positions'],
            'trades_executed': len(trade_results['trades_executed'])
        })
        
        return {
            'rebalanced': True,
            'trade_results': trade_results,
            'portfolio_summary': portfolio_summary
        }
    
    def _should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced"""
        
        if self.last_rebalance is None:
            return True
        
        if self.config.rebalance_frequency == 'daily':
            return True  # Always rebalance daily
        if self.config.rebalance_frequency == 'weekly':
            return (datetime.now() - self.last_rebalance).days >= 7
        if self.config.rebalance_frequency == 'monthly':
            return (datetime.now() - self.last_rebalance).days >= 30
        
        return False
    
    def get_performance_metrics(self) -> dict[str, Any]:
        """Get portfolio performance metrics"""
        
        if len(self.portfolio_history) < 2:
            return {'error': 'Insufficient history'}
        
        # Calculate returns
        values = [h['total_value'] for h in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()
        
        # Performance metrics with numerical safety
        total_return = (values[-1] - values[0]) / values[0]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Safe Sharpe ratio calculation
        if len(returns) < 2 or volatility < 1e-12:
            sharpe_ratio = float("nan")
        else:
            sharpe_ratio = annualized_return / (volatility + 1e-12)
        
        # Drawdown
        peak = pd.Series(values).expanding().max()
        drawdown = (pd.Series(values) - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': sum(h['trades_executed'] for h in self.portfolio_history),
            'avg_position_count': np.mean([h['num_positions'] for h in self.portfolio_history])
        }
