"""
Enhanced Reward Calculator

Implements sophisticated reward shaping for trading systems:
r_t = ΔPortfolioValue_t - λ_var·σ²_t - λ_dd·max_dd_t - λ_turn·turnover_t - costs_t

Based on the framework for continuous control, high-risk domain, offline-heavy data regime.
"""

import numpy as np
from typing import Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EnhancedTradeResult:
    """Enhanced trade result with risk metrics"""
    timestamp: datetime
    action: str
    position_size: float
    price_change: float
    reward: float
    features: dict[str, float]
    market_context: dict[str, float]
    success: bool
    risk_metrics: dict[str, float]
    operational_costs: dict[str, float]


class EnhancedRewardCalculator:
    """
    Enhanced reward calculator implementing sophisticated reward shaping
    
    Key features:
    - Risk-adjusted rewards with variance, drawdown, and turnover penalties
    - Operational cost modeling (commissions, slippage, borrowing)
    - Portfolio risk budgeting with target Sharpe ratios
    - Offline RL compatibility with clear reward signals
    """
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        
        # Risk parameters (λ coefficients)
        self.lambda_variance = config.get('lambda_variance', 0.1)
        self.lambda_drawdown = config.get('lambda_drawdown', 0.2)
        self.lambda_turnover = config.get('lambda_turnover', 0.05)
        self.lambda_leverage = config.get('lambda_leverage', 0.1)
        
        # Risk budgeting
        self.target_sharpe = config.get('target_sharpe', 1.5)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.15)
        self.volatility_threshold = config.get('volatility_threshold', 0.05)
        
        # Operational costs
        self.transaction_cost_bps = config.get('transaction_cost_bps', 10)
        self.slippage_bps = config.get('slippage_bps', 5)
        self.borrowing_cost_bps = config.get('borrowing_cost_bps', 2)
        
        # Reward thresholds
        self.profit_threshold_high = config.get('profit_threshold_high', 0.005)  # 0.5%
        self.profit_threshold_low = config.get('profit_threshold_low', 0.002)   # 0.2%
        self.loss_threshold_high = config.get('loss_threshold_high', -0.005)    # -0.5%
        self.loss_threshold_low = config.get('loss_threshold_low', -0.002)      # -0.2%
        
        # Trade tracking
        self.successful_trades: list[EnhancedTradeResult] = []
        self.failed_trades: list[EnhancedTradeResult] = []
        self.portfolio_history: list[dict[str, float]] = []
        
        # Risk metrics tracking
        self.rolling_volatility = []
        self.rolling_returns = []
        self.max_drawdown_so_far = 0.0
        self.peak_portfolio_value = 0.0
    
    def calculate_reward(self,
                        action: str,
                        price_change: float,
                        position_size: float,
                        portfolio_value: float,
                        features: dict[str, float],
                        market_context: dict[str, float],
                        timestamp: datetime) -> float:
        """
        Calculate sophisticated reward with risk shaping and operational costs
        
        Implements: r_t = ΔPortfolioValue_t - λ_var·σ²_t - λ_dd·max_dd_t - λ_turn·turnover_t - costs_t
        """
        
        # Skip if no position
        if position_size == 0:
            return 0.0

        # 1. Base PnL (realized returns after costs)
        if action == 'BUY':
            raw_pnl = price_change * position_size
        elif action == 'SELL':
            raw_pnl = -price_change * position_size  # Profit when price falls
        else:  # HOLD
            raw_pnl = 0

        # 2. Operational costs
        operational_costs = self._calculate_operational_costs(action, position_size, market_context)
        
        # 3. Risk terms
        risk_metrics = self._calculate_risk_metrics(portfolio_value, market_context)
        
        # 4. Risk penalties
        var_penalty = self.lambda_variance * (risk_metrics['volatility'] ** 2)
        dd_penalty = self.lambda_drawdown * risk_metrics['max_drawdown']
        turnover_penalty = self.lambda_turnover * abs(position_size)
        leverage_penalty = self._calculate_leverage_penalty(position_size)
        
        # 5. Composite reward
        net_pnl = raw_pnl - operational_costs['total_cost']
        reward = net_pnl - var_penalty - dd_penalty - turnover_penalty - leverage_penalty
        
        # 6. Risk budgeting normalization
        reward = self._apply_risk_budgeting(reward, risk_metrics)
        
        # 7. Clear reward signals (avoid tiny rewards that confuse models)
        reward = self._apply_reward_thresholds(reward)
        
        # 8. Update portfolio history and risk metrics
        self._update_portfolio_history(portfolio_value, timestamp)
        
        # 9. Track trade result for strategy analysis
        success = reward > 0.1  # Clear success threshold
        trade_result = EnhancedTradeResult(
            timestamp=timestamp,
            action=action,
            position_size=position_size,
            price_change=price_change,
            reward=reward,
            features=features.copy(),
            market_context=market_context.copy(),
            success=success,
            risk_metrics=risk_metrics,
            operational_costs=operational_costs
        )

        if success:
            self.successful_trades.append(trade_result)
        else:
            self.failed_trades.append(trade_result)

        return reward
    
    def _calculate_operational_costs(self, action: str, position_size: float, 
                                   market_context: dict[str, float]) -> dict[str, float]:
        """Calculate operational costs (commissions, slippage, borrowing)"""
        
        # Transaction costs
        transaction_cost = abs(position_size) * self.transaction_cost_bps / 10000
        
        # Slippage costs
        slippage_cost = abs(position_size) * self.slippage_bps / 10000
        
        # Borrowing costs (for short positions)
        borrowing_cost = 0.0
        if action == 'SELL' and position_size > 0:
            borrowing_cost = position_size * self.borrowing_cost_bps / 10000
        
        # Market impact (estimated from volatility)
        volatility = market_context.get('volatility', 0.02)
        market_impact = abs(position_size) * volatility * 0.1  # 10% of volatility
        
        total_cost = transaction_cost + slippage_cost + borrowing_cost + market_impact
        
        return {
            'transaction_cost': transaction_cost,
            'slippage_cost': slippage_cost,
            'borrowing_cost': borrowing_cost,
            'market_impact': market_impact,
            'total_cost': total_cost
        }
    
    def _calculate_risk_metrics(self, portfolio_value: float, 
                              market_context: dict[str, float]) -> dict[str, float]:
        """Calculate risk metrics for reward shaping"""
        
        # Current volatility
        volatility = market_context.get('volatility', 0.02)
        
        # Rolling volatility (if we have history)
        if len(self.rolling_volatility) > 0:
            rolling_vol = np.mean(self.rolling_volatility[-20:])  # 20-day rolling
        else:
            rolling_vol = volatility
        
        # Max drawdown
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            self.max_drawdown_so_far = 0.0
        else:
            current_dd = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            self.max_drawdown_so_far = max(self.max_drawdown_so_far, current_dd)
        
        # Tail risk (CVaR approximation)
        if len(self.rolling_returns) > 0:
            returns_array = np.array(self.rolling_returns[-20:])
            cvar_95 = np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)])
        else:
            cvar_95 = -0.02  # Default 2% tail risk
        
        return {
            'volatility': volatility,
            'rolling_volatility': rolling_vol,
            'max_drawdown': self.max_drawdown_so_far,
            'cvar_95': cvar_95,
            'portfolio_value': portfolio_value
        }
    
    def _calculate_leverage_penalty(self, position_size: float) -> float:
        """Calculate leverage penalty for excessive position sizes"""
        
        if position_size <= 1.0:
            return 0.0
        
        # Penalty increases quadratically with leverage
        excess_leverage = position_size - 1.0
        penalty = self.lambda_leverage * (excess_leverage ** 2)
        
        return penalty
    
    def _apply_risk_budgeting(self, reward: float, risk_metrics: dict[str, float]) -> float:
        """Apply risk budgeting normalization"""
        
        volatility = risk_metrics['volatility']
        
        if volatility > 0:
            # Risk-adjusted reward
            risk_adjusted_reward = reward / volatility
            
            # Check against target Sharpe ratio
            if risk_adjusted_reward < self.target_sharpe * 0.1:  # Below target
                reward *= 0.5  # Penalize low risk-adjusted returns
            
            # Check against volatility threshold
            if volatility > self.volatility_threshold:
                reward *= 0.7  # Reduce reward in high volatility
        
        # Check against drawdown limit
        if risk_metrics['max_drawdown'] > self.max_drawdown_limit:
            reward *= 0.3  # Heavy penalty for exceeding drawdown limit
        
        return reward
    
    def _apply_reward_thresholds(self, reward: float) -> float:
        """Apply clear reward thresholds to avoid tiny rewards that confuse models"""
        
        if reward > self.profit_threshold_high:
            return 1.0  # Strong positive signal
        if reward > self.profit_threshold_low:
            return 0.5  # Moderate positive signal
        if reward < self.loss_threshold_high:
            return -1.0  # Strong negative signal
        if reward < self.loss_threshold_low:
            return -0.5  # Moderate negative signal
        return 0.0  # Neutral (noise)
    
    def _update_portfolio_history(self, portfolio_value: float, timestamp: datetime):
        """Update portfolio history for risk calculations"""
        
        # Calculate return
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1]['value']
            if prev_value > 0:
                return_pct = (portfolio_value - prev_value) / prev_value
                self.rolling_returns.append(return_pct)
                
                # Keep only last 100 returns
                if len(self.rolling_returns) > 100:
                    self.rolling_returns = self.rolling_returns[-100:]
        
        # Update portfolio history
        self.portfolio_history.append({
            'timestamp': timestamp,
            'value': portfolio_value
        })
        
        # Keep only last 100 entries
        if len(self.portfolio_history) > 100:
            self.portfolio_history = self.portfolio_history[-100:]
    
    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        if not self.successful_trades and not self.failed_trades:
            return {'error': 'No trades recorded'}
        
        all_trades = self.successful_trades + self.failed_trades
        
        # Basic metrics
        total_trades = len(all_trades)
        successful_trades = len(self.successful_trades)
        win_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        # Reward metrics
        rewards = [trade.reward for trade in all_trades]
        avg_reward = np.mean(rewards)
        reward_std = np.std(rewards)
        sharpe_ratio = avg_reward / reward_std if reward_std > 0 else 0
        
        # Risk metrics
        max_dd = self.max_drawdown_so_far
        avg_volatility = np.mean([trade.risk_metrics['volatility'] for trade in all_trades])
        
        # Cost analysis
        total_costs = [trade.operational_costs['total_cost'] for trade in all_trades]
        avg_cost = np.mean(total_costs)
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'avg_volatility': avg_volatility,
            'avg_operational_cost': avg_cost,
            'risk_budget_utilization': max_dd / self.max_drawdown_limit if self.max_drawdown_limit > 0 else 0
        }
    
    def reset(self):
        """Reset the calculator state"""
        self.successful_trades.clear()
        self.failed_trades.clear()
        self.portfolio_history.clear()
        self.rolling_volatility.clear()
        self.rolling_returns.clear()
        self.max_drawdown_so_far = 0.0
        self.peak_portfolio_value = 0.0
