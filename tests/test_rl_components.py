"""
Tests for RL components

Tests reward calculation, state management, strategy analysis, and Q-learning.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock

from core.rl import (
    TradingRewardCalculator,
    TradingStateManager,
    StrategyAnalyzer,
    QLearningTrader,
    QLearningConfig
)
from core.rl.reward_calculator import TradeResult


class TestTradingRewardCalculator:
    """Test reward calculation functionality"""
    
    def setup_method(self):
        self.config = {
            'risk_free_rate': 0.02,
            'transaction_cost_bps': 10,
            'min_reward_threshold': 0.001
        }
        self.calculator = TradingRewardCalculator(self.config)
    
    def test_calculate_reward_buy_positive(self):
        """Test reward calculation for profitable BUY"""
        reward = self.calculator.calculate_reward(
            action='BUY',
            price_change=0.05,  # 5% gain
            position_size=0.1,
            portfolio_value=10000,
            features={'rsi': 30},
            market_context={'volatility': 0.02},
            timestamp=datetime.now()
        )
        
        # Should be positive for profitable trade
        assert reward > 0
        assert len(self.calculator.successful_trades) == 1
    
    def test_calculate_reward_sell_negative(self):
        """Test reward calculation for profitable SELL (price falls)"""
        reward = self.calculator.calculate_reward(
            action='SELL',
            price_change=-0.03,  # 3% loss (good for short)
            position_size=0.1,
            portfolio_value=10000,
            features={'rsi': 70},
            market_context={'volatility': 0.02},
            timestamp=datetime.now()
        )
        
        # Should be positive for profitable short
        assert reward > 0
    
    def test_calculate_reward_hold(self):
        """Test reward calculation for HOLD"""
        reward = self.calculator.calculate_reward(
            action='HOLD',
            price_change=0.02,
            position_size=0.0,
            portfolio_value=10000,
            features={'rsi': 50},
            market_context={'volatility': 0.02},
            timestamp=datetime.now()
        )
        
        # HOLD should have minimal reward (just transaction cost)
        assert abs(reward) < 0.001
    
    def test_transaction_cost_penalty(self):
        """Test that transaction costs are applied"""
        reward_with_cost = self.calculator.calculate_reward(
            action='BUY',
            price_change=0.0,  # No price change
            position_size=0.1,
            portfolio_value=10000,
            features={},
            market_context={},
            timestamp=datetime.now()
        )
        
        # Should be negative due to transaction costs
        assert reward_with_cost < 0
    
    def test_analyze_strategy_patterns(self):
        """Test strategy pattern analysis"""
        # Add some mock trades
        self.calculator.successful_trades = [
            TradeResult(
                timestamp=datetime.now(),
                action='BUY',
                position_size=0.1,
                price_change=0.05,
                reward=0.004,
                features={'rsi': 30, 'momentum': 0.8},
                market_context={'volatility': 0.02},
                success=True
            )
        ]
        
        self.calculator.failed_trades = [
            TradeResult(
                timestamp=datetime.now(),
                action='BUY',
                position_size=0.1,
                price_change=-0.02,
                reward=-0.003,
                features={'rsi': 70, 'momentum': -0.2},
                market_context={'volatility': 0.03},
                success=False
            )
        ]
        
        analysis = self.calculator.analyze_strategy_patterns()
        
        assert 'success_rate' in analysis
        assert analysis['success_rate'] == 0.5
        assert 'feature_analysis' in analysis
        assert 'action_analysis' in analysis


class TestTradingStateManager:
    """Test state management functionality"""
    
    def setup_method(self):
        self.feature_builder = Mock()
        self.feature_builder.build_features.return_value = pd.DataFrame({
            'rsi': [30, 35, 40],
            'momentum': [0.8, 0.6, 0.4]
        })
        
        self.state_manager = TradingStateManager(self.feature_builder)
    
    def test_get_state(self):
        """Test state generation"""
        # Mock price data
        price_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        state = self.state_manager.get_state('SPY', datetime.now(), price_data)
        
        assert 'features' in state
        assert 'portfolio' in state
        assert 'market_context' in state
        assert 'timestamp' in state
        assert 'symbol' in state
    
    def test_get_state_vector(self):
        """Test state vector conversion"""
        state = {
            'features': {'rsi': 30, 'momentum': 0.8},
            'portfolio': {
                'cash': 10000,
                'position': 0.1,
                'unrealized_pnl': 100,
                'total_value': 10100,
                'max_drawdown': 0.05,
                'sharpe_ratio': 1.2,
                'volatility': 0.02
            },
            'market_context': {
                'volatility': 0.02,
                'trend': 0.1,
                'volume_ratio': 1.2,
                'market_regime': 'trending',
                'sector_performance': 0.05,
                'correlation_to_spy': 0.8
            }
        }
        
        vector = self.state_manager.get_state_vector(state)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
        assert vector.dtype == np.float32


class TestStrategyAnalyzer:
    """Test strategy analysis functionality"""
    
    def setup_method(self):
        self.config = {
            'min_pattern_frequency': 2,
            'min_success_rate': 0.6,
            'min_confidence': 0.7
        }
        self.analyzer = StrategyAnalyzer(self.config)
    
    def test_analyze_trades_empty(self):
        """Test analysis with no trades"""
        analysis = self.analyzer.analyze_trades([])
        assert 'error' in analysis
    
    def test_analyze_trades_with_data(self):
        """Test analysis with trade data"""
        trades = [
            TradeResult(
                timestamp=datetime.now(),
                action='BUY',
                position_size=0.1,
                price_change=0.05,
                reward=0.004,
                features={'rsi': 30, 'momentum': 0.8},
                market_context={'volatility': 0.02, 'market_regime': 'trending'},
                success=True
            ),
            TradeResult(
                timestamp=datetime.now(),
                action='BUY',
                position_size=0.1,
                price_change=-0.02,
                reward=-0.003,
                features={'rsi': 70, 'momentum': -0.2},
                market_context={'volatility': 0.03, 'market_regime': 'volatile'},
                success=False
            )
        ]
        
        analysis = self.analyzer.analyze_trades(trades)
        
        assert 'patterns' in analysis
        assert 'recommendations' in analysis
        assert 'statistics' in analysis
        assert 'top_patterns' in analysis
    
    def test_find_feature_patterns(self):
        """Test feature pattern detection"""
        # Create mock DataFrame with feature patterns
        df = pd.DataFrame({
            'rsi': [30, 30, 70, 70, 30, 70],
            'momentum': [0.8, 0.9, -0.2, -0.1, 0.7, -0.3],
            'action': ['BUY', 'BUY', 'SELL', 'SELL', 'BUY', 'SELL'],
            'reward': [0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
            'success': [True, True, True, True, True, True]
        })
        
        patterns = self.analyzer._find_feature_patterns(df)
        
        # Should find patterns for RSI and momentum
        assert len(patterns) > 0
        for pattern in patterns:
            assert pattern.success_rate >= 0.6
            assert pattern.frequency >= 2


class TestQLearningTrader:
    """Test Q-learning functionality"""
    
    def setup_method(self):
        self.config = QLearningConfig(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1,
            state_size=10,
            action_size=3
        )
        self.agent = QLearningTrader(self.config)
    
    def test_choose_action(self):
        """Test action selection"""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        action = self.agent.choose_action(state)
        
        assert action in ['BUY', 'SELL', 'HOLD']
    
    def test_update(self):
        """Test Q-table update"""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        next_state = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        
        # Initial Q-values should be zero
        initial_q = self.agent.q_table.copy()
        
        # Update with positive reward
        self.agent.update(state, 'BUY', 0.1, next_state, done=False)
        
        # Q-values should have changed
        assert not np.array_equal(initial_q, self.agent.q_table)
    
    def test_get_q_values(self):
        """Test Q-value retrieval"""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        q_values = self.agent.get_q_values(state)
        
        assert 'BUY' in q_values
        assert 'SELL' in q_values
        assert 'HOLD' in q_values
        assert all(isinstance(v, float) for v in q_values.values())
    
    def test_get_best_action(self):
        """Test best action selection"""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Manually set Q-values to test
        state_idx = self.agent._discretize_state(state)
        self.agent.q_table[state_idx, 0] = 0.1  # BUY
        self.agent.q_table[state_idx, 1] = 0.3  # SELL (highest)
        self.agent.q_table[state_idx, 2] = 0.2  # HOLD
        
        best_action = self.agent.get_best_action(state)
        assert best_action == 'SELL'
    
    def test_get_action_confidence(self):
        """Test action confidence calculation"""
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        confidence = self.agent.get_action_confidence(state, 'BUY')
        
        assert 0 <= confidence <= 1
    
    def test_get_training_stats(self):
        """Test training statistics"""
        # Add some training history
        self.agent.training_history = [
            {'reward': 0.1, 'q_value': 0.05, 'epsilon': 0.1},
            {'reward': 0.2, 'q_value': 0.1, 'epsilon': 0.09}
        ]
        
        stats = self.agent.get_training_stats()
        
        assert 'total_steps' in stats
        assert 'current_epsilon' in stats
        assert 'avg_reward' in stats
        assert 'q_table_stats' in stats
    
    def test_save_load_model(self):
        """Test model saving and loading"""
        # Set some Q-values
        self.agent.q_table[0, 0] = 0.5
        self.agent.q_table[1, 1] = 0.7
        
        # Save model
        self.agent.save_model('test_model.pkl')
        
        # Create new agent and load
        new_agent = QLearningTrader(self.config)
        new_agent.load_model('test_model.pkl')
        
        # Check that Q-values were loaded correctly
        assert np.array_equal(self.agent.q_table, new_agent.q_table)
        assert self.agent.epsilon == new_agent.epsilon
        
        # Clean up
        import os
        os.remove('test_model.pkl')


if __name__ == "__main__":
    pytest.main([__file__])


