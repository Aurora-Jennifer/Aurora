"""
Profit-Optimized Trading Model

Implements models that actually learn to maximize trading profits,
not just predict training data actions.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


@dataclass
class TradingDecision:
    """Represents a trading decision with expected profit"""
    action: int  # 0=BUY, 1=SELL, 2=HOLD
    confidence: float
    expected_profit: float
    position_size: float


class ProfitOptimizedModel:
    """
    Model that learns to predict expected profits for each action,
    then selects the action with highest expected profit.
    """
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.action_names = ['BUY', 'SELL', 'HOLD']
        self.is_trained = False
        
        # Initialize models for each action
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize profit prediction models for each action"""
        
        # Random Forest for each action
        self.models['rf_buy'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf_sell'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf_hold'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost for each action
        self.models['xgb_buy'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.models['xgb_sell'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.models['xgb_hold'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    def fit_profit_based(self, X: np.ndarray, rewards: np.ndarray, 
                        actions: np.ndarray, market_context: list[dict]) -> None:
        """
        Train models to predict expected profits for each action
        
        Args:
            X: Feature matrix
            rewards: Actual rewards achieved
            actions: Actions taken (0=BUY, 1=SELL, 2=HOLD)
            market_context: Market context for each sample
        """
        
        print("Training profit-optimized models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models for each action
        for action_idx, action_name in enumerate(self.action_names):
            print(f"Training models for {action_name}...")
            
            # Get samples where this action was taken
            action_mask = (actions == action_idx)
            if np.sum(action_mask) < 5:  # Need at least 5 samples
                print(f"  Warning: Only {np.sum(action_mask)} samples for {action_name}, skipping")
                continue
            
            X_action = X_scaled[action_mask]
            y_action = rewards[action_mask]
            
            # Train Random Forest
            try:
                self.models[f'rf_{action_name.lower()}'].fit(X_action, y_action)
                print(f"  RF {action_name}: {np.sum(action_mask)} samples, avg reward: {np.mean(y_action):.4f}")
            except Exception as e:
                print(f"  RF {action_name} failed: {e}")
            
            # Train XGBoost
            try:
                self.models[f'xgb_{action_name.lower()}'].fit(X_action, y_action)
                print(f"  XGB {action_name}: {np.sum(action_mask)} samples, avg reward: {np.mean(y_action):.4f}")
            except Exception as e:
                print(f"  XGB {action_name} failed: {e}")
        
        self.is_trained = True
        print("Profit-optimized training completed!")
    
    def predict_profits(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict expected profits for each action
        
        Returns:
            Tuple of (buy_profits, sell_profits, hold_profits)
        """
        
        if not self.is_trained:
            # Return neutral predictions if not trained
            return np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        
        # Predict profits for each action
        buy_profits = np.zeros(len(X))
        sell_profits = np.zeros(len(X))
        hold_profits = np.zeros(len(X))
        
        # Use ensemble of RF and XGB for each action
        for action_name in ['buy', 'sell', 'hold']:
            rf_model = self.models.get(f'rf_{action_name}')
            xgb_model = self.models.get(f'xgb_{action_name}')
            
            if rf_model is not None and hasattr(rf_model, 'predict'):
                try:
                    rf_pred = rf_model.predict(X_scaled)
                except:
                    rf_pred = np.zeros(len(X))
            else:
                rf_pred = np.zeros(len(X))
            
            if xgb_model is not None and hasattr(xgb_model, 'predict'):
                try:
                    xgb_pred = xgb_model.predict(X_scaled)
                except:
                    xgb_pred = np.zeros(len(X))
            else:
                xgb_pred = np.zeros(len(X))
            
            # Ensemble prediction (average of RF and XGB)
            ensemble_pred = (rf_pred + xgb_pred) / 2
            
            if action_name == 'buy':
                buy_profits = ensemble_pred
            elif action_name == 'sell':
                sell_profits = ensemble_pred
            else:  # hold
                hold_profits = ensemble_pred
        
        return buy_profits, sell_profits, hold_profits
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict optimal actions based on expected profits
        
        Returns:
            Tuple of (actions, confidence_scores)
        """
        
        # Get expected profits for each action
        buy_profits, sell_profits, hold_profits = self.predict_profits(X)
        
        # Stack profits for each action
        profit_matrix = np.column_stack([buy_profits, sell_profits, hold_profits])
        
        # Select action with highest expected profit
        optimal_actions = np.argmax(profit_matrix, axis=1)
        
        # Calculate confidence as the difference between best and second-best profit
        sorted_profits = np.sort(profit_matrix, axis=1)
        confidence = sorted_profits[:, -1] - sorted_profits[:, -2]
        
        # Normalize confidence to [0, 1]
        confidence = np.clip(confidence, 0, 1)
        
        return optimal_actions, confidence
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from the models"""
        
        importance = {}
        
        # Get importance from Random Forest models
        for action_name in ['buy', 'sell', 'hold']:
            rf_model = self.models.get(f'rf_{action_name}')
            if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
                importance[f'rf_{action_name}'] = rf_model.feature_importances_.tolist()
        
        return importance
    
    def analyze_profit_potential(self, X: np.ndarray) -> dict[str, Any]:
        """
        Analyze the profit potential of different actions
        
        Returns:
            Dictionary with profit analysis
        """
        
        buy_profits, sell_profits, hold_profits = self.predict_profits(X)
        
        analysis = {
            'avg_buy_profit': np.mean(buy_profits),
            'avg_sell_profit': np.mean(sell_profits),
            'avg_hold_profit': np.mean(hold_profits),
            'max_buy_profit': np.max(buy_profits),
            'max_sell_profit': np.max(sell_profits),
            'max_hold_profit': np.max(hold_profits),
            'profitable_buy_opportunities': np.sum(buy_profits > 0.01),
            'profitable_sell_opportunities': np.sum(sell_profits > 0.01),
            'profitable_hold_opportunities': np.sum(hold_profits > 0.01),
            'total_samples': len(X)
        }
        
        return analysis


class ExplorationModel:
    """
    Model that balances exploration (trying new strategies) 
    with exploitation (using known profitable strategies)
    """
    
    def __init__(self, base_model: ProfitOptimizedModel, config: dict[str, Any]):
        self.base_model = base_model
        self.config = config
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.exploration_decay = config.get('exploration_decay', 0.99)
        self.min_exploration_rate = config.get('min_exploration_rate', 0.01)
        
        # Track performance for adaptive exploration
        self.action_performance = {'BUY': [], 'SELL': [], 'HOLD': []}
        self.total_trades = 0
        self.successful_trades = 0
    
    def predict_with_exploration(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict actions with exploration vs exploitation balance
        
        Returns:
            Tuple of (actions, confidence_scores)
        """
        
        # Get base model predictions
        optimal_actions, confidence = self.base_model.predict_with_confidence(X)
        
        # Add exploration
        exploration_mask = np.random.random(len(X)) < self.exploration_rate
        
        if np.any(exploration_mask):
            # Randomly explore other actions
            num_explore = np.sum(exploration_mask)
            random_actions = np.random.randint(0, 3, num_explore)
            optimal_actions[exploration_mask] = random_actions
            
            # Lower confidence for exploratory actions
            confidence[exploration_mask] *= 0.5
        
        return optimal_actions, confidence
    
    def update_performance(self, actions: np.ndarray, rewards: np.ndarray):
        """Update performance tracking for adaptive exploration"""
        
        self.total_trades += len(actions)
        self.successful_trades += np.sum(rewards > 0)
        
        # Update action-specific performance
        for i, action in enumerate(actions):
            action_name = self.base_model.action_names[action]
            self.action_performance[action_name].append(rewards[i])
        
        # Decay exploration rate based on performance
        if self.total_trades > 100:
            success_rate = self.successful_trades / self.total_trades
            if success_rate > 0.6:  # Good performance, reduce exploration
                self.exploration_rate *= self.exploration_decay
            elif success_rate < 0.4:  # Poor performance, increase exploration
                self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
            
            # Ensure exploration rate stays within bounds
            self.exploration_rate = max(self.min_exploration_rate, 
                                      min(0.3, self.exploration_rate))
    
    def get_exploration_stats(self) -> dict[str, Any]:
        """Get exploration statistics"""
        
        stats = {
            'exploration_rate': self.exploration_rate,
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / max(1, self.total_trades),
            'action_performance': {}
        }
        
        for action_name, rewards in self.action_performance.items():
            if rewards:
                stats['action_performance'][action_name] = {
                    'count': len(rewards),
                    'avg_reward': np.mean(rewards),
                    'success_rate': np.sum(np.array(rewards) > 0) / len(rewards)
                }
            else:
                stats['action_performance'][action_name] = {
                    'count': 0,
                    'avg_reward': 0,
                    'success_rate': 0
                }
        
        return stats
