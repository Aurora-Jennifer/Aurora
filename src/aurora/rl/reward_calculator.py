"""
Trading Reward Calculator

Implements reward functions that optimize for positive returns and track
what contributes to successful trades.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TradeResult:
    """Result of a single trade decision"""
    timestamp: datetime
    action: str  # 'BUY', 'SELL', 'HOLD'
    position_size: float
    price_change: float
    reward: float
    features: dict[str, float]
    market_context: dict[str, float]
    success: bool  # True if positive return


class TradingRewardCalculator:
    """
    Calculates rewards based on positive returns and tracks strategy patterns
    """
    
    def __init__(self, config: dict[str, Any]):
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.transaction_cost_bps = config.get('transaction_cost_bps', 10)
        self.min_reward_threshold = config.get('min_reward_threshold', 0.001)
        
        # Strategy analysis tracking
        self.successful_trades = []
        self.failed_trades = []
        self.strategy_patterns = {}
        
    def calculate_reward(self, 
                        action: str,
                        price_change: float, 
                        position_size: float,
                        portfolio_value: float,
                        features: dict[str, float],
                        market_context: dict[str, float],
                        timestamp: datetime) -> float:
        """
        Calculate meaningful reward based on clear profit/loss signals
        
        Args:
            action: 'BUY', 'SELL', or 'HOLD'
            price_change: Percentage change in price
            position_size: Size of position (0-1)
            portfolio_value: Current portfolio value
            features: Technical indicators at decision time
            market_context: Market conditions (volatility, trend, etc.)
            timestamp: When decision was made
            
        Returns:
            Reward value (positive for good decisions, negative for bad)
        """
        
        # Skip if no position
        if position_size == 0:
            return 0.0
        
        # Calculate raw P&L
        if action == 'BUY':
            raw_pnl = price_change * position_size
        elif action == 'SELL':
            raw_pnl = -price_change * position_size  # Profit when price falls
        else:  # HOLD
            raw_pnl = 0
            
        # Transaction cost (more realistic)
        transaction_cost = abs(position_size) * self.transaction_cost_bps / 10000
        
        # Risk-adjusted reward with clear thresholds
        net_pnl = raw_pnl - transaction_cost
        
        # Clear reward signals (avoid tiny rewards that confuse models)
        if net_pnl > 0.005:  # 0.5% profit
            reward = 1.0  # Strong positive signal
        elif net_pnl > 0.002:  # 0.2% profit
            reward = 0.5  # Moderate positive signal
        elif net_pnl < -0.005:  # 0.5% loss
            reward = -1.0  # Strong negative signal
        elif net_pnl < -0.002:  # 0.2% loss
            reward = -0.5  # Moderate negative signal
        else:
            reward = 0.0  # Neutral (noise)
        
        # Anti-greed penalty: penalize excessive position sizes
        if position_size > 0.5:  # More than 50% position
            greed_penalty = (position_size - 0.5) * 0.5
            reward -= greed_penalty
        
        # Volatility penalty: avoid trading in extreme volatility
        volatility = market_context.get('volatility', 0.02)
        if volatility > 0.05:  # High volatility
            reward *= 0.5  # Reduce reward in volatile conditions
        
        # Track trade result for strategy analysis
        success = reward > 0.1  # Clear success threshold
        trade_result = TradeResult(
            timestamp=timestamp,
            action=action,
            position_size=position_size,
            price_change=price_change,
            reward=reward,
            features=features.copy(),
            market_context=market_context.copy(),
            success=success
        )
        
        if success:
            self.successful_trades.append(trade_result)
        else:
            self.failed_trades.append(trade_result)
            
        return reward
    
    def analyze_strategy_patterns(self) -> dict[str, Any]:
        """
        Analyze what patterns lead to successful trades
        
        Returns:
            Dictionary with strategy insights
        """
        if not self.successful_trades:
            return {"error": "No successful trades to analyze"}
            
        analysis = {
            "total_trades": len(self.successful_trades) + len(self.failed_trades),
            "success_rate": len(self.successful_trades) / (len(self.successful_trades) + len(self.failed_trades)),
            "avg_successful_reward": np.mean([t.reward for t in self.successful_trades]),
            "avg_failed_reward": np.mean([t.reward for t in self.failed_trades]),
            "feature_analysis": {},
            "action_analysis": {},
            "market_condition_analysis": {}
        }
        
        # Analyze features that contribute to success
        if self.successful_trades:
            successful_features = pd.DataFrame([t.features for t in self.successful_trades])
            failed_features = pd.DataFrame([t.features for t in self.failed_trades])
            
            for feature in successful_features.columns:
                if feature in failed_features.columns:
                    success_mean = successful_features[feature].mean()
                    failed_mean = failed_features[feature].mean()
                    analysis["feature_analysis"][feature] = {
                        "success_avg": success_mean,
                        "failed_avg": failed_mean,
                        "difference": success_mean - failed_mean,
                        "contribution_score": abs(success_mean - failed_mean) / (abs(success_mean) + abs(failed_mean) + 1e-8)
                    }
        
        # Analyze actions
        action_counts = {}
        for trade in self.successful_trades:
            action_counts[trade.action] = action_counts.get(trade.action, 0) + 1
        analysis["action_analysis"] = action_counts
        
        # Analyze market conditions
        if self.successful_trades:
            successful_context = pd.DataFrame([t.market_context for t in self.successful_trades])
            failed_context = pd.DataFrame([t.market_context for t in self.failed_trades])
            
            for context in successful_context.columns:
                if context in failed_context.columns:
                    success_mean = successful_context[context].mean()
                    failed_mean = failed_context[context].mean()
                    analysis["market_condition_analysis"][context] = {
                        "success_avg": success_mean,
                        "failed_avg": failed_mean,
                        "difference": success_mean - failed_mean
                    }
        
        return analysis
    
    def get_top_strategies(self, top_n: int = 5) -> dict[str, Any]:
        """
        Identify the most profitable strategy patterns
        
        Args:
            top_n: Number of top strategies to return
            
        Returns:
            Dictionary with top strategy patterns
        """
        analysis = self.analyze_strategy_patterns()
        
        if "error" in analysis:
            return analysis
            
        # Sort features by contribution score
        feature_contributions = []
        for feature, data in analysis["feature_analysis"].items():
            feature_contributions.append({
                "feature": feature,
                "contribution_score": data["contribution_score"],
                "success_avg": data["success_avg"],
                "failed_avg": data["failed_avg"]
            })
        
        feature_contributions.sort(key=lambda x: x["contribution_score"], reverse=True)
        
        return {
            "top_features": feature_contributions[:top_n],
            "success_rate": analysis["success_rate"],
            "avg_reward": analysis["avg_successful_reward"],
            "total_trades": analysis["total_trades"]
        }
    
    def reset_analysis(self):
        """Reset all tracking data"""
        self.successful_trades = []
        self.failed_trades = []
        self.strategy_patterns = {}
