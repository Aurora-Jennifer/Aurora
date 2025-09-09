"""
Trading State Manager

Manages the state representation for RL trading, including features,
portfolio state, and market context.
"""

import pandas as pd
import numpy as np
from typing import Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    position: float  # Current position size (-1 to 1)
    unrealized_pnl: float
    total_value: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float


@dataclass
class MarketContext:
    """Market conditions and context"""
    volatility: float
    trend: float  # -1 to 1 (bearish to bullish)
    volume_ratio: float  # Current vs average volume
    market_regime: str  # 'trending', 'ranging', 'volatile'
    sector_performance: float
    correlation_to_spy: float


class TradingStateManager:
    """
    Manages state representation for RL trading decisions
    """
    
    def __init__(self, feature_builder, portfolio_tracker=None):
        self.feature_builder = feature_builder
        self.portfolio_tracker = portfolio_tracker
        self.state_history = []
        
    def get_state(self, symbol: str, timestamp: datetime, 
                  price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Build complete state representation
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            price_data: Historical price data
            
        Returns:
            Complete state dictionary
        """
        # Build technical features
        features = self.feature_builder.build_features(price_data)
        
        # Get latest feature values
        if not features.empty:
            latest_features = features.iloc[-1].to_dict()
        else:
            latest_features = {}
            
        # Build portfolio state
        portfolio_state = self._get_portfolio_state(symbol, timestamp, price_data)
        
        # Build market context
        market_context = self._get_market_context(symbol, timestamp, price_data)
        
        # Combine into state
        state = {
            'features': latest_features,
            'portfolio': portfolio_state.__dict__,
            'market_context': market_context.__dict__,
            'timestamp': timestamp,
            'symbol': symbol
        }
        
        # Store state history
        self.state_history.append(state)
        
        return state
    
    def _get_portfolio_state(self, symbol: str, timestamp: datetime, 
                           price_data: pd.DataFrame) -> PortfolioState:
        """Get current portfolio state"""
        if self.portfolio_tracker:
            # Use actual portfolio tracker if available
            return self.portfolio_tracker.get_state()
        # Mock portfolio state for testing
        return PortfolioState(
            cash=10000.0,
            position=0.0,
            unrealized_pnl=0.0,
            total_value=10000.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            volatility=0.02
        )
    
    def _get_market_context(self, symbol: str, timestamp: datetime,
                          price_data: pd.DataFrame) -> MarketContext:
        """Calculate market context from price data"""
        if len(price_data) < 20:
            return MarketContext(
                volatility=0.02,
                trend=0.0,
                volume_ratio=1.0,
                market_regime='unknown',
                sector_performance=0.0,
                correlation_to_spy=0.0
            )
            
        # Calculate volatility (20-day rolling)
        returns = price_data['Close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Calculate trend (20-day vs 5-day SMA)
        sma_5 = price_data['Close'].rolling(5).mean()
        sma_20 = price_data['Close'].rolling(20).mean()
        trend = (sma_5.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
        trend = np.clip(trend, -1, 1)  # Normalize to -1 to 1
        
        # Volume ratio
        if 'Volume' in price_data.columns:
            avg_volume = price_data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = price_data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0
            
        # Market regime classification
        if abs(trend) > 0.05:
            regime = 'trending'
        elif volatility > 0.03:
            regime = 'volatile'
        else:
            regime = 'ranging'
            
        # Mock sector performance and correlation (would need sector data)
        sector_performance = 0.0
        correlation_to_spy = 0.0
        
        return MarketContext(
            volatility=volatility,
            trend=trend,
            volume_ratio=volume_ratio,
            market_regime=regime,
            sector_performance=sector_performance,
            correlation_to_spy=correlation_to_spy
        )
    
    def get_state_vector(self, state: dict[str, Any]) -> np.ndarray:
        """
        Convert state dictionary to numerical vector for RL algorithms
        
        Args:
            state: State dictionary
            
        Returns:
            Numerical state vector
        """
        vector = []
        
        # Add features
        features = state['features']
        for feature_name in sorted(features.keys()):
            value = features[feature_name]
            if pd.isna(value):
                vector.append(0.0)
            else:
                vector.append(float(value))
                
        # Add portfolio state
        portfolio = state['portfolio']
        vector.extend([
            portfolio['cash'] / 10000.0,  # Normalize
            portfolio['position'],
            portfolio['unrealized_pnl'] / 1000.0,  # Normalize
            portfolio['total_value'] / 10000.0,  # Normalize
            portfolio['max_drawdown'],
            portfolio['sharpe_ratio'],
            portfolio['volatility']
        ])
        
        # Add market context
        market = state['market_context']
        vector.extend([
            market['volatility'],
            market['trend'],
            market['volume_ratio'],
            market['sector_performance'],
            market['correlation_to_spy']
        ])
        
        # Encode market regime as one-hot
        regime_encoding = {
            'trending': [1, 0, 0],
            'ranging': [0, 1, 0], 
            'volatile': [0, 0, 1],
            'unknown': [0, 0, 0]
        }
        vector.extend(regime_encoding.get(market['market_regime'], [0, 0, 0]))
        
        return np.array(vector, dtype=np.float32)
    
    def get_state_size(self) -> int:
        """Get the size of the state vector"""
        # This should match the vector created in get_state_vector
        # For now, return a reasonable estimate
        return 20  # Will be calculated dynamically in practice
    
    def reset_history(self):
        """Reset state history"""
        self.state_history = []

