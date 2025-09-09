"""
Baseline Strategy Implementations
Provides simple baseline strategies for sanity checking alpha generation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BaselineStrategy:
    """Base class for baseline strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trading signals."""
        raise NotImplementedError
    
    def compute_returns(self, data: pd.DataFrame, signals: pd.Series, 
                       costs_bps: float = 0.0) -> pd.Series:
        """Compute strategy returns with costs."""
        returns = data['returns'].copy()
        
        # Apply signals (assuming signals are position weights)
        strategy_returns = signals.shift(1) * returns
        
        # Apply transaction costs
        if costs_bps > 0:
            position_changes = signals.diff().abs()
            costs = position_changes * (costs_bps / 10000)
            strategy_returns -= costs
        
        return strategy_returns


class EqualWeightStrategy(BaselineStrategy):
    """1/N Equal Weight Strategy."""
    
    def __init__(self):
        super().__init__("1/N Equal Weight")
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate equal weight signals."""
        # For single asset, this is just buy and hold
        return pd.Series(1.0, index=data.index)


class MomentumStrategy(BaselineStrategy):
    """12-1 Momentum Strategy (12-month return minus 1-month return)."""
    
    def __init__(self, lookback_long: int = 252, lookback_short: int = 21):
        super().__init__(f"{lookback_long//21}-{lookback_short//21} Momentum")
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum signals."""
        prices = data['close']
        
        # Calculate long-term and short-term returns
        long_return = prices.pct_change(self.lookback_long)
        short_return = prices.pct_change(self.lookback_short)
        
        # Momentum signal: long-term minus short-term
        momentum = long_return - short_return
        
        # Convert to position weights (normalize to [-1, 1])
        signals = momentum.rolling(63).rank(pct=True) * 2 - 1  # 3-month ranking
        
        return signals.fillna(0)


class MeanReversionStrategy(BaselineStrategy):
    """Mean Reversion Strategy based on price deviation from moving average."""
    
    def __init__(self, lookback: int = 63, threshold: float = 2.0):
        super().__init__(f"Mean Reversion ({lookback}d)")
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate mean reversion signals."""
        prices = data['close']
        
        # Calculate moving average and standard deviation
        ma = prices.rolling(self.lookback).mean()
        std = prices.rolling(self.lookback).std()
        
        # Z-score of price deviation
        z_score = (prices - ma) / std
        
        # Mean reversion signal: opposite of z-score
        signals = -z_score / self.threshold
        
        # Clip to [-1, 1] range
        signals = np.clip(signals, -1, 1)
        
        return signals.fillna(0)


class BuyAndHoldStrategy(BaselineStrategy):
    """Simple Buy and Hold Strategy."""
    
    def __init__(self):
        super().__init__("Buy and Hold")
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate buy and hold signals."""
        return pd.Series(1.0, index=data.index)


class RandomStrategy(BaselineStrategy):
    """Random Strategy for sanity checking."""
    
    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.seed = seed
    
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate random signals."""
        np.random.seed(self.seed)
        signals = np.random.choice([-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3])
        return pd.Series(signals, index=data.index)


def run_baseline_strategies(data: pd.DataFrame, costs_bps: float = 0.0) -> Dict[str, Dict]:
    """
    Run all baseline strategies and return performance metrics.
    
    Args:
        data: DataFrame with 'close' and 'returns' columns
        costs_bps: Transaction costs in basis points
        
    Returns:
        Dictionary with strategy results
    """
    strategies = [
        EqualWeightStrategy(),
        MomentumStrategy(),
        MeanReversionStrategy(),
        BuyAndHoldStrategy(),
        RandomStrategy()
    ]
    
    results = {}
    
    for strategy in strategies:
        try:
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Compute returns
            strategy_returns = strategy.compute_returns(data, signals, costs_bps)
            
            # Calculate performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
            
            # Count trades (position changes)
            trades = signals.diff().abs().sum()
            
            results[strategy.name] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades': trades,
                'signals': signals,
                'returns': strategy_returns
            }
            
            logger.info(f"{strategy.name}: Sharpe={sharpe_ratio:.3f}, Return={annualized_return:.3f}, Trades={trades:.0f}")
            
        except Exception as e:
            logger.error(f"Error running {strategy.name}: {e}")
            results[strategy.name] = {
                'error': str(e),
                'sharpe_ratio': np.nan,
                'annualized_return': np.nan,
                'trades': 0
            }
    
    return results


def compare_with_baselines(strategy_results: Dict, baseline_results: Dict) -> pd.DataFrame:
    """
    Compare strategy results with baseline strategies.
    
    Args:
        strategy_results: Results from main strategy
        baseline_results: Results from baseline strategies
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    # Add main strategy results
    for strategy_name, results in strategy_results.items():
        comparison_data.append({
            'strategy': strategy_name,
            'type': 'main',
            'sharpe_ratio': results.get('sharpe_ratio', np.nan),
            'annualized_return': results.get('annualized_return', np.nan),
            'volatility': results.get('volatility', np.nan),
            'max_drawdown': results.get('max_drawdown', np.nan),
            'trades': results.get('trades', 0)
        })
    
    # Add baseline results
    for strategy_name, results in baseline_results.items():
        comparison_data.append({
            'strategy': strategy_name,
            'type': 'baseline',
            'sharpe_ratio': results.get('sharpe_ratio', np.nan),
            'annualized_return': results.get('annualized_return', np.nan),
            'volatility': results.get('volatility', np.nan),
            'max_drawdown': results.get('max_drawdown', np.nan),
            'trades': results.get('trades', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Add relative performance metrics
    if len(df) > 0:
        best_baseline_sharpe = df[df['type'] == 'baseline']['sharpe_ratio'].max()
        if not np.isnan(best_baseline_sharpe):
            df['vs_best_baseline'] = df['sharpe_ratio'] - best_baseline_sharpe
            df['vs_buy_hold'] = df['sharpe_ratio'] - df[df['strategy'] == 'Buy and Hold']['sharpe_ratio'].iloc[0] if 'Buy and Hold' in df['strategy'].values else np.nan
    
    return df


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_days = 252
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_days))
    returns = pd.Series(np.random.normal(0.0005, 0.02, n_days), index=dates)
    
    data = pd.DataFrame({
        'close': prices,
        'returns': returns
    }, index=dates)
    
    # Run baseline strategies
    baseline_results = run_baseline_strategies(data, costs_bps=3.0)
    
    # Print results
    print("\nBaseline Strategy Results:")
    print("=" * 50)
    for name, results in baseline_results.items():
        if 'error' not in results:
            print(f"{name:20s}: Sharpe={results['sharpe_ratio']:6.3f}, "
                  f"Return={results['annualized_return']:6.3f}, "
                  f"Trades={results['trades']:6.0f}")
        else:
            print(f"{name:20s}: ERROR - {results['error']}")
