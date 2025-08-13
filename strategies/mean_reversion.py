"""
Mean Reversion Strategy
"""

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, StrategyParams


@dataclass
class MeanReversionParams(StrategyParams):
    """Parameters for Mean Reversion strategy."""
    lookback_period: int = 20
    std_dev_threshold: float = 2.0  # Number of standard deviations


class MeanReversion(BaseStrategy):
    """
    Mean Reversion Strategy.
    
    Generates long signals when price is below mean by threshold,
    short signals when above mean by threshold, flat otherwise.
    """
    
    def __init__(self, params: MeanReversionParams):
        super().__init__(params)
        self.lookback_period = params.lookback_period
        self.std_dev_threshold = params.std_dev_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals."""
        close = df['Close'].dropna()
        
        # Calculate rolling mean and standard deviation
        rolling_mean = close.rolling(self.lookback_period).mean()
        rolling_std = close.rolling(self.lookback_period).std()
        
        # Calculate z-score
        z_score = (close - rolling_mean) / rolling_std
        
        # Generate signals using numpy for better compatibility with MultiIndex
        signals = np.zeros(len(close))
        long_mask = z_score < -self.std_dev_threshold
        short_mask = z_score > self.std_dev_threshold
        
        # Convert to numpy arrays and flatten if needed
        long_indices = np.where(long_mask.values.flatten())[0]
        short_indices = np.where(short_mask.values.flatten())[0]
        
        signals[long_indices] = 1  # Long
        signals[short_indices] = -1   # Short
        
        return pd.Series(signals, index=close.index)
    
    def get_default_params(self) -> MeanReversionParams:
        """Return default mean reversion parameters."""
        return MeanReversionParams(lookback_period=20, std_dev_threshold=2.0)
    
    def get_param_ranges(self) -> Dict[str, Any]:
        """Return parameter ranges for optimization."""
        return {
            'lookback_period': range(10, 51, 5),
            'std_dev_threshold': [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
        }
    
    def validate_params(self, params: MeanReversionParams) -> bool:
        """Validate mean reversion parameters."""
        return (params.lookback_period > 0 and 
                params.std_dev_threshold > 0)
    
    def get_description(self) -> str:
        """Return strategy description."""
        return f"Mean Reversion (lookback={self.lookback_period}, threshold={self.std_dev_threshold}σ)"
