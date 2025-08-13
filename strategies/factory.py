"""
Strategy Factory for managing and creating trading strategies.
"""

from typing import Dict, Type, Any, Optional
from .base import BaseStrategy, StrategyParams
from .sma_crossover import SMACrossover, SMAParams
from .momentum import Momentum, MomentumParams
from .mean_reversion import MeanReversion, MeanReversionParams
from .ensemble_strategy import EnsembleStrategy, EnsembleStrategyParams
from .regime_aware_ensemble import RegimeAwareEnsembleStrategy, RegimeAwareEnsembleParams


class StrategyFactory:
    """Factory for creating and managing trading strategies."""
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._param_classes: Dict[str, Type[StrategyParams]] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register the default strategies."""
        self.register_strategy('sma', SMACrossover, SMAParams)
        self.register_strategy('momentum', Momentum, MomentumParams)
        self.register_strategy('mean_reversion', MeanReversion, MeanReversionParams)
        self.register_strategy('ensemble', EnsembleStrategy, EnsembleStrategyParams)
        self.register_strategy('regime_ensemble', RegimeAwareEnsembleStrategy, RegimeAwareEnsembleParams)
    
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy], 
                         param_class: Type[StrategyParams]):
        """Register a new strategy."""
        self._strategies[name] = strategy_class
        self._param_classes[name] = param_class
    
    def get_available_strategies(self) -> list[str]:
        """Get list of available strategy names."""
        return list(self._strategies.keys())
    
    def create_strategy(self, name: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """
        Create a strategy instance.
        
        Args:
            name: Strategy name
            params: Strategy parameters (optional, uses defaults if not provided)
            
        Returns:
            BaseStrategy: Strategy instance
        """
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {self.get_available_strategies()}")
        
        strategy_class = self._strategies[name]
        param_class = self._param_classes[name]
        
        if params is None:
            # Create default parameters
            param_instance = param_class()
        else:
            # Create parameters from dict
            param_instance = param_class(**params)
        
        return strategy_class(param_instance)
    
    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """Get information about a strategy."""
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        strategy_class = self._strategies[name]
        param_class = self._param_classes[name]
        
        # Create default instance to get info
        default_params = param_class()
        strategy = strategy_class(default_params)
        
        return {
            'name': name,
            'class': strategy_class.__name__,
            'description': strategy.get_description(),
            'default_params': default_params,
            'param_ranges': strategy.get_param_ranges()
        }
    
    def list_strategies(self) -> Dict[str, str]:
        """List all available strategies with descriptions."""
        strategies = {}
        for name in self.get_available_strategies():
            try:
                info = self.get_strategy_info(name)
                strategies[name] = info['description']
            except Exception as e:
                strategies[name] = f"Error loading strategy: {e}"
        return strategies


# Global factory instance
strategy_factory = StrategyFactory()
