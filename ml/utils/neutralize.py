"""
Factor and Benchmark Neutralization

Implements neutralization of predictions against market factors, sectors,
and other systematic exposures to isolate alpha.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def neutralize(predictions: np.ndarray, 
              exposures: np.ndarray,
              method: str = 'orthogonal') -> np.ndarray:
    """
    Neutralize predictions against systematic exposures.
    
    Args:
        predictions: Raw predictions [n_samples]
        exposures: Exposure matrix [n_samples, n_factors]
        method: 'orthogonal' or 'residual'
    
    Returns:
        Neutralized predictions
    """
    if exposures.shape[0] != len(predictions):
        raise ValueError(f"Exposure rows {exposures.shape[0]} != predictions length {len(predictions)}")
    
    if method == 'orthogonal':
        # Project predictions onto space orthogonal to exposures
        # This removes all systematic exposure
        try:
            # Use pseudo-inverse for numerical stability
            beta = np.linalg.pinv(exposures) @ predictions
            neutralized = predictions - exposures @ beta
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in orthogonal projection, using residual method")
            method = 'residual'
    
    if method == 'residual':
        # Use linear regression residuals
        reg = LinearRegression(fit_intercept=False)
        reg.fit(exposures, predictions)
        neutralized = predictions - reg.predict(exposures)
    
    return neutralized


def create_market_exposures(returns: np.ndarray, 
                          market_returns: np.ndarray,
                          lookback: int = 20) -> np.ndarray:
    """
    Create market beta exposures.
    
    Args:
        returns: Asset returns [n_samples]
        market_returns: Market returns [n_samples]
        lookback: Rolling window for beta calculation
    
    Returns:
        Market beta exposures [n_samples]
    """
    if len(returns) != len(market_returns):
        raise ValueError("Returns and market returns must have same length")
    
    # Rolling beta calculation
    betas = np.full(len(returns), np.nan)
    
    for i in range(lookback, len(returns)):
        # Get rolling window
        asset_window = returns[i-lookback:i]
        market_window = market_returns[i-lookback:i]
        
        # Calculate beta
        if np.std(market_window) > 1e-8:
            beta = np.cov(asset_window, market_window)[0, 1] / np.var(market_window)
            betas[i] = beta
        else:
            betas[i] = 0.0
    
    # Forward fill
    betas = pd.Series(betas).fillna(method='ffill').fillna(0.0).values
    
    return betas


def create_sector_exposures(symbols: List[str], 
                          sector_map: Dict[str, str]) -> np.ndarray:
    """
    Create sector exposure matrix.
    
    Args:
        symbols: List of symbols
        sector_map: Dict of symbol -> sector
    
    Returns:
        Sector exposure matrix [n_samples, n_sectors]
    """
    # Get unique sectors
    sectors = sorted(set(sector_map.values()))
    n_sectors = len(sectors)
    
    # Create one-hot encoding
    exposures = np.zeros((len(symbols), n_sectors))
    
    for i, symbol in enumerate(symbols):
        sector = sector_map.get(symbol, 'other')
        if sector in sectors:
            j = sectors.index(sector)
            exposures[i, j] = 1.0
    
    return exposures


def create_size_exposures(market_caps: np.ndarray,
                         quantiles: List[float] = [0.2, 0.4, 0.6, 0.8]) -> np.ndarray:
    """
    Create size factor exposures.
    
    Args:
        market_caps: Market capitalizations
        quantiles: Size quantile boundaries
    
    Returns:
        Size exposure matrix [n_samples, n_size_buckets]
    """
    n_buckets = len(quantiles) + 1
    exposures = np.zeros((len(market_caps), n_buckets))
    
    # Create size buckets
    for i, mc in enumerate(market_caps):
        if np.isnan(mc):
            exposures[i, 0] = 1.0  # Missing data bucket
        else:
            # Find appropriate bucket
            bucket = 0
            for j, q in enumerate(quantiles):
                if mc <= np.quantile(market_caps[~np.isnan(market_caps)], q):
                    bucket = j
                    break
            else:
                bucket = len(quantiles)  # Largest bucket
            
            exposures[i, bucket] = 1.0
    
    return exposures


def create_momentum_exposures(returns: np.ndarray,
                            lookbacks: List[int] = [5, 10, 20]) -> np.ndarray:
    """
    Create momentum factor exposures.
    
    Args:
        returns: Asset returns
        lookbacks: Momentum lookback periods
    
    Returns:
        Momentum exposure matrix [n_samples, n_lookbacks]
    """
    n_lookbacks = len(lookbacks)
    exposures = np.zeros((len(returns), n_lookbacks))
    
    for i, lookback in enumerate(lookbacks):
        # Calculate momentum
        momentum = np.full(len(returns), np.nan)
        
        for j in range(lookback, len(returns)):
            momentum[j] = np.sum(returns[j-lookback:j])
        
        # Forward fill and normalize
        momentum = pd.Series(momentum).fillna(method='ffill').fillna(0.0).values
        
        # Standardize
        if np.std(momentum) > 1e-8:
            momentum = (momentum - np.mean(momentum)) / np.std(momentum)
        
        exposures[:, i] = momentum
    
    return exposures


class FactorNeutralizer:
    """
    Factor neutralization system.
    """
    
    def __init__(self, 
                 neutralize_market: bool = True,
                 neutralize_sectors: bool = True,
                 neutralize_size: bool = False,
                 neutralize_momentum: bool = False):
        self.neutralize_market = neutralize_market
        self.neutralize_sectors = neutralize_sectors
        self.neutralize_size = neutralize_size
        self.neutralize_momentum = neutralize_momentum
        
        self.sector_map = None
        self.is_fitted = False
    
    def fit(self, 
            symbols: List[str],
            sector_map: Optional[Dict[str, str]] = None) -> 'FactorNeutralizer':
        """Fit neutralizer with symbol and sector information."""
        self.symbols = symbols
        self.sector_map = sector_map or {}
        self.is_fitted = True
        
        logger.info(f"Factor neutralizer fitted for {len(symbols)} symbols")
        return self
    
    def create_exposures(self, 
                        returns: np.ndarray,
                        market_returns: np.ndarray,
                        market_caps: Optional[np.ndarray] = None,
                        symbols: Optional[List[str]] = None) -> np.ndarray:
        """
        Create exposure matrix for neutralization.
        
        Args:
            returns: Asset returns
            market_returns: Market returns
            market_caps: Market capitalizations (optional)
        
        Returns:
            Exposure matrix [n_samples, n_factors]
        """
        if not self.is_fitted:
            raise ValueError("Neutralizer not fitted yet")
        
        exposures_list = []
        
        # Market exposure
        if self.neutralize_market:
            market_beta = create_market_exposures(returns, market_returns)
            exposures_list.append(market_beta.reshape(-1, 1))
        
        # Sector exposures
        if self.neutralize_sectors and self.sector_map:
            # Use provided symbols or default to fitted symbols
            symbols_to_use = symbols if symbols is not None else self.symbols
            sector_exposures = create_sector_exposures(symbols_to_use, self.sector_map)
            exposures_list.append(sector_exposures)
        
        # Size exposures
        if self.neutralize_size and market_caps is not None:
            size_exposures = create_size_exposures(market_caps)
            exposures_list.append(size_exposures)
        
        # Momentum exposures
        if self.neutralize_momentum:
            momentum_exposures = create_momentum_exposures(returns)
            exposures_list.append(momentum_exposures)
        
        if not exposures_list:
            logger.warning("No exposures created, returning identity")
            return np.ones((len(returns), 1))
        
        # Combine exposures
        exposures = np.hstack(exposures_list)
        
        logger.info(f"Created exposure matrix: {exposures.shape}")
        return exposures
    
    def neutralize(self, 
                  predictions: np.ndarray,
                  exposures: np.ndarray,
                  method: str = 'orthogonal') -> np.ndarray:
        """Neutralize predictions against exposures."""
        return neutralize(predictions, exposures, method)
    
    def neutralize_batch(self, 
                        predictions_dict: Dict[str, np.ndarray],
                        exposures: np.ndarray,
                        method: str = 'orthogonal') -> Dict[str, np.ndarray]:
        """Neutralize multiple predictions."""
        neutralized_dict = {}
        
        for name, preds in predictions_dict.items():
            neutralized_dict[name] = self.neutralize(preds, exposures, method)
        
        return neutralized_dict


def evaluate_neutralization(original_preds: np.ndarray,
                          neutralized_preds: np.ndarray,
                          exposures: np.ndarray) -> Dict[str, float]:
    """
    Evaluate neutralization effectiveness.
    
    Args:
        original_preds: Original predictions
        neutralized_preds: Neutralized predictions
        exposures: Exposure matrix
    
    Returns:
        Dict of evaluation metrics
    """
    # Calculate correlations with factors
    original_corrs = []
    neutralized_corrs = []
    
    for i in range(exposures.shape[1]):
        orig_corr = np.corrcoef(original_preds, exposures[:, i])[0, 1]
        neut_corr = np.corrcoef(neutralized_preds, exposures[:, i])[0, 1]
        
        original_corrs.append(orig_corr)
        neutralized_corrs.append(neut_corr)
    
    # Summary statistics
    metrics = {
        'original_max_corr': float(np.max(np.abs(original_corrs))),
        'neutralized_max_corr': float(np.max(np.abs(neutralized_corrs))),
        'original_mean_abs_corr': float(np.mean(np.abs(original_corrs))),
        'neutralized_mean_abs_corr': float(np.mean(np.abs(neutralized_corrs))),
        'correlation_reduction': float(np.mean(np.abs(original_corrs)) - np.mean(np.abs(neutralized_corrs))),
        'original_correlations': [float(c) for c in original_corrs],
        'neutralized_correlations': [float(c) for c in neutralized_corrs]
    }
    
    logger.info(f"Neutralization evaluation:")
    logger.info(f"  Original max correlation: {metrics['original_max_corr']:.4f}")
    logger.info(f"  Neutralized max correlation: {metrics['neutralized_max_corr']:.4f}")
    logger.info(f"  Correlation reduction: {metrics['correlation_reduction']:.4f}")
    
    return metrics
