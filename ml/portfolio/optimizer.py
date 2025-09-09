"""
Portfolio Optimizer with Turnover Penalty

Implements turnover-penalized portfolio optimization for risk-aware
position sizing with transaction costs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def ridge_turnover_opt(expected_returns: np.ndarray,
                      cov_matrix: np.ndarray,
                      previous_weights: np.ndarray,
                      lambda_risk: float = 5.0,
                      lambda_turnover: float = 10.0,
                      weight_cap: float = 0.03,
                      long_only: bool = False) -> np.ndarray:
    """
    Ridge regression with turnover penalty for portfolio optimization.
    
    Args:
        expected_returns: Expected returns [n_assets]
        cov_matrix: Covariance matrix [n_assets, n_assets]
        previous_weights: Previous period weights [n_assets]
        lambda_risk: Risk penalty parameter
        lambda_turnover: Turnover penalty parameter
        weight_cap: Maximum absolute weight per asset
        long_only: Whether to enforce long-only constraints
    
    Returns:
        Optimal weights [n_assets]
    """
    n_assets = len(expected_returns)
    
    if len(previous_weights) != n_assets:
        raise ValueError(f"Previous weights length {len(previous_weights)} != assets {n_assets}")
    
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(f"Covariance matrix shape {cov_matrix.shape} != ({n_assets}, {n_assets})")
    
    # Closed-form solution for ridge regression with turnover penalty
    # minimize: lambda_risk * w'Σw - μ'w + lambda_turnover * ||w - w_prev||²
    # Solution: (2*lambda_risk*Σ + 2*lambda_turnover*I) w = μ + 2*lambda_turnover*w_prev
    
    # Build system matrix
    A = 2 * lambda_risk * cov_matrix + 2 * lambda_turnover * np.eye(n_assets)
    b = expected_returns + 2 * lambda_turnover * previous_weights
    
    # Solve linear system
    try:
        weights = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix, using pseudo-inverse")
        weights = np.linalg.pinv(A) @ b
    
    # Apply constraints
    if long_only:
        weights = np.maximum(weights, 0.0)
    
    # Apply weight caps
    weights = np.clip(weights, -weight_cap, weight_cap)
    
    # Normalize to sum to 1 (if long-only) or sum of absolute values to 1
    if long_only:
        total_weight = np.sum(weights)
        if total_weight > 1e-12:
            weights = weights / total_weight
    else:
        total_weight = np.sum(np.abs(weights))
        if total_weight > 1.0:
            weights = weights / total_weight
    
    return weights


def mean_variance_optimize(expected_returns: np.ndarray,
                          cov_matrix: np.ndarray,
                          previous_weights: np.ndarray,
                          lambda_risk: float = 5.0,
                          lambda_turnover: float = 10.0,
                          weight_cap: float = 0.03,
                          long_only: bool = False) -> np.ndarray:
    """
    Mean-variance optimization with turnover penalty using scipy.
    
    Args:
        expected_returns: Expected returns [n_assets]
        cov_matrix: Covariance matrix [n_assets, n_assets]
        previous_weights: Previous period weights [n_assets]
        lambda_risk: Risk penalty parameter
        lambda_turnover: Turnover penalty parameter
        weight_cap: Maximum absolute weight per asset
        long_only: Whether to enforce long-only constraints
    
    Returns:
        Optimal weights [n_assets]
    """
    n_assets = len(expected_returns)
    
    # Objective function
    def objective(w):
        # Risk term
        risk_term = lambda_risk * w.T @ cov_matrix @ w
        
        # Expected return term (negative for maximization)
        return_term = -np.dot(expected_returns, w)
        
        # Turnover penalty
        turnover_term = lambda_turnover * np.sum((w - previous_weights) ** 2)
        
        return risk_term + return_term + turnover_term
    
    # Constraints
    constraints = []
    
    # Budget constraint (sum to 1)
    if long_only:
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    else:
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1.0})
    
    # Bounds
    if long_only:
        bounds = [(0, weight_cap) for _ in range(n_assets)]
    else:
        bounds = [(-weight_cap, weight_cap) for _ in range(n_assets)]
    
    # Initial guess
    x0 = previous_weights.copy()
    
    # Optimize
    result = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if result.success:
        weights = result.x
    else:
        logger.warning(f"Optimization failed: {result.message}, using previous weights")
        weights = previous_weights.copy()
    
    return weights


def estimate_covariance(returns: np.ndarray, 
                       method: str = 'ledoit_wolf',
                       lookback: int = 60) -> np.ndarray:
    """
    Estimate covariance matrix from returns.
    
    Args:
        returns: Return matrix [n_periods, n_assets]
        method: Estimation method ('sample', 'ledoit_wolf', 'shrunk')
        lookback: Lookback period for estimation
    
    Returns:
        Covariance matrix [n_assets, n_assets]
    """
    if returns.shape[0] < lookback:
        lookback = returns.shape[0]
    
    # Use recent data
    recent_returns = returns[-lookback:]
    
    if method == 'sample':
        cov_matrix = np.cov(recent_returns.T)
    
    elif method == 'ledoit_wolf':
        lw = LedoitWolf()
        cov_matrix = lw.fit(recent_returns).covariance_
    
    elif method == 'shrunk':
        # Simple shrinkage towards identity
        sample_cov = np.cov(recent_returns.T)
        n_assets = sample_cov.shape[0]
        
        # Shrinkage target (identity matrix scaled by average variance)
        target = np.eye(n_assets) * np.trace(sample_cov) / n_assets
        
        # Shrinkage intensity
        alpha = 0.1
        cov_matrix = (1 - alpha) * sample_cov + alpha * target
    
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    
    # Ensure positive semi-definite
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
    cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    return cov_matrix


class PortfolioOptimizer:
    """
    Portfolio optimizer with turnover penalty and risk management.
    """
    
    def __init__(self,
                 lambda_risk: float = 5.0,
                 lambda_turnover: float = 10.0,
                 weight_cap: float = 0.03,
                 long_only: bool = False,
                 cov_method: str = 'ledoit_wolf',
                 cov_lookback: int = 60):
        self.lambda_risk = lambda_risk
        self.lambda_turnover = lambda_turnover
        self.weight_cap = weight_cap
        self.long_only = long_only
        self.cov_method = cov_method
        self.cov_lookback = cov_lookback
        
        self.previous_weights = None
        self.asset_names = None
        self.is_initialized = False
    
    def initialize(self, asset_names: List[str]) -> 'PortfolioOptimizer':
        """Initialize optimizer with asset names."""
        self.asset_names = asset_names
        n_assets = len(asset_names)
        
        # Initialize with equal weights
        if self.long_only:
            self.previous_weights = np.ones(n_assets) / n_assets
        else:
            self.previous_weights = np.zeros(n_assets)
        
        self.is_initialized = True
        
        logger.info(f"Portfolio optimizer initialized for {n_assets} assets")
        return self
    
    def optimize(self, 
                expected_returns: np.ndarray,
                returns_history: np.ndarray,
                method: str = 'ridge') -> np.ndarray:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected returns [n_assets]
            returns_history: Historical returns [n_periods, n_assets]
            method: Optimization method ('ridge' or 'scipy')
        
        Returns:
            Optimal weights [n_assets]
        """
        if not self.is_initialized:
            raise ValueError("Optimizer not initialized")
        
        n_assets = len(expected_returns)
        if n_assets != len(self.asset_names):
            raise ValueError(f"Expected returns length {n_assets} != assets {len(self.asset_names)}")
        
        # Estimate covariance matrix
        cov_matrix = estimate_covariance(returns_history, self.cov_method, self.cov_lookback)
        
        # Optimize weights
        if method == 'ridge':
            weights = ridge_turnover_opt(
                expected_returns, cov_matrix, self.previous_weights,
                self.lambda_risk, self.lambda_turnover, self.weight_cap, self.long_only
            )
        elif method == 'scipy':
            weights = mean_variance_optimize(
                expected_returns, cov_matrix, self.previous_weights,
                self.lambda_risk, self.lambda_turnover, self.weight_cap, self.long_only
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Update previous weights
        self.previous_weights = weights.copy()
        
        # Log optimization results
        turnover = np.sum(np.abs(weights - self.previous_weights))
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        expected_return = np.dot(expected_returns, weights)
        
        logger.info(f"Portfolio optimization completed:")
        logger.info(f"  Expected return: {expected_return:.4f}")
        logger.info(f"  Portfolio risk: {portfolio_risk:.4f}")
        logger.info(f"  Turnover: {turnover:.4f}")
        logger.info(f"  Max weight: {np.max(np.abs(weights)):.4f}")
        
        return weights
    
    def get_weights_dict(self, weights: np.ndarray) -> Dict[str, float]:
        """Convert weights array to dictionary."""
        if not self.is_initialized:
            raise ValueError("Optimizer not initialized")
        
        return dict(zip(self.asset_names, weights))
    
    def get_turnover(self, new_weights: np.ndarray) -> float:
        """Calculate turnover from previous weights."""
        if self.previous_weights is None:
            return 0.0
        
        return float(np.sum(np.abs(new_weights - self.previous_weights)))
    
    def get_portfolio_metrics(self, 
                            weights: np.ndarray,
                            expected_returns: np.ndarray,
                            cov_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio metrics."""
        expected_return = np.dot(expected_returns, weights)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = expected_return / (portfolio_risk + 1e-8)
        
        turnover = self.get_turnover(weights)
        
        metrics = {
            'expected_return': float(expected_return),
            'portfolio_risk': float(portfolio_risk),
            'sharpe_ratio': float(sharpe_ratio),
            'turnover': float(turnover),
            'max_weight': float(np.max(np.abs(weights))),
            'min_weight': float(np.min(weights)),
            'num_positions': int(np.sum(np.abs(weights) > 1e-6)),
            'gross_exposure': float(np.sum(np.abs(weights))),
            'net_exposure': float(np.sum(weights))
        }
        
        return metrics
