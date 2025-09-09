"""
Baseline strategies for performance comparison
"""
from typing import Tuple

import numpy as np


def sharpe_from_prices(prices: np.ndarray, rf_daily: float = 0.0) -> float:
    """Calculate Sharpe ratio from price series"""
    if len(prices) < 2:
        return 0.0
    
    rets = np.diff(prices) / prices[:-1]
    mu, sd = rets.mean(), rets.std(ddof=1) or 1e-12
    return ((mu - rf_daily) / sd) * np.sqrt(252)


def sharpe_from_returns(rets: np.ndarray) -> float:
    """Calculate Sharpe ratio from returns series"""
    if len(rets) == 0:
        return 0.0
    
    mu, sd = rets.mean(), rets.std(ddof=1) or 1e-12
    return (mu / sd) * np.sqrt(252)


def buy_and_hold_sharpe(prices: np.ndarray, rf: float = 0.0) -> float:
    """Legacy alias for sharpe_from_prices"""
    return sharpe_from_prices(prices, rf / 252)


def model_sharpe(returns: np.ndarray) -> float:
    """Legacy alias for sharpe_from_returns"""
    return sharpe_from_returns(returns)


def always_hold_return() -> Tuple[float, float]:
    """
    Always-hold strategy (no trades)
    """
    return 0.0, 0.0  # return, sharpe


def random_strategy_sharpe(prices: np.ndarray, seed: int = 42) -> float:
    """
    Random trading strategy Sharpe ratio
    """
    np.random.seed(seed)
    rets = np.diff(prices) / prices[:-1]
    random_signs = np.random.choice([-1, 1], len(rets))
    random_returns = random_signs * rets
    
    mu, sigma = random_returns.mean(), random_returns.std(ddof=1) or 1e-12
    return mu / sigma * np.sqrt(252)


def assert_baseline_domination(
    model_return: float,
    model_sharpe: float,
    prices: np.ndarray,
    margin: float = 0.10
) -> bool:
    """
    Assert that model beats all baselines with margin
    """
    # Always-hold baseline
    hold_return, hold_sharpe = always_hold_return()
    
    # Buy-and-hold baseline
    bh_sharpe = buy_and_hold_sharpe(prices)
    bh_return = (prices[-1] / prices[0] - 1) if len(prices) > 1 else 0.0
    
    # Random strategy baseline
    random_sharpe = random_strategy_sharpe(prices)
    
    # Check domination
    beats_hold = model_return > hold_return + margin and model_sharpe > hold_sharpe + margin
    beats_buy_hold = model_return > bh_return + margin and model_sharpe > bh_sharpe + margin
    beats_random = model_sharpe > random_sharpe + margin
    
    print("Baseline Comparison:")
    print(f"  Always-hold: return={hold_return:.4f}, sharpe={hold_sharpe:.4f}")
    print(f"  Buy-and-hold: return={bh_return:.4f}, sharpe={bh_sharpe:.4f}")
    print(f"  Random: sharpe={random_sharpe:.4f}")
    print(f"  Model: return={model_return:.4f}, sharpe={model_sharpe:.4f}")
    print(f"  Beats always-hold: {beats_hold}")
    print(f"  Beats buy-and-hold: {beats_buy_hold}")
    print(f"  Beats random: {beats_random}")
    
    if not (beats_hold and beats_buy_hold and beats_random):
        raise AssertionError("Model does not dominate all baselines")
    
    return True
