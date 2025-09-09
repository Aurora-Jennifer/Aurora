#!/usr/bin/env python3
"""
Trade Counting & Metrics Helpers

Implements proper trade counting and metrics calculation:
1. Count entries, exits, flips, round-trips
2. Calculate daily PnL Sharpe ratios
3. Add metrics invariants
"""

import numpy as np
from typing import Dict, Any, List

def trade_stats_from_positions(positions: np.ndarray) -> Dict[str, int]:
    """
    Count trades from position array
    
    Args:
        positions: Position array (-1, 0, 1)
    
    Returns:
        Dictionary with trade statistics
    """
    prev = 0
    entries = exits = flips = 0
    
    for p in positions:
        if p == prev:
            continue
        
        if prev == 0 and p != 0:
            entries += 1
        elif prev != 0 and p == 0:
            exits += 1
        else:
            flips += 1  # switch +1 <-> -1
        
        prev = p
    
    round_trips = min(entries, exits + flips)
    total_execs = entries + exits + 2 * flips
    
    return {
        'entries': entries,
        'exits': exits,
        'flips': flips,
        'round_trips': round_trips,
        'total_execs': total_execs
    }

def sharpe_from_daily(daily_pnl: np.ndarray) -> float:
    """
    Calculate Sharpe ratio from daily PnL
    
    Args:
        daily_pnl: Daily PnL array
    
    Returns:
        Annualized Sharpe ratio
    """
    mu = daily_pnl.mean()
    sd = daily_pnl.std(ddof=1)
    
    if sd == 0:
        return 0.0
    
    return (mu / sd) * (252 ** 0.5)

def calculate_daily_pnl(portfolio_values: np.ndarray) -> np.ndarray:
    """
    Calculate daily PnL from portfolio values
    
    Args:
        portfolio_values: Portfolio value array
    
    Returns:
        Daily PnL array
    """
    if len(portfolio_values) < 2:
        return np.array([0.0])
    
    return np.diff(portfolio_values) / portfolio_values[:-1]

def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        portfolio_values: Portfolio value array
    
    Returns:
        Maximum drawdown as fraction
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    
    return float(np.min(drawdown))

def calculate_profit_factor(gross_profits: float, gross_losses: float) -> float:
    """
    Calculate profit factor with better edge case handling
    
    Args:
        gross_profits: Total gross profits
        gross_losses: Total gross losses (absolute value)
    
    Returns:
        Profit factor (0.0 if no trades, inf if no losses, ratio otherwise)
    """
    if gross_profits == 0 and gross_losses == 0:
        return 0.0  # No trades
    elif gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    elif gross_profits == 0:
        return 0.0  # Only losses
    
    return gross_profits / gross_losses

def assert_metrics_invariants(sharpes: List[float], trade_counts: List[int], 
                            positions: List[np.ndarray], model_classes: np.ndarray):
    """
    Assert metrics invariants to catch regressions
    
    Args:
        sharpes: Sharpe ratios across folds
        trade_counts: Trade counts across folds
        positions: Position arrays across folds
        model_classes: Model classes
    """
    # Sharpe variance
    sharpe_std = np.std(sharpes)
    if sharpe_std < 0.05:
        raise AssertionError(f"Sharpe variance too low: {sharpe_std:.4f} < 0.05")
    
    # Trade count variance
    unique_trade_counts = len(set(trade_counts))
    if unique_trade_counts <= 1:
        raise AssertionError(f"Trade counts constant across folds: {trade_counts}")
    
    # Model classes
    if len(model_classes) != 3:
        raise AssertionError(f"Expected 3 classes, got {len(model_classes)}: {model_classes}")
    
    # Check for all-zero positions
    all_zero_positions = any(np.all(pos == 0) for pos in positions)
    if all_zero_positions:
        print("Warning: Some folds have all-zero positions")

def calculate_baseline_metrics(returns: np.ndarray, transaction_cost_bps: float = 4.0) -> Dict[str, float]:
    """
    Calculate baseline metrics (buy-and-hold, simple rule)
    
    Args:
        returns: Daily returns
        transaction_cost_bps: Transaction cost in basis points
    
    Returns:
        Dictionary with baseline metrics
    """
    # Buy-and-hold
    bh_return = np.prod(1 + returns) - 1
    bh_sharpe = sharpe_from_daily(returns)
    
    # Simple momentum rule (5-day lookback, no lookahead)
    if len(returns) >= 6:  # Need 5 for lookback + 1 for first signal
        # Calculate 5-day momentum using past returns only
        momentum_signals = np.zeros_like(returns)
        for i in range(5, len(returns)):
            # Use past 5 days to predict current day
            past_momentum = np.sum(returns[i-5:i])
            momentum_signals[i] = np.sign(past_momentum)
        
        # Apply signals to returns (with 1-day delay to avoid lookahead)
        momentum_returns = momentum_signals * returns
        momentum_return = np.prod(1 + momentum_returns) - 1
        momentum_sharpe = sharpe_from_daily(momentum_returns)
    else:
        momentum_return = bh_return
        momentum_sharpe = bh_sharpe
    
    return {
        'buy_hold_return': bh_return,
        'buy_hold_sharpe': bh_sharpe,
        'momentum_return': momentum_return,
        'momentum_sharpe': momentum_sharpe
    }

def print_fold_diagnostics(fold_id: int, params: Dict[str, float], 
                          trade_stats: Dict[str, int], edge_stats: Dict[str, float],
                          net_sharpe: float, bh_sharpe: float, rule_sharpe: float,
                          mdd: float, pf: float, model_classes: np.ndarray):
    """
    Print compact fold diagnostics
    
    Args:
        fold_id: Fold identifier
        params: Calibrated parameters
        trade_stats: Trade statistics
        edge_stats: Edge statistics
        net_sharpe: Net Sharpe ratio
        bh_sharpe: Buy-and-hold Sharpe
        rule_sharpe: Rule Sharpe
        mdd: Maximum drawdown
        pf: Profit factor
        model_classes: Model classes
    """
    turnover = trade_stats['total_execs'] / 30  # Approximate daily turnover
    
    print(f"Fold {fold_id} | H=5 eps={params['eps']:.6f} T=1.0 tau_in={params['tau_enter']:.4f} tau_out={params['tau_exit']:.4f}")
    print(f"  turnover={turnover:.3f} entries={trade_stats['entries']} exits={trade_stats['exits']} flips={trade_stats['flips']} roundtrips={trade_stats['round_trips']}")
    print(f"  edge_mean={edge_stats['mean']:.4f} edge_std={edge_stats['std']:.4f} pct(|edge|>tau)={edge_stats['above_tau']:.1%}")
    print(f"  net_sharpe={net_sharpe:.3f} bh_sharpe={bh_sharpe:.3f} rule_sharpe={rule_sharpe:.3f} mdd={mdd:.1%} pf={pf:.2f}")
    print(f"  classes_={model_classes}")
    print()
