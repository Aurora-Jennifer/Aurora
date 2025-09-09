#!/usr/bin/env python3
"""
Train-Only Calibration Helpers

Implements proper ε (neutral band) and τ (threshold) calibration:
1. Compute ε from train excess returns
2. Compute τ to hit target turnover on train
3. Implement hysteresis for position management
"""

import numpy as np
from typing import Tuple, Dict, Any

def pick_eps_from_train(abs_excess_future: np.ndarray, q: float = 0.5) -> float:
    """
    Pick epsilon (neutral band) from train excess returns
    
    Args:
        abs_excess_future: Absolute excess returns on training data
        q: Quantile to use for epsilon (0.5 = median)
    
    Returns:
        Epsilon value for neutral band
    """
    eps = float(np.quantile(abs_excess_future, q))
    
    # If epsilon is too small, use a reasonable default
    if eps <= 0:
        eps = 0.001  # 0.1% default epsilon
        print(f"Warning: Computed eps <= 0, using default: {eps}")
    
    return eps

def pick_tau_from_train(edges_train: np.ndarray, target_turnover: Tuple[float, float] = (0.05, 0.20)) -> float:
    """
    Pick tau (threshold) to hit target turnover on train data
    
    Args:
        edges_train: Training edges
        target_turnover: Target turnover range (min, max)
    
    Returns:
        Optimal tau value
    """
    abs_e = np.abs(edges_train)
    
    if np.allclose(abs_e, 0.0):
        raise AssertionError("Train edges are all zero; check proba/feature alignment.")
    
    # Candidate thresholds at percentiles
    cands = np.quantile(abs_e, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    
    # Target center of turnover range
    center = 0.5 * (target_turnover[0] + target_turnover[1])
    
    best, best_gap = cands[0], float("inf")
    
    for tau in cands:
        # Calculate turnover for this threshold
        pos = np.sign(edges_train) * (abs_e > tau)
        turnover = np.mean(np.abs(np.diff(pos)))
        
        # Choose tau closest to center of target range
        gap = abs(turnover - center)
        if gap < best_gap:
            best, best_gap = float(tau), gap
    
    return best

def decide_with_hysteresis(edges: np.ndarray, tau_enter: float, tau_exit: float = None) -> np.ndarray:
    """
    Implement hysteresis for position decisions
    
    Args:
        edges: Edge array
        tau_enter: Enter threshold
        tau_exit: Exit threshold (defaults to 0.5 * tau_enter)
    
    Returns:
        Position array (-1, 0, 1)
    """
    if tau_exit is None:
        tau_exit = 0.5 * tau_enter
    
    pos = 0
    out = []
    
    for e in edges:
        if pos == 0:
            # No position - enter if edge is strong enough
            if e > tau_enter:
                pos = 1
            elif e < -tau_enter:
                pos = -1
        else:
            # Have position - exit if edge is weak enough
            if abs(e) < tau_exit:
                pos = 0
            elif pos == 1 and e < -tau_enter:
                pos = -1
            elif pos == -1 and e > tau_enter:
                pos = 1
        
        out.append(pos)
    
    return np.asarray(out, dtype=int)

def calibrate_fold_parameters(edges_train: np.ndarray, abs_excess_train: np.ndarray, 
                            target_turnover: Tuple[float, float] = (0.05, 0.20),
                            eps_quantile: float = 0.5) -> Dict[str, float]:
    """
    Calibrate all fold parameters on training data
    
    Args:
        edges_train: Training edges
        abs_excess_train: Absolute excess returns on training data
        target_turnover: Target turnover range
        eps_quantile: Quantile for epsilon calculation
    
    Returns:
        Dictionary with calibrated parameters
    """
    # Calibrate epsilon
    eps = pick_eps_from_train(abs_excess_train, eps_quantile)
    
    # Calibrate tau
    tau_enter = pick_tau_from_train(edges_train, target_turnover)
    tau_exit = 0.5 * tau_enter
    
    return {
        'eps': eps,
        'tau_enter': tau_enter,
        'tau_exit': tau_exit,
        'target_turnover': target_turnover,
        'eps_quantile': eps_quantile
    }

def validate_calibration(edges_train: np.ndarray, params: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate calibration parameters and compute diagnostics
    
    Args:
        edges_train: Training edges
        params: Calibrated parameters
    
    Returns:
        Validation diagnostics
    """
    # Test hysteresis with calibrated parameters
    positions = decide_with_hysteresis(edges_train, params['tau_enter'], params['tau_exit'])
    
    # Calculate actual turnover
    actual_turnover = np.mean(np.abs(np.diff(positions)))
    
    # Calculate edge statistics
    edge_stats = {
        'mean': np.mean(edges_train),
        'std': np.std(edges_train),
        'abs_mean': np.mean(np.abs(edges_train)),
        'above_tau': np.mean(np.abs(edges_train) > params['tau_enter'])
    }
    
    return {
        'actual_turnover': actual_turnover,
        'target_turnover': params['target_turnover'],
        'turnover_gap': abs(actual_turnover - np.mean(params['target_turnover'])),
        'edge_stats': edge_stats,
        'positions': positions
    }

def print_calibration_diagnostics(params: Dict[str, float], validation: Dict[str, Any]):
    """
    Print calibration diagnostics
    
    Args:
        params: Calibrated parameters
        validation: Validation results
    """
    print(f"Calibrated parameters:")
    print(f"  eps: {params['eps']:.6f}")
    print(f"  tau_enter: {params['tau_enter']:.4f}")
    print(f"  tau_exit: {params['tau_exit']:.4f}")
    print(f"  target_turnover: {params['target_turnover']}")
    
    print(f"Validation results:")
    print(f"  actual_turnover: {validation['actual_turnover']:.3f}")
    print(f"  turnover_gap: {validation['turnover_gap']:.3f}")
    print(f"  edge_stats: {validation['edge_stats']}")
