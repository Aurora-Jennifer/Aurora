"""
Score to weight discipline with monotone mapping and turnover smoothing.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


def monotone_score_to_weight(scores: np.ndarray, 
                            max_weight: float = 0.02,
                            use_tanh: bool = True,
                            slope: float = 2.0) -> np.ndarray:
    """
    Convert scores to weights using monotone mapping.
    
    Args:
        scores: Raw model scores
        max_weight: Maximum absolute weight
        use_tanh: Use tanh mapping vs linear
        slope: Slope parameter for mapping
        
    Returns:
        Mapped weights
    """
    if len(scores) == 0:
        return np.array([])
    
    # Rank-based normalization to [-1, 1]
    ranks = stats.rankdata(scores, method='average')
    normalized = 2 * (ranks - 1) / (len(ranks) - 1) - 1 if len(ranks) > 1 else np.array([0.0])
    
    if use_tanh:
        # Tanh mapping for smooth monotone transform
        weights = max_weight * np.tanh(slope * normalized)
    else:
        # Linear mapping
        weights = max_weight * normalized
    
    return weights


def apply_turnover_smoothing(current_weights: Dict[str, float],
                           target_weights: Dict[str, float], 
                           smoothing_alpha: float = 0.2) -> Dict[str, float]:
    """
    Apply EWMA smoothing to reduce turnover.
    
    Args:
        current_weights: Current position weights
        target_weights: Target weights from model
        smoothing_alpha: Smoothing parameter (0-1, higher = more responsive)
        
    Returns:
        Smoothed weights
    """
    smoothed_weights = {}
    
    all_symbols = set(current_weights.keys()) | set(target_weights.keys())
    
    for symbol in all_symbols:
        current = current_weights.get(symbol, 0.0)
        target = target_weights.get(symbol, 0.0)
        
        # EWMA: new = α * target + (1-α) * current
        smoothed = smoothing_alpha * target + (1 - smoothing_alpha) * current
        
        # Only store non-trivial weights
        if abs(smoothed) > 1e-6:
            smoothed_weights[symbol] = smoothed
    
    return smoothed_weights


def calculate_turnover(prev_weights: Dict[str, float], 
                      new_weights: Dict[str, float]) -> float:
    """Calculate one-way turnover between weight vectors."""
    all_symbols = set(prev_weights.keys()) | set(new_weights.keys())
    
    turnover = 0.0
    for symbol in all_symbols:
        prev = prev_weights.get(symbol, 0.0)
        new = new_weights.get(symbol, 0.0)
        turnover += abs(new - prev)
    
    return turnover / 2.0  # One-way turnover


def test_score_weight_discipline():
    """Test score-to-weight mapping and turnover control."""
    print("🧪 TESTING SCORE-WEIGHT DISCIPLINE")
    print("="*50)
    
    # Test data
    np.random.seed(42)
    scores = np.random.normal(0, 1, 100)
    
    # Test monotone mapping
    weights_tanh = monotone_score_to_weight(scores, max_weight=0.02, use_tanh=True)
    weights_linear = monotone_score_to_weight(scores, max_weight=0.02, use_tanh=False)
    
    print(f"✅ Score to weight mapping:")
    print(f"   Tanh weights range: [{weights_tanh.min():.4f}, {weights_tanh.max():.4f}]")
    print(f"   Linear weights range: [{weights_linear.min():.4f}, {weights_linear.max():.4f}]")
    
    # Test monotonicity
    sorted_indices = np.argsort(scores)
    tanh_monotonic = np.all(np.diff(weights_tanh[sorted_indices]) >= 0)
    linear_monotonic = np.all(np.diff(weights_linear[sorted_indices]) >= 0)
    
    print(f"   Tanh monotonic: {'✅' if tanh_monotonic else '❌'}")
    print(f"   Linear monotonic: {'✅' if linear_monotonic else '❌'}")
    
    # Test turnover smoothing
    symbols = [f'SYM{i:03d}' for i in range(20)]
    current_weights = {sym: np.random.uniform(-0.01, 0.01) for sym in symbols[:15]}
    target_weights = {sym: np.random.uniform(-0.02, 0.02) for sym in symbols}
    
    # Calculate turnover without smoothing
    raw_turnover = calculate_turnover(current_weights, target_weights)
    
    # Apply smoothing
    smoothed_weights = apply_turnover_smoothing(current_weights, target_weights, smoothing_alpha=0.2)
    smooth_turnover = calculate_turnover(current_weights, smoothed_weights)
    
    print(f"\n🔄 Turnover control:")
    print(f"   Raw turnover: {raw_turnover:.3f}")
    print(f"   Smoothed turnover: {smooth_turnover:.3f}")
    print(f"   Reduction: {(1 - smooth_turnover/raw_turnover)*100:.1f}%")
    
    print(f"\n✅ Score-weight discipline test completed")
    return True


if __name__ == "__main__":
    test_score_weight_discipline()
