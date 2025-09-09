#!/usr/bin/env python3
"""
Decision & Calibration Module

Implements:
- Probability alignment to canonical order
- Temperature scaling
- Edge calculation (P(BUY) - P(SELL))
- Tau selection on training data only
- Hysteresis decision making
"""

import numpy as np
from typing import Tuple, Dict, Any


# Canonical class order: SELL, HOLD, BUY
CANON = (-1, 0, 1)


def align_proba(model, proba: np.ndarray) -> np.ndarray:
    """
    Align model probabilities to canonical order
    
    Args:
        model: Trained model with classes_ attribute or regression model
        proba: Raw predict_proba output (N, 3)
    
    Returns:
        Aligned probabilities in [SELL, HOLD, BUY] order
    
    Raises:
        ValueError: If model classes don't match expected format
    """
    cls = getattr(model, "classes_", None)
    if cls is None:
        # For regression models, assume standard order [SELL, HOLD, BUY]
        # This should already be the case from our baseline models
        P = proba
    else:
        # Find indices for each canonical label
        idx = []
        for lbl in CANON:
            j = np.where(cls == lbl)[0]
            if len(j) == 0:
                raise ValueError(f"Missing class {lbl} in classes_={cls}")
            idx.append(j[0])
        
        # Reorder probabilities
        P = proba[:, idx]
    
    # Validate normalization
    if not np.allclose(P.sum(1), 1.0, atol=1e-6):
        raise ValueError("Probability rows not normalized")
    
    return P


def temperature_scale(P: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature scaling to probabilities
    
    Args:
        P: Probability array (N, 3)
        T: Temperature parameter
    
    Returns:
        Temperature-scaled probabilities
    """
    # Apply temperature on logits recovered via log(P)
    logP = np.log(np.clip(P, 1e-12, 1.0))
    logP /= T
    P2 = np.exp(logP)
    P2 /= P2.sum(1, keepdims=True)  # Renormalize
    return P2


def edge_from_P(P_aligned: np.ndarray) -> np.ndarray:
    """
    Calculate edges from aligned probabilities
    
    Args:
        P_aligned: Aligned probabilities in [SELL, HOLD, BUY] order
    
    Returns:
        Edge values (P(BUY) - P(SELL))
    
    Raises:
        AssertionError: If edges are nearly constant
    """
    e = P_aligned[:, 2] - P_aligned[:, 0]  # BUY - SELL
    
    if np.std(e) < 1e-6:
        raise AssertionError("Edges nearly constant; check model/scaler")
    
    return e


def pick_tau_from_train(edges_train: np.ndarray, turnover_band: Tuple[float, float] = (0.08, 0.18)) -> float:
    """
    Pick tau threshold from training data to target turnover
    
    Args:
        edges_train: Training edges
        turnover_band: Target turnover range (min, max)
    
    Returns:
        Optimal tau value
    """
    ae = np.abs(edges_train)
    
    # Candidate thresholds at percentiles
    cands = np.quantile(ae, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    target = 0.5 * (turnover_band[0] + turnover_band[1])
    
    best, gap = cands[0], 1e9
    
    # Find tau that yields turnover closest to target
    for tau in cands:
        pos = np.sign(edges_train) * (ae > tau)
        turnover = np.mean(np.abs(np.diff(pos)))  # 0..1
        g = abs(turnover - target)
        if g < gap:
            best, gap = float(tau), g
    
    return best


def decide_hysteresis(edges: np.ndarray, tau_in: float, tau_out: float) -> np.ndarray:
    """
    Make trading decisions with hysteresis
    
    Args:
        edges: Edge values
        tau_in: Entry threshold
        tau_out: Exit threshold (should be < tau_in)
    
    Returns:
        Position array: -1 (SELL), 0 (HOLD), +1 (BUY)
    """
    pos = 0
    out = []
    
    for e in edges:
        if pos == 0:
            # No position: enter if edge exceeds tau_in
            if e > tau_in:
                pos = 1
            elif e < -tau_in:
                pos = -1
        else:
            # In position: exit if edge below tau_out, or flip if opposite signal strong
            if abs(e) < tau_out:
                pos = 0
            elif pos == 1 and e < -tau_in:
                pos = -1  # Flip from BUY to SELL
            elif pos == -1 and e > tau_in:
                pos = 1   # Flip from SELL to BUY
        
        out.append(pos)
    
    return np.asarray(out, dtype=int)


def calibrate_decision_parameters(edges_train: np.ndarray, turnover_band: Tuple[float, float] = (0.08, 0.18)) -> Dict[str, float]:
    """
    Calibrate decision parameters from training data
    
    Args:
        edges_train: Training edges
        turnover_band: Target turnover range
    
    Returns:
        Dictionary with calibrated parameters
    """
    # Pick tau_in from training data
    tau_in = pick_tau_from_train(edges_train, turnover_band)
    
    # Set tau_out as fraction of tau_in (hysteresis)
    tau_out = tau_in * 0.5
    
    return {
        'tau_in': tau_in,
        'tau_out': tau_out,
        'turnover_band': turnover_band
    }


def make_decisions(model, X: np.ndarray, params: Dict[str, float], 
                  temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make trading decisions from model predictions
    
    Args:
        model: Trained model
        X: Feature matrix
        params: Decision parameters (tau_in, tau_out)
        temperature: Temperature scaling parameter
    
    Returns:
        Tuple of (positions, edges, probabilities)
    """
    # Get model probabilities
    proba_raw = model.predict_proba(X)
    
    # Align to canonical order
    proba_aligned = align_proba(model, proba_raw)
    
    # Apply temperature scaling
    if temperature != 1.0:
        proba_aligned = temperature_scale(proba_aligned, temperature)
    
    # Calculate edges
    edges = edge_from_P(proba_aligned)
    
    # Make decisions with hysteresis
    positions = decide_hysteresis(edges, params['tau_in'], params['tau_out'])
    
    return positions, edges, proba_aligned


def validate_decisions(positions: np.ndarray, edges: np.ndarray, proba: np.ndarray) -> None:
    """
    Validate decision outputs
    
    Args:
        positions: Position array
        edges: Edge array
        proba: Probability array
    
    Raises:
        AssertionError: If decisions are invalid
    """
    # Check positions are valid
    if not np.all(np.isin(positions, [-1, 0, 1])):
        raise AssertionError("Invalid positions detected")
    
    # Check edges are finite
    if not np.all(np.isfinite(edges)):
        raise AssertionError("Non-finite edges detected")
    
    # Check probabilities are valid
    if not np.all(np.isfinite(proba)):
        raise AssertionError("Non-finite probabilities detected")
    
    if not np.allclose(proba.sum(1), 1.0, atol=1e-6):
        raise AssertionError("Probabilities not normalized")
    
    # Check edge variance
    if np.std(edges) < 1e-6:
        raise AssertionError("Edges too constant")


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                              n_informative=8, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create sample edges
    edges_train = np.random.randn(500) * 0.1
    
    # Calibrate parameters
    params = calibrate_decision_parameters(edges_train)
    print(f"Calibrated parameters: {params}")
    
    # Make decisions
    positions, edges, proba = make_decisions(model, X[:100], params, temperature=1.5)
    
    print(f"Positions: {np.bincount(positions + 1)}")  # Count [-1,0,1] as [0,1,2]
    print(f"Edge stats: mean={edges.mean():.4f}, std={edges.std():.4f}")
    
    # Validate
    validate_decisions(positions, edges, proba)
    print("✅ Decision validation passed!")
