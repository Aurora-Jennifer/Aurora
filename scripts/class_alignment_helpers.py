#!/usr/bin/env python3
"""
Class-Probability Alignment Helpers

Critical fixes for edge-based trading:
1. Ensure predict_proba columns match our internal class mapping
2. Compute edges correctly from aligned probabilities
3. Add assertions to catch alignment issues
"""

import numpy as np
from typing import Tuple, Optional

# Canonical class order: SELL, HOLD, BUY
# Note: sklearn models often use [0, 1, 2] instead of [-1, 0, 1]
CANON_LABELS = (-1, 0, 1)
SKLEARN_LABELS = (0, 1, 2)  # Common sklearn mapping

def aligned_proba(model, proba: np.ndarray) -> np.ndarray:
    """
    Ensure columns of proba match CANON_LABELS order
    
    Args:
        model: Trained sklearn model with classes_ attribute
        proba: Raw predict_proba output (N, 3)
    
    Returns:
        Aligned probabilities in [SELL, HOLD, BUY] order
    """
    cls = getattr(model, "classes_", None)
    if cls is None:
        raise ValueError("Model has no classes_; cannot align proba.")
    
    # Handle both [-1, 0, 1] and [0, 1, 2] class mappings
    if np.array_equal(cls, SKLEARN_LABELS):
        # sklearn mapping: [0, 1, 2] -> [SELL, HOLD, BUY]
        # No reordering needed, just verify
        P = proba
    elif np.array_equal(cls, CANON_LABELS):
        # Canonical mapping: [-1, 0, 1] -> [SELL, HOLD, BUY]
        # No reordering needed
        P = proba
    else:
        # Find indices for each canonical label
        idx = []
        for lbl in CANON_LABELS:
            where = np.where(cls == lbl)[0]
            if len(where) == 0:
                raise ValueError(f"Missing class {lbl} in model.classes_={cls}")
            idx.append(where[0])
        
        # Reorder columns to match canonical order
        P = proba[:, idx]
    
    # Verify probabilities sum to ~1
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("predict_proba rows do not sum to 1 after alignment.")
    
    return P

def compute_edge_from_proba(P: np.ndarray) -> np.ndarray:
    """
    Compute edge = P(BUY) - P(SELL) from aligned probabilities
    
    Args:
        P: Aligned probabilities in [SELL, HOLD, BUY] order
    
    Returns:
        Edge array (N,)
    """
    if P.shape[1] != 3:
        raise ValueError(f"Expected 3 classes, got {P.shape[1]}")
    
    edge = P[:, 2] - P[:, 0]  # BUY - SELL
    
    # Critical assertion: edges should not all be zero
    if np.allclose(edge, 0.0):
        raise AssertionError("Edges are all zero; check mapping/scaler.")
    
    return edge

def get_model_edge(model, X_scaled: np.ndarray) -> np.ndarray:
    """
    Get properly aligned edges from model predictions
    
    Args:
        model: Trained sklearn model
        X_scaled: Scaled features (N, features)
    
    Returns:
        Edge array (N,)
    """
    # Get raw probabilities
    proba_raw = model.predict_proba(X_scaled)
    
    # Align to canonical order
    proba_aligned = aligned_proba(model, proba_raw)
    
    # Compute edges
    edges = compute_edge_from_proba(proba_aligned)
    
    return edges

def assert_edge_quality(edges: np.ndarray, min_nonzero_ratio: float = 0.01):
    """
    Assert that edges have reasonable quality
    
    Args:
        edges: Edge array
        min_nonzero_ratio: Minimum ratio of non-zero edges
    """
    nonzero_ratio = np.mean(np.abs(edges) > 1e-6)
    if nonzero_ratio < min_nonzero_ratio:
        raise AssertionError(f"Too few non-zero edges: {nonzero_ratio:.3f} < {min_nonzero_ratio}")
    
    edge_std = np.std(edges)
    if edge_std < 1e-6:
        raise AssertionError(f"Edge standard deviation too low: {edge_std:.6f}")

def print_edge_diagnostics(edges: np.ndarray, model_classes: np.ndarray):
    """
    Print diagnostic information about edges and model classes
    
    Args:
        edges: Edge array
        model_classes: Model's classes_ attribute
    """
    print(f"Model classes: {model_classes}")
    print(f"Edge stats: mean={np.mean(edges):.4f}, std={np.std(edges):.4f}")
    print(f"Edge range: [{np.min(edges):.4f}, {np.max(edges):.4f}]")
    print(f"Non-zero edges: {np.mean(np.abs(edges) > 1e-6):.1%}")
    print(f"Strong edges (|edge| > 0.1): {np.mean(np.abs(edges) > 0.1):.1%}")
