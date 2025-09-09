"""
Walkforward Framework Compatibility Module

This module provides compatibility for tests that import from scripts.walkforward_framework.
It re-exports the actual implementations from scripts.walkforward/walk_core.py.
"""

# Import the actual implementations
from scripts.walkforward.walk_core import (
    Fold,
    gen_walkforward,
    build_feature_table,
    LeakageProofPipeline
)

# Re-export for compatibility
__all__ = [
    'Fold',
    'gen_walkforward', 
    'build_feature_table',
    'LeakageProofPipeline'
]
