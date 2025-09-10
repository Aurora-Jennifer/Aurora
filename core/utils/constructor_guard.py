"""
Constructor guard utility to prevent duplicate initialization.

This module provides a simple mechanism to detect and prevent
duplicate construction of critical components.
"""

import uuid
from typing import Dict

_CONSTRUCT_IDS: Dict[str, str] = {}

def construct_once(key: str) -> str:
    """
    Ensure a component is constructed only once.
    
    Args:
        key: Unique identifier for the component
        
    Returns:
        Unique construction ID
        
    Raises:
        RuntimeError: If component is constructed more than once
    """
    if key in _CONSTRUCT_IDS:
        raise RuntimeError(f"Double construction of {key}")
    _CONSTRUCT_IDS[key] = uuid.uuid4().hex[:6]
    return _CONSTRUCT_IDS[key]

def reset_construct_ids():
    """Reset construction IDs (for testing)."""
    global _CONSTRUCT_IDS
    _CONSTRUCT_IDS.clear()
