"""
Statistical analysis modules for Aurora trading system.

Provides professional-grade statistical validation tools including:
- IC (Information Coefficient) analysis with HAC standard errors
- Block bootstrap confidence intervals
- Multiple testing corrections
- Regime analysis
"""

from .ic_validator import ICValidator, ICResult

__all__ = [
    "ICValidator", 
    "ICResult"
]
