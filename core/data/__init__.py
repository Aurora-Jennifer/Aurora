"""
Data Management Package

Provides data acquisition, management, and processing capabilities.
"""

from .multi_symbol_manager import MultiSymbolDataManager, MultiSymbolConfig, SymbolData

__all__ = ['MultiSymbolDataManager', 'MultiSymbolConfig', 'SymbolData']