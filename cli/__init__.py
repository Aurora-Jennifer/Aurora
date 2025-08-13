"""
CLI Module
Command-line interface components
"""

from .backtest import main as backtest_main
from .paper import main as paper_main

__all__ = ["paper_main", "backtest_main"]
