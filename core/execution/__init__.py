"""
Execution Infrastructure Module

This module provides the complete execution infrastructure for converting
trading signals into actual orders on Alpaca and managing the paper trading portfolio.

Components:
- Order Management: Create, submit, and track orders
- Position Sizing: Convert signals to appropriate position sizes  
- Risk Management: Enforce limits and safety checks
- Portfolio Management: Track positions, P&L, and portfolio value
- Execution Engine: Orchestrate the complete execution flow

Usage:
    from core.execution import ExecutionEngine, OrderManager, PositionSizer
    
    # Initialize execution engine
    engine = ExecutionEngine(config)
    
    # Execute signals
    result = engine.execute_signals(signals)
"""

from .order_types import Order, OrderType, OrderSide, OrderStatus
from .order_manager import OrderManager
from .position_sizing import PositionSizer
from .risk_manager import RiskManager
from .portfolio_manager import PortfolioManager
from .execution_engine import ExecutionEngine

__all__ = [
    'Order',
    'OrderType', 
    'OrderSide',
    'OrderStatus',
    'OrderManager',
    'PositionSizer',
    'RiskManager', 
    'PortfolioManager',
    'ExecutionEngine'
]
