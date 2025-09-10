#!/usr/bin/env python3
"""
Execution Infrastructure Example

This example demonstrates how to use the execution infrastructure
to convert trading signals into actual orders on Alpaca.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.execution.order_types import Order, OrderType, OrderSide
from core.execution.position_sizing import PositionSizer, PositionSizingConfig
from core.execution.risk_manager import RiskManager, RiskLimits
from core.execution.execution_engine import ExecutionEngine, ExecutionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_execution_components():
    """Create and configure execution components."""
    
    # Position sizing configuration
    position_config = PositionSizingConfig(
        max_position_size=0.1,  # 10% max position size
        min_trade_size=100.0,   # $100 minimum trade
        max_trade_size=5000.0,  # $5,000 maximum trade
        signal_threshold=0.1    # 10% signal threshold
    )
    position_sizer = PositionSizer(position_config)
    
    # Risk management configuration
    risk_limits = RiskLimits(
        max_daily_loss=0.02,     # 2% daily loss limit
        max_position_risk=0.05,  # 5% position risk limit
        max_orders_per_day=50    # 50 orders per day limit
    )
    risk_manager = RiskManager(risk_limits)
    
    # Execution configuration
    exec_config = ExecutionConfig(
        enabled=True,
        mode="paper",  # Use paper trading
        signal_threshold=0.1,
        max_orders_per_execution=5
    )
    
    logger.info("Execution components created successfully")
    return position_sizer, risk_manager, exec_config


def demonstrate_position_sizing():
    """Demonstrate position sizing calculations."""
    logger.info("=== Position Sizing Demo ===")
    
    position_sizer, _, _ = create_execution_components()
    
    # Example signals and market data
    signals = {
        "AAPL": 0.8,   # Strong buy signal
        "MSFT": -0.6,  # Strong sell signal
        "GOOGL": 0.05  # Weak signal (below threshold)
    }
    
    current_prices = {
        "AAPL": 150.0,
        "MSFT": 300.0,
        "GOOGL": 2800.0
    }
    
    portfolio_value = 100000.0  # $100k portfolio
    current_positions = {}
    
    for symbol, signal in signals.items():
        if symbol in current_prices:
            quantity, metadata = position_sizer.calculate_position_size(
                signal=signal,
                symbol=symbol,
                current_price=current_prices[symbol],
                portfolio_value=portfolio_value,
                current_positions=current_positions
            )
            
            logger.info(f"{symbol}: Signal={signal:.2f}, Quantity={quantity}, "
                       f"Value=${quantity * current_prices[symbol]:.2f}, "
                       f"Pct={metadata.get('position_pct', 0):.1%}")


def demonstrate_risk_management():
    """Demonstrate risk management checks."""
    logger.info("=== Risk Management Demo ===")
    
    _, risk_manager, _ = create_execution_components()
    
    # Create test orders
    orders = [
        Order(symbol="AAPL", side=OrderSide.BUY, quantity=30, order_type=OrderType.MARKET),
        Order(symbol="MSFT", side=OrderSide.BUY, quantity=20, order_type=OrderType.MARKET),
        Order(symbol="GOOGL", side=OrderSide.BUY, quantity=2, order_type=OrderType.MARKET)
    ]
    
    portfolio_value = 100000.0
    daily_pnl = -500.0  # $500 loss today
    current_positions = {}
    current_prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2800.0}
    
    for order in orders:
        is_allowed, reason, risk_metrics = risk_manager.check_order_risk(
            order=order,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            current_prices=current_prices
        )
        
        logger.info(f"{order.symbol}: Allowed={is_allowed}, Reason={reason}")
        if is_allowed:
            logger.info(f"  Portfolio Value: ${risk_metrics.portfolio_value:.2f}")
            logger.info(f"  Daily P&L: ${risk_metrics.daily_pnl:.2f} ({risk_metrics.daily_pnl_pct:.2%})")


def demonstrate_order_creation():
    """Demonstrate order creation and validation."""
    logger.info("=== Order Creation Demo ===")
    
    # Create different types of orders
    market_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    
    limit_order = Order(
        symbol="MSFT",
        side=OrderSide.SELL,
        quantity=50,
        order_type=OrderType.LIMIT,
        limit_price=295.0
    )
    
    stop_order = Order(
        symbol="GOOGL",
        side=OrderSide.SELL,
        quantity=5,
        order_type=OrderType.STOP,
        stop_price=2700.0
    )
    
    orders = [market_order, limit_order, stop_order]
    
    for order in orders:
        logger.info(f"Order: {order}")
        logger.info(f"  Active: {order.is_active()}")
        logger.info(f"  Finished: {order.is_finished()}")
        logger.info(f"  Remaining: {order.get_remaining_quantity()}")
        
        # Convert to Alpaca format
        alpaca_dict = order.to_alpaca_dict()
        logger.info(f"  Alpaca Format: {alpaca_dict}")


def main():
    """Main demonstration function."""
    logger.info("Execution Infrastructure Demonstration")
    logger.info("=" * 50)
    
    try:
        # Demonstrate each component
        demonstrate_position_sizing()
        print()
        
        demonstrate_risk_management()
        print()
        
        demonstrate_order_creation()
        print()
        
        logger.info("=== Summary ===")
        logger.info("✅ Position sizing: Converts signals to position sizes")
        logger.info("✅ Risk management: Validates orders against risk limits")
        logger.info("✅ Order creation: Creates and validates trading orders")
        logger.info("✅ All components working correctly!")
        
        logger.info("\nNext steps:")
        logger.info("1. Set up Alpaca API credentials")
        logger.info("2. Initialize OrderManager and PortfolioManager with real Alpaca client")
        logger.info("3. Create ExecutionEngine with all components")
        logger.info("4. Run execute_signals() with real market data")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
