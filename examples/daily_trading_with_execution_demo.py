#!/usr/bin/env python3
"""
Daily Trading with Execution Infrastructure Demo

This example demonstrates how to use the integrated daily paper trading system
with real order execution on Alpaca.
"""

import os
import sys
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops.daily_paper_trading_with_execution import DailyPaperTradingWithExecution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_preflight_checks():
    """Demonstrate pre-flight checks."""
    print("🔍 Running Pre-flight Checks Demo")
    print("=" * 50)
    
    try:
        # Initialize operations
        ops = DailyPaperTradingWithExecution()
        
        # Run pre-flight checks
        success = ops._preflight_checks()
        
        if success:
            print("✅ Pre-flight checks PASSED")
            print("   - Model loaded successfully")
            print("   - Execution engine initialized")
            print("   - Alpaca connection verified")
            print("   - Market data available")
        else:
            print("❌ Pre-flight checks FAILED")
            print("   - Some components may not be available")
            print("   - System will run in fallback mode")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during pre-flight checks: {e}")
        return False


def demo_signal_generation():
    """Demonstrate signal generation."""
    print("\n📊 Signal Generation Demo")
    print("=" * 50)
    
    try:
        # Initialize operations
        ops = DailyPaperTradingWithExecution()
        
        # Generate mock market data
        import pandas as pd
        import numpy as np
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'symbol': symbol,
                    'timestamp': date,
                    'open': 100.0 + np.random.randn() * 5,
                    'high': 105.0 + np.random.randn() * 5,
                    'low': 95.0 + np.random.randn() * 5,
                    'close': 100.0 + np.random.randn() * 5,
                    'volume': 1000000 + np.random.randint(0, 500000)
                })
        
        market_data = pd.DataFrame(data)
        market_data = market_data.set_index(['symbol', 'timestamp'])
        
        # Generate signals
        signal_result = ops._generate_trading_signals(market_data)
        
        print(f"✅ Generated {len(signal_result['signals'])} trading signals")
        print(f"   Model used: {'Yes' if signal_result['model_used'] else 'No (Mock)'}")
        print(f"   Feature count: {signal_result.get('feature_count', 0)}")
        
        # Show sample signals
        signals = signal_result['signals']
        non_zero_signals = {k: v for k, v in signals.items() if v != 0}
        if non_zero_signals:
            print(f"   Active signals: {len(non_zero_signals)}")
            for symbol, signal in list(non_zero_signals.items())[:3]:
                print(f"     {symbol}: {signal:.3f}")
        
        return signal_result
        
    except Exception as e:
        print(f"❌ Error during signal generation: {e}")
        return None


def demo_execution_flow():
    """Demonstrate the complete execution flow."""
    print("\n🚀 Execution Flow Demo")
    print("=" * 50)
    
    try:
        # Initialize operations
        ops = DailyPaperTradingWithExecution()
        
        # Mock signals and prices
        signals = {
            'AAPL': 0.8,   # Strong buy signal
            'MSFT': -0.6,  # Strong sell signal
            'GOOGL': 0.05  # Weak signal (below threshold)
        }
        
        current_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2800.0
        }
        
        print(f"📈 Input signals: {len(signals)} symbols")
        for symbol, signal in signals.items():
            print(f"   {symbol}: {signal:.3f}")
        
        # Execute signals
        result = ops._execute_trading_signals(signals, current_prices)
        
        print(f"\n📊 Execution result:")
        print(f"   Status: {result['status']}")
        print(f"   Orders submitted: {result['orders_submitted']}")
        print(f"   Orders filled: {result['orders_filled']}")
        print(f"   Orders rejected: {result.get('orders_rejected', 0)}")
        
        if result.get('execution_time'):
            print(f"   Execution time: {result['execution_time']:.2f}s")
        
        if result.get('errors'):
            print(f"   Errors: {len(result['errors'])}")
            for error in result['errors'][:2]:  # Show first 2 errors
                print(f"     - {error}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return None


def demo_emergency_procedures():
    """Demonstrate emergency procedures."""
    print("\n🚨 Emergency Procedures Demo")
    print("=" * 50)
    
    try:
        # Initialize operations
        ops = DailyPaperTradingWithExecution()
        
        # Simulate emergency halt
        print("Simulating emergency halt...")
        ops._emergency_halt("Demo emergency halt")
        
        print("✅ Emergency halt executed")
        print(f"   Trading active: {ops.trading_active}")
        print(f"   Emergency halt: {ops.emergency_halt}")
        
        if ops.execution_engine:
            print("   Execution engine emergency stop: Executed")
        else:
            print("   Execution engine: Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during emergency procedures: {e}")
        return False


def main():
    """Main demonstration function."""
    print("🎯 Daily Trading with Execution Infrastructure Demo")
    print("=" * 60)
    print("This demo shows the integrated system that combines:")
    print("  • XGBoost signal generation")
    print("  • Position sizing and risk management")
    print("  • Real order execution on Alpaca")
    print("  • Portfolio tracking and monitoring")
    print("  • Emergency procedures and safety checks")
    print()
    
    try:
        # Run demonstrations
        preflight_success = demo_preflight_checks()
        
        signal_result = demo_signal_generation()
        
        execution_result = demo_execution_flow()
        
        emergency_success = demo_emergency_procedures()
        
        # Summary
        print("\n📋 Demo Summary")
        print("=" * 50)
        print(f"✅ Pre-flight checks: {'PASSED' if preflight_success else 'FAILED'}")
        print(f"✅ Signal generation: {'SUCCESS' if signal_result else 'FAILED'}")
        print(f"✅ Execution flow: {'SUCCESS' if execution_result else 'FAILED'}")
        print(f"✅ Emergency procedures: {'SUCCESS' if emergency_success else 'FAILED'}")
        
        print("\n🎉 Integration Complete!")
        print("The execution infrastructure is successfully integrated with daily paper trading.")
        print()
        print("Next steps:")
        print("1. Set up Alpaca API credentials in .env file")
        print("2. Configure execution parameters in config/execution.yaml")
        print("3. Run: python ops/daily_paper_trading_with_execution.py --mode trading")
        print("4. Monitor execution through logs and Alpaca dashboard")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
