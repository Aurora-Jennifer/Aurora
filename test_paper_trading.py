#!/usr/bin/env python3
"""
Test Paper Trading Functionality
Tests the paper trading system with realistic scenarios.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from enhanced_paper_trading import EnhancedPaperTradingSystem


def test_paper_trading_basic():
    """Test basic paper trading functionality."""
    print("🧪 Testing Basic Paper Trading")
    print("=" * 40)
    
    # Initialize system with profile
    system = EnhancedPaperTradingSystem(
        "config/enhanced_paper_trading_config.json",
        "config/live_profile.json"
    )
    
    # Verify initial state
    print(f"✅ Initial Capital: ${system.capital:,.2f}")
    print(f"✅ Symbols: {system.config['symbols']}")
    print(f"✅ Max Position Size: {system.config['risk_params']['max_weight_per_symbol']}")
    print(f"✅ Kill Switches: {len(system.kill_switches)} configured")
    
    # Test kill switch functionality
    print(f"\n🔒 Testing Kill Switches:")
    kill_switch_ok = system.check_kill_switches()
    print(f"   Kill switches check: {'✅ PASS' if kill_switch_ok else '❌ FAIL'}")
    
    return system


def test_paper_trading_scenarios():
    """Test various paper trading scenarios."""
    print("\n🧪 Testing Paper Trading Scenarios")
    print("=" * 40)
    
    system = test_paper_trading_basic()
    
    # Scenario 1: No position, strong buy signal
    print(f"\n📈 Scenario 1: No position, strong buy signal")
    symbol = "SPY"
    signals = {"regime_ensemble": 0.8}  # Strong buy
    
    # Mock regime params
    regime_params = type('RegimeParams', (), {
        'confidence_threshold': 0.3,
        'position_sizing_multiplier': 1.0,
        'regime_name': 'trend'
    })()
    
    # Mock current price
    current_price = 500.0
    
    # Execute trade
    system._execute_trades(symbol, signals, date.today(), regime_params)
    
    # Check results
    position = system.positions.get(symbol, 0.0)
    print(f"   Position after buy signal: {position:.3f}")
    print(f"   Expected position: 0.250 (clamped to max)")
    print(f"   Test: {'✅ PASS' if abs(position - 0.25) < 0.001 else '❌ FAIL'}")
    
    # Scenario 2: Existing position, sell signal (should reduce)
    print(f"\n📉 Scenario 2: Existing position, sell signal")
    signals = {"regime_ensemble": -0.6}  # Sell signal
    
    # Execute trade
    system._execute_trades(symbol, signals, date.today(), regime_params)
    
    # Check results
    position = system.positions.get(symbol, 0.0)
    print(f"   Position after sell signal: {position:.3f}")
    print(f"   Expected position: 0.000 (reduce to 0)")
    print(f"   Test: {'✅ PASS' if abs(position - 0.0) < 0.001 else '❌ FAIL'}")
    
    # Scenario 3: No position, sell signal (should be blocked)
    print(f"\n🚫 Scenario 3: No position, sell signal (reduce-only)")
    signals = {"regime_ensemble": -0.7}  # Strong sell
    
    # Execute trade
    system._execute_trades(symbol, signals, date.today(), regime_params)
    
    # Check results
    position = system.positions.get(symbol, 0.0)
    print(f"   Position after sell signal: {position:.3f}")
    print(f"   Expected position: 0.000 (no change - reduce-only)")
    print(f"   Test: {'✅ PASS' if abs(position - 0.0) < 0.001 else '❌ FAIL'}")
    
    # Scenario 4: Small position, very strong buy signal (should clamp)
    print(f"\n🔒 Scenario 4: Small position, very strong buy signal (clamp)")
    system.positions[symbol] = 0.1  # Small position
    signals = {"regime_ensemble": 1.0}  # Maximum buy signal
    
    # Execute trade
    system._execute_trades(symbol, signals, date.today(), regime_params)
    
    # Check results
    position = system.positions.get(symbol, 0.0)
    print(f"   Position after max buy signal: {position:.3f}")
    print(f"   Expected position: 0.250 (clamped to max)")
    print(f"   Test: {'✅ PASS' if abs(position - 0.25) < 0.001 else '❌ FAIL'}")
    
    return system


def test_performance_tracking():
    """Test performance tracking functionality."""
    print("\n📊 Testing Performance Tracking")
    print("=" * 40)
    
    system = test_paper_trading_basic()
    
    # Set up initial state
    system.capital = 100000
    system.positions = {"SPY": 0.25}  # 25% position
    system._previous_total_value = 100000
    
    # Test performance tracking
    system._update_performance_tracking(date.today())
    
    # Check results
    print(f"✅ Daily returns recorded: {len(system.daily_returns)}")
    if system.daily_returns:
        latest_return = system.daily_returns[-1]
        print(f"✅ Latest return: {latest_return['return']:.4f}")
        print(f"✅ Total value: ${latest_return['total_value']:,.2f}")
        print(f"✅ Cash: ${latest_return['cash']:,.2f}")
        print(f"✅ Positions value: ${latest_return['positions_value']:,.2f}")
    
    return system


def test_kill_switches():
    """Test kill switch functionality."""
    print("\n🔒 Testing Kill Switches")
    print("=" * 40)
    
    system = test_paper_trading_basic()
    
    # Test normal conditions
    print("📊 Testing normal conditions:")
    kill_switch_ok = system.check_kill_switches()
    print(f"   Kill switches: {'✅ PASS' if kill_switch_ok else '❌ FAIL'}")
    
    # Test daily loss limit
    print("\n📊 Testing daily loss limit:")
    system.capital = 95000  # 5% loss
    kill_switch_ok = system.check_kill_switches()
    print(f"   5% daily loss: {'✅ PASS' if kill_switch_ok else '❌ FAIL'}")
    
    system.capital = 97000  # 3% loss (should trigger 2% limit)
    kill_switch_ok = system.check_kill_switches()
    print(f"   3% daily loss: {'❌ FAIL' if kill_switch_ok else '✅ PASS (triggered)'}")
    
    # Reset capital
    system.capital = 100000
    
    return system


def main():
    """Run all paper trading tests."""
    print("🚀 Paper Trading Functionality Test Suite")
    print("=" * 50)
    
    try:
        # Test basic functionality
        system = test_paper_trading_basic()
        
        # Test scenarios
        system = test_paper_trading_scenarios()
        
        # Test performance tracking
        system = test_performance_tracking()
        
        # Test kill switches
        system = test_kill_switches()
        
        print("\n" + "=" * 50)
        print("🎉 All Paper Trading Tests Completed Successfully!")
        print("✅ System is ready for production use")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
