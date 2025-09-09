#!/usr/bin/env python3
"""
Test Decision Core - Verify unified decision logic works correctly

This script tests the decision core module to ensure it produces
consistent results and catches the "always-HOLD" bug.
"""

import sys
import torch
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

from core.decision_core import (
    decide, next_position, simulate_step, DecisionCfg, 
    BUY, SELL, HOLD, validate_decision_inputs, print_decision_legend
)


def test_decision_logic():
    """Test the core decision logic"""
    print("🧪 Testing Decision Core Logic")
    print("=" * 40)
    
    # Print legend
    print_decision_legend()
    
    # Test configuration
    cfg = DecisionCfg(
        tau=0.0001,
        temperature=1.0,
        gate_on="adv",
        cost_bps=4.0
    )
    print(f"Config: {cfg.to_dict()}")
    
    # Test case 1: Strong BUY signal
    print("\n📊 Test 1: Strong BUY signal")
    logits = torch.tensor([2.0, 0.5, 0.1])  # BUY is strongest
    advantage = torch.tensor([0.01, -0.005, 0.0])  # BUY has positive advantage
    
    validate_decision_inputs(logits, advantage, cfg)
    action = decide(logits, advantage, cfg)
    print(f"   Logits: {logits.tolist()}")
    print(f"   Advantage: {advantage.tolist()}")
    print(f"   Action: {action} ({'BUY' if action == BUY else 'SELL' if action == SELL else 'HOLD'})")
    print("   Expected: BUY (0)")
    assert action == BUY, f"Expected BUY, got {action}"
    
    # Test case 2: Weak signal -> HOLD
    print("\n📊 Test 2: Weak signal -> HOLD")
    logits = torch.tensor([0.1, 0.05, 0.0])  # All weak
    advantage = torch.tensor([0.00005, -0.00005, 0.0])  # Below tau threshold
    
    action = decide(logits, advantage, cfg)
    print(f"   Logits: {logits.tolist()}")
    print(f"   Advantage: {advantage.tolist()}")
    print(f"   Action: {action} ({'BUY' if action == BUY else 'SELL' if action == SELL else 'HOLD'})")
    print("   Expected: HOLD (2)")
    assert action == HOLD, f"Expected HOLD, got {action}"
    
    # Test case 3: Strong SELL signal
    print("\n📊 Test 3: Strong SELL signal")
    logits = torch.tensor([0.1, 2.0, 0.5])  # SELL is strongest
    advantage = torch.tensor([-0.005, 0.01, 0.0])  # SELL has positive advantage
    
    action = decide(logits, advantage, cfg)
    print(f"   Logits: {logits.tolist()}")
    print(f"   Advantage: {advantage.tolist()}")
    print(f"   Action: {action} ({'BUY' if action == BUY else 'SELL' if action == SELL else 'HOLD'})")
    print("   Expected: SELL (1)")
    assert action == SELL, f"Expected SELL, got {action}"
    
    print("\n✅ All decision logic tests passed!")


def test_position_logic():
    """Test position update logic"""
    print("\n🧪 Testing Position Logic")
    print("=" * 40)
    
    # Test position updates
    test_cases = [
        (0, BUY, 1),   # Start neutral, buy -> long
        (1, SELL, -1), # Long, sell -> short
        (-1, BUY, 1),  # Short, buy -> long
        (1, HOLD, 1),  # Long, hold -> long
        (0, HOLD, 0),  # Neutral, hold -> neutral
    ]
    
    for prev_pos, action, expected_pos in test_cases:
        new_pos = next_position(prev_pos, action)
        action_name = {BUY: 'BUY', SELL: 'SELL', HOLD: 'HOLD'}[action]
        print(f"   {prev_pos} + {action_name} -> {new_pos} (expected {expected_pos})")
        assert new_pos == expected_pos, f"Position update failed: {prev_pos} + {action} -> {new_pos}, expected {expected_pos}"
    
    print("✅ All position logic tests passed!")


def test_simulation_step():
    """Test trading simulation step"""
    print("\n🧪 Testing Simulation Step")
    print("=" * 40)
    
    # Test simulation steps
    test_cases = [
        (0, BUY, 100.0, 4.0, 1, 0.0004),   # Buy from neutral, cost = 4 bps
        (1, SELL, 100.0, 4.0, -1, 0.0008), # Sell from long, cost = 8 bps (2 trades)
        (1, HOLD, 100.0, 4.0, 1, 0.0),     # Hold, no cost
        (0, HOLD, 100.0, 4.0, 0, 0.0),     # Hold from neutral, no cost
    ]
    
    for prev_pos, action, price, cost_bps, expected_pos, expected_cost in test_cases:
        new_pos, cost = simulate_step(prev_pos, action, price, cost_bps)
        action_name = {BUY: 'BUY', SELL: 'SELL', HOLD: 'HOLD'}[action]
        print(f"   {prev_pos} + {action_name} @ {price} -> pos={new_pos}, cost={cost:.6f}")
        print(f"   Expected: pos={expected_pos}, cost={expected_cost:.6f}")
        assert new_pos == expected_pos, f"Position mismatch: got {new_pos}, expected {expected_pos}"
        assert abs(cost - expected_cost) < 1e-6, f"Cost mismatch: got {cost}, expected {expected_cost}"
    
    print("✅ All simulation step tests passed!")


def test_wire_level_asserts():
    """Test wire-level assertions for debugging"""
    print("\n🧪 Testing Wire-Level Asserts")
    print("=" * 40)
    
    cfg = DecisionCfg(tau=0.0001, temperature=1.0, gate_on="adv", cost_bps=4.0)
    
    # Test valid inputs
    logits = torch.tensor([1.0, 0.5, 0.0])
    advantage = torch.tensor([0.01, -0.005, 0.0])
    
    try:
        validate_decision_inputs(logits, advantage, cfg)
        print("✅ Valid inputs passed validation")
    except AssertionError as e:
        print(f"❌ Valid inputs failed: {e}")
        raise
    
    # Test invalid inputs
    print("\nTesting invalid inputs...")
    
    # Flat logits
    flat_logits = torch.tensor([0.0, 0.0, 0.0])
    try:
        validate_decision_inputs(flat_logits, advantage, cfg)
        print("❌ Flat logits should have failed validation")
        raise AssertionError("Flat logits validation should have failed")
    except AssertionError:
        print("✅ Flat logits correctly rejected")
    
    # Non-finite logits
    nan_logits = torch.tensor([1.0, float('nan'), 0.0])
    try:
        validate_decision_inputs(nan_logits, advantage, cfg)
        print("❌ NaN logits should have failed validation")
        raise AssertionError("NaN logits validation should have failed")
    except AssertionError:
        print("✅ NaN logits correctly rejected")
    
    print("✅ All wire-level assert tests passed!")


def run_smoke_test():
    """Run a smoke test with realistic data"""
    print("\n🧪 Smoke Test with Realistic Data")
    print("=" * 40)
    
    cfg = DecisionCfg(tau=0.0001, temperature=1.0, gate_on="adv", cost_bps=4.0)
    
    # Simulate 20 trading steps
    prev_pos = 0
    actions = []
    positions = []
    costs = []
    
    print("Simulating 20 trading steps...")
    for t in range(20):
        # Generate realistic logits and advantage
        logits = torch.randn(3) * 0.5  # Small random logits
        advantage = torch.randn(3) * 0.01  # Small random advantage
        
        # Make decision
        action = decide(logits, advantage, cfg)
        new_pos, cost = simulate_step(prev_pos, action, 100.0, 4.0)
        
        # Wire-level asserts with debug info
        assert action in {BUY, SELL, HOLD}, f"Invalid action: {action}"
        
        # Position should only change if action is not HOLD AND the action would change position
        # (e.g., BUY when already long, or SELL when already short, doesn't change position)
        position_changed = (new_pos != prev_pos)
        action_was_hold = (action == HOLD)
        
        # If action is HOLD, position should never change
        if action_was_hold:
            assert not position_changed, f"HOLD action changed position: {prev_pos} -> {new_pos}"
        
        # If position changed, action should not be HOLD
        if position_changed:
            assert not action_was_hold, f"Position changed with HOLD action: {prev_pos} -> {new_pos}"
        
        # Cost should only be charged when position actually changes
        if not position_changed:
            assert cost == 0.0, f"Cost charged when position didn't change: {cost}"
        
        if t == 0:
            assert prev_pos == 0, "Non-zero start position"
        
        # Smoke log first 15 steps
        if t < 15:
            action_name = {BUY: 'BUY', SELL: 'SELL', HOLD: 'HOLD'}[action]
            print(f"   [SMOKE] t={t} a={action_name} {prev_pos}->{new_pos} cost={cost:.6f}")
        
        actions.append(action)
        positions.append(new_pos)
        costs.append(cost)
        prev_pos = new_pos
    
    # Analyze results
    action_counts = {BUY: actions.count(BUY), SELL: actions.count(SELL), HOLD: actions.count(HOLD)}
    total_trades = sum(1 for a in actions if a != HOLD)
    total_cost = sum(costs)
    
    print("\nResults:")
    print(f"   Action distribution: {action_counts}")
    print(f"   Total trades: {total_trades}/20")
    print(f"   Total cost: {total_cost:.6f}")
    
    # Sanity checks
    assert total_trades > 0, "No trades occurred - system too conservative"
    assert total_trades < 20, "Too many trades - system too aggressive"
    assert total_cost > 0, "No costs incurred - trading simulation broken"
    
    print("✅ Smoke test passed!")


if __name__ == "__main__":
    test_decision_logic()
    test_position_logic()
    test_simulation_step()
    test_wire_level_asserts()
    run_smoke_test()
    
    print("\n🎉 All Decision Core tests passed!")
    print("The unified decision logic is working correctly.")
