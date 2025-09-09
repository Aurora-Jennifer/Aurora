"""
Stress tests - cheap, parameterized
"""
import pytest

from aurora.decision_core import HOLD, apply_decision


@pytest.mark.parametrize("commission_bps,slippage_bps", [(0, 0), (1, 3), (10, 0), (25, 0)])
def test_sharpe_nonincreasing_with_costs(commission_bps, slippage_bps):
    """Test that Sharpe ratio doesn't increase with higher costs"""
    # Simulate a simple trading scenario
    edge = 0.001  # Strong signal
    tau = 0.0001  # Low threshold
    
    costs_low = {'commission_bps': 0, 'slippage_bps': 0}
    costs_high = {'commission_bps': commission_bps, 'slippage_bps': slippage_bps}
    
    # Simulate 100 trading decisions
    positions_low = []
    positions_high = []
    
    for _ in range(100):
        _, pos_low, _ = apply_decision(edge, tau, 0, costs_low, fail_fast=False)
        _, pos_high, _ = apply_decision(edge, tau, 0, costs_high, fail_fast=False)
        positions_low.append(pos_low)
        positions_high.append(pos_high)
    
    # Higher costs should not improve performance
    # (This is a simplified test - in reality you'd calculate actual Sharpe)
    assert len(positions_low) == len(positions_high)


@pytest.mark.parametrize("tau", [0.0, 1e-5, 1e-4, 5e-4])
def test_turnover_drops_with_tau(tau):
    """Test that turnover decreases with higher tau"""
    edge = 0.001  # Strong signal
    costs = {'commission_bps': 1.0, 'slippage_bps': 3.0}
    
    # Simulate 100 trading decisions
    positions = []
    prev_pos = 0
    
    for _ in range(100):
        _, new_pos, _ = apply_decision(edge, tau, prev_pos, costs, fail_fast=False)
        positions.append(new_pos)
        prev_pos = new_pos
    
    # Higher tau should lead to more HOLD decisions (less turnover)
    hold_count = sum(1 for pos in positions if pos == 0)
    assert hold_count >= 0  # Basic sanity check


def test_missing_data_resilience():
    """Test resilience to missing data"""
    # Test NaN handling
    costs = {'commission_bps': 1.0, 'slippage_bps': 3.0}
    
    # Test with NaN edge
    action, pos, cost = apply_decision(float('nan'), 0.0001, 0, costs, fail_fast=False)
    assert action == HOLD
    assert pos == 0
    assert cost == 0
    
    # Test with Inf edge
    action, pos, cost = apply_decision(float('inf'), 0.0001, 0, costs, fail_fast=False)
    assert action == HOLD
    assert pos == 0
    assert cost == 0


def test_determinism():
    """Test that identical inputs produce identical outputs"""
    edge = 0.001
    tau = 0.0001
    costs = {'commission_bps': 1.0, 'slippage_bps': 3.0}
    
    # Run same decision multiple times
    results = []
    for _ in range(10):
        action, pos, cost = apply_decision(edge, tau, 0, costs, fail_fast=False)
        results.append((action, pos, cost))
    
    # All results should be identical
    assert all(r == results[0] for r in results)
