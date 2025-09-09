"""Test decision core functionality"""
from aurora.decision_core import BUY, HOLD, SELL, apply_decision


def test_decision_core_basic():
    """Test basic decision core functionality"""
    costs = {'commission_bps': 1.0, 'slippage_bps': 3.0}
    
    # Test BUY
    action, pos, cost = apply_decision(0.001, 0.0001, 0, costs, fail_fast=False)
    assert action == BUY
    assert pos == 1
    assert cost > 0
    
    # Test SELL
    action, pos, cost = apply_decision(-0.001, 0.0001, 0, costs, fail_fast=False)
    assert action == SELL
    assert pos == -1
    assert cost > 0
    
    # Test HOLD
    action, pos, cost = apply_decision(0.00005, 0.0001, 0, costs, fail_fast=False)
    assert action == HOLD
    assert pos == 0
    assert cost == 0


def test_nan_handling():
    """Test NaN handling"""
    costs = {'commission_bps': 1.0, 'slippage_bps': 3.0}
    
    action, pos, cost = apply_decision(float('nan'), 0.0001, 0, costs, fail_fast=False)
    assert action == HOLD
    assert pos == 0
    assert cost == 0
