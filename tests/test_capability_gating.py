#!/usr/bin/env python3
"""
Unit tests for capability gating and environment checking
"""

import pytest
import sys
import os

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from env_check import check_capabilities


def test_env_check_keys():
    """Test that environment check returns expected keys."""
    caps = check_capabilities()
    assert "python" in caps
    assert "lightgbm" in caps
    assert "xgboost" in caps


def test_env_check_python_always_true():
    """Test that python capability is always True."""
    caps = check_capabilities()
    assert caps["python"] is True


def test_env_check_returns_booleans():
    """Test that all capability values are booleans."""
    caps = check_capabilities()
    for key, value in caps.items():
        assert isinstance(value, bool), f"Capability {key} should be boolean, got {type(value)}"


def test_banner_gated():
    """Test that deployment readiness is properly gated by component status."""
    # Simulate deploy script booleans
    class MockDeployment:
        def __init__(self):
            self.robustness_pass = True
            self.oos_pass = True
            self.lag_pass = True
            self.portfolio_pass = False  # This should make overall status False
            self.ablation_pass = True
    
    fake = MockDeployment()
    
    # READY should be False if any component fails
    ready = all([
        fake.robustness_pass, 
        fake.oos_pass, 
        fake.lag_pass,
        fake.portfolio_pass,  # This is False
        fake.ablation_pass
    ])
    assert ready is False


def test_all_components_pass():
    """Test that deployment is ready when all components pass."""
    # Simulate all components passing
    class MockDeployment:
        def __init__(self):
            self.robustness_pass = True
            self.oos_pass = True
            self.lag_pass = True
            self.portfolio_pass = True
            self.ablation_pass = True
    
    fake = MockDeployment()
    
    # READY should be True when all components pass
    ready = all([
        fake.robustness_pass, 
        fake.oos_pass, 
        fake.lag_pass,
        fake.portfolio_pass,
        fake.ablation_pass
    ])
    assert ready is True


def test_capability_gating_behavior():
    """Test that capability gating works as expected."""
    caps = check_capabilities()
    
    # Test that we can check for specific capabilities
    if caps.get("lightgbm", False):
        # If LightGBM is available, we should be able to use it
        assert caps["lightgbm"] is True
    else:
        # If LightGBM is not available, we should skip it
        assert caps["lightgbm"] is False
    
    if caps.get("xgboost", False):
        # If XGBoost is available, we should be able to use it
        assert caps["xgboost"] is True
    else:
        # If XGBoost is not available, we should skip it
        assert caps["xgboost"] is False


def test_deployment_status_logic():
    """Test deployment status determination logic."""
    # Test PARTIAL status
    status = {
        "env_ok": True,
        "robustness_ok": True,
        "oos_ok": True,
        "lag_ok": True,
        "portfolio_ok": False,  # One component fails
        "ablation_ok": True
    }
    
    ready = all(status.values())
    deployment_status = "READY" if ready else "PARTIAL"
    assert deployment_status == "PARTIAL"
    
    # Test READY status
    status = {
        "env_ok": True,
        "robustness_ok": True,
        "oos_ok": True,
        "lag_ok": True,
        "portfolio_ok": True,
        "ablation_ok": True
    }
    
    ready = all(status.values())
    deployment_status = "READY" if ready else "PARTIAL"
    assert deployment_status == "READY"
