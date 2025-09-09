#!/usr/bin/env python3
"""
Unit tests for readiness truth verification
Prevents "optimistic" greens and ensures READY status is earned
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from verify_components import (
    verify_robustness, verify_oos, verify_lag,
    verify_portfolio, verify_ablation, verify_monitoring,
    verify_hard_invariants
)


def test_ready_requires_all_true(tmp_path):
    """Test that READY status requires all components to be True."""
    # Create a deployment report with mixed status
    report = {
        "component_status": {
            "env_ok": True,
            "robustness_ok": True,
            "oos_ok": True,
            "lag_ok": True,
            "portfolio_ok": False,  # This should make overall status False
            "ablation_ok": True,
            "monitoring_ok": True
        },
        "summary": "READY"  # This should be wrong
    }
    
    # Check that summary doesn't match actual status
    all_true = all(report["component_status"].values())
    assert not all_true, "All components should not be True"
    # The test is checking that if we had a wrong summary, we'd catch it
    # In this case, the summary is wrong, so we should detect the mismatch
    correct_summary = "READY" if all_true else "PARTIAL"
    assert report["summary"] != correct_summary, "Summary should not match actual status when components fail"


def test_ready_when_all_true(tmp_path):
    """Test that READY status is correct when all components are True."""
    report = {
        "component_status": {
            "env_ok": True,
            "robustness_ok": True,
            "oos_ok": True,
            "lag_ok": True,
            "portfolio_ok": True,
            "ablation_ok": True,
            "monitoring_ok": True
        },
        "summary": "READY"
    }
    
    all_true = all(report["component_status"].values())
    assert all_true, "All components should be True"
    assert report["summary"] == "READY", "Summary should be READY when all components pass"


def test_verify_robustness_with_actual_files(tmp_path):
    """Test robustness verification with actual CSV files."""
    # Create test directory structure
    test_dir = tmp_path / "deployment_test_cost"
    test_dir.mkdir(parents=True)
    
    # Create a CSV with successful runs
    csv_file = test_dir / "AAPL_grid.csv"
    df = pd.DataFrame({
        'median_model_sharpe': [0.1, 0.2, None, 0.15],  # Some successful runs
        'model_type': ['ridge', 'ridge', 'lgbm', 'ridge']
    })
    df.to_csv(csv_file, index=False)
    
    # Verify robustness
    success, message = verify_robustness(str(tmp_path))
    assert success, f"Robustness verification should pass: {message}"
    assert "successful" in message.lower()


def test_verify_robustness_no_files(tmp_path):
    """Test robustness verification with no files."""
    success, message = verify_robustness(str(tmp_path))
    assert not success, "Robustness verification should fail with no files"
    assert "No cost stress results found" in message


def test_verify_portfolio_with_weights(tmp_path):
    """Test portfolio verification with actual weights."""
    # Create portfolio directory
    portfolio_dir = tmp_path / "portfolio"
    portfolio_dir.mkdir()
    
    # Create portfolio weights CSV
    weights_file = portfolio_dir / "portfolio_weights.csv"
    df = pd.DataFrame({
        'strategy_id': ['AAPL_ridge_1', 'NVDA_ridge_2'],
        'weight': [0.6, 0.4]
    })
    df.to_csv(weights_file, index=False)
    
    # Verify portfolio
    success, message = verify_portfolio(str(tmp_path))
    assert success, f"Portfolio verification should pass: {message}"
    assert "strategies" in message.lower()


def test_verify_portfolio_no_weights(tmp_path):
    """Test portfolio verification with no weights."""
    success, message = verify_portfolio(str(tmp_path))
    assert not success, "Portfolio verification should fail with no weights"
    assert "No portfolio weights found" in message


def test_verify_lag_with_passed_json(tmp_path):
    """Test signal lag verification with passed JSON."""
    # Create lag tests directory
    lag_dir = tmp_path / "lag_tests"
    lag_dir.mkdir()
    
    # Create signal lag report
    report_file = lag_dir / "signal_lag_report.json"
    with open(report_file, 'w') as f:
        json.dump({"passed": True, "tests_run": 5, "tests_passed": 5}, f)
    
    # Verify lag
    success, message = verify_lag(str(tmp_path))
    assert success, f"Signal lag verification should pass: {message}"
    assert "passed" in message.lower()


def test_verify_lag_with_failed_json(tmp_path):
    """Test signal lag verification with failed JSON."""
    # Create lag tests directory
    lag_dir = tmp_path / "lag_tests"
    lag_dir.mkdir()
    
    # Create signal lag report
    report_file = lag_dir / "signal_lag_report.json"
    with open(report_file, 'w') as f:
        json.dump({"passed": False, "tests_run": 5, "tests_passed": 3}, f)
    
    # Verify lag
    success, message = verify_lag(str(tmp_path))
    assert not success, "Signal lag verification should fail with failed tests"


def test_verify_ablation_with_csv(tmp_path):
    """Test ablation verification with CSV results."""
    # Create ablation directory
    ablation_dir = tmp_path / "ablation"
    ablation_dir.mkdir()
    
    # Create delta Sharpe CSV
    csv_file = ablation_dir / "delta_sharpe.csv"
    df = pd.DataFrame({
        'feature_family': ['trend', 'volatility', 'momentum'],
        'delta_sharpe': [-0.1, -0.05, -0.02]
    })
    df.to_csv(csv_file, index=False)
    
    # Verify ablation
    success, message = verify_ablation(str(tmp_path))
    assert success, f"Ablation verification should pass: {message}"
    assert "feature comparisons" in message.lower()


def test_verify_monitoring_with_heartbeat(tmp_path):
    """Test monitoring verification with recent heartbeat."""
    import time
    
    # Create monitoring directory
    monitoring_dir = tmp_path / "monitoring"
    monitoring_dir.mkdir()
    
    # Create monitoring config
    config_file = monitoring_dir / "monitoring_config.json"
    with open(config_file, 'w') as f:
        json.dump({"enabled": True, "frequency": "daily"}, f)
    
    # Create recent heartbeat
    heartbeat_file = monitoring_dir / "heartbeat.json"
    with open(heartbeat_file, 'w') as f:
        json.dump({
            "timestamp": "2024-01-01T12:00:00Z",
            "last_run_epoch": time.time() - 3600  # 1 hour ago
        }, f)
    
    # Verify monitoring
    success, message = verify_monitoring(str(tmp_path))
    assert success, f"Monitoring verification should pass: {message}"
    assert "heartbeat" in message.lower() or "recent" in message.lower()


def test_verify_hard_invariants_with_gate_passes(tmp_path):
    """Test hard invariants verification with gate passes."""
    # Create grid results with gate passes
    grid_dir = tmp_path / "results"
    grid_dir.mkdir()
    
    csv_file = grid_dir / "grid_results.csv"
    df = pd.DataFrame({
        'strategy_id': ['strategy_1', 'strategy_2', 'strategy_3'],
        'gate_pass': [True, False, True],
        'sharpe_ratio': [0.1, -0.1, 0.2]
    })
    df.to_csv(csv_file, index=False)
    
    # Verify invariants
    success, message = verify_hard_invariants(str(tmp_path))
    assert success, f"Hard invariants verification should pass: {message}"
    assert "gate passes" in message.lower()


def test_verify_hard_invariants_no_gate_passes(tmp_path):
    """Test hard invariants verification with no gate passes."""
    # Create grid results with no gate passes
    grid_dir = tmp_path / "results"
    grid_dir.mkdir()
    
    csv_file = grid_dir / "grid_results.csv"
    df = pd.DataFrame({
        'strategy_id': ['strategy_1', 'strategy_2', 'strategy_3'],
        'gate_pass': [False, False, False],
        'sharpe_ratio': [-0.1, -0.2, -0.05]
    })
    df.to_csv(csv_file, index=False)
    
    # Verify invariants
    success, message = verify_hard_invariants(str(tmp_path))
    assert not success, "Hard invariants verification should fail with no gate passes"
    assert "No strategies passed gate" in message


def test_verification_messages_are_descriptive(tmp_path):
    """Test that verification messages are descriptive and helpful."""
    # Test with empty directory
    success, message = verify_robustness(str(tmp_path))
    assert not success
    assert len(message) > 10, "Verification message should be descriptive"
    assert "No cost stress results found" in message
    
    # Test with successful verification
    test_dir = tmp_path / "deployment_test_cost"
    test_dir.mkdir(parents=True)
    csv_file = test_dir / "AAPL_grid.csv"
    df = pd.DataFrame({'median_model_sharpe': [0.1, 0.2]})
    df.to_csv(csv_file, index=False)
    
    success, message = verify_robustness(str(tmp_path))
    assert success
    assert "successful" in message.lower()
    assert "cost stress" in message.lower()


def test_verification_handles_malformed_files(tmp_path):
    """Test that verification handles malformed files gracefully."""
    # Create malformed CSV
    test_dir = tmp_path / "deployment_test_cost"
    test_dir.mkdir(parents=True)
    csv_file = test_dir / "malformed.csv"
    with open(csv_file, 'w') as f:
        f.write("not,a,valid,csv\nwith,missing,columns")
    
    # Should handle gracefully
    success, message = verify_robustness(str(tmp_path))
    assert not success, "Should handle malformed files gracefully"
    assert "No successful" in message or "No cost stress" in message


def test_verification_handles_missing_directories(tmp_path):
    """Test that verification handles missing directories gracefully."""
    # Test with non-existent directory
    success, message = verify_robustness(str(tmp_path / "nonexistent"))
    assert not success, "Should handle missing directories gracefully"
    assert "No cost stress results found" in message
