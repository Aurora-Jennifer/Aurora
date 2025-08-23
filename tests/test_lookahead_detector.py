"""
Tests for the improved lookahead detection algorithm.
"""

import pytest
import pandas as pd
import numpy as np
from core.data_sanity.lookahead_detector import detect_lookahead, detect_lookahead_with_context


def test_detect_lookahead_empty_series():
    """Test lookahead detection with empty series."""
    empty_series = pd.Series(dtype=float)
    result = detect_lookahead(empty_series)
    
    assert result.suspicious_match_rate == 0.0
    assert result.n_suspicious == 0
    assert result.n_total == 0
    assert result.passed is True


def test_detect_lookahead_single_value():
    """Test lookahead detection with single value."""
    single_series = pd.Series([0.01])
    result = detect_lookahead(single_series)
    
    assert result.suspicious_match_rate == 0.0
    assert result.n_suspicious == 0
    assert result.n_total == 1
    assert result.passed is True


def test_detect_lookahead_legitimate_zero_runs():
    """Test that legitimate zero-return runs are ignored."""
    # Create series with legitimate zero returns (stable prices)
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    returns = pd.Series([0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.03, 0.0, 0.0], index=dates)
    
    result = detect_lookahead(returns, eps=0.0, min_run=2)
    
    # Should pass because the zero runs are legitimate
    assert result.passed == True
    assert result.stable_runs_count > 0
    assert result.n_suspicious == 0


def test_detect_lookahead_actual_lookahead():
    """Test that actual lookahead contamination is detected."""
    # Create series with actual lookahead (returns equal future returns)
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], index=dates)
    
    # Introduce lookahead: make some returns equal to their future values
    returns.iloc[2] = returns.iloc[3]  # r[2] = r[3]
    returns.iloc[5] = returns.iloc[6]  # r[5] = r[6]
    
    result = detect_lookahead(returns, eps=0.0, min_run=2)
    
    # Should fail because of actual lookahead
    assert result.passed == False
    assert result.n_suspicious > 0
    assert result.suspicious_match_rate > 0.001


def test_detect_lookahead_mixed_scenario():
    """Test mixed scenario with both legitimate zeros and actual lookahead."""
    dates = pd.date_range('2023-01-01', periods=15, freq='D')
    returns = pd.Series([
        0.01, 0.0, 0.0, 0.0,  # Legitimate zero run
        0.02, 0.03, 0.04,     # Normal returns
        0.05, 0.05,           # Actual lookahead (r[7] = r[8])
        0.06, 0.0, 0.0,       # Legitimate zero run
        0.07, 0.08, 0.09      # Normal returns
    ], index=dates)
    
    result = detect_lookahead(returns, eps=0.0, min_run=2)
    
    # Should detect the actual lookahead but ignore legitimate zeros
    assert result.n_suspicious > 0
    assert result.stable_runs_count > 0


def test_detect_lookahead_with_context():
    """Test lookahead detection with close price context."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    returns = pd.Series([0.01, 0.0, 0.0, 0.02, 0.03, 0.0, 0.0, 0.04, 0.05, 0.06], index=dates)
    close_prices = pd.Series([100, 100, 100, 102, 105, 105, 105, 109, 114, 120], index=dates)
    
    result = detect_lookahead_with_context(returns, close_prices, eps=0.0, min_run=2)
    
    assert "context" in result
    assert result["passed"] == True  # Should pass with legitimate zero runs


def test_detect_lookahead_identical_closes():
    """Test that identical close prices are correctly identified."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    returns = pd.Series([0.01, 0.0, 0.0, 0.02, 0.03, 0.0, 0.0, 0.04, 0.05, 0.06], index=dates)
    close_prices = pd.Series([100, 100, 100, 102, 105, 105, 105, 109, 114, 120], index=dates)
    
    result = detect_lookahead_with_context(returns, close_prices, eps=0.0, min_run=2)
    
    # Should identify that identical returns correspond to identical close prices
    if result["n_suspicious"] > 0:
        assert "context" in result
        assert "likely_legitimate" in result["context"]


def test_detect_lookahead_threshold_configuration():
    """Test different threshold configurations."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
    
    # Introduce some lookahead
    returns.iloc[10] = returns.iloc[11]
    returns.iloc[20] = returns.iloc[21]
    
    # Test with strict threshold
    result_strict = detect_lookahead(returns, eps=0.0, min_run=2)
    
    # Test with lenient threshold (higher max_suspicious_rate)
    # We'll test this by checking the raw rate
    rate = result_strict.suspicious_match_rate
    
    # Should be detected with strict threshold
    assert rate > 0.001
    
    # But would pass with lenient threshold
    assert rate < 0.1  # Assuming our test doesn't create too much lookahead


def test_detect_lookahead_epsilon_configuration():
    """Test epsilon configuration for zero return detection."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    returns = pd.Series([0.01, 0.001, 0.001, 0.001, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], index=dates)
    
    # With eps=0.0, small returns are not considered zero
    result_strict = detect_lookahead(returns, eps=0.0, min_run=2)
    
    # With eps=0.002, small returns are considered zero
    result_lenient = detect_lookahead(returns, eps=0.002, min_run=2)
    
    # The lenient version should identify more stable runs
    assert result_lenient.stable_runs_count >= result_strict.stable_runs_count


def test_detect_lookahead_min_run_configuration():
    """Test minimum run length configuration."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    returns = pd.Series([0.01, 0.0, 0.02, 0.0, 0.0, 0.03, 0.0, 0.04, 0.05, 0.06], index=dates)
    
    # With min_run=2, single zeros are not stable runs
    result_min2 = detect_lookahead(returns, eps=0.0, min_run=2)
    
    # With min_run=1, single zeros are stable runs
    result_min1 = detect_lookahead(returns, eps=0.0, min_run=1)
    
    # The min_run=1 version should identify more stable runs
    assert result_min1.stable_runs_count >= result_min2.stable_runs_count


if __name__ == "__main__":
    pytest.main([__file__])
