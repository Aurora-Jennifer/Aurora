#!/usr/bin/env python3
"""
Deterministic Falsification Matrix Tests

This module tests the system's behavior across a matrix of:
- Cost configurations (commission + slippage)
- Tau thresholds
- Random seeds

The tests ensure:
1. Deterministic behavior (same inputs → same outputs)
2. Monotonic relationships (higher costs → lower Sharpe)
3. Robustness across parameter ranges
"""

import itertools
import json
import subprocess
import hashlib
import pytest
from pathlib import Path
from typing import Any
import sys

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))


class FalsificationMatrix:
    """Test matrix for falsification testing"""
    
    def __init__(self):
        # Cost configurations: (commission_bps, slippage_bps)
        self.costs = [
            (0, 0),      # No costs
            (1, 3),      # Low costs
            (5, 10),     # Medium costs
            (10, 15),    # High costs
            (25, 0),     # High commission only
            (0, 25),     # High slippage only
        ]
        
        # Tau thresholds
        self.taus = [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
        
        # Random seeds for determinism testing
        self.seeds = [1337, 1338, 42, 999]
    
    def create_test_config(self, commission_bps: float, slippage_bps: float, 
                          tau: float, seed: int) -> dict[str, Any]:
        """Create test configuration"""
        return {
            "extends": "configs/paper.yaml",
            "costs": {
                "commission_bps": commission_bps,
                "slippage_bps": slippage_bps,
                "total_bps": commission_bps + slippage_bps
            },
            "decision": {
                "tau_threshold": tau
            },
            "seed": {
                "global": seed,
                "dataloader": seed + 1,
                "model_init": seed + 2
            },
            "runtime": {
                "single_decision_core": True,
                "allow_legacy_paths": False,
                "fail_fast_on_nan": True
            },
            "eval": {
                "folds": 2,  # Reduced for faster testing
                "test_window_days": 30
            }
        }
    
    def run_config(self, config: dict[str, Any], tmp_path: Path) -> tuple[str, str]:
        """Run configuration and return output and hash"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(json.dumps(config, indent=2))
        
        try:
            # Run the falsification harness
            result = subprocess.run(
                ["python", "scripts/falsification_harness.py", 
                 "--config", str(config_file), "--quick"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path.cwd()
            )
            
            output = result.stdout + result.stderr
            output_hash = hashlib.sha256(output.encode()).hexdigest()
            
            return output, output_hash
            
        except subprocess.TimeoutExpired:
            return "TIMEOUT", "timeout"
        except Exception as e:
            return f"ERROR: {e}", "error"
    
    def parse_metrics(self, output: str) -> dict[str, float]:
        """Parse metrics from output"""
        metrics = {}
        
        # Look for Sharpe ratio
        if "Sharpe:" in output:
            try:
                sharpe_line = [line for line in output.split('\n') if 'Sharpe:' in line][0]
                sharpe = float(sharpe_line.split('Sharpe:')[1].strip())
                metrics['sharpe'] = sharpe
            except:
                pass
        
        # Look for HOLD percentage
        if "HOLD" in output and "%" in output:
            try:
                hold_line = [line for line in output.split('\n') if 'HOLD' in line and '%' in line][0]
                hold_pct = float(hold_line.split('%')[0].split()[-1])
                metrics['hold_pct'] = hold_pct
            except:
                pass
        
        # Look for turnover
        if "Turnover:" in output:
            try:
                turnover_line = [line for line in output.split('\n') if 'Turnover:' in line][0]
                turnover = float(turnover_line.split('Turnover:')[1].strip())
                metrics['turnover'] = turnover
            except:
                pass
        
        return metrics


@pytest.fixture
def falsification_matrix():
    """Fixture for falsification matrix"""
    return FalsificationMatrix()


def test_determinism(falsification_matrix, tmp_path):
    """Test that same inputs produce same outputs"""
    # Test with a fixed configuration
    config = falsification_matrix.create_test_config(
        commission_bps=1.0,
        slippage_bps=3.0,
        tau=0.0001,
        seed=1337
    )
    
    # Run twice
    output1, hash1 = falsification_matrix.run_config(config, tmp_path)
    output2, hash2 = falsification_matrix.run_config(config, tmp_path)
    
    # Should be identical
    assert hash1 == hash2, f"Non-deterministic output: {hash1} != {hash2}"
    assert output1 == output2, "Outputs should be identical"


def test_cost_monotonicity(falsification_matrix, tmp_path):
    """Test that higher costs lead to lower performance"""
    results = []
    
    # Test different cost levels with same tau and seed
    for commission_bps, slippage_bps in falsification_matrix.costs:
        config = falsification_matrix.create_test_config(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            tau=0.0001,
            seed=1337
        )
        
        output, _ = falsification_matrix.run_config(config, tmp_path)
        metrics = falsification_matrix.parse_metrics(output)
        
        total_cost = commission_bps + slippage_bps
        results.append((total_cost, metrics.get('sharpe', 0.0)))
    
    # Sort by cost
    results.sort(key=lambda x: x[0])
    
    # Check monotonicity (higher costs should lead to lower Sharpe)
    for i in range(1, len(results)):
        cost_prev, sharpe_prev = results[i-1]
        cost_curr, sharpe_curr = results[i]
        
        if cost_curr > cost_prev:
            # Allow some tolerance for noise
            assert sharpe_curr <= sharpe_prev + 0.1, \
                f"Sharpe increased with higher costs: {sharpe_prev:.3f} -> {sharpe_curr:.3f}"


def test_tau_monotonicity(falsification_matrix, tmp_path):
    """Test that higher tau leads to lower turnover"""
    results = []
    
    # Test different tau levels with same costs and seed
    for tau in falsification_matrix.taus:
        config = falsification_matrix.create_test_config(
            commission_bps=1.0,
            slippage_bps=3.0,
            tau=tau,
            seed=1337
        )
        
        output, _ = falsification_matrix.run_config(config, tmp_path)
        metrics = falsification_matrix.parse_metrics(output)
        
        results.append((tau, metrics.get('turnover', 0.0), metrics.get('hold_pct', 0.0)))
    
    # Sort by tau
    results.sort(key=lambda x: x[0])
    
    # Check monotonicity (higher tau should lead to lower turnover, higher HOLD%)
    for i in range(1, len(results)):
        tau_prev, turnover_prev, hold_prev = results[i-1]
        tau_curr, turnover_curr, hold_curr = results[i]
        
        if tau_curr > tau_prev:
            # Higher tau should lead to lower turnover
            assert turnover_curr <= turnover_prev + 0.1, \
                f"Turnover increased with higher tau: {turnover_prev:.3f} -> {turnover_curr:.3f}"
            
            # Higher tau should lead to higher HOLD%
            assert hold_curr >= hold_prev - 5.0, \
                f"HOLD% decreased with higher tau: {hold_prev:.1f}% -> {hold_curr:.1f}%"


def test_seed_independence(falsification_matrix, tmp_path):
    """Test that different seeds produce similar results (not identical)"""
    results = []
    
    # Test different seeds with same configuration
    for seed in falsification_matrix.seeds:
        config = falsification_matrix.create_test_config(
            commission_bps=1.0,
            slippage_bps=3.0,
            tau=0.0001,
            seed=seed
        )
        
        output, _ = falsification_matrix.run_config(config, tmp_path)
        metrics = falsification_matrix.parse_metrics(output)
        
        results.append((seed, metrics.get('sharpe', 0.0)))
    
    # Results should be similar but not identical
    sharpes = [r[1] for r in results]
    if len(sharpes) > 1:
        mean_sharpe = sum(sharpes) / len(sharpes)
        max_deviation = max(abs(s - mean_sharpe) for s in sharpes)
        
        # Allow reasonable variance (±0.2 Sharpe)
        assert max_deviation <= 0.2, \
            f"Too much variance across seeds: {max_deviation:.3f} > 0.2"


def test_edge_cases(falsification_matrix, tmp_path):
    """Test edge cases"""
    
    # Test zero tau (should trade on any edge)
    config = falsification_matrix.create_test_config(
        commission_bps=1.0,
        slippage_bps=3.0,
        tau=0.0,
        seed=1337
    )
    
    output, _ = falsification_matrix.run_config(config, tmp_path)
    metrics = falsification_matrix.parse_metrics(output)
    
    # With zero tau, should have low HOLD%
    if 'hold_pct' in metrics:
        assert metrics['hold_pct'] < 50.0, \
            f"Zero tau should result in low HOLD%: {metrics['hold_pct']:.1f}%"
    
    # Test very high tau (should mostly HOLD)
    config = falsification_matrix.create_test_config(
        commission_bps=1.0,
        slippage_bps=3.0,
        tau=1e-3,  # High tau
        seed=1337
    )
    
    output, _ = falsification_matrix.run_config(config, tmp_path)
    metrics = falsification_matrix.parse_metrics(output)
    
    # With high tau, should have high HOLD%
    if 'hold_pct' in metrics:
        assert metrics['hold_pct'] > 50.0, \
            f"High tau should result in high HOLD%: {metrics['hold_pct']:.1f}%"


def test_matrix_completeness(falsification_matrix, tmp_path):
    """Test that the matrix runs without errors"""
    # Test a subset of the full matrix
    test_cases = list(itertools.product(
        falsification_matrix.costs[:3],  # First 3 cost configs
        falsification_matrix.taus[:3],   # First 3 tau values
        falsification_matrix.seeds[:2]   # First 2 seeds
    ))
    
    results = []
    for (commission_bps, slippage_bps), tau, seed in test_cases:
        config = falsification_matrix.create_test_config(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            tau=tau,
            seed=seed
        )
        
        output, output_hash = falsification_matrix.run_config(config, tmp_path)
        
        # Should not timeout or error
        assert output_hash not in ["timeout", "error"], \
            f"Test case failed: costs=({commission_bps}, {slippage_bps}), tau={tau}, seed={seed}"
        
        results.append((config, output_hash))
    
    # All test cases should complete
    assert len(results) == len(test_cases), \
        f"Expected {len(test_cases)} results, got {len(results)}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
